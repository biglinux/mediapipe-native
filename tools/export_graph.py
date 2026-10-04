#!/usr/bin/env python3
"""Turn a MediaPipe ONNX model into a plan the Rust executor can run.

Offline only. Nothing this script imports becomes a dependency of the Rust
crate or of anything that links it.

Writes two files into the output directory:

  graph.json   the operations in order, tensor shapes, and where each block of
               numbers lives inside the weight file
  weights.bin  those numbers, little-endian 32-bit floats, each block starting
               on a 64-byte boundary

Three things happen here rather than at run time, because they are properties
of the model and never change between frames.

*Layout.* ONNX stores images channel-plane by channel-plane. The executor wants
the channels of one pixel side by side, which is what makes a 1x1 convolution a
plain matrix multiply. Weights are transposed here once; the transposes the
model itself contains then describe a layout change that no longer happens, so
they become aliases and disappear.

*Activation folding.* Where a convolution is followed by a parametric ReLU that
nothing else reads, the slopes move into the convolution. The executor then
applies them to each output row while that row is still in the nearest cache,
instead of walking the whole tensor a second time.

*Buffer assignment.* Tensor lifetimes are worked out here and every tensor is
given a slot in a small pool of reusable buffers. The face mesh needs 45 MB if
each tensor keeps its own storage and 3.2 MB once they share.

Usage:
    export_graph.py MODEL.onnx OUTPUT_DIR [--source "note about provenance"]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

FORMAT = "mediapipe-native-plan/1"

# Transposes between the two layouts. The executor is NHWC throughout, so a
# node that only reorders axes describes work that no longer needs doing.
NHWC_TO_NCHW = [0, 3, 1, 2]
NCHW_TO_NHWC = [0, 2, 3, 1]

# Single-file plan layout, mirrored by Plan::from_container in src/plan.rs.
MAGIC = b"MPNPLAN1"


def container(description: bytes, weights: bytes) -> bytes:
    """Pack the description and the weights into one .mpplan file.

    Magic, then the description length as a 32-bit little-endian count, then
    the description, then zero padding up to the next 64-byte boundary, then
    the weights. The padding is what lets the reader use the weights in place
    instead of copying them to an aligned buffer first.
    """
    head = MAGIC + len(description).to_bytes(4, "little") + description
    head += b"\0" * (-len(head) % 64)
    return head + weights


class Exporter:
    def __init__(self, model: onnx.ModelProto) -> None:
        inferred = onnx.shape_inference.infer_shapes(model)
        self.graph = model.graph
        self.const = {t.name: numpy_helper.to_array(t) for t in self.graph.initializer}
        self.shape = {}
        for value in (
            list(inferred.graph.input)
            + list(inferred.graph.value_info)
            + list(inferred.graph.output)
        ):
            dims = [d.dim_value for d in value.type.tensor_type.shape.dim]
            if dims:
                dims[0] = 1  # the batch axis is symbolic in these exports
            self.shape[value.name] = dims

        self.blob = bytearray()
        self.ops: list[dict] = []
        self.alias: dict[str, str] = {}  # tensor -> the tensor it shares storage with
        self.size: dict[str, int] = {}  # root tensor -> element count
        self.hwc: dict[str, tuple[int, int, int]] = {}

    # ── weight blob ─────────────────────────────────────────────────────────

    def put(self, array: np.ndarray) -> list[int]:
        """Append an array; return its [start, length] in floats."""
        while len(self.blob) % 64:
            self.blob.append(0)
        start = len(self.blob) // 4
        data = np.ascontiguousarray(array, dtype=np.float32)
        if not np.isfinite(data).all():
            raise SystemExit("refusing to export a weight that is not a finite number")
        self.blob.extend(data.tobytes())
        return [start, data.size]

    # ── tensors ─────────────────────────────────────────────────────────────

    def root(self, name: str) -> str:
        while name in self.alias:
            name = self.alias[name]
        return name

    def note(self, name: str, h: int, w: int, c: int) -> None:
        self.hwc[name] = (h, w, c)
        key = self.root(name)
        self.size[key] = max(self.size.get(key, 0), h * w * c)

    def note_nchw(self, name: str) -> tuple[int, int, int]:
        """Record an ONNX NCHW tensor by its NHWC extent."""
        _, c, h, w = self.shape[name]
        self.note(name, h, w, c)
        return h, w, c

    def dims(self, name: str) -> tuple[int, int, int]:
        return self.hwc[name]

    def flat(self, name: str) -> int:
        h, w, c = self.hwc[name]
        return h * w * c

    def alias_to(self, out: str, src: str, count: int | None = None) -> None:
        self.alias[out] = src
        if count is None:
            self.hwc[out] = self.hwc[src]
        else:
            self.hwc[out] = (1, 1, count)
        key = self.root(out)
        self.size[key] = max(self.size.get(key, 0), self.flat(out))

    # ── node translation ────────────────────────────────────────────────────

    def attrs(self, node) -> dict:
        return {a.name: (list(a.ints) if len(a.ints) else a.i) for a in node.attribute}

    def convert(self) -> None:
        source = self.graph.input[0].name
        _, h, w, c = self.shape[source]  # the model boundary is already NHWC
        self.note(source, h, w, c)

        for node in self.graph.node:
            handler = getattr(self, f"on_{node.op_type.lower()}", None)
            if handler is None:
                raise SystemExit(f"no rule for ONNX operation {node.op_type!r}")
            handler(node)

    def on_transpose(self, node) -> None:
        perm = self.attrs(node)["perm"]
        if perm not in (NHWC_TO_NCHW, NCHW_TO_NHWC):
            raise SystemExit(f"transpose {perm} reorders more than the layout")
        self.alias_to(node.output[0], node.input[0])

    def on_reshape(self, node) -> None:
        target = self.shape[node.output[0]]
        count = int(np.prod([d for d in target if d]))
        if count != self.flat(node.input[0]):
            raise SystemExit("reshape changes the element count")
        self.alias_to(node.output[0], node.input[0], count)

    def on_conv(self, node) -> None:
        weight = self.const[node.input[1]]
        bias = self.const[node.input[2]] if len(node.input) > 2 else None
        out_channels, in_per_group, kh, kw = weight.shape
        a = self.attrs(node)
        group = a.get("group", 1)
        sh, sw = a.get("strides", [1, 1])
        pt, pl, _pb, _pr = a.get("pads", [0, 0, 0, 0])
        oh, ow, oc = self.note_nchw(node.output[0])
        ih, iw, ic = self.dims(node.input[0])
        depthwise = group > 1 and in_per_group == 1

        if depthwise:
            if not group == ic == out_channels:
                raise SystemExit(
                    "grouped convolutions that are not depthwise are unsupported"
                )
            packed = weight.reshape(out_channels, kh, kw).transpose(1, 2, 0)
        else:
            if group != 1:
                raise SystemExit(
                    "grouped convolutions that are not depthwise are unsupported"
                )
            packed = weight.transpose(2, 3, 1, 0)  # [kh][kw][in][out]

        self.emit(
            "dwconv" if depthwise else "conv",
            node.input[0],
            node.output[0],
            kh=kh,
            kw=kw,
            sh=sh,
            sw=sw,
            pt=pt,
            pl=pl,
            ih=ih,
            iw=iw,
            ic=ic,
            oh=oh,
            ow=ow,
            oc=oc,
            w=self.put(packed),
            b=self.put(bias) if bias is not None else [0, 0],
        )

    def on_prelu(self, node) -> None:
        oh, ow, oc = self.note_nchw(node.output[0])
        ih, iw, ic = self.dims(node.input[0])
        self.emit(
            "prelu",
            node.input[0],
            node.output[0],
            ih=ih,
            iw=iw,
            ic=ic,
            oh=oh,
            ow=ow,
            oc=oc,
            w=self.put(self.const[node.input[1]].reshape(-1)),
        )

    def on_relu(self, node) -> None:
        oh, ow, oc = self.note_nchw(node.output[0])
        ih, iw, ic = self.dims(node.input[0])
        self.emit(
            "relu",
            node.input[0],
            node.output[0],
            ih=ih,
            iw=iw,
            ic=ic,
            oh=oh,
            ow=ow,
            oc=oc,
        )

    def on_add(self, node) -> None:
        oh, ow, oc = self.note_nchw(node.output[0])
        ih, iw, ic = self.dims(node.input[0])
        self.emit(
            "add",
            node.input[0],
            node.output[0],
            second=node.input[1],
            ih=ih,
            iw=iw,
            ic=ic,
            oh=oh,
            ow=ow,
            oc=oc,
        )

    def on_maxpool(self, node) -> None:
        a = self.attrs(node)
        if any(p for p in a.get("pads", [0, 0, 0, 0])):
            raise SystemExit("padded max pooling is unsupported")
        kh, kw = a["kernel_shape"]
        sh, sw = a["strides"]
        oh, ow, oc = self.note_nchw(node.output[0])
        ih, iw, ic = self.dims(node.input[0])
        self.emit(
            "maxpool",
            node.input[0],
            node.output[0],
            kh=kh,
            kw=kw,
            sh=sh,
            sw=sw,
            ih=ih,
            iw=iw,
            ic=ic,
            oh=oh,
            ow=ow,
            oc=oc,
        )

    def on_pad(self, node) -> None:
        pads = [int(v) for v in self.const[node.input[1]]]
        begin, end = pads[:4], pads[4:]
        if any(begin) or end[0] or end[2] or end[3] or end[1] <= 0:
            raise SystemExit(f"only trailing channel padding is supported, got {pads}")
        oh, ow, oc = self.note_nchw(node.output[0])
        ih, iw, ic = self.dims(node.input[0])
        self.emit(
            "padc",
            node.input[0],
            node.output[0],
            ih=ih,
            iw=iw,
            ic=ic,
            oh=oh,
            ow=ow,
            oc=oc,
        )

    def on_sigmoid(self, node) -> None:
        oh, ow, oc = self.note_nchw(node.output[0])
        ih, iw, ic = self.dims(node.input[0])
        self.emit(
            "sigmoid",
            node.input[0],
            node.output[0],
            ih=ih,
            iw=iw,
            ic=ic,
            oh=oh,
            ow=ow,
            oc=oc,
        )

    def on_concat(self, node) -> None:
        a = self.attrs(node)
        target = self.shape[node.output[0]]
        if len(node.input) != 2:
            raise SystemExit("only two-way concatenation is supported")
        if a.get("axis") != 1 or len(target) != 3:
            raise SystemExit(
                "concatenation is only supported on the leading axis of a [1,N,C]"
            )
        count = int(np.prod([d for d in target if d]))
        self.note(node.output[0], 1, 1, count)
        ih, iw, ic = self.dims(node.input[0])
        self.emit(
            "concat",
            node.input[0],
            node.output[0],
            second=node.input[1],
            ih=ih,
            iw=iw,
            ic=ic,
            oh=1,
            ow=1,
            oc=count,
            n2=self.flat(node.input[1]),
        )

    def emit(
        self, kind: str, x: str, y: str, second: str | None = None, **fields
    ) -> None:
        op = {"kind": kind, "_x": x, "_y": y, **fields}
        if second is not None:
            op["_x2"] = second
        self.ops.append(op)

    # ── graph-level passes ──────────────────────────────────────────────────

    def fold_activations(self) -> None:
        """Move a parametric ReLU into the operation that feeds it."""
        readers: dict[str, int] = {}
        for op in self.ops:
            for key in ("_x", "_x2"):
                if key in op:
                    readers[op[key]] = readers.get(op[key], 0) + 1

        # The producer need not be the previous operation: TFLite graphs can
        # list another branch's operations between a convolution and its PReLU.
        producers = {op["_y"]: op for op in self.ops}
        folded = []
        for op in self.ops:
            src = producers.get(op["_x"]) if op["kind"] == "prelu" else None
            if (
                src is not None
                and src["kind"] in ("conv", "dwconv", "add")
                and "act" not in src
                and readers.get(op["_x"], 0) == 1
            ):
                src["act"] = op["w"]
                src["_y"] = op["_y"]
                continue
            folded.append(op)
        self.ops = folded

    def assign_buffers(self) -> tuple[list[int], dict[str, int]]:
        """Give every tensor a slot in a pool of reusable buffers.

        A slot is handed out before the operation's inputs are released, so an
        operation never writes a buffer it is reading. The executor relies on
        that, and re-checks it when the plan is loaded.
        """
        for op in self.ops:
            for key in ("_x", "_x2", "_y"):
                if key in op:
                    op[key] = self.root(op[key])

        last: dict[str, int] = {}
        for index, op in enumerate(self.ops):
            for key in ("_x", "_x2", "_y"):
                if key in op:
                    last[op[key]] = index
        results = [self.root(o.name) for o in self.graph.output]
        for name in results:
            last[name] = len(self.ops)

        pool: list[int] = []
        free: list[int] = []
        slot: dict[str, int] = {}
        live: dict[str, int] = {}

        source = self.root(self.graph.input[0].name)
        slot[source] = 0
        pool.append(self.size[source])
        live[source] = 0

        for index, op in enumerate(self.ops):
            need = self.size[op["_y"]]
            if op["_y"] in slot:
                chosen = slot[op["_y"]]  # an alias already placed this tensor
                pool[chosen] = max(pool[chosen], need)
            elif free:
                roomy = [s for s in free if pool[s] >= need]
                chosen = min(roomy or free, key=lambda s: pool[s])
                free.remove(chosen)
                pool[chosen] = max(pool[chosen], need)
            else:
                chosen = len(pool)
                pool.append(need)
            slot[op["_y"]] = chosen
            live[op["_y"]] = chosen
            for key in ("_x", "_x2"):
                name = op.get(key)
                if name is not None and last[name] == index and name in live:
                    released = live.pop(name)
                    if released not in free:
                        free.append(released)
        return pool, slot

    def write(self, out_dir: Path, name: str, source_note: str) -> dict:
        self.convert()
        self.fold_activations()
        pool, slot = self.assign_buffers()

        for op in self.ops:
            op["x"] = slot[op.pop("_x")]
            op["y"] = slot[op.pop("_y")]
            if "_x2" in op:
                op["x2"] = slot[op.pop("_x2")]

        source = self.root(self.graph.input[0].name)
        _, h, w, c = self.shape[self.graph.input[0].name]
        plan = {
            "format": FORMAT,
            "source": source_note,
            "input": {"slot": slot[source], "h": h, "w": w, "c": c},
            "outputs": [
                {
                    "slot": slot[self.root(o.name)],
                    "n": self.flat(self.root(o.name)),
                    "name": o.name,
                }
                for o in self.graph.output
            ],
            "pool": pool,
            "ops": self.ops,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        description = json.dumps(plan).encode()
        (out_dir / "graph.json").write_bytes(description)
        (out_dir / "weights.bin").write_bytes(bytes(self.blob))
        (out_dir / f"{name}.mpplan").write_bytes(
            container(description, bytes(self.blob))
        )
        return plan


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model", type=Path, help="MediaPipe model in ONNX form")
    parser.add_argument("out_dir", type=Path, help="directory to write the plan into")
    parser.add_argument(
        "--source", default="", help="note recorded in the plan for provenance"
    )
    args = parser.parse_args()

    exporter = Exporter(onnx.load(args.model))
    plan = exporter.write(args.out_dir, args.model.stem, args.source)

    scratch = sum(plan["pool"]) * 4
    separate = sum(exporter.size.values()) * 4
    print(f"plan         {args.out_dir / (args.model.stem + '.mpplan')}")
    print(f"operations   {len(plan['ops'])}")
    print(f"weights      {len(exporter.blob) / 1e6:.2f} MB")
    print(
        f"scratch      {scratch / 1e6:.2f} MB in {len(plan['pool'])} buffers"
        f"  (separate storage would need {separate / 1e6:.1f} MB)"
    )
    for out in plan["outputs"]:
        print(f"output       {out['n']:>6} values  {out['name']}")


if __name__ == "__main__":
    main()
