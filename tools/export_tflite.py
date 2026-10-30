#!/usr/bin/env python3
"""Turn a MediaPipe TFLite model into a plan the Rust executor runs.

Offline, development only, like export_graph.py whose tensor bookkeeping,
PReLU folding, buffer assignment and file layout this reuses unchanged. Only
the graph reading differs:

*Layout.* TFLite is already NHWC, so reshapes are pure aliases and filters only
need reordering to [kh][kw][in][out] (dense) or [kh][kw][c] (depthwise).

*Constants.* The bundles store weights as float16 behind DEQUANTIZE and some
as sparse tensors behind DENSIFY. TensorFlow's own interpreter materialises
both once; the exact float32 values it computes are what gets exported, so
nothing here re-implements sparse decoding or rounds a weight.

*Folding.* A spatial PAD whose only readers are convolutions becomes their top
and left padding (bottom and right fall out of the output size). A per-channel
constant MUL is a 1x1 depthwise convolution. FULLY_CONNECTED is a 1x1
convolution on a one-pixel image, and a spatial MEAN a whole-image average
pool. RELU6 becomes an explicit clamp to [0, 6].

*Format 4.* HARD_SWISH and a MUL of a feature map by a computed per-channel
vector (squeeze-and-excite) need plan format 4, which older executors refuse.
There a RELU, HARD_SWISH or LOGISTIC that is the only reader of its input is
folded into the convolution or addition producing it, also across a
DEPTH_TO_SPACE. MediaPipe's Convolution2DTransposeBias, with kernel equal to
stride, becomes a 1x1 convolution to stride^2 times the channels followed by
DEPTH_TO_SPACE.

Anything else is refused by name rather than approximated.

Usage:
    export_tflite.py MODEL.tflite OUTPUT_DIR [--format 1|3|4] [--source "note about provenance"]

The format defaults to 3, or 4 when the model needs it. `--format 1` stamps
the original plan format, whose schedule the face plans' reference outputs
pin; the executor refuses a format-1 plan holding a newer operation.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import flatbuffers
import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as S

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_graph

BUILTIN = {v: k for k, v in vars(S.BuiltinOperator).items() if not k.startswith("_")}
ACTIVATION = {
    v: k for k, v in vars(S.ActivationFunctionType).items() if not k.startswith("_")
}
CONSTANT_PRODUCERS = ("DEQUANTIZE", "DENSIFY")
# Operations only plan format 4 holds.
FORMAT4_KINDS = ("hardswish", "channelmul")
# TfLitePadding, as MediaPipe's custom operations store it in raw options.
TFLITE_PADDING_SAME, TFLITE_PADDING_VALID = 1, 2


def options(op, cls):
    table = op.BuiltinOptions()
    value = cls()
    value.Init(table.Bytes, table.Pos)
    return value


def transpose_conv_options(raw: bytes) -> tuple[int, int, int]:
    """Padding, stride_w and stride_h of a raw TfLiteTransposeConvParams."""
    if len(raw) != 12:
        raise SystemExit(f"transposed convolution options are {len(raw)} bytes")
    return struct.unpack("<3i", raw)


def runnable(raw: bytes) -> bytes:
    """The model with MediaPipe's Convolution2DTransposeBias replaced by
    TFLite's builtin TRANSPOSE_CONV (version 3 takes a bias), so TensorFlow's
    interpreter can run it. MediaPipe's kernel is TFLite's reference
    transposed convolution seeded with the bias, with the same padding rule
    for SAME and VALID. Existing tensor and buffer indices are kept."""
    model = S.ModelT.InitFromPackedBuf(raw, 0)
    custom = {
        k
        for k, code in enumerate(model.operatorCodes)
        if code.customCode == b"Convolution2DTransposeBias"
    }
    if not custom:
        return raw
    code = S.OperatorCodeT()
    code.builtinCode = code.deprecatedBuiltinCode = S.BuiltinOperator.TRANSPOSE_CONV
    code.version = 3
    model.operatorCodes.append(code)
    for sub in model.subgraphs:
        for op in sub.operators:
            if op.opcodeIndex not in custom:
                continue
            padding, sw, sh = transpose_conv_options(bytes(op.customOptions or b""))
            x, weight, bias = op.inputs
            shape = S.BufferT()
            shape.data = list(
                np.array(sub.tensors[op.outputs[0]].shape, "<i4").tobytes()
            )
            model.buffers.append(shape)
            tensor = S.TensorT()
            tensor.shape, tensor.type = [4], S.TensorType.INT32
            tensor.buffer, tensor.name = len(model.buffers) - 1, b"output_shape"
            sub.tensors.append(tensor)
            builtin = S.TransposeConvOptionsT()
            builtin.padding = {
                TFLITE_PADDING_SAME: S.Padding.SAME,
                TFLITE_PADDING_VALID: S.Padding.VALID,
            }[padding]
            builtin.strideW, builtin.strideH = sw, sh
            op.opcodeIndex = len(model.operatorCodes) - 1
            op.inputs = [len(sub.tensors) - 1, weight, x, bias]
            op.builtinOptionsType = S.BuiltinOptions.TransposeConvOptions
            op.builtinOptions, op.customOptions = builtin, None
    builder = flatbuffers.Builder(2 * len(raw))
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    return bytes(builder.Output())


def blocks_as_pointwise(
    weights: np.ndarray, bias: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """A transposed convolution whose kernel equals its stride, as a 1x1
    convolution: `weights` is [out][kh][kw][in], the result [in][kh*kw*out]
    with column (i * kw + j) * out + o for output o at offset (i, j) of the
    block, the channel order DEPTH_TO_SPACE then spreads into place."""
    out, kh, kw, ic = weights.shape
    packed = weights.transpose(3, 1, 2, 0).reshape(ic, kh * kw * out)
    return packed, np.tile(bias, kh * kw)


def same_padding(size: int, out: int, stride: int, kernel: int) -> int:
    """TFLite SAME: the smaller half of the total padding goes first."""
    return max((out - 1) * stride + kernel - size, 0) // 2


class TfliteExporter(export_graph.Exporter):
    def __init__(self, path: Path, plan_format: int | None = None) -> None:
        # Overrides Exporter's ONNX reader. `plan_format` None picks 3, or 4
        # if the model needs it.
        self.format = plan_format
        raw = path.read_bytes()
        self.model = S.Model.GetRootAsModel(raw, 0)
        if self.model.SubgraphsLength() != 1:
            raise SystemExit("only single-subgraph models are supported")
        self.sub = self.model.Subgraphs(0)
        if self.sub.InputsLength() != 1:
            raise SystemExit("only single-input models are supported")

        # Without the default XNNPACK delegate: it consumes the dequantized
        # weights internally and leaves those tensors unwritten.
        interpreter = tf.lite.Interpreter(
            model_content=runnable(raw),
            experimental_preserve_all_tensors=True,
            num_threads=1,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        )
        interpreter.allocate_tensors()
        interpreter.invoke()  # constants do not depend on the (zero) input
        self.interpreter = interpreter

        self.names = [
            self.tensor(i).Name().decode() for i in range(self.sub.TensorsLength())
        ]
        if len(set(self.names)) != len(self.names):
            self.names = [f"{n}#{i}" for i, n in enumerate(self.names)]
        self.readers: dict[int, list[int]] = {}
        self.constant: set[int] = set()
        for k in range(self.sub.OperatorsLength()):
            op = self.sub.Operators(k)
            for i in self.inputs(op):
                self.readers.setdefault(i, []).append(k)
        for i in range(self.sub.TensorsLength()):
            if self.model.Buffers(self.tensor(i).Buffer()).DataLength():
                self.constant.add(i)
        for k in range(self.sub.OperatorsLength()):
            op = self.sub.Operators(k)
            if self.kind(op) in CONSTANT_PRODUCERS:
                if not all(i in self.constant for i in self.inputs(op)):
                    raise SystemExit(f"op {k} dequantizes a non-constant tensor")
                self.constant.update(self.outputs(op))

        source = self.sub.Inputs(0)
        outputs = [self.sub.Outputs(k) for k in range(self.sub.OutputsLength())]
        self.graph = SimpleNamespace(
            input=[SimpleNamespace(name=self.names[source])],
            output=[SimpleNamespace(name=self.names[i]) for i in outputs],
        )
        self.shape = {self.names[source]: [1, *self.hwc_of(source)]}
        self.blob = bytearray()
        self.ops: list[dict] = []
        self.alias: dict[str, str] = {}
        self.size: dict[str, int] = {}
        self.hwc: dict[str, tuple[int, int, int]] = {}
        self.spatial_pad: dict[
            int, tuple[int, int, int]
        ] = {}  # output -> (input, top, left)

    # ── reading the flatbuffer ──────────────────────────────────────────────

    def tensor(self, i: int):
        return self.sub.Tensors(i)

    def inputs(self, op) -> list[int]:
        return [op.Inputs(j) for j in range(op.InputsLength()) if op.Inputs(j) >= 0]

    def outputs(self, op) -> list[int]:
        return [op.Outputs(j) for j in range(op.OutputsLength())]

    def kind(self, op) -> str:
        code = self.model.OperatorCodes(op.OpcodeIndex())
        return BUILTIN[max(code.BuiltinCode(), code.DeprecatedBuiltinCode())]

    def dims(self, i: int) -> list[int]:
        return (
            self.tensor(i).ShapeAsNumpy().tolist()
            if self.tensor(i).ShapeLength()
            else []
        )

    def dims4(self, i: int) -> list[int]:
        d = self.dims(i)
        if len(d) != 4 or d[0] != 1:
            raise SystemExit(f"tensor {self.names[i]!r} is {d}, expected [1, h, w, c]")
        return d

    def value(self, i: int) -> np.ndarray:
        if i not in self.constant:
            raise SystemExit(f"tensor {self.names[i]!r} should be a constant")
        return np.asarray(self.interpreter.get_tensor(i))

    def hwc_of(self, i: int) -> tuple[int, int, int]:
        """NHWC extent of a tensor: [1, n, c] is one row of n points, lower
        ranks are one pixel."""
        d = self.dims(i)
        if not d or d[0] != 1:
            raise SystemExit(f"tensor {self.names[i]!r} has shape {d}")
        if len(d) == 4:
            return d[1], d[2], d[3]
        if len(d) == 3:
            return 1, d[1], d[2]
        return 1, 1, int(np.prod(d[1:]))

    def record(self, i: int) -> tuple[int, int, int]:
        h, w, c = self.hwc_of(i)
        self.note(self.names[i], h, w, c)
        return h, w, c

    # ── translation ─────────────────────────────────────────────────────────

    def convert(self) -> None:
        source = self.sub.Inputs(0)
        self.record(source)
        # Graph operations in order, without the constant producers; a rule
        # that recognises a longer pattern consumes the ones after it.
        self.order = [
            k
            for k in range(self.sub.OperatorsLength())
            if self.kind(self.sub.Operators(k)) not in CONSTANT_PRODUCERS
        ]
        self.consumed: set[int] = set()
        for at, k in enumerate(self.order):
            if k in self.consumed:
                continue
            self.at = at
            op = self.sub.Operators(k)
            kind = self.kind(op)
            # Only this class's rules: the ONNX ones it inherits read ONNX nodes.
            handler = TfliteExporter.__dict__.get(f"on_{kind.lower()}")
            if handler is None:
                raise SystemExit(f"op {k}: no rule for TFLite operation {kind}")
            handler(self, op)
        for pad in self.spatial_pad:
            if pad in {self.sub.Outputs(k) for k in range(self.sub.OutputsLength())}:
                raise SystemExit("a spatially padded tensor is a model output")
        needs4 = any(op["kind"] in FORMAT4_KINDS for op in self.ops)
        if self.format is None:
            self.format = 4 if needs4 else 3
        elif needs4 and self.format < 4:
            raise SystemExit("this model needs plan format 4")
        # write() stamps the module's format tag after convert().
        export_graph.FORMAT = f"mediapipe-native-plan/{self.format}"
        self.fold_into_producers()

    def fold_into_producers(self) -> None:
        """Move a clamp into the convolution or addition that feeds it, the
        way export_graph folds PReLU slopes, so it runs on rows still in cache.
        In format 4 a ReLU, hard-swish or sigmoid moves there too, also across
        a DEPTH_TO_SPACE: it only moves values, so it commutes with them."""
        readers: dict[str, int] = {}
        for op in self.ops:
            for key in ("_x", "_x2"):
                if key in op:
                    readers[op[key]] = readers.get(op[key], 0) + 1
        results = {o.name for o in self.graph.output}
        # As for PReLU, the producer need not be the previous operation.
        producers = {op["_y"]: op for op in self.ops}
        folded = []
        if self.format >= 4:
            foldable = ("clip", "relu", "hardswish", "sigmoid")
        else:
            foldable = ("clip",)

        def private(name: str) -> bool:
            return readers.get(name) == 1 and name not in results

        for op in self.ops:
            src = producers.get(op["_x"]) if op["kind"] in foldable else None
            renamed = src
            if (
                self.format >= 4
                and src is not None
                and src["kind"] == "depthtospace"
                and private(src["_x"])
            ):
                src = producers.get(src["_x"])
            if (
                src is not None
                and src["kind"] in ("conv", "dwconv", "add")
                and not {"act", "clip", "fused"} & src.keys()
                and private(op["_x"])
            ):
                if op["kind"] == "clip":
                    src["clip"] = op["w"]
                else:
                    src["fused"] = op["kind"]
                renamed["_y"] = op["_y"]
                continue
            folded.append(op)
        self.ops = folded

    def following(self, kinds: list[str]):
        """The operations right after the current one, if they are `kinds` in
        that order; they are then marked consumed. None otherwise."""
        ahead = self.order[self.at + 1 : self.at + 1 + len(kinds)]
        ops = [self.sub.Operators(k) for k in ahead]
        if len(ops) != len(kinds) or [self.kind(o) for o in ops] != kinds:
            return None
        self.consumed.update(ahead)
        return ops

    def reduces(self, op, axis: int) -> bool:
        """A keep-dims MEAN or SUM over exactly `axis` (negative from the end)."""
        x, axes = self.inputs(op)
        rank = len(self.dims(x))
        want = sorted(
            a % rank for a in np.asarray(self.value(axes)).reshape(-1).tolist()
        )
        return want == [axis % rank] and options(op, S.ReducerOptions).KeepDims()

    def activate(self, op, cls, out: int) -> str:
        """Name for the operation's own result, adding the fused activation."""
        act = ACTIVATION[options(op, cls).FusedActivationFunction()]
        name = self.names[out]
        if act == "NONE":
            return name
        raw = f"{name}#pre"
        h, w, c = self.hwc_of(out)
        self.note(raw, h, w, c)
        if act == "RELU":
            self.emit("relu", raw, name, ih=h, iw=w, ic=c, oh=h, ow=w, oc=c)
        elif act == "RELU6":
            self.emit(
                "clip",
                raw,
                name,
                ih=h,
                iw=w,
                ic=c,
                oh=h,
                ow=w,
                oc=c,
                w=self.put(np.array([0.0, 6.0], dtype=np.float32)),
            )
        else:
            raise SystemExit(f"fused activation {act} is unsupported")
        return raw

    def padded_source(self, i: int) -> tuple[int, int, int]:
        """(tensor actually read, extra top rows, extra left columns)."""
        return self.spatial_pad.get(i, (i, 0, 0))

    def convolution(self, op, depthwise: bool) -> None:
        x, weight, *rest = self.inputs(op)
        out = self.outputs(op)[0]
        cls = S.DepthwiseConv2DOptions if depthwise else S.Conv2DOptions
        o = options(op, cls)
        if o.DilationHFactor() != 1 or o.DilationWFactor() != 1:
            raise SystemExit("dilated convolutions are unsupported")
        w = self.value(weight).astype(np.float32)
        bias = self.value(rest[0]).astype(np.float32) if rest else None
        src, top, left = self.padded_source(x)
        ih, iw, ic = self.hwc_of(src)
        _, xh, xw, _ = self.dims4(x)
        oh, ow, oc = self.record(out)
        if depthwise:
            if o.DepthMultiplier() != 1 or w.shape[3] != ic or oc != ic:
                raise SystemExit("depthwise multipliers other than one are unsupported")
            _, kh, kw, _ = w.shape
            packed = w.reshape(kh, kw, oc)
        else:
            _, kh, kw, wi = w.shape
            if wi != ic:
                raise SystemExit("grouped convolutions are unsupported")
            packed = w.transpose(1, 2, 3, 0)  # [kh][kw][in][out]
        sh, sw = o.StrideH(), o.StrideW()
        if o.Padding() == S.Padding.SAME:
            top += same_padding(xh, oh, sh, kh)
            left += same_padding(xw, ow, sw, kw)
        y = self.activate(op, cls, out)
        self.emit(
            "dwconv" if depthwise else "conv",
            self.names[src],
            y,
            kh=kh,
            kw=kw,
            sh=sh,
            sw=sw,
            pt=top,
            pl=left,
            ih=ih,
            iw=iw,
            ic=ic,
            oh=oh,
            ow=ow,
            oc=oc,
            w=self.put(packed),
            b=self.put(bias) if bias is not None else [0, 0],
        )
        if y != self.names[out]:
            self.ops.append(self.ops.pop(-2))  # the activation runs after its producer

    def on_conv_2d(self, op) -> None:
        self.convolution(op, depthwise=False)

    def on_depthwise_conv_2d(self, op) -> None:
        self.convolution(op, depthwise=True)

    def on_fully_connected(self, op) -> None:
        x, weight, *rest = self.inputs(op)
        out = self.outputs(op)[0]
        w = self.value(weight).astype(np.float32)  # [out][in]
        bias = self.value(rest[0]).astype(np.float32) if rest else None
        ih, iw, ic = self.hwc_of(x)
        if (ih, iw) != (1, 1) or w.shape[1] != ic:
            raise SystemExit("fully connected input must be one flat vector")
        _, _, oc = self.record(out)
        y = self.activate(op, S.FullyConnectedOptions, out)
        self.emit(
            "conv",
            self.names[x],
            y,
            kh=1,
            kw=1,
            sh=1,
            sw=1,
            pt=0,
            pl=0,
            ih=1,
            iw=1,
            ic=ic,
            oh=1,
            ow=1,
            oc=oc,
            w=self.put(w.T),
            b=self.put(bias) if bias is not None else [0, 0],
        )
        if y != self.names[out]:
            self.ops.append(self.ops.pop(-2))

    def on_add(self, op) -> None:
        a, b = self.inputs(op)
        out = self.outputs(op)[0]
        if a in self.constant:
            a, b = b, a
        if b in self.constant:
            self.add_constant(op, a, b, out)
            return
        if self.hwc_of(a) != self.hwc_of(b):
            raise SystemExit(
                "only additions of two equally shaped tensors are supported"
            )
        h, w, c = self.record(out)
        y = self.activate(op, S.AddOptions, out)
        self.emit(
            "add",
            self.names[a],
            y,
            second=self.names[b],
            ih=h,
            iw=w,
            ic=c,
            oh=h,
            ow=w,
            oc=c,
        )
        if y != self.names[out]:
            self.ops.append(self.ops.pop(-2))

    def add_constant(self, op, x: int, constant: int, out: int) -> None:
        """Per-channel shift: the bias of the scale just before it, else of a
        1x1 depthwise identity. `t + x * s` is the value TFLite's `x * s + t`
        computes."""
        h, w, c = self.record(out)
        shift = self.value(constant).astype(np.float32).reshape(-1)
        if shift.size != c or self.hwc_of(x) != (h, w, c):
            raise SystemExit("only per-channel constant addition is supported")
        last = self.ops[-1] if self.ops else None
        outputs = {self.sub.Outputs(k) for k in range(self.sub.OutputsLength())}
        scale_only = (
            last is not None
            and last["kind"] == "dwconv"
            and last["kh"] == 1
            and last["b"] == [0, 0]
            and last["_y"] == self.names[x]
            and self.readers.get(x) == [self.readers[x][0]]
            and x not in outputs
        )
        y = self.activate(op, S.AddOptions, out)
        if scale_only:
            last["_y"], last["b"] = y, self.put(shift)
            return
        self.emit(
            "dwconv",
            self.names[x],
            y,
            kh=1,
            kw=1,
            sh=1,
            sw=1,
            pt=0,
            pl=0,
            ih=h,
            iw=w,
            ic=c,
            oh=h,
            ow=w,
            oc=c,
            w=self.put(np.ones(c, np.float32)),
            b=self.put(shift),
        )
        if y != self.names[out]:
            self.ops.append(self.ops.pop(-2))

    def on_mul(self, op) -> None:
        a, b = self.inputs(op)
        if a in self.constant:
            a, b = b, a
        out = self.outputs(op)[0]
        h, w, c = self.record(out)
        if b not in self.constant:
            return self.channel_mul(op, a, b, out)
        scale = self.value(b).astype(np.float32).reshape(-1)
        if scale.size == 1:
            scale = np.full(c, scale[0], np.float32)
        if scale.size != c or self.hwc_of(a) != (h, w, c):
            raise SystemExit("only per-channel constant multiplication is supported")
        y = self.activate(op, S.MulOptions, out)
        self.emit(
            "dwconv",
            self.names[a],
            y,
            kh=1,
            kw=1,
            sh=1,
            sw=1,
            pt=0,
            pl=0,
            ih=h,
            iw=w,
            ic=c,
            oh=h,
            ow=w,
            oc=c,
            w=self.put(scale),
            b=[0, 0],
        )
        if y != self.names[out]:
            self.ops.append(self.ops.pop(-2))

    def channel_mul(self, op, a: int, b: int, out: int) -> None:
        """A feature map times a computed vector of one value per channel,
        the gate of a squeeze-and-excite block."""
        if self.hwc_of(a)[:2] == (1, 1):
            a, b = b, a
        h, w, c = self.hwc_of(out)
        if self.hwc_of(a) != (h, w, c) or self.hwc_of(b) != (1, 1, c):
            raise SystemExit(
                "only a feature map times a per-channel vector is supported"
            )
        y = self.activate(op, S.MulOptions, out)
        self.emit(
            "channelmul",
            self.names[a],
            y,
            second=self.names[b],
            ih=h,
            iw=w,
            ic=c,
            oh=h,
            ow=w,
            oc=c,
        )
        if y != self.names[out]:
            self.ops.append(self.ops.pop(-2))

    def on_hard_swish(self, op) -> None:
        x, out = self.inputs(op)[0], self.outputs(op)[0]
        h, w, c = self.record(out)
        self.emit(
            "hardswish",
            self.names[x],
            self.names[out],
            ih=h,
            iw=w,
            ic=c,
            oh=h,
            ow=w,
            oc=c,
        )

    def on_custom(self, op) -> None:
        code = self.model.OperatorCodes(op.OpcodeIndex()).CustomCode().decode()
        if code != "Convolution2DTransposeBias":
            raise SystemExit(f"no rule for custom operation {code}")
        self.transpose_conv(op)

    def transpose_conv(self, op) -> None:
        """MediaPipe's transposed convolution with bias, for kernel == stride
        and no padding: every input pixel writes its own stride-by-stride
        block, which is a 1x1 convolution to stride^2 x out channels, then
        DEPTH_TO_SPACE. The options are a raw TfLiteTransposeConvParams:
        padding, stride_w, stride_h as 32-bit integers."""
        if len(self.inputs(op)) != 3:
            raise SystemExit("a transposed convolution takes input, weights and bias")
        x, weight, bias = self.inputs(op)
        out = self.outputs(op)[0]
        raw = op.CustomOptionsAsNumpy().tobytes() if op.CustomOptionsLength() else b""
        padding, sw, sh = transpose_conv_options(raw)
        w = self.value(weight).astype(np.float32)  # [out][kh][kw][in]
        b = self.value(bias).astype(np.float32).reshape(-1)
        if w.ndim != 4:
            raise SystemExit("transposed convolution weights must be [out, kh, kw, in]")
        _, ih, iw, ic = self.dims4(x)
        _, oh, ow, oc = self.dims4(out)
        self.record(out)
        _, kh, kw, wi = w.shape
        if padding not in (TFLITE_PADDING_SAME, TFLITE_PADDING_VALID):
            raise SystemExit(f"transposed convolution padding {padding} is unknown")
        # MediaPipe pads SAME by kh - (ih - 1) % sh - 1 (VALID by nothing):
        # with kernel == stride, zero exactly when the stride divides the input.
        same = padding == TFLITE_PADDING_SAME
        if (
            (kh, kw) != (sh, sw)
            or sh != sw
            or sh < 2
            or (same and (ih % sh or iw % sw))
            or (wi, w.shape[0], b.size) != (ic, oc, oc)
            or (oh, ow) != (ih * sh, iw * sw)
        ):
            raise SystemExit(
                "only transposed convolutions whose kernel equals the stride, "
                f"unpadded, are supported (kernel {kh}x{kw}, stride {sh}x{sw})"
            )
        packed, tiled = blocks_as_pointwise(w, b)
        blocks = f"{self.names[out]}#blocks"
        self.note(blocks, ih, iw, sh * sw * oc)
        self.emit(
            "conv",
            self.names[x],
            blocks,
            kh=1,
            kw=1,
            sh=1,
            sw=1,
            pt=0,
            pl=0,
            ih=ih,
            iw=iw,
            ic=ic,
            oh=ih,
            ow=iw,
            oc=sh * sw * oc,
            w=self.put(packed),
            b=self.put(tiled),
        )
        self.emit(
            "depthtospace",
            blocks,
            self.names[out],
            ih=ih,
            iw=iw,
            ic=sh * sw * oc,
            oh=oh,
            ow=ow,
            oc=oc,
        )

    def on_prelu(self, op) -> None:
        x, alpha = self.inputs(op)
        out = self.outputs(op)[0]
        h, w, c = self.record(out)
        slope = self.value(alpha).astype(np.float32).reshape(-1)
        if slope.size != c:
            raise SystemExit("PReLU slopes must be one per channel")
        self.emit(
            "prelu",
            self.names[x],
            self.names[out],
            ih=h,
            iw=w,
            ic=c,
            oh=h,
            ow=w,
            oc=c,
            w=self.put(slope),
        )

    def on_relu(self, op) -> None:
        x, out = self.inputs(op)[0], self.outputs(op)[0]
        h, w, c = self.record(out)
        self.emit(
            "relu", self.names[x], self.names[out], ih=h, iw=w, ic=c, oh=h, ow=w, oc=c
        )

    def on_logistic(self, op) -> None:
        x, out = self.inputs(op)[0], self.outputs(op)[0]
        h, w, c = self.record(out)
        self.emit(
            "sigmoid",
            self.names[x],
            self.names[out],
            ih=h,
            iw=w,
            ic=c,
            oh=h,
            ow=w,
            oc=c,
        )

    def on_max_pool_2d(self, op) -> None:
        self.pool(op, "maxpool")

    def on_average_pool_2d(self, op) -> None:
        self.pool(op, "avgpool")

    def pool(self, op, kind: str) -> None:
        """Pooling over windows that lie inside the image, as SAME pads none
        when the windows fit."""
        x, out = self.inputs(op)[0], self.outputs(op)[0]
        o = options(op, S.Pool2DOptions)
        if ACTIVATION[o.FusedActivationFunction()] != "NONE":
            raise SystemExit("pooling with a fused activation is unsupported")
        kh, kw, sh, sw = o.FilterHeight(), o.FilterWidth(), o.StrideH(), o.StrideW()
        ih, iw, ic = self.hwc_of(x)
        oh, ow, oc = self.record(out)
        if o.Padding() == S.Padding.SAME and (
            same_padding(ih, oh, sh, kh)
            or same_padding(iw, ow, sw, kw)
            or (oh - 1) * sh + kh > ih
            or (ow - 1) * sw + kw > iw
        ):
            raise SystemExit("padded pooling is unsupported")
        self.emit(
            kind,
            self.names[x],
            self.names[out],
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

    def on_pad(self, op) -> None:
        x, pads = self.inputs(op)
        out = self.outputs(op)[0]
        p = self.value(pads).astype(np.int64).reshape(4, 2)
        if (p < 0).any() or p[0].any():
            raise SystemExit(f"unsupported padding {p.tolist()}")
        if not p[1].any() and not p[2].any() and p[3, 0] == 0 and p[3, 1] > 0:
            ih, iw, ic = self.hwc_of(x)
            oh, ow, oc = self.record(out)
            self.emit(
                "padc",
                self.names[x],
                self.names[out],
                ih=ih,
                iw=iw,
                ic=ic,
                oh=oh,
                ow=ow,
                oc=oc,
            )
            return
        if p[3].any():
            raise SystemExit(f"mixed spatial and channel padding {p.tolist()}")
        readers = self.readers.get(out, [])
        if not readers or any(
            self.kind(self.sub.Operators(k)) not in ("CONV_2D", "DEPTHWISE_CONV_2D")
            or self.sub.Operators(k).Inputs(0) != out
            for k in readers
        ):
            raise SystemExit("spatial padding feeds something other than a convolution")
        src, top, left = self.padded_source(x)
        self.spatial_pad[out] = (src, top + int(p[1, 0]), left + int(p[2, 0]))

    def on_resize_bilinear(self, op) -> None:
        x, _size = self.inputs(op)
        out = self.outputs(op)[0]
        o = options(op, S.ResizeBilinearOptions)
        if o.AlignCorners() or not o.HalfPixelCenters():
            raise SystemExit("only half-pixel-centre bilinear resizing is supported")
        ih, iw, ic = self.hwc_of(x)
        oh, ow, oc = self.record(out)
        self.emit(
            "resize",
            self.names[x],
            self.names[out],
            ih=ih,
            iw=iw,
            ic=ic,
            oh=oh,
            ow=ow,
            oc=oc,
        )

    def on_depth_to_space(self, op) -> None:
        x, out = self.inputs(op)[0], self.outputs(op)[0]
        ih, iw, ic = self.hwc_of(x)
        oh, ow, oc = self.record(out)
        if options(op, S.DepthToSpaceOptions).BlockSize() != oh // ih:
            raise SystemExit("depth-to-space block does not match the shapes")
        self.emit(
            "depthtospace",
            self.names[x],
            self.names[out],
            ih=ih,
            iw=iw,
            ic=ic,
            oh=oh,
            ow=ow,
            oc=oc,
        )

    def on_strided_slice(self, op) -> None:
        x, begin, end, strides = self.inputs(op)
        out = self.outputs(op)[0]
        o = options(op, S.StridedSliceOptions)
        if o.EllipsisMask() or o.NewAxisMask():
            raise SystemExit("strided slice masks beyond begin/end are unsupported")
        shape = self.dims(x)
        rank = len(shape)
        b, e, s = (
            self.value(t).astype(np.int64).tolist() for t in (begin, end, strides)
        )
        lo = [0 if o.BeginMask() >> a & 1 else b[a] for a in range(rank)]
        hi = [
            shape[a] if o.EndMask() >> a & 1 else min(e[a], shape[a])
            for a in range(rank)
        ]
        if s != [1] * rank:
            raise SystemExit("only unit-stride slices are supported")
        # Leading whole axes of size one, then a prefix of one axis, then whole
        # axes: the first elements of the flat tensor, which is an alias.
        cut = next((a for a in range(rank) if (lo[a], hi[a]) != (0, shape[a])), rank)
        if cut == rank or (
            all(shape[a] == 1 for a in range(cut))
            and lo[cut] == 0
            and all((lo[a], hi[a]) == (0, shape[a]) for a in range(cut + 1, rank))
        ):
            count = int(np.prod(hi[cut:])) if cut < rank else int(np.prod(shape))
            self.alias_to(self.names[out], self.names[x], count)
            return
        if (
            o.ShrinkAxisMask()
            or rank != 4
            or any(lo[a] != 0 or hi[a] != shape[a] for a in range(3))
        ):
            raise SystemExit(
                "only unit-stride slices of the channel axis are supported"
            )
        ih, iw, ic = self.hwc_of(x)
        oh, ow, oc = self.record(out)
        if oc != hi[3] - lo[3]:
            raise SystemExit("slice output disagrees with its bounds")
        self.emit(
            "channelslice",
            self.names[x],
            self.names[out],
            ih=ih,
            iw=iw,
            ic=ic,
            oh=oh,
            ow=ow,
            oc=oc,
            begin=lo[3],
        )

    def on_mean(self, op) -> None:
        x, axes = self.inputs(op)
        out = self.outputs(op)[0]
        if len(self.dims(x)) == 3 and self.reduces(op, -2):
            return self.center_scale(op)
        if len(self.dims(x)) == 4 and self.reduces(op, -1):
            return self.layer_norm(op)
        if sorted(self.value(axes).reshape(-1).tolist()) != [1, 2]:
            raise SystemExit("only a spatial mean is supported")
        ih, iw, ic = self.hwc_of(x)
        _, _, oc = self.record(out)
        self.emit(
            "avgpool",
            self.names[x],
            self.names[out],
            kh=ih,
            kw=iw,
            sh=1,
            sw=1,
            ih=ih,
            iw=iw,
            ic=ic,
            oh=1,
            ow=1,
            oc=oc,
        )

    def center_scale(self, mean) -> None:
        """MEAN over points, SUB, MUL by itself, SUM over coordinates, SQRT,
        MEAN over points, DIV: points centred and divided by their mean
        distance from the centre."""
        ops = self.following(["SUB", "MUL", "SUM", "SQRT", "MEAN", "DIV"])
        if ops is None:
            raise SystemExit("a mean over points outside the landmark normalization")
        sub, square, total, root, spread, div = ops
        x, centre = self.inputs(mean)[0], self.outputs(mean)[0]
        d = self.outputs(sub)[0]
        wired = (
            self.inputs(sub) == [x, centre]
            and self.inputs(square) == [d, d]
            and self.inputs(total)[0] == self.outputs(square)[0]
            and self.reduces(total, -1)
            and self.inputs(root) == [self.outputs(total)[0]]
            and self.inputs(spread)[0] == self.outputs(root)[0]
            and self.reduces(spread, -2)
            and self.inputs(div) == [d, self.outputs(spread)[0]]
        )
        if not wired:
            raise SystemExit(
                "landmark normalization is wired differently than expected"
            )
        out = self.outputs(div)[0]
        h, w, c = self.record(out)
        self.emit(
            "centerscale",
            self.names[x],
            self.names[out],
            ih=h,
            iw=w,
            ic=c,
            oh=h,
            ow=w,
            oc=c,
        )

    def layer_norm(self, mean) -> None:
        """TFLite's expanded LayerNormalization without beta: m = MEAN(x),
        NEG(m), SQUARED_DIFFERENCE(x, m), MEAN, ADD eps, RSQRT, MUL gamma,
        then x * s + (-m) * s."""
        ops = self.following(
            [
                "NEG",
                "SQUARED_DIFFERENCE",
                "MEAN",
                "ADD",
                "RSQRT",
                "MUL",
                "MUL",
                "MUL",
                "ADD",
            ]
        )
        if ops is None:
            raise SystemExit("a channel mean outside a layer normalization")
        neg, sqdiff, var, add_eps, rsqrt, gain, xs, ms, total = ops
        x, m = self.inputs(mean)[0], self.outputs(mean)[0]
        nm, s = self.outputs(neg)[0], self.outputs(gain)[0]
        v, ve, r = (
            self.outputs(var)[0],
            self.outputs(add_eps)[0],
            self.outputs(rsqrt)[0],
        )
        eps = self.inputs(add_eps)[1]
        gamma = self.inputs(gain)[1]
        wired = (
            self.inputs(neg) == [m]
            and self.inputs(sqdiff) == [x, m]
            and self.inputs(var)[0] == self.outputs(sqdiff)[0]
            and self.reduces(var, -1)
            and self.inputs(add_eps)[0] == v
            and eps in self.constant
            and self.inputs(rsqrt) == [ve]
            and self.inputs(gain)[0] == r
            and gamma in self.constant
            and self.inputs(xs) == [x, s]
            and self.inputs(ms) == [nm, s]
            and self.inputs(total) == [self.outputs(xs)[0], self.outputs(ms)[0]]
        )
        if not wired:
            raise SystemExit("layer normalization is wired differently than expected")
        out = self.outputs(total)[0]
        h, w, c = self.record(out)
        g = self.value(gamma).astype(np.float32).reshape(-1)
        e = self.value(eps).astype(np.float32).reshape(-1)
        if g.size != c or e.size != 1:
            raise SystemExit("layer normalization constants have the wrong size")
        self.emit(
            "layernorm",
            self.names[x],
            self.names[out],
            ih=h,
            iw=w,
            ic=c,
            oh=h,
            ow=w,
            oc=c,
            w=self.put(np.concatenate([g, e])),
        )

    def on_transpose(self, op) -> None:
        x, perm = self.inputs(op)
        out = self.outputs(op)[0]
        ih, iw, ic = self.hwc_of(x)
        if self.value(perm).reshape(-1).tolist() != [0, 1, 3, 2] or ih != 1:
            raise SystemExit("only width/channel transposes of one row are supported")
        oh, ow, oc = self.record(out)
        self.emit(
            "transpose",
            self.names[x],
            self.names[out],
            ih=ih,
            iw=iw,
            ic=ic,
            oh=oh,
            ow=ow,
            oc=oc,
        )

    def on_reshape(self, op) -> None:
        x, out = self.inputs(op)[0], self.outputs(op)[0]
        count = int(np.prod(self.dims(out)))
        if count != self.flat(self.names[x]):
            raise SystemExit("reshape changes the element count")
        self.alias_to(self.names[out], self.names[x], count)

    def on_concatenation(self, op) -> None:
        parts = self.inputs(op)
        out = self.outputs(op)[0]
        o = options(op, S.ConcatenationOptions)
        shape = self.dims(out)
        if ACTIVATION[o.FusedActivationFunction()] != "NONE":
            raise SystemExit("concatenation with a fused activation is unsupported")
        # Points of one row, [1, 1, n, c] along n, lay out end to end as well.
        one_row = len(shape) == 4 and shape[1] == 1 and o.Axis() in (2, -2)
        if (
            one_row
            and len(parts) == 2
            and parts[0] in self.constant
            and parts[1] not in self.constant
        ):
            head = self.value(parts[0]).astype(np.float32).reshape(-1)
            h, w, c = self.record(out)
            ih, iw, ic = self.hwc_of(parts[1])
            self.emit(
                "constconcat",
                self.names[parts[1]],
                self.names[out],
                ih=ih,
                iw=iw,
                ic=ic,
                oh=h,
                ow=w,
                oc=c,
                w=self.put(head),
            )
            return
        if not one_row and (len(shape) != 3 or o.Axis() not in (1, -2)):
            raise SystemExit(
                "concatenation is only supported on the leading axis of a [1,N,C]"
            )
        if any(part in self.constant for part in parts):
            raise SystemExit(
                "constant concatenation parts are only supported first, in one row"
            )
        head = self.names[parts[0]]
        for k, part in enumerate(parts[1:], 1):
            name = self.names[out] if k == len(parts) - 1 else f"{self.names[out]}#{k}"
            count = self.flat(head) + self.flat(self.names[part])
            self.note(name, 1, 1, count)
            self.emit(
                "concat",
                head,
                name,
                second=self.names[part],
                ih=1,
                iw=1,
                ic=self.flat(head),
                oh=1,
                ow=1,
                oc=count,
                n2=self.flat(self.names[part]),
            )
            head = name


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model", type=Path, help="MediaPipe model in TFLite form")
    parser.add_argument("out_dir", type=Path, help="directory to write the plan into")
    parser.add_argument(
        "--source", default="", help="note recorded in the plan for provenance"
    )
    parser.add_argument(
        "--format",
        type=int,
        choices=(1, 3, 4),
        default=None,
        help="plan format to write (default 3, or 4 when the model needs it)",
    )
    args = parser.parse_args()

    exporter = TfliteExporter(args.model, args.format)
    plan = exporter.write(args.out_dir, args.model.stem, args.source)
    if not np.frombuffer(bytes(exporter.blob), dtype=np.float32).any():
        raise SystemExit(
            "every exported weight is zero: constants were not materialised"
        )

    scratch = sum(plan["pool"]) * 4
    print(f"plan         {args.out_dir / (args.model.stem + '.mpplan')}")
    print(f"format       {plan['format']}")
    print(f"operations   {len(plan['ops'])}")
    print(f"weights      {len(exporter.blob) / 1e6:.2f} MB")
    print(f"scratch      {scratch / 1e6:.2f} MB in {len(plan['pool'])} buffers")
    for out in plan["outputs"]:
        print(f"output       {out['n']:>6} values  {out['name']}")


if __name__ == "__main__":
    main()
