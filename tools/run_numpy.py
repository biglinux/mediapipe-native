#!/usr/bin/env python3
"""Run a .mpplan with numpy alone, as a reference and as a measuring stick.

This is the slow, obvious version of what `src/ops.rs` does fast. It exists so
that someone with numpy and no Rust can still:

  * check that they read the plan format correctly, by reproducing the numbers
    the Rust executor produces;
  * see the **activations**, not just the weights — the ranges and crest
    factors of what flows between layers, which is the half of the
    narrow-number question that the weights alone cannot answer;
  * try an idea on one layer and confirm it is still the same arithmetic
    before anyone writes a kernel for it.

Speed is not a goal here. Correctness and legibility are.

Usage:
    run_numpy.py MODEL.mpplan                  # run once, print the outputs
    run_numpy.py MODEL.mpplan --stats          # per-layer activation report
    run_numpy.py MODEL.mpplan --save out.npz   # every intermediate tensor
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

MAGIC = b"MPNPLAN1"


def load(path: Path) -> tuple[dict, np.ndarray]:
    raw = path.read_bytes()
    if raw[:8] != MAGIC:
        raise SystemExit(f"{path} does not start with {MAGIC!r}")
    (json_len,) = struct.unpack_from("<I", raw, 8)
    plan = json.loads(raw[12 : 12 + json_len])
    start = -(-(12 + json_len) // 64) * 64
    return plan, np.frombuffer(raw, dtype="<f4", offset=start)


def pseudo_image(count: int) -> np.ndarray:
    """The same fixed image `mpbench` uses, so the two can be compared.

    A 32-bit xorshift, exactly as in `src/bin/mpbench.rs`.
    """
    out = np.empty(count, dtype=np.float32)
    state = np.uint32(0x2545F491)
    for i in range(count):
        state ^= np.uint32(state << np.uint32(13))
        state ^= np.uint32(state >> np.uint32(17))
        state ^= np.uint32(state << np.uint32(5))
        out[i] = np.float32(state >> np.uint32(8)) / np.float32(16777216.0)
    return out


def block(weights: np.ndarray, spec: list[int] | None) -> np.ndarray | None:
    if not spec or spec[1] == 0:
        return None
    return weights[spec[0] : spec[0] + spec[1]]


def pad_for(op: dict, x: np.ndarray) -> np.ndarray:
    """Add the rows above and columns left that the operation asks for.

    Only the leading padding is recorded in the plan; whatever trailing
    padding a tap needs is implied by the output size, so the array is grown
    to whatever the kernel will actually reach.
    """
    top, left = op.get("pt", 0), op.get("pl", 0)
    need_h = (op["oh"] - 1) * op["sh"] + op["kh"]
    need_w = (op["ow"] - 1) * op["sw"] + op["kw"]
    bottom = max(0, need_h - (x.shape[0] + top))
    right = max(0, need_w - (x.shape[1] + left))
    if top or left or bottom or right:
        x = np.pad(x, ((top, bottom), (left, right), (0, 0)))
    return x


def conv(op: dict, x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Dense convolution. Weights are stored [kh][kw][in][out]."""
    w = block(weights, op["w"]).reshape(op["kh"], op["kw"], op["ic"], op["oc"])
    bias = block(weights, op.get("b"))
    x = pad_for(op, x)
    y = np.zeros((op["oh"], op["ow"], op["oc"]), dtype=x.dtype)
    for ky in range(op["kh"]):
        for kx in range(op["kw"]):
            rows = slice(ky, ky + op["oh"] * op["sh"], op["sh"])
            cols = slice(kx, kx + op["ow"] * op["sw"], op["sw"])
            y += x[rows, cols] @ w[ky, kx]
    if bias is not None:
        y += bias
    return y


def dwconv(op: dict, x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Depthwise convolution. Weights are stored [kh][kw][channel]."""
    w = block(weights, op["w"]).reshape(op["kh"], op["kw"], op["oc"])
    bias = block(weights, op.get("b"))
    x = pad_for(op, x)
    y = np.zeros((op["oh"], op["ow"], op["oc"]), dtype=x.dtype)
    for ky in range(op["kh"]):
        for kx in range(op["kw"]):
            rows = slice(ky, ky + op["oh"] * op["sh"], op["sh"])
            cols = slice(kx, kx + op["ow"] * op["sw"], op["sw"])
            y += x[rows, cols] * w[ky, kx]
    if bias is not None:
        y += bias
    return y


def maxpool(op: dict, x: np.ndarray) -> np.ndarray:
    y = np.full((op["oh"], op["ow"], op["oc"]), -np.inf, dtype=x.dtype)
    for ky in range(op["kh"]):
        for kx in range(op["kw"]):
            rows = slice(ky, ky + op["oh"] * op["sh"], op["sh"])
            cols = slice(kx, kx + op["ow"] * op["sw"], op["sw"])
            y = np.maximum(y, x[rows, cols])
    return y


def prelu(x: np.ndarray, slope: np.ndarray) -> np.ndarray:
    return np.where(x < 0, x * slope, x)


def resize(op: dict, x: np.ndarray) -> np.ndarray:
    """Bilinear, half-pixel centres, clamped like TFLite's reference kernel."""

    def axis(out: int, size: int):
        at = (np.arange(out) + 0.5) * (size / out) - 0.5
        low = np.maximum(np.floor(at), 0).astype(int)
        high = np.minimum(np.ceil(at), size - 1).astype(int)
        return low, high, (at - low).astype(x.dtype)

    y0, y1, dy = axis(op["oh"], op["ih"])
    x0, x1, dx = axis(op["ow"], op["iw"])
    dy, dx = dy[:, None, None], dx[None, :, None]
    top = x[y0][:, x0] * (1 - dx) + x[y0][:, x1] * dx
    bottom = x[y1][:, x0] * (1 - dx) + x[y1][:, x1] * dx
    return top * (1 - dy) + bottom * dy


def run(
    plan: dict,
    weights: np.ndarray,
    image: np.ndarray,
    keep: bool = False,
    dtype=np.float32,
) -> tuple[list[np.ndarray], dict[int, np.ndarray]]:
    """Execute the plan. Returns the declared outputs, and every tensor if asked.

    `dtype=np.float64` runs the same graph and float32 weights in double
    precision: a yardstick for how far any float32 engine strays."""
    pool: dict[int, np.ndarray] = {}
    spec = plan["input"]
    pool[spec["slot"]] = image.reshape(spec["h"], spec["w"], spec["c"]).astype(dtype)
    intermediates: dict[int, np.ndarray] = {}

    for index, op in enumerate(plan["ops"]):
        kind = op["kind"]
        x = pool[op["x"]]
        # Every tensor is NHWC, and a buffer may be wider than the tensor in
        # it, so each operation takes the shape it was told to expect.
        x = x.reshape(-1)[: op["ih"] * op["iw"] * op["ic"]].reshape(
            op["ih"], op["iw"], op["ic"]
        )

        if kind == "conv":
            y = conv(op, x, weights)
        elif kind == "dwconv":
            y = dwconv(op, x, weights)
        elif kind == "prelu":
            y = prelu(x, block(weights, op["w"]))
        elif kind == "relu":
            y = np.maximum(x, 0.0)
        elif kind == "add":
            other = pool[op["x2"]].reshape(-1)[: x.size].reshape(x.shape)
            y = x + other
        elif kind == "maxpool":
            y = maxpool(op, x)
        elif kind == "padc":
            y = np.zeros((op["oh"], op["ow"], op["oc"]), dtype=dtype)
            y[:, :, : op["ic"]] = x
        elif kind == "sigmoid":
            y = 1.0 / (1.0 + np.exp(-x.astype(np.float64)))
        elif kind == "clip":
            low, high = block(weights, op["w"])
            y = np.clip(x, low, high)
        elif kind == "avgpool":
            y = np.zeros((op["oh"], op["ow"], op["oc"]), dtype=dtype)
            for ky in range(op["kh"]):
                for kx in range(op["kw"]):
                    rows = slice(ky, ky + op["oh"] * op["sh"], op["sh"])
                    cols = slice(kx, kx + op["ow"] * op["sw"], op["sw"])
                    y += x[rows, cols]
            y /= op["kh"] * op["kw"]
        elif kind == "resize":
            y = resize(op, x)
        elif kind == "depthtospace":
            b = op["oh"] // op["ih"]
            y = x.reshape(op["ih"], op["iw"], b, b, op["oc"]).transpose(0, 2, 1, 3, 4)
            y = y.reshape(op["oh"], op["ow"], op["oc"])
        elif kind == "channelslice":
            y = x[:, :, op["begin"] : op["begin"] + op["oc"]]
        elif kind == "transpose":
            y = x.transpose(0, 2, 1)
        elif kind == "layernorm":
            params = block(weights, op["w"])
            gamma, eps = params[: op["oc"]], params[op["oc"]]
            mean = x.mean(axis=2, keepdims=True)
            s = gamma / np.sqrt(((x - mean) ** 2).mean(axis=2, keepdims=True) + eps)
            y = x * s + (-mean) * s
        elif kind == "centerscale":
            d = x - x.mean(axis=1, keepdims=True)
            y = d / np.sqrt((d * d).sum(axis=2)).mean()
        elif kind == "constconcat":
            head = block(weights, op["w"]).astype(dtype)
            y = np.concatenate([head, x.reshape(-1)]).reshape(
                op["oh"], op["ow"], op["oc"]
            )
        elif kind == "concat":
            head = x.reshape(-1)
            tail = pool[op["x2"]].reshape(-1)[: op["n2"]]
            y = np.concatenate([head, tail]).reshape(1, 1, -1)
        else:
            raise SystemExit(f"unhandled operation {kind!r}")

        # Convolutions may carry a parametric ReLU folded into their tail,
        # and (format 3) a clamp after it.
        folded = block(weights, op.get("act"))
        if folded is not None:
            y = prelu(y, folded)
        bounds = block(weights, op.get("clip"))
        if bounds is not None:
            y = np.clip(y, bounds[0], bounds[1])

        pool[op["y"]] = y.astype(dtype)
        if keep:
            intermediates[index] = pool[op["y"]]

    outputs = [pool[o["slot"]].reshape(-1)[: o["n"]] for o in plan["outputs"]]
    return outputs, intermediates


def describe(op: dict) -> str:
    if op["kind"] == "conv":
        return (
            f"conv {op['kh']}x{op['kw']} s{op['sh']} "
            f"{op['ic']}->{op['oc']} at {op['oh']}x{op['ow']}"
        )
    if op["kind"] == "dwconv":
        return (
            f"depthwise {op['kh']}x{op['kw']} s{op['sh']} "
            f"{op['oc']}ch at {op['oh']}x{op['ow']}"
        )
    return op["kind"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("plan", type=Path)
    parser.add_argument(
        "--stats",
        action="store_true",
        help="per-layer activation range and crest factor",
    )
    parser.add_argument("--save", type=Path, help="write every tensor to a .npz")
    args = parser.parse_args()

    plan, weights = load(args.plan)
    spec = plan["input"]
    image = pseudo_image(spec["h"] * spec["w"] * spec["c"])
    outputs, tensors = run(plan, weights, image, keep=args.stats or bool(args.save))

    print(f"# {args.plan.name}: {len(plan['ops'])} operations")
    for out, meta in zip(outputs, plan["outputs"]):
        print(
            f"output {meta['name']!r}: {out.size} values, "
            f"first four {np.array2string(out[:4], precision=6)}"
        )

    if args.stats:
        print("\n# activations on the fixed test image")
        print(
            f"{'idx':>4}  {'layer':<40} {'values':>8} {'peak':>10} "
            f"{'rms':>10} {'crest':>7}  zero%"
        )
        for index, tensor in tensors.items():
            flat = tensor.reshape(-1).astype(np.float64)
            rms = float(np.sqrt(np.mean(flat**2)))
            peak = float(np.abs(flat).max())
            crest = peak / rms if rms > 0 else float("inf")
            zeros = 100.0 * float((np.abs(flat) < 1e-6).mean())
            print(
                f"{index:>4}  {describe(plan['ops'][index]):<40} {flat.size:>8} "
                f"{peak:>10.4f} {rms:>10.4f} {crest:>7.2f}  {zeros:5.1f}"
            )

    if args.save:
        np.savez_compressed(args.save, **{str(k): v for k, v in tensors.items()})
        print(f"\nwrote {args.save}")


if __name__ == "__main__":
    main()
