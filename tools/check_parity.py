#!/usr/bin/env python3
"""Check the native executor against a reference engine, on identical pixels.

`mpbench` writes the image it used and every result it produced into the plan
directory. This feeds the same image to ONNX Runtime and compares, value by
value, in the order the model declares its outputs.

Both sides are 32-bit floating point, and they add numbers in different orders,
so they will not agree to the last bit. What matters is the size of the gap.
For the face mesh, whose outputs are pixel coordinates on a 256-pixel image, a
gap of a thousandth of a pixel is arithmetic noise and a gap of a tenth is a
bug. The threshold below is deliberately far tighter than anything that could
move a landmark.

Usage:
    check_parity.py MODEL.onnx PLAN_DIR [--tolerance 0.01]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model", type=Path)
    parser.add_argument("plan_dir", type=Path)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="largest acceptable difference in output units",
    )
    args = parser.parse_args()

    image_path = args.plan_dir / "bench_input.bin"
    if not image_path.exists():
        print(f"no {image_path}; run mpbench on this plan first", file=sys.stderr)
        return 2
    image = np.fromfile(image_path, dtype=np.float32)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(args.model), options, providers=["CPUExecutionProvider"]
    )
    spec = session.get_inputs()[0]
    height, width, channels = (d if isinstance(d, int) else 1 for d in spec.shape[1:])
    reference = session.run(
        None, {spec.name: image.reshape(1, height, width, channels)}
    )

    mine = sorted(args.plan_dir.glob("bench_out_*.bin"))
    if len(mine) != len(reference):
        print(
            f"native produced {len(mine)} outputs, the model declares {len(reference)}",
            file=sys.stderr,
        )
        return 2

    worst = 0.0
    for index, (path, want) in enumerate(zip(mine, reference)):
        got = np.fromfile(path, dtype=np.float32)
        want = np.asarray(want).reshape(-1)
        if got.shape != want.shape:
            print(
                f"output {index}: {got.size} values against {want.size}",
                file=sys.stderr,
            )
            return 1
        if not np.isfinite(got).all():
            print(
                f"output {index}: native produced a value that is not a number",
                file=sys.stderr,
            )
            return 1
        gap = np.abs(got - want)
        worst = max(worst, float(gap.max()))
        print(
            f"output {index}  {got.size:>6} values   "
            f"average gap {gap.mean():.6f}   largest {gap.max():.6f}   "
            f"reference spans +-{np.abs(want).max():.1f}"
        )

    if worst > args.tolerance:
        print(f"FAIL: largest gap {worst:.6f} is over the {args.tolerance} tolerance")
        return 1
    print(f"PASS: largest gap {worst:.6f}, tolerance {args.tolerance}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
