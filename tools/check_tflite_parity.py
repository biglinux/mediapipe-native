#!/usr/bin/env python3
"""Compare a native plan with TensorFlow Lite on the probe's 24 inputs.

`examples/probe` runs the plan on 24 deterministic images (noise, constants,
a checkerboard, a ramp, a single lit pixel, a denormal) and writes every
output. This rebuilds the same images, runs the original .tflite model in
TensorFlow's interpreter with one thread, and reports the largest gap per
output. Both sides are float32 with different summation orders, so they are
compared by size of gap, not by bits.

Some graphs amplify rounding (the pose segmentation logits reach +-900 on
these synthetic images), so a fixed gap alone cannot tell rounding from a bug.
The yardstick is the same plan run in float64 by tools/run_numpy.py. Each
engine's float32 error against it is a maximum over 24 synthetic images, which
moves with rounding luck; the measured native/TFLite ratios span 0.3-1.6. An
output passes when the native error is within the tolerance, or at most twice
the worse of TFLite's own XNNPACK and reference kernel errors.

Usage:
    check_tflite_parity.py MODEL.tflite PLAN.mpplan [--probe target/release/examples/probe]
                           [--tolerance 0.005] [--work artifacts/parity/NAME | --native DIR]

Set MEDIAPIPE_NATIVE_TIER to choose the probe's tier; each run is a new process.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_numpy


def probe_image(case: int, h: int, w: int, c: int) -> np.ndarray:
    """The image examples/probe.rs builds for `case`, value for value."""
    n = h * w * c
    i = np.arange(n)
    if case == 1:
        return np.zeros(n, np.float32)
    if case == 2:
        return np.ones(n, np.float32)
    if case == 3:
        return np.full(n, 0.5, np.float32)
    if case == 4:
        return ((i // c // w + i // c % w) % 2).astype(np.float32)
    if case == 5:
        return ((i // c % w) / np.float32(w)).astype(np.float32)
    if case == 6:
        return (i // c == h * w // 2).astype(np.float32)
    if case == 7:
        return np.full(n, 1.0e-30, np.float32)
    state = 0x2545F491 if case == 0 else (case * 0x9E3779B9) & 0xFFFFFFFF
    out = np.empty(n, np.float32)
    for k in range(n):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        out[k] = (state >> 8) / 16_777_216.0
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model", type=Path)
    parser.add_argument("plan", type=Path)
    parser.add_argument(
        "--probe", type=Path, default=Path("target/release/examples/probe")
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.005,
        help="largest acceptable absolute difference in output units",
    )
    parser.add_argument("--work", type=Path, default=None)
    parser.add_argument(
        "--skip-cases",
        default="",
        help="comma-separated probe cases the model cannot take, e.g. the "
        "constant images 1,2,3,7 as landmark sets for face_blendshapes",
    )
    parser.add_argument(
        "--native",
        type=Path,
        default=None,
        help="existing probe output directory (e.g. from another machine); "
        "skips running the probe",
    )
    args = parser.parse_args()

    work = args.native or args.work or Path("artifacts/parity") / args.plan.stem
    skip = {int(c) for c in args.skip_cases.split(",") if c}
    if args.native is None:
        command = [str(args.probe), str(args.plan), str(work)]
        if skip:
            command.append("--allow-nonfinite")
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)

    interpreter = tf.lite.Interpreter(model_path=str(args.model), num_threads=1)
    interpreter.allocate_tensors()
    reference = tf.lite.Interpreter(
        model_path=str(args.model),
        num_threads=1,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
    )
    reference.allocate_tensors()
    spec = interpreter.get_input_details()[0]
    shape = [int(d) for d in spec["shape"]]
    # [1, n, c] is one row of n points, as the exporter reads it.
    _, h, w, c = shape if len(shape) == 4 else [1, 1, *shape[1:]]
    outputs = interpreter.get_output_details()

    plan, weights = run_numpy.load(args.plan)
    native_err = [0.0] * len(outputs)
    tflite_err = [0.0] * len(outputs)
    gap = [0.0] * len(outputs)
    span = [0.0] * len(outputs)
    for case in range(24):
        if case in skip:
            continue
        image = probe_image(case, h, w, c).reshape(shape)
        for engine in (interpreter, reference):
            engine.set_tensor(spec["index"], image)
            engine.invoke()
        truth, _ = run_numpy.run(plan, weights, image.reshape(-1), dtype=np.float64)
        for k, detail in enumerate(outputs):
            want = interpreter.get_tensor(detail["index"]).reshape(-1)
            got = np.fromfile(work / f"case{case:02}-out{k}.bin", dtype=np.float32)
            if got.shape != want.shape:
                print(
                    f"output {k}: {got.size} values against {want.size}",
                    file=sys.stderr,
                )
                return 1
            other = reference.get_tensor(
                reference.get_output_details()[k]["index"]
            ).reshape(-1)
            exact = truth[k].reshape(-1)
            gap[k] = max(gap[k], float(np.abs(got - want).max()))
            native_err[k] = max(native_err[k], float(np.abs(got - exact).max()))
            tflite_err[k] = max(
                tflite_err[k],
                float(np.abs(want - exact).max()),
                float(np.abs(other - exact).max()),
            )
            span[k] = max(span[k], float(np.abs(exact).max()))

    failed = False
    for k, detail in enumerate(outputs):
        allowed = max(args.tolerance, 2 * tflite_err[k])
        failed |= native_err[k] > allowed
        # The yardstick itself must agree with TFLite, or it measures nothing.
        if tflite_err[k] > max(args.tolerance, 1e-3 * span[k]):
            failed = True
            print(
                f"output {k}: float64 run disagrees with TFLite by {tflite_err[k]:.6g}; "
                "the plan or run_numpy.py does not describe this model"
            )
        print(
            f"output {k} {detail['name']:<24} {int(np.prod(detail['shape'])):>7} values  "
            f"float64 error: native {native_err[k]:.6f}  TFLite {tflite_err[k]:.6f}  "
            f"ratio {native_err[k] / max(tflite_err[k], 1e-12):.2f}  "
            f"| native-XNNPACK {gap[k]:.6f}  spans +-{span[k]:.3f}"
        )
    verdict = "FAIL" if failed else "PASS"
    print(
        f"{verdict}: {24 - len(skip)} cases; native float64 error within {args.tolerance} "
        "or twice TFLite's own"
    )
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
