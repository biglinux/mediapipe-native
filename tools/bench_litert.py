#!/usr/bin/env python3
"""CPU time per frame of a .tflite model on LiteRT, one thread, XNNPACK.

Runs 20 warm-up frames, then FRAMES frames on a seeded random input, and
prints milliseconds of process CPU time per frame, as `mpbench` does.

Usage:
    bench_litert.py MODEL.tflite [FRAMES]
"""

from __future__ import annotations

import sys
import time

import numpy as np
from ai_edge_litert.interpreter import Interpreter


def main() -> None:
    if len(sys.argv) not in (2, 3):
        raise SystemExit(__doc__)
    frames = int(sys.argv[2]) if len(sys.argv) == 3 else 300
    interpreter = Interpreter(model_path=sys.argv[1], num_threads=1)
    interpreter.allocate_tensors()
    rng = np.random.default_rng(0)
    for detail in interpreter.get_input_details():
        value = rng.random(detail["shape"], dtype=np.float32)
        interpreter.set_tensor(detail["index"], value)
    for _ in range(20):
        interpreter.invoke()
    started = time.process_time()
    for _ in range(frames):
        interpreter.invoke()
    print(f"{(time.process_time() - started) / frames * 1e3:.3f}")


if __name__ == "__main__":
    main()
