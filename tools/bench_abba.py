#!/usr/bin/env python3
"""Interleaved, single-CPU ABBA benchmark. Needs two mpbench binaries with --json."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

# Model name -> plan file under plans/.
PLANS = {
    "face_landmarks": "face_landmarks/face_landmarks_detector.mpplan",
    "face_detector": "face_detector/face_detector.mpplan",
    "hand_detector": "hand_detector/hand_detector.mpplan",
    "hand_landmarks": "hand_landmarks/hand_landmarks_detector.mpplan",
    "hand_roi_refinement": "hand_roi_refinement/hand_roi_refinement.mpplan",
    "pose_detector": "pose_detector/pose_detector.mpplan",
    "pose_landmarks": "pose_landmarks/pose_landmarks_detector.mpplan",
    "holistic_face_landmarks": "holistic_face_landmarks/face_landmarks_detector.mpplan",
    "face_blendshapes": "face_blendshapes/face_blendshapes.mpplan",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cgroup(name: str) -> str:
    """A root cgroup without a parent exposes no limit file; record that as-is."""
    path = Path("/sys/fs/cgroup") / name
    return path.read_text().strip() if path.exists() else "unavailable"


def write(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse completed ABBA blocks; repeat an interrupted block",
    )
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument(
        "--tiers", type=int, nargs="+", default=[2, 3], choices=range(4)
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["face_landmarks", "face_detector"],
        choices=list(PLANS),
    )
    args = parser.parse_args()
    if args.frames < 1 or args.blocks < 1:
        parser.error("frames and blocks must be positive")
    if args.cpu not in os.sched_getaffinity(0):
        parser.error("requested CPU is outside the available affinity mask")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    binaries = {
        "A": args.baseline.resolve(strict=True),
        "B": args.candidate.resolve(strict=True),
    }
    result = {
        "status": "RUNNING",
        "system": platform.uname()._asdict(),
        "cpu": args.cpu,
        "frames_per_run": args.frames,
        "blocks": args.blocks,
        "pattern": "ABBA",
        "cpuinfo": Path("/proc/cpuinfo").read_text(),
        "cpu_max": cgroup("cpu.max"),
        "memory_max": cgroup("memory.max"),
        "binaries": {
            k: {"path": str(p), "sha256": digest(p)} for k, p in binaries.items()
        },
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runs": [],
        "summary": [],
    }
    result["plans"] = {
        model: digest(args.root / "plans" / PLANS[model]) for model in args.models
    }
    result["models"] = args.models
    result["tiers"] = args.tiers
    if args.resume:
        previous = json.loads(args.output.read_text())
        for key in (
            "cpu",
            "frames_per_run",
            "blocks",
            "pattern",
            "binaries",
            "system",
            "cpu_max",
            "memory_max",
        ):
            if previous[key] != result[key]:
                raise ValueError(f"cannot resume: {key} changed")
        for key in ("models", "tiers"):
            if key in previous and previous[key] != result[key]:
                raise ValueError(f"cannot resume: {key} changed")
        if previous.get("plans", result["plans"]) != result["plans"]:
            raise ValueError("cannot resume: model files changed")
        previous.update(
            models=args.models,
            tiers=args.tiers,
            plans=result["plans"],
            status="RUNNING",
        )
        previous.setdefault("resumed_utc", []).append(result["started_utc"])
        previous["summary"] = []
        result = previous
    elif args.output.exists():
        raise FileExistsError(
            "output already exists; select a new path or use --resume"
        )
    write(args.output, result)
    for model in args.models:
        plan = (args.root / "plans" / PLANS[model]).resolve(strict=True)
        for tier in args.tiers:
            group = []
            block_cpu_reduction = []
            for block in range(args.blocks):
                current = [
                    r
                    for r in result["runs"]
                    if r["model"] == model and r["tier"] == tier and r["block"] == block
                ]
                if current and "".join(r["binary"] for r in current) != "ABBA":
                    # Never join two halves of an ABBA block across an interruption.
                    result.setdefault("interrupted_runs", []).extend(current)
                    result["runs"] = [r for r in result["runs"] if r not in current]
                    current = []
                if current:
                    group.extend(current)
                for label in "" if current else "ABBA":
                    env = dict(os.environ, MEDIAPIPE_NATIVE_TIER=str(tier))
                    command = [
                        "taskset",
                        "-c",
                        str(args.cpu),
                        str(binaries[label]),
                        str(plan),
                        str(args.frames),
                        "--json",
                    ]
                    started = time.monotonic()
                    run = subprocess.run(
                        command,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=60,
                        check=False,
                    )
                    if run.returncode:
                        result.update(
                            status="FAILED",
                            failed_command=command,
                            stdout=run.stdout,
                            stderr=run.stderr,
                        )
                        write(args.output, result)
                        raise RuntimeError(f"benchmark failed: {run.stderr}")
                    rows = [
                        json.loads(line)
                        for line in run.stdout.splitlines()
                        if line.startswith("{")
                    ]
                    if len(rows) != 1:
                        raise ValueError("expected one JSON measurement from mpbench")
                    measurement = rows[0]
                    if (
                        measurement["tier"] != tier
                        or measurement["allocations"] != 0
                        or measurement.get("first_frame_allocations", 0) != 0
                    ):
                        raise ValueError(
                            "requested tier unavailable or allocation gate failed"
                        )
                    if measurement["frames"] != args.frames or any(
                        not isinstance(measurement.get(key), (int, float))
                        or not math.isfinite(measurement[key])
                        or measurement[key] <= 0
                        for key in ("p50_ms", "p95_ms", "p99_ms", "cpu_ms_per_frame")
                    ):
                        raise ValueError(
                            "missing, nonfinite or invalid timing measurement"
                        )
                    samples = measurement.get("samples_ms", [])
                    if len(samples) != args.frames or any(
                        not math.isfinite(t) or t <= 0 for t in samples
                    ):
                        raise ValueError("missing or invalid per-frame samples")
                    entry = {
                        "model": model,
                        "tier": tier,
                        "block": block,
                        "binary": label,
                        "elapsed_s": time.monotonic() - started,
                        "measurement": measurement,
                    }
                    group.append(entry)
                    current.append(entry)
                    result["runs"].append(entry)
                    write(args.output, result)
                cpu_a = statistics.mean(
                    current[i]["measurement"]["cpu_ms_per_frame"] for i in (0, 3)
                )
                cpu_b = statistics.mean(
                    current[i]["measurement"]["cpu_ms_per_frame"] for i in (1, 2)
                )
                block_cpu_reduction.append(100 * (1 - cpu_b / cpu_a))
            metrics = ("p50_ms", "p95_ms", "p99_ms", "cpu_ms_per_frame")
            summary = {
                "model": model,
                "tier": tier,
                "plan_sha256": digest(plan),
                "runs_per_binary": 2 * args.blocks,
                "block_cpu_reduction_percent": block_cpu_reduction,
            }
            for label in binaries:
                sample = [r["measurement"] for r in group if r["binary"] == label]
                summary[label] = {
                    key: statistics.median(x[key] for x in sample) for key in metrics
                }
            summary["p50_reduction_percent"] = 100 * (
                1 - summary["B"]["p50_ms"] / summary["A"]["p50_ms"]
            )
            summary["cpu_reduction_percent"] = 100 * (
                1 - summary["B"]["cpu_ms_per_frame"] / summary["A"]["cpu_ms_per_frame"]
            )
            result["summary"].append(summary)
            write(args.output, result)
            print(json.dumps(summary), flush=True)
    result["status"] = "PASS"
    result["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write(args.output, result)


if __name__ == "__main__":
    main()
