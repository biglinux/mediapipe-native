#!/usr/bin/env python3
"""Compare every float and every bit of probe outputs, without numpy or a runtime."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path


def compare(left: Path, right: Path) -> dict:
    a = {p.relative_to(left): p for p in left.rglob("case*-out*.bin")}
    b = {p.relative_to(right): p for p in right.rglob("case*-out*.bin")}
    if not a or a.keys() != b.keys():
        raise ValueError(f"Corpus file sets differ or are empty: {len(a)} / {len(b)}")
    values = changed = 0
    worst = 0.0
    rows = []
    for name in sorted(a):
        x, y = a[name].read_bytes(), b[name].read_bytes()
        if len(x) != len(y) or len(x) % 4:
            raise ValueError(f"{name}: malformed or unequal output lengths")
        changes = 0
        gap = 0.0
        for i, ((u,), (v,)) in enumerate(
            zip(struct.iter_unpack("<f", x), struct.iter_unpack("<f", y))
        ):
            if not math.isfinite(u) or not math.isfinite(v):
                raise ValueError(f"{name}: non-finite value at {i}")
            changes += x[i * 4 : i * 4 + 4] != y[i * 4 : i * 4 + 4]
            gap = max(gap, abs(u - v))
        values += len(x) // 4
        changed += changes
        worst = max(worst, gap)
        rows.append(
            {
                "file": str(name),
                "values": len(x) // 4,
                "changed_bits_values": changes,
                "max_abs_error": gap,
            }
        )
    return {
        "files": len(rows),
        "values": values,
        "changed_bits_values": changed,
        "max_abs_error": worst,
        "bit_exact": changed == 0,
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.original, args.candidate)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "results"}))
    if not result["bit_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
