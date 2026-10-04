#!/usr/bin/env python3
"""Inspect named x86 kernel symbols. This is not a transitive ISA proof or CPU emulator."""

from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assembly", type=Path)
    args = parser.parse_args()
    text = subprocess.run(
        ["objdump", "-d", "--no-show-raw-insn", "-C", str(args.binary)],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout
    if args.assembly:
        args.assembly.write_text(text)
    sections = []
    current = None
    for line in text.splitlines():
        match = re.match(r"^([0-9a-f]+) <(.+)>:$", line)
        if match:
            current = {"address": match[1], "symbol": match[2], "lines": []}
            sections.append(current)
        elif current is not None:
            current["lines"].append(line)
    rows = []
    for section in sections:
        if "mediapipe_native::ops::" not in section["symbol"]:
            continue
        counts = collections.Counter()
        nofma = bool(
            re.search(r"(?:_avx(?:[^2\w]|$)|::store8_full)", section["symbol"])
        )
        unexpected = []
        for line in section["lines"]:
            match = re.match(r"\s*[0-9a-f]+:\s+(\S+)\s*(.*)", line)
            if not match:
                continue
            mnemonic, operands = match.groups()
            counts[mnemonic] += 1
            if re.search(r"%zmm|%k[0-7]\b", operands) or mnemonic.startswith(
                ("tileload", "tilestore", "tdp", "ldtilecfg")
            ):
                unexpected.append(line.strip())
            if (
                nofma
                and mnemonic.startswith("vp")
                and "%ymm" in operands
                and not mnemonic.startswith(("vpermil", "vperm2f128", "vptest"))
            ):
                unexpected.append(line.strip())
        fma = sum(
            n
            for instruction, n in counts.items()
            if instruction.startswith(("vfmadd", "vfmsub", "vfnmadd", "vfnmsub"))
        )
        if nofma and fma:
            unexpected.append("FMA in a named non-FMA AVX kernel")
        rows.append(
            {
                "symbol": section["symbol"],
                "address": section["address"],
                "instruction_count": sum(counts.values()),
                "fma": fma,
                "avx_without_fma_selected": nofma,
                "mask_operations": counts["vmaskmovps"],
                "unaligned_vector_moves": counts["vmovups"],
                "vzeroupper": counts["vzeroupper"],
                "unexpected": unexpected,
            }
        )
    violations = [r for r in rows if r["unexpected"]]
    result = {
        "status": "PASS" if rows and not violations else "FAIL",
        "scope": "named ops symbols only; not a full call-graph or scalar/SSE proof",
        "note": "masked instructions in partial-pixel tails are allowed; full-pixel PReLU loops use VMOVUPS",
        "kernels": len(rows),
        "avx_without_fma_kernels": sum(r["avx_without_fma_selected"] for r in rows),
        "violations": violations,
        "results": rows,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "results"}))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
