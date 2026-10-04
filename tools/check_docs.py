#!/usr/bin/env python3
"""Check that every relative link in the maintained Markdown files resolves.

Only local file targets are checked: external URLs and heading fragments are
skipped, and links inside code spans or fenced blocks are ignored. Also checks
that .gitignore still keeps exported plans and local artifacts out of Git.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

LINK = re.compile(r"!?\[[^\]\n]*\]\(([^\n)]+)\)")


def prose(text: str) -> str:
    """Exclude fenced examples and inline code before examining link paths."""
    fence_char, fence_len = "", 0
    result = []
    for line in text.splitlines():
        fence = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if fence:
            token = fence.group(1)
            if not fence_char:
                fence_char, fence_len = token[0], len(token)
            elif token[0] == fence_char and len(token) >= fence_len:
                fence_char = ""
            continue
        if not fence_char:
            result.append(re.sub(r"`+[^`]*`+", "", line))
    return "\n".join(result)


def links(text: str) -> list[str]:
    targets = []
    for match in LINK.finditer(prose(text)):
        value = match.group(1).strip()
        if value.startswith("<"):
            end = value.find(">")
            if end < 0:
                raise ValueError("unclosed angle-bracket link")
            value = value[1:end]
        else:
            value = value.split()[0]  # optional title after a plain target
        targets.append(value)
    return targets


def path_errors(root: Path, source: Path, text: str) -> list[str]:
    errors = []
    for target in links(text):
        parsed = urlsplit(target)
        if parsed.scheme in ("http", "https", "mailto") or target.startswith("//"):
            continue
        if parsed.scheme:
            errors.append(f"{source}: nonportable link scheme: {target}")
            continue
        if not parsed.path:  # heading fragments are outside this validator's scope
            continue
        path = (source.parent / unquote(parsed.path)).resolve()
        if not path.is_relative_to(root.resolve()):
            errors.append(f"{source}: target escapes repository: {target}")
        elif not path.exists():
            errors.append(f"{source}: missing target: {target}")
    return errors


def maintained(root: Path) -> list[Path]:
    files = [root / p for p in ("README.md", "CONTRIBUTING.md")]
    files += [root / ".github/pull_request_template.md"]

    def fail(error: OSError) -> None:
        raise error

    for directory, names, filenames in os.walk(root / "docs", onerror=fail):
        for name in filenames:
            if name.endswith(".md"):
                files.append(Path(directory) / name)
    return sorted(files)


def check(root: Path) -> tuple[int, list[str]]:
    errors = []
    files = maintained(root)
    for path in files:
        if not path.is_file():
            errors.append(f"missing maintained guide: {path}")
            continue
        errors += path_errors(root, path, path.read_text())
    ignored = (root / ".gitignore").read_text().splitlines()
    if "/plans" not in ignored or "/artifacts" not in ignored:
        errors.append("preserve /plans and /artifacts ignore rules")
    return len(files), errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    count, errors = check(args.root.resolve())
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"PASS: {count} Markdown files; external links and fragments not checked")


if __name__ == "__main__":
    main()
