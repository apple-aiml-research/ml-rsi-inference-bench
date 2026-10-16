#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Cheap local smoke check for your candidate edit."""
from __future__ import annotations


import py_compile
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TREE = ROOT / "repo" / "python" / "sglang"


def main() -> int:
    if not TREE.is_dir():
        print(f"missing: {TREE.relative_to(ROOT)}")
        return 1

    checked = 0
    failures = 0
    for path in sorted(TREE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        checked += 1
        try:
            py_compile.compile(str(path), cfile=None, doraise=True)
        except py_compile.PyCompileError as exc:
            print(f"compile error: {exc}", file=sys.stderr)
            failures += 1

    if failures:
        print(f"{failures} of {checked} file(s) failed to compile")
        return 1

    print(f"ok ({checked} files compile)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
