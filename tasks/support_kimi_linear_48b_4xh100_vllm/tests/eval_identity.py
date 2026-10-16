#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The restored evaluation tree and the grader's copy hold the same files, byte for byte.

    eval_identity.py <restored tree> <grader's copy>

Regular files are compared by content; bytecode caches are ignored. The restored tree's symbolic links
(its model-author source and dataset pointers, recorded in the base commit) are the base's by the git
restore that precedes this check and are listed, not followed. Exit 1 on any difference.
"""
import hashlib
import sys
from pathlib import Path


def files(root: Path) -> dict[str, str]:
    out = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if "__pycache__" in rel.parts or path.is_symlink() or not path.is_file():
            continue
        out[str(rel)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def main() -> int:
    restored, copy = Path(sys.argv[1]), Path(sys.argv[2])
    a, b = files(restored), files(copy)
    links = sorted(str(p.relative_to(restored)) for p in restored.rglob("*") if p.is_symlink())
    diff = sorted(set(a) ^ set(b)) + sorted(k for k in set(a) & set(b) if a[k] != b[k])
    print(f"[eval-identity] {len(a)} files, {len(links)} links {links}; differences: {diff or 'none'}")
    return 1 if diff else 0


if __name__ == "__main__":
    sys.exit(main())
