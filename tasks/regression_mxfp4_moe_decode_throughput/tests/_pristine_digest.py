#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Content digest of a pristine engine tree, and the check that it has not been altered."""
from __future__ import annotations

import os
import argparse
import hashlib
import json
import sys
from pathlib import Path


def tree_digest(root: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    for path in sorted(root.rglob("*.py")):
        if path.is_symlink() or not path.is_file():
            continue
        parts = path.relative_to(root).parts
        if ".git" in parts or "__pycache__" in parts:
            continue
        h.update(
            str(path.relative_to(root)).encode()
            + b"\0"
            + hashlib.sha256(path.read_bytes()).hexdigest().encode()
            + b"\n"
        )
        n += 1
    return h.hexdigest(), n


def _expected(path: Path) -> str | None:
    """First 64-hex token in the sealed file, so it may carry a comment beside the digest."""
    for raw in path.read_text(encoding="utf-8").splitlines():
        token = raw.split("#", 1)[0].strip().split()
        if token and len(token[0]) == 64 and all(c in "0123456789abcdef" for c in token[0]):
            return token[0]
    return None


def symlink_inventory(root: Path) -> tuple[str, int, list[str]]:
    """Every symlink in the tree, as `<relpath> -> <target>`, digested."""
    h = hashlib.sha256()
    n = 0
    entries: list[str] = []
    for path in sorted(root.rglob("*")):
        parts = path.relative_to(root).parts
        if ".git" in parts or "__pycache__" in parts:
            continue
        if not path.is_symlink():
            continue
        try:
            target = os.readlink(path)
        except OSError:
            target = "<unreadable>"
        entry = f"{path.relative_to(root)} -> {target}"
        entries.append(entry)
        h.update(entry.encode() + b"\n")
        n += 1
    return h.hexdigest(), n, entries

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tree")
    ap.add_argument("--expect", help="file holding the sealed digest")
    ap.add_argument("--out", help="write a JSON record of the result here")
    args = ap.parse_args()

    root = Path(args.tree)
    _ENGINE_MARKERS = ("vllm", "python/sglang")
    if not any((root / marker).is_dir() for marker in _ENGINE_MARKERS):
        print(f"[pristine-digest] FATAL {root} is not an engine tree "
              f"(looked for {' or '.join(_ENGINE_MARKERS)})", file=sys.stderr)
        return 2
    digest, files = tree_digest(root)
    symlink_digest, symlink_count, symlink_entries = symlink_inventory(root)
    sealed_requested = args.expect is not None
    seal_path = Path(args.expect) if sealed_requested else None
    want = _expected(seal_path) if seal_path is not None and seal_path.is_file() else None
    if sealed_requested and want is None:
        print(f"[pristine-digest] tree={root} files={files} digest={digest} "
              f"expected=<UNSEALED> match=False")
        print(f"[pristine-digest] FATAL {args.expect} seals no digest (missing, empty, or still "
              f"UNPINNED). A run cannot be scored against an unverified pristine tree. Re-derive it "
              f"by running this script over a fresh checkout of task.toml's baseline_commit and "
              f"committing the digest it prints.",
              file=sys.stderr)
        return 4
    ok = (not sealed_requested) or want == digest
    print(f"[pristine-digest] tree={root} files={files} digest={digest} "
          f"expected={want or '<none sealed>'} match={ok}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "tree": str(root), "files": files, "digest": digest,
            "symlink_digest": symlink_digest, "symlinks": symlink_count,
            "symlink_entries": symlink_entries[:200],
            "symlink_digest": symlink_digest, "symlinks": symlink_count,
            "symlink_entries": symlink_entries[:200],
            "expected": want, "match": bool(ok),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not ok:
        print("[pristine-digest] FATAL the tree about to be benched as the PRISTINE baseline is "
              "not the pinned tree. Its throughput is the denominator of the scored ratio and the "
              "reference for the no-op gate, so this run cannot be scored.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
