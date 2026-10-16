#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The pristine tree's content digest — the one authority the agent cannot reach."""
from __future__ import annotations

import os
import hashlib
import sys
from pathlib import Path


def resolve_root(root: Path) -> Path:
    """Accept EITHER shape of this engine's tree and return the one that becomes `repo/`."""
    if (root / "python" / "sglang").is_dir() and (root / "python" / "pyproject.toml").is_file():
        return root / "python"
    return root


def tree_digest(root: Path) -> str:
    """sha256 over the (path, content) pairs of every regular file under `root`."""
    outer = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if "__pycache__" in path.parts or path.suffix == ".pyc" or ".git" in path.parts:
            continue
        if path.is_symlink() or not path.is_file():
            continue
        inner = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                inner.update(chunk)
        rel = path.relative_to(root).as_posix()
        outer.update(rel.encode("utf-8") + b"\0" + inner.hexdigest().encode("ascii") + b"\n")
    return outer.hexdigest()


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

def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <tree-root>", file=sys.stderr)
        return 2
    root = Path(argv[1])
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2
    resolved = resolve_root(root)
    if resolved != root:
        print(f"[pristine-digest] monorepo root given; digesting {resolved}", file=sys.stderr)
    print(tree_digest(resolved))
    sym_digest, sym_count, sym_entries = symlink_inventory(resolved)
    print(f"[pristine-digest] symlinks={sym_count} digest={sym_digest}", file=sys.stderr)
    for entry in sym_entries[:200]:
        print(f"[pristine-digest]   {entry}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
