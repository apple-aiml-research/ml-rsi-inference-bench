#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Digest the tree that was benched as the scored PRISTINE arm."""
from __future__ import annotations

import os
import argparse
import hashlib
import json
import sys
from pathlib import Path


def tree_digest(root: Path) -> tuple[str, int, list[str]]:
    h = hashlib.sha256()
    n = 0
    seen: list[str] = []
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
        seen.append(str(path.relative_to(root)))
        n += 1
    return h.hexdigest(), n, seen


def _expected(path: Path) -> tuple[str | None, int | None]:
    """The sealed digest and, when the seal carries it, the sealed `*.py` COUNT."""
    digest: str | None = None
    count: int | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        token = raw.split("#", 1)[0].strip().split()
        if token and len(token[0]) == 64 and all(c in "0123456789abcdef" for c in token[0]):
            digest = token[0]
            if len(token) > 1 and token[1].isdigit():
                count = int(token[1])
            break
    return digest, count



def _expected_symlinks(path: Path) -> tuple[str | None, int | None]:
    """The sealed SYMLINK digest and count, or (None, None) when the seal does not carry one."""
    for raw in path.read_text(encoding="utf-8").splitlines():
        token = raw.split("#", 1)[0].strip().split()
        if len(token) >= 2 and token[0] == "symlinks" and len(token[1]) == 64:
            count = int(token[2]) if len(token) > 2 and token[2].isdigit() else None
            return token[1], count
    return None, None

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
    if not (root / "vllm").is_dir():
        print(f"[pristine-digest] FATAL {root} is not an engine tree", file=sys.stderr)
        return 2
    digest, files, paths = tree_digest(root)
    symlink_digest, symlink_count, symlink_entries = symlink_inventory(root)
    sealed_requested = args.expect is not None
    seal_path = Path(args.expect) if sealed_requested else None
    want, want_files = (_expected(seal_path)
                        if seal_path is not None and seal_path.is_file() else (None, None))
    if sealed_requested and want is None:
        print(f"[pristine-digest] tree={root} files={files} digest={digest} "
              f"expected=<UNSEALED> match=False")
        print(f"[pristine-digest] FATAL {args.expect} seals no digest (missing, empty, or still "
              f"UNPINNED). A run cannot be scored against an unverified pristine tree. The recipe is "
              f"in the seal file's own header, under `TO RE-DERIVE:` -- a shallow fetch of task.toml's "
              f"baseline_commit, then this script.",
              file=sys.stderr)
        return 4
    want_sym, want_sym_count = (_expected_symlinks(seal_path)
                                if seal_path is not None and seal_path.is_file() else (None, None))
    sym_ok = want_sym is None or (
        want_sym == symlink_digest and (want_sym_count is None or want_sym_count == symlink_count)
    )
    ok = (not sealed_requested) or (
        want == digest and (want_files is None or want_files == files) and sym_ok
    )
    print(f"[pristine-digest] tree={root} files={files} digest={digest} "
          f"expected={want or '<none sealed>'} match={ok}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        record: dict = {
            "tree": str(root), "files": files, "digest": digest,
            "symlink_digest": symlink_digest, "symlinks": symlink_count,
            "symlink_entries": symlink_entries[:200],
            "expected": want, "expected_files": want_files, "match": bool(ok),
            "expected_symlink_digest": want_sym, "expected_symlinks": want_sym_count,
            "symlink_match": bool(sym_ok),
        }
        if not ok:
            if not sym_ok:
                record["mismatch_kind"] = "symlink_drift"
                record["mismatch_note"] = (
                    f"the content digest matches but the symlink inventory does not: "
                    f"{symlink_count} links digesting to {symlink_digest[:16]}, against a sealed "
                    f"{want_sym_count} at {(want_sym or '')[:16]}. `tree_digest` cannot see this "
                    f"channel -- a path absent from the pinned tree arrives as a link and moves "
                    f"neither the digest nor the file count. `symlink_entries` below is the list."
                )
            elif want_files is not None and want_files != files:
                record["mismatch_kind"] = "file_set_drift"
                record["mismatch_note"] = (
                    f"the staged tree has {files} *.py files and the seal covers {want_files}. "
                    f"A different SET of files is a staging difference, not an edit -- compare "
                    f"`paths` below against a fresh checkout of task.toml's baseline_commit."
                )
            else:
                record["mismatch_kind"] = "content_drift"
                record["mismatch_note"] = (
                    "the file count matches the seal and the digest does not, so some *.py file's "
                    "CONTENT differs: an edit to the tree about to be benched as the baseline."
                )
            record["paths"] = paths
        Path(args.out).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                                  encoding="utf-8")
    if not ok:
        kind = ("file_set_drift" if (want_files is not None and want_files != files)
                else "content_drift")
        detail = (f"{files} *.py files against a sealed {want_files}: a tree STAGED differently, not "
                  f"an edit. See `paths` in the JSON record."
                  if kind == "file_set_drift" else
                  "the file count matches and the digest does not, so a *.py file's CONTENT differs.")
        print(f"[pristine-digest] FATAL ({kind}) the tree about to be benched as the PRISTINE "
              f"baseline is not the sealed tree. {detail} Its throughput is the denominator of the "
              f"scored ratio and the reference for the no-op gate, so this run cannot be scored.",
              file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
