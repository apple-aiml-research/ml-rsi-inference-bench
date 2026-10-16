#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Reproduce the supplied kernel preparation of support_minimax_m3_4xb200_vllm from upstream objects (maker-only).

The prepared base is the upstream base commit plus ONE kernel-only preparation commit. Its content is taken
directly from upstream Git objects: every supplied path is either a whole file at the source revision's content or
named hunks of the source revision's change to that path, applied to the base content. Nothing is produced by
landing a larger change and reverting it, so no withheld content exists in any object this script writes.

kernel-materials.json (beside this script) is the manifest. For every path it records the operation, the hunk
selection, the base blob, the source blob and the supplied blob; the script asserts every preimage blob (base and
source) and, unless --record is given, every supplied blob, and fails on any mismatch, on a hunk count that differs
from the manifest, or on any change outside the listed paths. --record writes the observed supplied blobs and the
patch digest into the manifest (first reconstruction only; the result is reviewed and committed).

  prepare_kernel_materials.py --repo <vllm clone> [--out <patch>] [--record]

The output patch is `git diff --binary <base> <prepared tree>`; the Dockerfile applies it and commits it with the
fixed identity. This script and the manifest are maker material: neither enters any image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "kernel-materials.json"
DEFAULT_OUT = HERE / "prepared-base" / "000-supplied-kernels.patch"


def git(repo: Path, *args: str, input: bytes | None = None, env: dict | None = None) -> bytes:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        input=input,
        capture_output=True,
        env={**os.environ, **(env or {})},
    )
    if proc.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {proc.stderr.decode(errors='replace')}")
    return proc.stdout


def blob_of(repo: Path, rev: str, path: str) -> str | None:
    proc = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--verify", "--quiet", f"{rev}:{path}"],
        capture_output=True,
    )
    return proc.stdout.decode().strip() or None


def split_hunks(diff: str) -> tuple[str, list[str]]:
    """(file header, hunks) of a single-file unified diff."""
    idx = [m.start() for m in re.finditer(r"^@@ ", diff, flags=re.M)]
    if not idx:
        raise SystemExit("no hunks in diff")
    header = diff[: idx[0]]
    hunks = [diff[a:b] for a, b in zip(idx, idx[1:] + [len(diff)])]
    return header, hunks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, required=True, help="a vLLM clone holding the base and source objects")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--record", action="store_true", help="record supplied blobs and the patch digest")
    args = ap.parse_args()

    man = json.loads(MANIFEST.read_text())
    base, source = man["upstream_base"], man["source_revision"]
    for rev in (base, source):
        git(args.repo, "cat-file", "-e", f"{rev}^{{commit}}")
    parents = git(args.repo, "rev-list", "--parents", "-n", "1", source).decode().split()
    if parents[1:] != [base]:
        raise SystemExit(f"source {source} is not a single-parent child of base {base}: {parents}")

    with tempfile.TemporaryDirectory() as tmp:
        wt = Path(tmp) / "prepared"
        git(args.repo, "worktree", "add", "--detach", str(wt), base)
        try:
            for e in man["supplied"]:
                path = e["path"]
                got_base = blob_of(args.repo, base, path)
                got_source = blob_of(args.repo, source, path)
                if got_base != e["base_blob"] or got_source != e["source_blob"]:
                    raise SystemExit(
                        f"{path}: preimage mismatch: base {got_base} (want {e['base_blob']}), "
                        f"source {got_source} (want {e['source_blob']})"
                    )
                target = wt / path
                if e["operation"] == "whole_file":
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(git(args.repo, "cat-file", "blob", got_source))
                elif e["operation"] == "hunks":
                    diff = git(args.repo, "diff", "--no-ext-diff", "-U3", base, source, "--", path).decode()
                    header, hunks = split_hunks(diff)
                    if len(hunks) != e["source_hunks"]:
                        raise SystemExit(f"{path}: {len(hunks)} hunks in the source change, manifest says {e['source_hunks']}")
                    chosen = [h for h in hunks if any(m in h for m in e["select_if_contains"])]
                    if len(chosen) != e["supplied_hunks"]:
                        raise SystemExit(f"{path}: selected {len(chosen)} hunks, manifest says {e['supplied_hunks']}")
                    git(wt, "apply", "--recount", "--whitespace=nowarn", "-", input=(header + "".join(chosen)).encode())
                else:
                    raise SystemExit(f"{path}: unknown operation {e['operation']}")
                git(wt, "add", "--", path)
                supplied = git(wt, "rev-parse", f":{path}").decode().strip()
                if args.record:
                    e["supplied_blob"] = supplied
                elif supplied != e["supplied_blob"]:
                    raise SystemExit(f"{path}: supplied blob {supplied}, manifest says {e['supplied_blob']}")
            changed = sorted(git(wt, "diff", "--cached", "--name-only", base).decode().split())
            listed = sorted(e["path"] for e in man["supplied"])
            if changed != listed:
                raise SystemExit(f"prepared tree changes {changed}, manifest lists {listed}")
            tree = git(wt, "write-tree").decode().strip()
            patch = git(wt, "diff", "--cached", "--no-ext-diff", "--binary", "--full-index", base)
        finally:
            git(args.repo, "worktree", "remove", "--force", str(wt))

    digest = hashlib.sha256(patch).hexdigest()
    if args.record:
        man["prepared_tree"] = tree
        man["patch"] = {"sha256": digest, "bytes": len(patch)}
        MANIFEST.write_text(json.dumps(man, indent=1) + "\n")
    else:
        if tree != man["prepared_tree"] or digest != man["patch"]["sha256"]:
            raise SystemExit(f"prepared tree {tree} / patch {digest} differ from the manifest")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(patch)
    print(f"prepared tree {tree}; patch {args.out} sha256 {digest} ({len(patch)} bytes); {len(man['supplied'])} paths")
    return 0


if __name__ == "__main__":
    sys.exit(main())
