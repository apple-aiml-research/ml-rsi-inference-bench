#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Restore protected source before importing any candidate package.

Start with /usr/bin/python3 -I -S. Only stdlib and Git are used. The separately
built verifier owns this file, the base identity, test.patch and public_eval.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def git(app, *args, env=None):
    return subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-C", str(app), *args],
                          env=env, capture_output=True, check=True).stdout


def file_manifest(root):
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"protected public code contains a symlink: {path}")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def verify_capture(app, base, artifacts):
    patch = artifacts / "model.patch"
    receipt = artifacts / "capture.json"
    if not patch.exists() and not receipt.exists():
        return {"submission": "base"}
    if not receipt.is_file() or not patch.is_file():
        raise ValueError("incomplete committed capture: both patch and receipt are required")
    record = json.loads(receipt.read_text())
    data = patch.read_bytes()
    if record.get("schema") != "gpt-oss-committed-capture/1" or record.get("base") != base:
        raise ValueError("capture does not name this prepared base")
    if record.get("patch_bytes") != len(data) or record.get("patch_sha256") != hashlib.sha256(data).hexdigest():
        raise ValueError("capture patch bytes disagree with receipt")
    if record.get("base_tree") != git(app, "rev-parse", f"{base}^{{tree}}").decode().strip():
        raise ValueError("capture base tree differs")
    with tempfile.TemporaryDirectory(prefix="gpt-oss-capture-") as scratch:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(scratch) / "index"))
        git(app, "read-tree", base, env=env)
        if data:
            git(app, "apply", "--cached", "--whitespace=nowarn", str(patch.resolve()), env=env)
        reconstructed = git(app, "write-tree", env=env).decode().strip()
    if reconstructed != record.get("tree"):
        raise ValueError("captured committed tree does not reconstruct from its patch")
    return {"submission": "committed", "tree": reconstructed,
            "patch_sha256": record["patch_sha256"]}


def protect(app, base, tests, artifacts):
    capture = verify_capture(app, base, artifacts)
    protected = ("python/sglang/test", "test", "eval")
    # Do not follow a submitted parent symlink while deleting a protected tree.
    for relative in ("python", "python/sglang"):
        parent = app / relative
        if parent.is_symlink() or not parent.is_dir():
            raise ValueError(f"protected package parent must be a directory: {relative}")
    for relative in protected:
        path = app / relative
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.exists():
            shutil.rmtree(path)
    git(app, "checkout", "--force", base, "--", *protected)
    test_patch = tests / "test.patch"
    if test_patch.stat().st_size:
        git(app, "apply", "--whitespace=nowarn", str(test_patch.resolve()))
    actual = file_manifest(app / "eval")
    expected = file_manifest(tests / "public_eval")
    if not expected or actual != expected:
        raise ValueError("restored public checking code differs from verifier-owned bytes")
    return {**capture, "public_eval_files": actual}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--app", type=Path, default=Path("/app"))
    p.add_argument("--tests", type=Path, default=Path("/tests"))
    p.add_argument("--artifacts", type=Path, default=Path("/logs/artifacts"))
    p.add_argument("--record", type=Path, default=Path("/logs/verifier/protected-source.json"))
    a = p.parse_args()
    base = json.loads((a.tests / "config.json").read_text())["base_commit"]
    record = protect(a.app, base, a.tests, a.artifacts)
    a.record.parent.mkdir(parents=True, exist_ok=True)
    a.record.write_text(json.dumps(record, indent=2) + "\n")
    print(f"[protected source] restored {len(record['public_eval_files'])} public files")
