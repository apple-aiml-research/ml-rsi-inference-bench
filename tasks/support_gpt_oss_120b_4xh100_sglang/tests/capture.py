#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Atomically capture committed engine and kernel changes against the prepared tree."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def capture(app: Path, prepared: Path, output: Path):
    app = app.resolve(strict=True)
    output.mkdir(parents=True, exist_ok=True)
    patch_path, record_path = output / "model.patch", output / "capture.json"
    # A failed new capture must not leave an earlier successful receipt usable.
    record_path.unlink(missing_ok=True)
    patch_path.unlink(missing_ok=True)
    pin = json.loads(prepared.read_text())
    base = pin["commit"]
    env = dict(os.environ, GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
    def git(*args):
        return subprocess.run(["git", "--no-pager", "-c", "core.hooksPath=/dev/null",
                               "-c", f"safe.directory={app}", *args], cwd=app, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
    if Path(git("rev-parse", "--show-toplevel").decode().strip()).resolve() != app:
        raise RuntimeError("capture root is not the prepared repository")
    if git("rev-parse", f"{base}^{{tree}}").decode().strip() != pin["tree"]:
        raise RuntimeError("prepared source tree identity mismatch")
    head = git("rev-parse", "HEAD").decode().strip()
    git("merge-base", "--is-ancestor", base, head)
    patch = git("diff", "--no-ext-diff", "--no-textconv", "--binary", "--full-index",
                "--no-renames", "--src-prefix=a/", "--dst-prefix=b/", base, head, "--")
    paths = git("diff", "--name-only", "--no-renames", "-z", base, head, "--").split(b"\0")
    dirty = git("status", "--porcelain=v1", "-z", "--untracked-files=normal").split(b"\0")
    record = {"schema": "gpt-oss-committed-capture/1", "base": base, "base_tree": pin["tree"],
              "head": head, "tree": git("rev-parse", "HEAD^{tree}").decode().strip(),
              "patch_sha256": hashlib.sha256(patch).hexdigest(), "patch_bytes": len(patch),
              "paths": [s.decode("utf-8", "surrogateescape") for s in paths if s],
              "uncommitted_status": [s.decode("utf-8", "surrogateescape") for s in dirty if s]}
    def install(path, data):
        fd, temporary = tempfile.mkstemp(prefix=".capture-", dir=output)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
    install(patch_path, patch)
    # This commit marker is installed last. Verification checks its digest and
    # reconstructed tree, so interrupted/inconsistent pairs cannot be accepted.
    install(record_path, (json.dumps(record, indent=2) + "\n").encode())
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", type=Path, default=Path("/app"))
    parser.add_argument("--prepared", type=Path, default=Path("/opt/infb/prepared.json"))
    parser.add_argument("--output", type=Path, default=Path("/logs/artifacts"))
    args = parser.parse_args()
    record = capture(args.app, args.prepared, args.output)
    print(f"[capture] committed {record['head']}: {record['patch_bytes']} bytes, {len(record['paths'])} paths")
    if record["uncommitted_status"]:
        print("[capture] uncommitted work was excluded; see capture.json")
