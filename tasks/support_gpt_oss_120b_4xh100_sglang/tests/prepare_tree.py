#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Create a deterministic captured source root from already prepared files."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


def prepare(root: Path, record: Path) -> None:
    if (root / ".git").exists():
        raise ValueError("prepared root must not carry a Git object store")
    files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*")
                   if p.is_file() or p.is_symlink())
    if any(".git" in Path(p).parts or "__pycache__" in Path(p).parts or p.endswith((".pyc", ".whl"))
           for p in files):
        raise ValueError("history or generated code is present in prepared source")
    env = {**os.environ, "GIT_AUTHOR_NAME": "Benchmark preparation", "GIT_COMMITTER_NAME": "Benchmark preparation",
           "GIT_AUTHOR_EMAIL": "preparation@localhost", "GIT_COMMITTER_EMAIL": "preparation@localhost",
           "GIT_AUTHOR_DATE": "2026-09-26T00:00:00+00:00", "GIT_COMMITTER_DATE": "2026-09-26T00:00:00+00:00"}

    def git(*args):
        return subprocess.check_output(["git", "--literal-pathspecs", "-C", str(root), *args], env=env, text=True).strip()

    git("init", "--initial-branch=main")
    git("config", "core.hooksPath", "/dev/null")
    with tempfile.NamedTemporaryFile() as spec:
        spec.write(b"\0".join(p.encode() for p in files) + b"\0")
        spec.flush()
        git("add", "--force", "--pathspec-from-file=" + spec.name, "--pathspec-file-nul")
        git("commit", "--only", "--pathspec-from-file=" + spec.name, "--pathspec-file-nul",
            "-m", "Prepared model-support source")
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({"schema": "gpt-oss-prepared-source/1", "commit": git("rev-parse", "HEAD"),
                                  "tree": git("rev-parse", "HEAD^{tree}"), "files": len(files)}, indent=2) + "\n")
    (root / ".git/info/exclude").write_text(".timer/\nscripts/start_agent_timer.sh\n")
    if git("status", "--porcelain") or git("rev-list", "--count", "HEAD") != "1":
        raise ValueError("prepared source must be clean with exactly one commit")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("/app"))
    ap.add_argument("--record", type=Path, default=Path("/opt/infb/prepared.json"))
    args = ap.parse_args()
    prepare(args.root, args.record)
