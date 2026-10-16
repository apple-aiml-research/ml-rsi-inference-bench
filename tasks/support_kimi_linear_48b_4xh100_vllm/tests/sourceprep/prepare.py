#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Build-only source reconstruction. This file and upstream objects never enter the agent image."""

import hashlib
import os
from pathlib import Path
import subprocess

UPSTREAM = "1994de99ea0bf8dd84257a19800f4f62526a7edf"
KERNEL_PARENT = "05e034f085b0255b171ff494938a775f2fe80ee1"
KERNEL = "e88bdd60d9a25d985168c9f4a60ab10095236d7c"
PATCH_SHA = "242871a3c31df220b284d7fc60f06e910cb88aafd68953f1e49ec2d48d66eed2"
TREE = "f02109e92a704dc7d6abe7d49d87662ff4146a24"
BASE = "d89bba62d459db1df4eb397a938022d7454b1214"


def git(path, *args, **kwargs):
    return subprocess.check_output(["git", "-C", str(path), *args], **kwargs)


def main():
    upstream, prepared = Path("/upstream"), Path("/prepared")
    upstream.mkdir()
    prepared.mkdir()
    git(upstream, "init", "--quiet")
    git(upstream, "remote", "add", "origin", "https://github.com/vllm-project/vllm.git")
    git(upstream, "fetch", "--depth", "1", "origin", UPSTREAM, KERNEL_PARENT, KERNEL)
    patch = git(upstream, "diff", "--no-ext-diff", "--no-color", "--binary", "--full-index",
                KERNEL_PARENT, KERNEL)
    assert hashlib.sha256(patch).hexdigest() == PATCH_SHA
    archive = subprocess.Popen(["git", "-C", str(upstream), "archive", UPSTREAM], stdout=subprocess.PIPE)
    subprocess.run(["tar", "-x", "-C", str(prepared)], stdin=archive.stdout, check=True)
    archive.stdout.close()
    assert archive.wait() == 0
    git(prepared, "init", "--quiet", "-b", "main")
    git(prepared, "apply", "--reverse", input=patch)
    paths = sorted(p.name for p in prepared.iterdir() if p.name != ".git")
    git(prepared, "add", "-f", "--", *paths)
    assert git(prepared, "write-tree").decode().strip() == TREE
    env = dict(os.environ, GIT_AUTHOR_NAME="Inference-Bench", GIT_AUTHOR_EMAIL="benchmark@localhost",
               GIT_COMMITTER_NAME="Inference-Bench", GIT_COMMITTER_EMAIL="benchmark@localhost",
               GIT_AUTHOR_DATE="2025-10-30T12:00:00+00:00", GIT_COMMITTER_DATE="2025-10-30T12:00:00+00:00")
    commit = git(prepared, "commit-tree", TREE,
                 input=b"Prepare vLLM model-support starting tree\n", env=env).decode().strip()
    assert commit == BASE, commit
    git(prepared, "update-ref", "refs/heads/main", commit)
    git(prepared, "config", "core.hooksPath", "/dev/null")
    (prepared / ".git/info/exclude").write_text(".timer/\nscripts/start_agent_timer.sh\n")
    assert not git(prepared, "status", "--porcelain").strip()
    assert git(prepared, "rev-list", "--all", "--count").strip() == b"1"
    print(f"prepared {commit}, tree {TREE}; one root commit, no upstream object database copied")


if __name__ == "__main__":
    main()
