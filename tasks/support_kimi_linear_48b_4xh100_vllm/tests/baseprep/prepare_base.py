#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Build-only: re-root the compiled engine tree at /app as the task's base commit.

The task layer of the recipe runs this on the engine stage's /app, which holds the adapted engine tree
at its own clean root commit. It adds exactly:

  * the three-line upstream test-registry entry (`test-registry.patch`), so upstream's initialization
    test for this model is collected at the base, without depending on a candidate's registry edit;
  * the evaluation tree `eval/kimi_linear/`, from the bind-mounted build context, and its three tracked
    symbolic links onto image-owned files (the model-author source in the checkpoint and the GSM8K
    splits the engine stage fetched and checked by sha256);

and then commits exactly those files -- the engine tree's own tracked paths plus the evaluation tree --
as ONE parentless commit with a fixed neutral identity and date, replacing /app/.git. The build products
the engine stage compiled stay git-ignored and uncommitted. The tree and commit are asserted; with
`--compute` they are printed instead (this file ships in no image).
"""

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ENGINE_BASE = "d89bba62d459db1df4eb397a938022d7454b1214"
ENGINE_TREE = "f02109e92a704dc7d6abe7d49d87662ff4146a24"
REGISTRY_PATCH_SHA = "f443e4de6f07177703478ec0266a7eb361052ef6b017538e22aa20f998dcfa6f"
TREE = "7f270f8da80f40edd60f6cc0e9e4c11f474045c4"
BASE = "acb5b28091e784d9f6f0a0d61338aa31b5f4b86f"
APP = Path("/app")
SOURCE = Path("/infb-eval/kimi_linear")
HERE = Path(__file__).resolve().parent
#: The evaluation tree's symbolic links: path under /app -> image-owned target.
LINKS = {
    "eval/kimi_linear/reference/modeling_kimi.py": "/model/modeling_kimi.py",
    "eval/kimi_linear/reference/configuration_kimi.py": "/model/configuration_kimi.py",
    "eval/kimi_linear/datasets/gsm8k": "/opt/infb/datasets/gsm8k",
}
IDENTITY = dict(GIT_AUTHOR_NAME="Inference-Bench", GIT_AUTHOR_EMAIL="benchmark@localhost",
                GIT_COMMITTER_NAME="Inference-Bench", GIT_COMMITTER_EMAIL="benchmark@localhost",
                GIT_AUTHOR_DATE="2025-10-30T12:00:00+00:00", GIT_COMMITTER_DATE="2025-10-30T12:00:00+00:00")


def git(*args, input=None, env=None):
    return subprocess.check_output(["git", "-C", str(APP), *args], input=input,
                                   env={**os.environ, "GIT_LITERAL_PATHSPECS": "1", **(env or {})})


def already_prepared():
    """Is /app already this task's prepared base? True only when every identity holds.

    A builder can hand the task layer an image this recipe already finished: it builds tests/Dockerfile with
    ENGINE set to the agent's own image, so this layer runs a second time on its own output. That second run
    is then a verified no-op. Anything short of the complete identity falls through to main()'s asserts, which are unchanged.
    """
    if git("rev-parse", "HEAD").decode().strip() != BASE:
        return False
    if git("rev-parse", "HEAD^{tree}").decode().strip() != TREE:
        return False
    if git("rev-list", "--all", "--count").strip() != b"1" or git("status", "--porcelain").strip():
        return False
    for rel, destination in LINKS.items():
        link = APP / rel
        if not link.is_symlink() or os.readlink(link) != destination or not link.exists():
            return False
    return True


def main():
    compute = "--compute" in sys.argv[1:]
    if not compute and already_prepared():
        print(f"already prepared: {BASE}, tree {TREE}, one root commit, clean, its {len(LINKS)} links resolving; "
              f"nothing to do")
        return
    assert git("rev-parse", "HEAD").decode().strip() == ENGINE_BASE
    assert git("rev-parse", "HEAD^{tree}").decode().strip() == ENGINE_TREE
    assert git("rev-list", "--all", "--count").strip() == b"1"
    assert not git("status", "--porcelain").strip()
    tracked = [p for p in git("ls-files", "-z").decode().split("\0") if p]

    patch = (HERE / "test-registry.patch").read_bytes()
    assert hashlib.sha256(patch).hexdigest() == REGISTRY_PATCH_SHA
    git("apply", "--whitespace=nowarn", "-", input=patch)

    target = APP / "eval" / "kimi_linear"
    assert not (APP / "eval").exists(), "the engine tree already has an eval/ directory"
    shutil.copytree(SOURCE, target, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for rel, destination in LINKS.items():
        link = APP / rel
        assert not link.exists() and not link.is_symlink(), f"{rel} is already present in the materials"
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(destination)
        if not compute:
            assert link.exists(), f"{rel} -> {destination} does not resolve in this image"
    added = []
    for root, dirs, names in os.walk(target):
        for name in sorted(dirs):
            if (Path(root) / name).is_symlink():
                added.append(str((Path(root) / name).relative_to(APP)))
        for name in sorted(names):
            added.append(str((Path(root) / name).relative_to(APP)))

    shutil.rmtree(APP / ".git")
    git("init", "--quiet", "-b", "main")
    with tempfile.NamedTemporaryFile("wb", delete=False) as listing:
        listing.write(b"\0".join(p.encode() for p in sorted(set(tracked) | set(added))) + b"\0")
    try:
        git("add", "-f", f"--pathspec-from-file={listing.name}", "--pathspec-file-nul")
    finally:
        os.unlink(listing.name)
    (APP / ".git/info/exclude").write_text(".timer/\nscripts/start_agent_timer.sh\n")
    tree = git("write-tree").decode().strip()
    commit = git("commit-tree", tree, input=b"Prepare vLLM model-support starting tree\n", env=IDENTITY).decode().strip()
    if compute:
        print(f"TREE = \"{tree}\"\nBASE = \"{commit}\"\n{len(tracked)} engine paths, {len(added)} evaluation paths")
        return
    assert tree == TREE, f"the prepared tree is {tree}, not {TREE}"
    assert commit == BASE, f"the prepared commit is {commit}, not {BASE}"
    git("update-ref", "refs/heads/main", commit)
    git("config", "core.hooksPath", "/dev/null")
    assert not git("status", "--porcelain").strip()
    assert git("rev-list", "--all", "--count").strip() == b"1"
    print(f"prepared {commit}, tree {tree}: the engine tree ({len(tracked)} paths), the registry entry and "
          f"eval/kimi_linear/ ({len(added)} paths, {len(LINKS)} of them links); one root commit")


if __name__ == "__main__":
    main()
