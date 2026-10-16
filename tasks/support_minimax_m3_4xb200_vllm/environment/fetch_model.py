#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Fetch one group of the checkpoint's files into the image's Hugging Face cache and verify them (image build only).

The manifest (model-manifest.json) lists every file of the pinned revision with its size, its git object id and,
for the Hub's LFS files, the sha256 the Hub reports. The files are partitioned deterministically into ``--of``
groups of similar size so each image layer holds one safetensors shard (the small files share one group); group
``--group`` is placed in the image's Hugging Face cache (``HF_HUB_CACHE``), and every file is verified: its size,
the sha256 of every LFS file, and the git object id of every other file. The sha256 of every file is computed and
appended to ``$HF_HUB_CACHE/.infb/manifest.json``, which preflight reads at grading time.

Source. By default the files are downloaded from the Hub at the pinned revision (``snapshot_download``). A builder
that already holds the files may bind that copy read-only at ``/infb-model-staging/<repo_id>/`` for the build (for
example ``podman build --volume <dir>:/infb-model-staging:ro``); a staged file is then copied instead of
downloaded. The source changes nothing else: a staged file is accepted only if it passes the same verification,
and one that does not is downloaded from the Hub. The cache layout is the one ``snapshot_download`` writes
(``blobs/<etag>`` plus a ``snapshots/<revision>/`` tree of relative symlinks), so ``/model`` and the repository id
resolve the same way whichever source filled it.

``--finalize`` additionally writes ``refs/main`` with the pinned revision and asserts the whole manifest is present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MANIFEST = Path(os.environ.get("INFB_MODEL_MANIFEST", "/opt/infb/model-manifest.json"))
HF_CACHE = Path(os.environ.get("HF_HUB_CACHE", "/hf/hub"))
RECORD = HF_CACHE / ".infb" / "manifest.json"
STAGING = Path("/infb-model-staging")


def repo_folder(repo_id: str) -> Path:
    return HF_CACHE / ("models--" + repo_id.replace("/", "--"))


def partition(files: list[dict], n: int) -> list[list[dict]]:
    """Greedy balanced partition by size; deterministic (largest first, lightest bin wins)."""
    bins: list[list[dict]] = [[] for _ in range(n)]
    loads = [0] * n
    for f in sorted(files, key=lambda f: (-int(f["size"]), f["path"])):
        i = min(range(n), key=lambda k: (loads[k], k))
        bins[i].append(f)
        loads[i] += int(f["size"])
    return [sorted(b, key=lambda f: f["path"]) for b in bins]


def digests(path: Path) -> tuple[str, str]:
    """(sha256 of the content, git object id of the content) in one read."""
    size = path.stat().st_size
    sha256 = hashlib.sha256()
    git = hashlib.sha1(b"blob %d\0" % size)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(64 << 20), b""):
            sha256.update(chunk)
            git.update(chunk)
    return sha256.hexdigest(), git.hexdigest()


def problem(f: dict, path: Path) -> str | None:
    """None when the file at ``path`` is exactly the manifest's file, else what is wrong."""
    if not path.is_file():
        return "missing"
    size = path.stat().st_size
    if size != int(f["size"]):
        return f"size {size} != {f['size']}"
    sha256, git = digests(path)
    if f.get("sha256"):
        if sha256 != f["sha256"]:
            return f"sha256 {sha256} != {f['sha256']}"
    elif git != f["blob_id"]:
        return f"git blob id {git} != {f['blob_id']}"
    return None


def etag(f: dict) -> str:
    """The name ``snapshot_download`` gives the blob: the LFS sha256, else the git object id."""
    return f.get("sha256") or f["blob_id"]


def place_from_staging(repo_id: str, revision: str, f: dict) -> bool:
    """Copy one staged file into the cache layout if it verifies; False when it is absent or does not."""
    src = STAGING / repo_id / f["path"]
    if not src.is_file():
        return False
    why = problem(f, src)
    if why:
        print(f"[fetch_model] staged {f['path']} rejected ({why}); fetching it from the Hub", flush=True)
        return False
    folder = repo_folder(repo_id)
    blob = folder / "blobs" / etag(f)
    link = folder / "snapshots" / revision / f["path"]
    blob.parent.mkdir(parents=True, exist_ok=True)
    link.parent.mkdir(parents=True, exist_ok=True)
    tmp = blob.with_name(blob.name + ".incomplete")
    shutil.copyfile(src, tmp)
    os.replace(tmp, blob)
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(os.path.relpath(blob, link.parent))
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--group", type=int, required=True)
    ap.add_argument("--of", type=int, required=True)
    ap.add_argument(
        "--finalize",
        action="store_true",
        help="after this group, write refs/main and assert the whole manifest is present",
    )
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    repo_id, revision = manifest["repo_id"], manifest["revision"]
    snapshot = repo_folder(repo_id) / "snapshots" / revision
    group = partition(manifest["files"], args.of)[args.group]
    total = sum(int(f["size"]) for f in group)
    print(
        f"[fetch_model] group {args.group}/{args.of}: {len(group)} files, "
        f"{total / 1e9:.1f} GB -> {snapshot}",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=8) as pool:
        staged = list(pool.map(lambda f: place_from_staging(repo_id, revision, f), group))
    from_hub = [f["path"] for f, ok in zip(group, staged) if not ok]
    sources = {f["path"]: ("staging" if ok else "hub") for f, ok in zip(group, staged)}
    if from_hub:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=from_hub,
            cache_dir=str(HF_CACHE),
            max_workers=8,
        )
    print(
        f"[fetch_model] {len(group) - len(from_hub)} files from the staged copy, {len(from_hub)} from the Hub",
        flush=True,
    )

    def check(f: dict) -> dict:
        p = snapshot / f["path"]
        why = problem(f, p)
        if why:
            raise SystemExit(f"[fetch_model] {p}: {why}")
        sha256, git = digests(p)
        return {
            "path": f["path"],
            "size": p.stat().st_size,
            "sha256": sha256,
            "git_blob_id": git,
            "hub_sha256": f.get("sha256"),
            "source": sources[f["path"]],
        }

    with ThreadPoolExecutor(max_workers=8) as pool:
        verified = list(pool.map(check, group))

    RECORD.parent.mkdir(parents=True, exist_ok=True)
    record = (
        json.loads(RECORD.read_text())
        if RECORD.exists()
        else {"repo_id": repo_id, "revision": revision, "files": {}}
    )
    for v in verified:
        record["files"][v["path"]] = v
    RECORD.write_text(json.dumps(record, indent=1, sort_keys=True))
    print(
        f"[fetch_model] verified {len(verified)} files; "
        f"recorded {len(record['files'])}/{len(manifest['files'])}",
        flush=True,
    )

    if args.finalize:
        missing = [f["path"] for f in manifest["files"] if f["path"] not in record["files"]]
        if missing:
            raise SystemExit(f"[fetch_model] manifest incomplete after the last group: {missing}")
        refs = repo_folder(repo_id) / "refs"
        refs.mkdir(parents=True, exist_ok=True)
        (refs / "main").write_text(revision)
        record["complete"] = True
        record["total_bytes"] = sum(int(v["size"]) for v in record["files"].values())
        record["refs_main"] = revision
        RECORD.write_text(json.dumps(record, indent=1, sort_keys=True))
        print(
            f"[fetch_model] complete: {len(record['files'])} files, "
            f"{record['total_bytes']} bytes; refs/main -> {revision}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
