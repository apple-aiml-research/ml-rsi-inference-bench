#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Download one group of the checkpoint's files from the Hub and verify them (image build only).

The manifest (model-manifest.json) lists every file of the pinned revision with its size and, for
LFS files, the sha256 the Hub reports. The files are partitioned deterministically into ``--of``
groups of similar size so each Docker layer stays a few GB; group ``--group`` is downloaded into
the image's Hugging Face cache (``HF_HUB_CACHE``), every file's size is checked, every file's
sha256 is computed and compared with the manifest where the manifest has one, and the computed
digests are appended to ``$HF_HUB_CACHE/.infb/manifest.json``, which preflight reads at grading
time.

Why the Hub cache layout and not a plain directory: the upstream acceptance test this task freezes
launches ``vllm serve Qwen/Qwen3.8-Flash-Next-FP8`` -- the repository id, not a path -- so the only
way to run that test *verbatim* offline is for the id to resolve from the Hub cache with
``HF_HUB_OFFLINE=1``. ``snapshot_download`` writes ``blobs/`` plus a ``snapshots/<revision>/`` tree
of symlinks; ``--finalize`` additionally writes ``refs/main`` with the pinned revision, which is
what a download at ``revision="main"`` would have written and what the offline resolver reads when
no revision is given. The revision is asserted to be the repository head at authoring time; the
landing record names it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MANIFEST = Path(os.environ.get("INFB_MODEL_MANIFEST", "/opt/infb/model-manifest.json"))
HF_CACHE = Path(os.environ.get("HF_HUB_CACHE", "/hf/hub"))
RECORD = HF_CACHE / ".infb" / "manifest.json"


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


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(64 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
    names = [f["path"] for f in group]
    total = sum(int(f["size"]) for f in group)
    print(
        f"[fetch_model] group {args.group}/{args.of}: {len(names)} files, "
        f"{total / 1e9:.1f} GB -> {snapshot}",
        flush=True,
    )

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        allow_patterns=names,
        cache_dir=str(HF_CACHE),
        max_workers=8,
    )

    def check(f: dict) -> dict:
        p = snapshot / f["path"]
        if not p.is_file():
            raise SystemExit(f"[fetch_model] missing after download: {p}")
        size = p.stat().st_size
        if size != int(f["size"]):
            raise SystemExit(f"[fetch_model] size mismatch {p}: {size} != {f['size']}")
        digest = sha256_of(p)
        if f.get("sha256") and digest != f["sha256"]:
            raise SystemExit(f"[fetch_model] sha256 mismatch {p}: {digest} != {f['sha256']}")
        return {
            "path": f["path"],
            "size": size,
            "sha256": digest,
            "hub_sha256": f.get("sha256"),
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
