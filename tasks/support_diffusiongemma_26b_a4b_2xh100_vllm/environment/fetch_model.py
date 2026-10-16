#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Download one group of a pinned checkpoint's files from the Hub and verify them (image build only).

The manifest (model-manifest.json) lists, for each of the task's two checkpoints, every file of the
pinned revision with its size and sha256 (the Hub's own LFS sha256 where the file is in LFS, and
the digest of the verified download otherwise, beside the Hub's git blob id). A repository's files
are partitioned deterministically into ``--of`` groups of similar size so each Docker layer stays a
few GB: for the DiffusionGemma checkpoint, twelve groups are one per safetensors shard plus one for
the nine small files. Group ``--group`` is downloaded into the image's Hugging Face cache
(``HF_HUB_CACHE``), every file's size and sha256 are checked, and the digests are recorded in
``$HF_HUB_CACHE/.infb/manifest.json``, which preflight reads at grading time. ``--finalize`` on a
repository's last group writes ``refs/main`` with the pinned revision and asserts the repository is
complete.

    fetch_model.py --group 0 --of 12                                   # the DiffusionGemma checkpoint
    fetch_model.py --repo Qwen/Qwen2.5-0.5B-Instruct --group 0 --of 1 --finalize
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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", help="repository id (default: the manifest's first model)")
    ap.add_argument("--group", type=int, required=True)
    ap.add_argument("--of", type=int, required=True)
    ap.add_argument("--finalize", action="store_true",
                    help="after this group, write refs/main and assert the repository is complete")
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    models = {m["repo_id"]: m for m in manifest["models"]}
    model = models[args.repo] if args.repo else manifest["models"][0]
    repo_id, revision = model["repo_id"], model["revision"]
    snapshot = repo_folder(repo_id) / "snapshots" / revision
    group = partition(model["files"], args.of)[args.group]
    names = [f["path"] for f in group]
    total = sum(int(f["size"]) for f in group)
    print(f"[fetch_model] {repo_id} group {args.group}/{args.of}: {len(names)} files, "
          f"{total / 1e9:.2f} GB -> {snapshot}", flush=True)

    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id, revision=revision, allow_patterns=names,
                      cache_dir=str(HF_CACHE), max_workers=8)

    def check(f: dict) -> dict:
        p = snapshot / f["path"]
        if not p.is_file():
            raise SystemExit(f"[fetch_model] missing after download: {p}")
        size = p.stat().st_size
        if size != int(f["size"]):
            raise SystemExit(f"[fetch_model] size mismatch {p}: {size} != {f['size']}")
        digest = sha256_of(p)
        if digest != f["sha256"]:
            raise SystemExit(f"[fetch_model] sha256 mismatch {p}: {digest} != {f['sha256']}")
        return {"path": f["path"], "size": size, "sha256": digest}

    with ThreadPoolExecutor(max_workers=8) as pool:
        verified = list(pool.map(check, group))

    RECORD.parent.mkdir(parents=True, exist_ok=True)
    record = json.loads(RECORD.read_text()) if RECORD.exists() else {"models": {}}
    entry = record["models"].setdefault(repo_id, {"revision": revision, "files": {}})
    for v in verified:
        entry["files"][v["path"]] = v
    RECORD.write_text(json.dumps(record, indent=1, sort_keys=True))
    print(f"[fetch_model] verified {len(verified)} files; recorded {len(entry['files'])}/{len(model['files'])} "
          f"of {repo_id}", flush=True)

    if args.finalize:
        missing = [f["path"] for f in model["files"] if f["path"] not in entry["files"]]
        if missing:
            raise SystemExit(f"[fetch_model] {repo_id} incomplete after its last group: {missing}")
        refs = repo_folder(repo_id) / "refs"
        refs.mkdir(parents=True, exist_ok=True)
        (refs / "main").write_text(revision)
        entry.update(complete=True, refs_main=revision,
                     total_bytes=sum(int(v["size"]) for v in entry["files"].values()))
        RECORD.write_text(json.dumps(record, indent=1, sort_keys=True))
        print(f"[fetch_model] {repo_id} complete: {len(entry['files'])} files, {entry['total_bytes']} bytes; "
              f"refs/main -> {revision}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
