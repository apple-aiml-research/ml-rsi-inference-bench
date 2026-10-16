#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Download one group of the checkpoint's files from the Hub and verify them (image build only).

The checkpoint goes into the **Hub cache layout** rather than a plain directory, and `/model` is a
symlink to the snapshot, so the same bytes are reachable two ways: by repo id
(`openai/gpt-oss-120b`, which is how the engine's launcher is given a model when it is given a repo
id) and by path (`/model`, which is what the instruction names). `--finalize` writes the `refs/main`
that `snapshot_download` omits for a commit-sha revision, creates the symlink, and proves both
routes resolve offline before the build continues.

The manifest (model-manifest.json) lists every file of the pinned revision with its size and, for
LFS files, the sha256 the Hub reports. The files are partitioned deterministically into ``--of``
groups of similar size so each image layer stays well under the registry's per-blob cap; group
``--group`` is downloaded, every file's size is checked, every file's sha256 is computed and compared
with the manifest where the manifest has one, and the computed digests are written to the record the
verifier's preflight reads at grading time.
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
CACHE_ROOT = Path(os.environ.get("HF_HUB_CACHE", "/model-cache"))
RECORD = Path(os.environ.get("INFB_MODEL_RECORD", "/opt/infb/checkpoint.json"))
MODEL_LINK = Path(os.environ.get("INFB_MODEL_LINK", "/model"))


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


def repo_cache_dir(repo_id: str) -> Path:
    return CACHE_ROOT / ("models--" + repo_id.replace("/", "--"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--group", type=int, required=True)
    ap.add_argument("--of", type=int, required=True)
    ap.add_argument("--finalize", action="store_true", help="after this group, assert the whole manifest is present, write refs/main and link /model")
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    repo_id, revision = manifest["repo_id"], manifest["revision"]
    group = partition(manifest["files"], args.of)[args.group]
    names = [f["path"] for f in group]
    total = sum(int(f["size"]) for f in group)
    print(f"[fetch_model] group {args.group}/{args.of}: {len(names)} files, {total / 1e9:.1f} GB -> {CACHE_ROOT}", flush=True)

    from huggingface_hub import snapshot_download

    snap = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=names,
            cache_dir=str(CACHE_ROOT),
            max_workers=8,
        )
    )

    def check(f: dict) -> dict:
        p = snap / f["path"]
        if not p.is_file():
            raise SystemExit(f"[fetch_model] missing after download: {p}")
        size = p.stat().st_size
        if size != int(f["size"]):
            raise SystemExit(f"[fetch_model] size mismatch {p}: {size} != {f['size']}")
        digest = sha256_of(p)
        if f.get("sha256") and digest != f["sha256"]:
            raise SystemExit(f"[fetch_model] sha256 mismatch {p}: {digest} != {f['sha256']}")
        return {"path": f["path"], "size": size, "sha256": digest, "hub_sha256": f.get("sha256")}

    with ThreadPoolExecutor(max_workers=8) as pool:
        verified = list(pool.map(check, group))

    RECORD.parent.mkdir(parents=True, exist_ok=True)
    record = json.loads(RECORD.read_text()) if RECORD.exists() else {
        "repo_id": repo_id, "revision": revision, "snapshot": str(snap), "files": {}}
    for v in verified:
        record["files"][v["path"]] = v
    RECORD.write_text(json.dumps(record, indent=1, sort_keys=True))
    print(f"[fetch_model] verified {len(verified)} files; recorded {len(record['files'])}/{len(manifest['files'])}", flush=True)

    if args.finalize:
        missing = [f["path"] for f in manifest["files"] if f["path"] not in record["files"]]
        if missing:
            raise SystemExit(f"[fetch_model] manifest incomplete after the last group: {missing}")
        refs = repo_cache_dir(repo_id) / "refs"
        refs.mkdir(parents=True, exist_ok=True)
        (refs / "main").write_text(revision)
        if MODEL_LINK.is_symlink() or MODEL_LINK.exists():
            MODEL_LINK.unlink()
        MODEL_LINK.symlink_to(snap)
        record["complete"] = True
        record["total_bytes"] = sum(int(v["size"]) for v in record["files"].values())
        RECORD.write_text(json.dumps(record, indent=1, sort_keys=True))
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_HUB_CACHE"] = str(CACHE_ROOT)
        from huggingface_hub import try_to_load_from_cache

        for rev in (revision, None):
            hit = try_to_load_from_cache(repo_id, "config.json", cache_dir=str(CACHE_ROOT),
                                         **({"revision": rev} if rev else {}))
            if not isinstance(hit, str):
                raise SystemExit(f"[fetch_model] config.json does not resolve from the cache at revision {rev!r}: {hit!r}")
        if not (MODEL_LINK / "config.json").is_file():
            raise SystemExit(f"[fetch_model] {MODEL_LINK}/config.json is not readable through the symlink")
        print(f"[fetch_model] complete: {len(record['files'])} files, {record['total_bytes']} bytes; "
              f"{repo_id} resolves offline by repo id and through {MODEL_LINK}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
