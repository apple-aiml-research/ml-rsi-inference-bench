#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Download one group of the checkpoint's files from the Hub and verify them (image build only).

The manifest (model-manifest.json) lists every file of the pinned revision with its size and, for
LFS files, the sha256 the Hub reports. The files are partitioned deterministically into ``--of``
groups of similar size so each Docker layer stays around 45 GB; group ``--group`` is downloaded
into DEFAULT_MODEL_CACHE_DIR/<repo id>, every file's size is checked, every file's sha256 is
computed and compared with the manifest where the manifest has one, and the computed digests are
appended to /model/.infb/manifest.json, which preflight reads at grading time.

One file of the model-author source is ADAPTED before this RUN ends, after its original bytes are
verified: ``inference/kernel.py`` has its decorated ``sparse_attn_kernel`` definition and its
``sparse_attn`` wrapper body replaced by signature-preserving unavailable-operation stubs (the
optimized sparse-attention computation is the solver's work in this task; every other author kernel
and ``model.py`` are untouched). Both the original and the adapted digests and sizes
go to /model/.infb/source-adaptations.json; the effective record in /model/.infb/manifest.json records the
supplied bytes. The original bytes are never left in the image: the file is rewritten in place in the same RUN
that downloaded it, and nothing else of the download is kept.
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
MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "/model"))
RECORD = MODEL_ROOT / ".infb" / "manifest.json"
ADAPTATIONS_RECORD = MODEL_ROOT / ".infb" / "source-adaptations.json"

_STUB_NOTE = ("omitted from this adapted model-author source (Inference-Bench, /model/.infb/source-adaptations.json); "
              "the operation is specified in /app/eval/deepseek_v4_flash/kernels/README.md")
ADAPTATIONS = {
    "inference/kernel.py": {
        "operation": "stub sparse_attn_kernel and the sparse_attn wrapper body",
        "original": {"sha256": "59b325083d7103975cba025bd0d60ea343bb82d8fff53088afb7c04bd380c0c2", "size": 22198},
        "adapted": {"sha256": "8726d5a7250bb2934be70c0ffe85422252547ead75ca718cea9420239efab9fb", "size": 19265},
        "begin": "@tilelang.jit(pass_configs=pass_configs)\ndef sparse_attn_kernel(",
        "middle": "\n\n\ndef sparse_attn(\n",
        "end": "\n\n\n@tilelang.jit(pass_configs=pass_configs)\ndef hc_split_sinkhorn_kernel(",
        "replacement": (
            "def sparse_attn_kernel(h: int, d: int, scale=None):\n"
            f'    """Unavailable: the sparse-attention kernel is {_STUB_NOTE}."""\n'
            f'    raise NotImplementedError("sparse_attn_kernel is {_STUB_NOTE}")\n'
            "\n\n"
            "def sparse_attn(\n"
            "    q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor, topk_idxs: torch.Tensor, softmax_scale: float\n"
            ") -> torch.Tensor:\n"
            f'    """Unavailable: the sparse-attention computation is {_STUB_NOTE}."""\n'
            f'    raise NotImplementedError("sparse_attn is {_STUB_NOTE}")'
        ),
    },
}


def adapt(text: str, spec: dict) -> str:
    """The adapted text: everything outside [begin, end) byte for byte, the bounded span replaced."""
    for anchor in ("begin", "middle", "end"):
        if text.count(spec[anchor]) != 1:
            raise SystemExit(f"[fetch_model] adaptation anchor {anchor!r} is not unique in the original: refusing")
    a, b, c = text.index(spec["begin"]), text.index(spec["middle"]), text.index(spec["end"])
    if not a < b < c:
        raise SystemExit("[fetch_model] adaptation anchors out of order: refusing")
    return text[:a] + spec["replacement"] + text[c:]


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
    ap.add_argument("--finalize", action="store_true", help="after this group, assert the whole manifest is present")
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    repo_id, revision = manifest["repo_id"], manifest["revision"]
    dest = MODEL_ROOT / repo_id
    dest.mkdir(parents=True, exist_ok=True)
    group = partition(manifest["files"], args.of)[args.group]
    names = [f["path"] for f in group]
    total = sum(int(f["size"]) for f in group)
    print(f"[fetch_model] group {args.group}/{args.of}: {len(names)} files, {total / 1e9:.1f} GB -> {dest}", flush=True)

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        allow_patterns=names,
        local_dir=str(dest),
        max_workers=8,
    )
    shutil.rmtree(dest / ".cache", ignore_errors=True)

    def check(f: dict) -> dict:
        p = dest / f["path"]
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

    adapted = []
    for v in verified:
        spec = ADAPTATIONS.get(v["path"])
        if spec is None:
            continue
        p = dest / v["path"]
        if v["sha256"] != spec["original"]["sha256"] or v["size"] != spec["original"]["size"]:
            raise SystemExit(f"[fetch_model] {p} is not the pinned original ({v['sha256']}, {v['size']} B): refusing to adapt")
        new = adapt(p.read_text(encoding="utf-8"), spec).encode("utf-8")
        got = hashlib.sha256(new).hexdigest()
        if got != spec["adapted"]["sha256"] or len(new) != spec["adapted"]["size"]:
            raise SystemExit(f"[fetch_model] the adapted {p} is {got}, {len(new)} B; expected {spec['adapted']}")
        p.write_bytes(new)
        if sha256_of(p) != got:
            raise SystemExit(f"[fetch_model] {p} did not read back as written")
        adapted.append({"path": v["path"], "operation": spec["operation"], "original": dict(spec["original"]),
                        "adapted": {"sha256": got, "size": len(new)}})
        v.update({"size": len(new), "sha256": got, "adapted_from_sha256": spec["original"]["sha256"]})
        print(f"[fetch_model] adapted {v['path']}: {spec['original']['sha256'][:12]} -> {got[:12]}", flush=True)
    if adapted:
        ADAPTATIONS_RECORD.parent.mkdir(parents=True, exist_ok=True)
        ADAPTATIONS_RECORD.write_text(json.dumps({
            "schema": "infb-source-adaptations/1", "repo_id": repo_id, "revision": revision,
            "note": "model-author source adapted for this task; the effective /model/.infb/manifest.json records the "
                    "supplied (adapted) bytes; environment/model-manifest.json keeps the Hub listing as provenance",
            "files": adapted}, indent=1, sort_keys=True))
    for junk in [q for q in dest.rglob("*") if q.name == "__pycache__" or q.suffix in (".pyc", ".pyo")]:
        shutil.rmtree(junk, ignore_errors=True) if junk.is_dir() else junk.unlink(missing_ok=True)

    RECORD.parent.mkdir(parents=True, exist_ok=True)
    record = json.loads(RECORD.read_text()) if RECORD.exists() else {
        "repo_id": repo_id, "revision": revision, "files": {}}
    for v in verified:
        record["files"][v["path"]] = v
    RECORD.write_text(json.dumps(record, indent=1, sort_keys=True))
    print(f"[fetch_model] verified {len(verified)} files; recorded {len(record['files'])}/{len(manifest['files'])}", flush=True)

    if args.finalize:
        missing = [f["path"] for f in manifest["files"] if f["path"] not in record["files"]]
        if missing:
            raise SystemExit(f"[fetch_model] manifest incomplete after the last group: {missing}")
        record["complete"] = True
        record["total_bytes"] = sum(int(v["size"]) for v in record["files"].values())
        RECORD.write_text(json.dumps(record, indent=1, sort_keys=True))
        print(f"[fetch_model] complete: {len(record['files'])} files, {record['total_bytes']} bytes", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
