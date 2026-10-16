#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Fetch the L3 instances' Hugging Face files into the verifier image's offline cache.

    python fetch_l3_hf.py l3-hf-manifest.json HF_HUB_CACHE

For every repository under "fetched": each file at the pinned revision, from huggingface.co, checked against its
sha256, written as <cache>/models--<org>--<name>/snapshots/<revision>/<file>, and refs/main set to the revision,
so an offline lookup of the repository resolves to exactly these files. Refuses (nonzero) on any mismatch.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path


def get(url: str) -> bytes:
    for attempt in range(6):
        try:
            return urllib.request.urlopen(url, timeout=300).read()
        except Exception:  # noqa: BLE001 -- retried, then raised
            if attempt == 5:
                raise
            time.sleep(10 * (attempt + 1))
    raise AssertionError


def main(manifest: str, cache: str) -> int:
    m = json.loads(Path(manifest).read_text())
    root = Path(cache)
    for repo, e in m["fetched"].items():
        rev = e["revision"]
        base = root / ("models--" + repo.replace("/", "--"))
        snap = base / "snapshots" / rev
        for name, f in e["files"].items():
            data = get(f"https://huggingface.co/{repo}/resolve/{rev}/{name}")
            got = hashlib.sha256(data).hexdigest()
            if got != f["sha256"] or len(data) != f["bytes"]:
                print(f"[fetch_l3_hf] {repo}@{rev}:{name}: sha256 {got} ({len(data)} B), expected {f['sha256']} ({f['bytes']} B)")
                return 3
            out = snap / name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
        (base / "refs").mkdir(parents=True, exist_ok=True)
        (base / "refs" / "main").write_text(rev)
        print(f"[fetch_l3_hf] {repo}@{rev}: {len(e['files'])} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
