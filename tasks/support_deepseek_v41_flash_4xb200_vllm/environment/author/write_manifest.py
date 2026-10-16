#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Write, and check, the manifest of the converted model-author shards (image build only).

    write_manifest.py --shards /model-author/deepseek-v41-mp4 --snapshot <checkpoint snapshot> \
        --fetch-record /hf/hub/.infb/manifest.json --expected expected-shards.json

The manifest (`<shards>/manifest.json`) links the four shards to the checkpoint they were converted from and to the
converter that produced them: the checkpoint's repository and revision with the digest of the image's fetch record
(every checkpoint file's size and sha256, verified when it was fetched), the converter `inference/convert.py` by
sha256, the exact command, and each shard's size and sha256. With `--expected`, every shard's digest must equal the
recorded one, so a drift in any input of the conversion fails the build instead of changing the shards silently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

COMMAND = ["python", "inference/convert.py", "--hf-ckpt-path", "<snapshot>", "--save-path", "/model-author/deepseek-v41-mp4",
           "--model-parallel", "4", "--expert-dtype", "fp4", "--tokenizer-path", "<snapshot>"]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(64 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--shards", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--fetch-record", required=True)
    ap.add_argument("--expected")
    a = ap.parse_args(argv)
    shards, snapshot = Path(a.shards), Path(a.snapshot)
    record = json.loads(Path(a.fetch_record).read_text())
    files = sorted(shards.glob("model*-mp4.safetensors"))
    if [f.name for f in files] != [f"model{i}-mp4.safetensors" for i in range(4)]:
        raise SystemExit(f"expected model0..3-mp4.safetensors in {shards}, found {[f.name for f in files]}")
    with ThreadPoolExecutor(max_workers=4) as pool:
        digests = list(pool.map(sha256_of, files))
    manifest = {
        "checkpoint": {"repo_id": record["repo_id"], "revision": record["revision"],
                       "fetch_record_sha256": sha256_of(Path(a.fetch_record)),
                       "files": len(record["files"]), "bytes": record.get("total_bytes")},
        "converter": {"path": "inference/convert.py", "sha256": sha256_of(snapshot / "inference" / "convert.py")},
        "command": COMMAND,
        "shards": [{"file": f.name, "bytes": f.stat().st_size, "sha256": d} for f, d in zip(files, digests)],
        "tokenizer_files": sorted(p.name for p in shards.iterdir() if p.name.startswith("tokenizer")),
    }
    if a.expected:
        expected = json.loads(Path(a.expected).read_text())
        want = {s["file"]: s["sha256"] for s in expected["shards"]}
        got = {s["file"]: s["sha256"] for s in manifest["shards"]}
        if want != got or expected["converter"]["sha256"] != manifest["converter"]["sha256"]:
            raise SystemExit(f"the converted shards differ from the recorded ones:\n  want {want}\n  got  {got}")
    (shards / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
