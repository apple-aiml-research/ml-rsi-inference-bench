#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Fetch the files that two groups of the engine's own Rust tests read at run time, so that no test needs the network
(image build only; standard library only).

  fetch_offline_test_files.py <offline-test-files.json>

- The Hugging Face files of the eight models `rust/src/chat/tests/roundtrip.rs` loads (the `vllm-chat` crate's
  `roundtrip_*` tests), at their pinned revisions, into the image's Hugging Face cache:
  ``$HF_HUB_CACHE/models--<org>--<name>/snapshots/<revision>/<file>`` with ``refs/main`` naming the revision. The Rust
  client resolves a model from this cache before it would reach for the Hub, and it does not honor
  ``HF_HUB_OFFLINE``, so without these files the tests fail offline and are rate-limit flaky online.
- The gpt-oss harmony encoding's vocabulary (the crate's ``output::harmony::tests``) into ``/opt/infb/tiktoken``,
  which the recipe's ``TIKTOKEN_ENCODINGS_BASE`` names. openai-harmony loads the file from that directory and checks
  it against the sha256 pinned in the crate, instead of downloading it from openaipublic.blob.core.windows.net.

Every file is pinned in the manifest by repository revision (or URL), size and sha256, and is verified before it is
written; any mismatch fails the build.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HUB = "https://huggingface.co"
HF_CACHE = Path(os.environ.get("HF_HUB_CACHE", "/hf/hub"))
TIKTOKEN_DIR = Path("/opt/infb/tiktoken")
RETRYABLE = (429, 500, 502, 503, 504)


def fetch(url: str, size: int, sha256: str, attempts: int = 10) -> bytes:
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "infb-image-build"})
            with urllib.request.urlopen(request, timeout=300) as response:
                data = response.read()
            break
        except urllib.error.HTTPError as exc:
            if exc.code not in RETRYABLE or attempt == attempts - 1:
                raise
        except (urllib.error.URLError, OSError):
            if attempt == attempts - 1:
                raise
        time.sleep(min(60, 5 * 2 ** attempt))
    digest = hashlib.sha256(data).hexdigest()
    if len(data) != size or digest != sha256:
        sys.exit(f"{url}: {len(data)} B, sha256 {digest}; the manifest pins {size} B, sha256 {sha256}")
    return data


def main() -> None:
    manifest = json.loads(Path(sys.argv[1]).read_text())
    for model in manifest["hub"]:
        repo, revision = model["repo"], model["revision"]
        folder = HF_CACHE / ("models--" + repo.replace("/", "--"))
        snapshot = folder / "snapshots" / revision
        snapshot.mkdir(parents=True, exist_ok=True)
        for entry in model["files"]:
            data = fetch(f"{HUB}/{repo}/resolve/{revision}/{entry['file']}", entry["bytes"], entry["sha256"])
            (snapshot / entry["file"]).write_bytes(data)
        (folder / "refs").mkdir(exist_ok=True)
        (folder / "refs" / "main").write_text(revision)
        print(f"{repo}@{revision[:12]}: {len(model['files'])} files", flush=True)
    TIKTOKEN_DIR.mkdir(parents=True, exist_ok=True)
    for entry in manifest["tiktoken"]:
        (TIKTOKEN_DIR / entry["file"]).write_bytes(fetch(entry["url"], entry["bytes"], entry["sha256"]))
        print(f"{entry['file']}: {entry['bytes']} B", flush=True)


if __name__ == "__main__":
    main()
