#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Assert the L3 fixture repositories are in this image's offline cache at their pins.

    python check_l3_cache.py <manifest.json> HF_HUB_CACHE

`fetch_l3_hf.py` already refuses on a sha256 or size mismatch while it downloads, so this does not
re-check bytes. It checks the thing a download cannot: that the files landed where an OFFLINE lookup
will look for them, for EVERY repository the manifest pins, with `refs/main` at the pinned revision.
A fetch that silently wrote nothing, or wrote one repository of seven, leaves no other trace in the
image, and the failure would surface only as a pass-to-pass regression the agent cannot reproduce.

Every expected path is derived from the manifest, so this cannot drift from the pins it guards.
Refuses (nonzero) naming every repository and file that is missing or the wrong size.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main(manifest: str, cache: str) -> int:
    m = json.loads(Path(manifest).read_text())
    root = Path(cache)
    problems: list[str] = []
    repos = files = 0

    for repo, entry in sorted(m["fetched"].items()):
        repos += 1
        rev = entry["revision"]
        base = root / ("models--" + repo.replace("/", "--"))
        snap = base / "snapshots" / rev

        ref = base / "refs" / "main"
        if not ref.is_file():
            problems.append(f"{repo}: no refs/main, so an offline lookup cannot resolve the repository")
        elif ref.read_text().strip() != rev:
            problems.append(f"{repo}: refs/main is {ref.read_text().strip()!r}, pinned {rev!r}")

        for name, meta in sorted(entry["files"].items()):
            files += 1
            path = snap / name
            if not path.is_file():
                problems.append(f"{repo}@{rev}: {name} is absent from {snap}")
                continue
            size = path.stat().st_size
            if size != meta["bytes"]:
                problems.append(f"{repo}@{rev}: {name} is {size} B, pinned {meta['bytes']} B")

    if problems:
        print(f"[check_l3_cache] {len(problems)} problem(s) in {cache}:")
        for p in problems:
            print(f"[check_l3_cache]   {p}")
        return 1

    print(f"[check_l3_cache] {repos} repositories, {files} files, all present at their pinned revisions in {cache}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
