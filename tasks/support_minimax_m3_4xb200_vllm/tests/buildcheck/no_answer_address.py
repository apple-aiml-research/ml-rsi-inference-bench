#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Fail the build if the answer's address is on the agent's surface.

Nothing the agent can read says which pull requests this task reconstructs: no task-authored file the image
ships names the landing or the repair commits, the pull requests by number (the landing, the two admitted-by-test
repair candidates and the two closed release-branch originals they relanded), the landing author, or the pinned
revision of the MSA package the landing fetched; the prepared commits' messages and identities name none of them;
and none of those commits is an object in /app's store (the recipe clones at the upstream head, drops later
branches, expires the reflog and prunes -- this proves the prune happened).

Scanned: /opt/infb (minus the tools venv of public packages), /public, /app/eval and the checkpoint's text files
(the model author's; a hit there would be a finding to report, not something to edit). Third-party venvs are not
scanned: no package published to PyPI can know this task. BIND-MOUNTED at build, never copied into the image.
"""

from __future__ import annotations

import bz2
import gzip
import lzma
import re
import subprocess
import sys
from pathlib import Path

FORBIDDEN_COMMITS = (
    "0a1c5034f5e4fe736db672010cda33d9d850f87e",
    "4c626633159887b0f2c962058c17c78f1434556d",
    "d8e422ccda9b39ca8a0756b5b195513045abcee8",
)
FORBIDDEN_IDENTITIES = FORBIDDEN_COMMITS + (
    "544eee5e09ae2dfa774d5b06739013f9b7402c57",
)
PATTERNS = {
    "reference source identity": re.compile("|".join(sha[:9] for sha in FORBIDDEN_IDENTITIES)),
    "reference PR by number": re.compile(r"(?:#|\bPR\s*#?|pull/|pulls/|issues/)(?:45381|45744|45718|45680|45713)\b",
                                         re.IGNORECASE),
    "landing PR author": re.compile(r"youkaichao", re.IGNORECASE),
}
ROOTS = [Path("/opt/infb"), Path("/public"), Path("/model"), Path("/app/eval")]
SKIP = [Path("/opt/infb/tools")]
TEXT_SUFFIXES = {".md", ".txt", ".json", ".jsonl", ".py", ".sh", ".toml", ".yaml", ".yml", ".lock",
                 ".patch", ".cfg", ".ini", ".jinja", ""}
COMPRESSED_SUFFIXES = {".xz": lzma.open, ".lzma": lzma.open, ".gz": gzip.open, ".bz2": bz2.open}
MAX_BYTES = 64 * 1024 * 1024


def read_scannable(path: Path) -> str | None:
    try:
        if path.stat().st_size > MAX_BYTES:
            return None
        opener = COMPRESSED_SUFFIXES.get(path.suffix)
        if opener is not None:
            try:
                with opener(path, "rb") as fh:
                    return fh.read(MAX_BYTES).decode("utf-8", errors="ignore")
            except (OSError, EOFError, lzma.LZMAError):
                pass
            return path.read_text(encoding="utf-8", errors="ignore")
        if path.suffix not in TEXT_SUFFIXES:
            return None
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None


def main() -> int:
    hits: list[str] = []
    scanned = 0
    for root in ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if any(path.is_relative_to(s) for s in SKIP) or not path.is_file():
                continue
            if path.suffix not in TEXT_SUFFIXES and path.suffix not in COMPRESSED_SUFFIXES:
                continue
            text = read_scannable(path)
            if text is None:
                continue
            scanned += 1
            hits += [f"{path}: {name}" for name, rx in PATTERNS.items() if rx.search(text)]

    log = subprocess.run(["git", "-C", "/app", "log", "--format=%an%n%ae%n%cn%n%ce%n%B", "-2"],
                         capture_output=True, text=True, check=True).stdout
    hits += [f"prepared commits: {n}" for n, rx in PATTERNS.items() if rx.search(log)]
    for sha in FORBIDDEN_COMMITS:
        if subprocess.run(["git", "-C", "/app", "cat-file", "-e", f"{sha}^{{commit}}"],
                          capture_output=True).returncode == 0:
            hits.append(f"/app/.git: excluded source {sha[:12]} remains in the object store")

    if hits:
        print("FATAL: the answer's address is on the agent's surface:", file=sys.stderr)
        for h in hits:
            print("  " + h, file=sys.stderr)
        return 1
    print(f"no answer address: {scanned} task-authored files, the prepared commits and the object store clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
