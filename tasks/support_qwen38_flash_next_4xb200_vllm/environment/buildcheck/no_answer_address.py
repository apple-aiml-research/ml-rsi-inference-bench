#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Fail the build if the answer's address is on the agent's surface.

The instruction promises the agent that nothing here says which pull request the task reconstructs.
This check makes the promise a build assertion instead of a claim:

  * no task-authored file the image ships names the landing squash, the PR by number, or its author
    -- `/opt/infb` (minus its pip venv of public packages), `/public`, and the checkpoint's own text
    files, which are the model author's and would be a finding to report, not something to edit;
  * the prepared-base commit's message and identity name none of them;
  * the landing squash's OBJECT is not in /app's store, so it cannot be recovered from history
    either (the recipe clones at the upstream head, then drops later branches, expires the reflog
    and prunes -- this proves the prune happened).

Third-party venvs are not scanned, deliberately: a five-digit number turns up by chance in gigabytes
of wheel data, and no package published to PyPI can know this task.
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
    "e126687a9a828d513c01a07cd69f025f27d63280",
    "f870b9297685bd3c968063b52b769315cb08fd7f",
    "28e605fb330093fee5a2b98aebc19d9c268c1822",
    "79468c20ef23227d4e051f807e10fd54fb24e24f",
    "7dbd0a8d22d20eb938994ad7a9158e02f30c9ce6",
    "bab9aff960e972d69b434e91c1b768d52d639ea0",
)
PATTERNS = {
    "reference source identity": re.compile('|'.join(sha[:8] for sha in FORBIDDEN_COMMITS)),
    "reference PR by number": re.compile(r"(?:#|\bPR\s*#?|pull/|pulls/)(?:53896|54517|55375|51565|58434)\b", re.IGNORECASE),
    "landing PR author": re.compile(r"peakcrosser7", re.IGNORECASE),
}
ROOTS = [Path("/opt/infb"), Path("/public"), Path("/model"), Path("/app/eval")]
SKIP = [Path("/opt/infb/tools")]  # the fetch/lock tooling venv: public packages only
TEXT_SUFFIXES = {".md", ".txt", ".json", ".jsonl", ".py", ".sh", ".toml", ".yaml", ".yml", ".lock",
                 ".patch", ".cfg", ".ini", ".jinja", ""}
#: COMPRESSED forms this check DECOMPRESSES rather than skips. The grader's measurement client ships
#: to the agent under /public/efficiency/ with a public natural-text sample as `sample.jsonl.xz`;
#: skipped by suffix, its 7 MB of text would have been the one part of /public this assertion never
#: looked at, and the assertion would have gone on passing.
COMPRESSED_SUFFIXES = {".xz": lzma.open, ".lzma": lzma.open, ".gz": gzip.open, ".bz2": bz2.open}
MAX_BYTES = 64 * 1024 * 1024  # the checkpoint's shards are skipped by suffix; this bounds the rest


def read_scannable(path: Path) -> str | None:
    """The text of a file this check can scan, or None when it is none of its business.

    A compressed file is decompressed; one that is NAMED compressed and is not is read as text, so a
    mislabelled file cannot slip past by its name alone.
    """
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


def scan_file(path: Path) -> list[str]:
    text = read_scannable(path)
    if text is None:
        return []
    return [f"{path}: {name}" for name, rx in PATTERNS.items() if rx.search(text)]


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
            scanned += 1
            hits += scan_file(path)

    head = subprocess.run(
        ["git", "-C", "/app", "log", "-1", "--format=%an%n%ae%n%cn%n%ce%n%B"],
        capture_output=True, text=True, check=True,
    ).stdout
    hits += [f"prepared-base commit: {n}" for n, rx in PATTERNS.items() if rx.search(head)]

    for sha in FORBIDDEN_COMMITS:
        present = subprocess.run(
            ["git", "-C", "/app", "cat-file", "-e", f"{sha}^{{commit}}"], capture_output=True
        ).returncode == 0
        if present:
            hits.append(f"/app/.git: excluded source {sha[:12]} remains in the object store")

    if hits:
        print("FATAL: the answer's address is on the agent's surface:", file=sys.stderr)
        for h in hits:
            print("  " + h, file=sys.stderr)
        return 1
    print(f"no answer address: {scanned} task-authored files, the prepared-base commit, and the "
          f"object store all clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
