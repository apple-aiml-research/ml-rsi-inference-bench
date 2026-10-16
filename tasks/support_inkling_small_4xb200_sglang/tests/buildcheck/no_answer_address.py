#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Fail the build if the answer's address is on the agent's surface.

Nothing the image ships may say which pull requests the task reconstructs, or let them be recovered:

  * no task-authored file the image ships names the landing squash, its predecessor, the repairs, the
    registered tests' pull requests or the related unmerged and later ones, by number or by SHA, or
    the landing author's handle -- `/opt/infb` (minus its pip venv of public packages), `/public`,
    `/app/eval`; the checkpoint's own text files are the model author's and are not scanned (its model
    card links only the engine's cookbook URL);
  * the prepared-base commit's message and identity name none of them;
  * none of the landing's, the repairs' or the registered tests' commit OBJECTS is in /app's store.

Third-party venvs are not scanned, deliberately: a five-digit number turns up by chance in gigabytes
of wheel data, and no package published to PyPI can know this task. Bind-mount this file; never copy
it into the image (it carries the addresses it looks for).
"""

from __future__ import annotations

import bz2
import gzip
import lzma
import re
import subprocess
import sys
from pathlib import Path

#: The landing, the two repair candidates, and the registered test's three pull requests.
FORBIDDEN_COMMITS = (
    "02236fa38cb065660a0b4156aab283a3807a5290",
    "25b70150646579f1137d6b00ddc3967493d13354",
    "ce84df0fa111c992eb8cc3efc03a2941c9ce18fa",
    "b20c375c10443e4f4a5656689a04d514194364fd",
    "c500674124c6b8c4edca5f4284e8ec0189fcce09",
    "fb72a37fdefa80ac00b23096130d5f815bbe2136",
)
#: The landing and its predecessor, the repairs and registered tests, and the related unmerged or later.
FORBIDDEN_PRS = ("31681", "31358", "34184", "33417", "34168", "32402", "34405", "33000", "32480", "35161", "31940")
PATTERNS = {
    "reference source identity": re.compile("|".join(sha[:8] for sha in FORBIDDEN_COMMITS)),
    "reference PR by number": re.compile(
        r"(?:#|\bPR\s*#?|pull/|pulls/)(?:" + "|".join(FORBIDDEN_PRS) + r")\b", re.IGNORECASE),
    "landing PR author": re.compile(r"\bch-wan\b", re.IGNORECASE),
}
ROOTS = [Path("/opt/infb"), Path("/public"), Path("/app/eval")]
SKIP = [Path("/opt/infb/tools")]  # the fetch/lock tooling venv: public packages only
TEXT_SUFFIXES = {".md", ".txt", ".json", ".jsonl", ".py", ".sh", ".toml", ".yaml", ".yml", ".lock",
                 ".patch", ".cfg", ".ini", ".jinja", ""}
COMPRESSED_SUFFIXES = {".xz": lzma.open, ".lzma": lzma.open, ".gz": gzip.open, ".bz2": bz2.open}
MAX_BYTES = 64 * 1024 * 1024


def read_scannable(path: Path) -> str | None:
    """The text of a file this check can scan, or None; compressed files are decompressed."""
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
