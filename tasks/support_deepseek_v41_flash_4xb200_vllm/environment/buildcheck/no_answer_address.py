#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Fail the build if the answer's address is on the agent's surface.

The instruction promises the agent that nothing here says which pull requests the task reconstructs.
This check makes the promise a build assertion instead of a claim.

Third-party venvs are not scanned, deliberately: a five-digit number turns up by chance in gigabytes
of wheel data, and no package published to PyPI can know this task.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

COMMITS = [
    "9b959b86577c082c0b2bf9e2c22263255a36ad83",
    "e77daef89e18e08321ae7b8b24827eedd5fe8673",
    "dc07f1638f73814b95776832b85df1cc92850416",
    "80447d27655918da6bfbccd0d3a40e975bda220a",
    "e7a3963339654aa18b713f9ff233f129af9c7c8b",
    "c4d424c2e27b392da235704ba0edca7ac48bd495",
    "12575123059c2142e39c9af4e217a4792c3c8176",
    "9ef37771beac1c1ce6a8b7ceae1f7ebeb6f51800",
    "66081d4c9c7d7c44f13fea402e5b622aa0f409c2",
]
PRS = ["56201", "56228", "56214", "56160", "57432", "56682", "56448", "56254", "58215", "56464"]
# Personal names are matched as whole words (a bare substring match reads GSM8K's "Andy loses" as the
# author "Andy Lo"); handles and e-mail fragments as they are.
NAMES = ["Roger Wang", "Yongye Zhu", "Jee Jee Li", "Woosuk Kwon", "Andy Lo", "Yicong Wang", "Jared Wen",
         "Juntian Liu", "Zhean Xu"]
HANDLES = ["rogerw.io", "zyy1102000", "jeejeelee", "andyluo7", "zjy0516", "riverclouds", "woosuk@",
           "andy@mistral", "wangyicong", "w13431838023", "juntianl777", "zheanxu"]
PATTERNS = {
    "landing or repair commit": re.compile("|".join(c[:8] for c in COMMITS)),
    "landing or repair PR by number": re.compile(
        r"(?:#|\bPR\s*#?|pull/|pulls/)(?:" + "|".join(PRS) + r")\b", re.IGNORECASE
    ),
    "landing or repair author": re.compile(
        "|".join([rf"\b{re.escape(n)}\b" for n in NAMES] + [re.escape(h) for h in HANDLES]), re.IGNORECASE
    ),
}
# /model/deepseek-ai/DeepSeek-V4.1-Flash is a SYMLINK onto the snapshot, and Path.rglob does not descend
# into a symlinked directory: the snapshot is named here resolved, or its text files would go unscanned.
ROOTS = [Path("/opt/infb"), Path("/app/eval/deepseek_v41_flash"), Path("/app/third_party/deepgemm_v41_sparse"),
         Path("/model"), Path("/model/deepseek-ai/DeepSeek-V4.1-Flash").resolve()]
SKIP = [Path("/opt/infb/tools")]  # the fetch/lock tooling venv: public packages only
TEXT_SUFFIXES = {".md", ".txt", ".json", ".jsonl", ".py", ".sh", ".toml", ".yaml", ".yml", ".lock",
                 ".patch", ".cfg", ".ini", ".jinja", ""}
MAX_BYTES = 64 * 1024 * 1024  # the checkpoint's shards are skipped by suffix; this bounds the rest


def scan_file(path: Path) -> list[str]:
    try:
        if path.stat().st_size > MAX_BYTES:
            return []
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
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
            if path.suffix not in TEXT_SUFFIXES:
                continue
            scanned += 1
            hits += scan_file(path)

    head = subprocess.run(
        ["git", "-C", "/app", "log", "-3", "--format=%an%n%ae%n%cn%n%ce%n%B"],
        capture_output=True, text=True, check=True,
    ).stdout
    hits += [f"a commit the agent starts from: {n}" for n, rx in PATTERNS.items() if rx.search(head)]

    for commit in COMMITS:
        present = subprocess.run(
            ["git", "-C", "/app", "cat-file", "-e", f"{commit}^{{commit}}"], capture_output=True
        ).returncode == 0
        if present:
            hits.append(f"/app/.git: {commit[:12]} is still in the object store")

    if hits:
        print("FATAL: the answer's address is on the agent's surface:", file=sys.stderr)
        for h in hits:
            print("  " + h, file=sys.stderr)
        return 1
    print(f"no answer address: {scanned} task-authored files, the materials and preparation commits, and the "
          f"object store all clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
