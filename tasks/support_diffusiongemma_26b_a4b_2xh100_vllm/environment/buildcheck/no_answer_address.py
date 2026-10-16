#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Fail the build if the answer's address is on the agent's surface.

The instruction names no pull request, and nothing the image ships may either. This check makes that a
build assertion instead of a claim:

  * no task-authored file the image ships names the landing, its repairs, the amendments or the
    FlashAttention fork's later changes -- by commit, by PR number or by the landing authors'
    handles: /opt/infb (minus its pip venv of public packages), /public, the checkpoint's own text
    files (the model author's; a hit there is a finding to report, not something to edit) and
    /app/eval;
  * the two preparation commits' messages and identities name none of them;
  * none of those commits' OBJECTS is in /app's store -- the landing squash, every repair and
    amendment, the two open PR heads and the landing PR's 37 commits (the recipe clones at the
    upstream head, then drops later branches, expires the reflog and prunes; this proves it).

Third-party venvs are not scanned, deliberately: a five-digit number turns up by chance in gigabytes of
wheel data, and no package published before the base can know this task. The vendored FlashAttention
fork at the engine's own pin is upstream source at the base, like /app's history, and is outside the
roots for the same reason /app's history is.
"""

from __future__ import annotations

import bz2
import gzip
import lzma
import re
import subprocess
import sys
from pathlib import Path

# Commits whose objects must be absent from /app's store and whose prefixes must appear nowhere.
VLLM_COMMITS = (
    "eb28452b10a1376d143b2847a78b31726db346dd",
    "521b88c29ef29b37efefa64efda7b59ec99d210c",
    "6c5872efc5fca7a53f5f3aa589e8cb7896d98618",
    "701a23d99f405668158d1395e11c30107dd65b75",
    "b136cc2c2c590593c9dfff2bd732edbb24bcceeb",
    "c5b66233b26824f463fcca8af0779ff9e9d7930c",
    "8fc1b2d046f4a991b5c24bb5470bf45efcfe9d01",
    "2c88fb131c7ae0be01907cd8c276911db5e7aad4",
    "1b3b88ec2b7457aa030db4d0e7d8aaf04f6d0fb8",
    "63d9ad0a3a435cdf3a44495028b10f390a38f960",
    "283637a6535d7cfe4b49f32cfe368d9e9fd5602a",
    "945f22d0a294f87dde449cddbaa4ecdbc3caadd3",
    "d42ae12f36d5a9136815832fc108d0f6873320e8",
    "d01679e56c68a4d1bdf0e8c4c56162405c054e50",
    "1c9aaab632997311aa3c088306986bb2640ff474",
    "5d304b53b9bf7a3458daeb3b28794dac3fe7cbbc",
    "812f082160fe6a225f7e7ad8e05ce24b98e96be4",
    "ba3e9b79995ca3e576729df60c65186b488146a5",
    "4275f3e1561bccb494f27969210b75d2a9836ea2",
    "97de79ef335e6d2758788f91289e7447ec514ecf",
    "1ca03cb852588ec44ab5d621822db5299627e93a",
    "e91a1c5a0ac6a0bb16031e3e0ec607e0ad2a9d42",
    "788dbac28ebfb47a567ee8189e7bba6914fe1f67",
    "c28885765c25af81dc77ea9d85c2ff0805066fc8",
    "c0b95277a8f31532f1b31927bbcd6317037a06d4",
    "95d74558dcc2699323041a891cfcd69d384fcb69",
    "ef420c1b2895c09c344dd43dd06957ee3c2cea3f",
    "efa4011ec0c530b8becae6e86eb4706c003a6024",
    "a275a28e052c58af2a5a89cb1268f92a677bfe6f",
    "5a89c3ed8f17f6d5af179b24734d2176f6a19ccd",
    "df75f7b8728bd4cf120f0a960fe1ece69c1d66b4",
    "4ae3822f3a5fe08cc77ab9fb27702d725eb031fa",
    "9cb0ec2cc708d0943fdd47d570d8043f3d0e3c43",
    "61588ddfe5a112b6aaa3da63a227cb94d6e10865",
    "297dd43314452249bac9d66b24c2ae26de067d3e",
    "043dc271329c97e7c99f70a13543615ae4c6ee06",
    "18e7d0bc1cbfc468fe737373842e432c52afab24",
    "31454092232026c9e621130a901c579305341ff8",
    "9601df4bd0634073877b70c372b23e9d93f05c1b",
    "d25326b1fcfbcdfdc4133e7263b0d95ec31c9b87",
    "ea7720c7f0e23bbee0ff0545a6f82ac6f0d78f65",
    "1f4df4d284aa4b0d30248f2bf529695025ec253e",
    "d79a675f332aa82cb4c849d46ce37dbf496e55fb",
    "c0b75dc8bca7c64aa87d58f2d73597027ee17479",
    "35f824da553c965de3dd4b8a005b5a5c9f843f8f",
    "0ace2015095aca54aae1531b7acf210982865833",
    "919fa56efa8ef7f9cad1b5406d680c2e58f31624",
    "633b6d05d45fda221e82b5ed6be5677a5c4ea52e",
    "74b5964f02c7e023fadd3004cfac8a61c52eef1f",
)
FORK_COMMITS = (
    "3cc79a594439f6cab338136e62ddb498899f6ea3",
    "803020a8fa15407871341d41eba4919ade2ee1ee",
    "1c726f3c48b155aa399e7e47f044cce0beb9fcbb",
    "2c839c33742309ec41e620bf837495ec9926c56e",
    "b3964b1d8b95d8e8447435668ab169a2700bab65",
)
PATTERNS = {
    "withheld source identity": re.compile("|".join(sha[:8] for sha in VLLM_COMMITS + FORK_COMMITS)),
    "withheld PR by number": re.compile(
        r"(?:#|\bPR\s*#?|pull/|pulls/)(?:45163|45468|45417|46177|45719|45965|47464|46659|57414|57250|57417|"
        r"51994|49202|58035|45689|46644)\b", re.IGNORECASE),
    "fork PR by number": re.compile(r"(?:flash-attention(?:/pull/|#)|\bFA\s*#|\bfork\s*#)(?:143|145|150|152|154)\b",
                                    re.IGNORECASE),
    "landing author": re.compile(r"LucasWilkinson|MatthewBonanni", re.IGNORECASE),
}
ROOTS = [Path("/opt/infb"), Path("/public"), Path("/model"), Path("/app/eval")]
SKIP = [Path("/opt/infb/tools")]  # the fetch/lock tooling venv: public packages only
TEXT_SUFFIXES = {".md", ".txt", ".json", ".jsonl", ".py", ".sh", ".toml", ".yaml", ".yml", ".lock",
                 ".patch", ".cfg", ".ini", ".jinja", ""}
#: Compressed forms are decompressed, not skipped: /public/efficiency/sample.jsonl.xz is text.
COMPRESSED_SUFFIXES = {".xz": lzma.open, ".lzma": lzma.open, ".gz": gzip.open, ".bz2": bz2.open}
MAX_BYTES = 64 * 1024 * 1024  # the checkpoint's shards are skipped by suffix; this bounds the rest


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

    heads = subprocess.run(["git", "-C", "/app", "log", "-2", "--format=%an%n%ae%n%cn%n%ce%n%B"],
                           capture_output=True, text=True, check=True).stdout
    hits += [f"preparation commits: {n}" for n, rx in PATTERNS.items() if rx.search(heads)]

    for sha in VLLM_COMMITS:
        present = subprocess.run(["git", "-C", "/app", "cat-file", "-e", f"{sha}^{{commit}}"],
                                 capture_output=True).returncode == 0
        if present:
            hits.append(f"/app/.git: withheld commit {sha[:12]} is in the object store")

    if hits:
        print("FATAL: the answer's address is on the agent's surface:", file=sys.stderr)
        for h in hits:
            print("  " + h, file=sys.stderr)
        return 1
    print(f"no answer address: {scanned} task-authored files, the preparation commits and the object "
          f"store ({len(VLLM_COMMITS)} withheld commits absent) all clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
