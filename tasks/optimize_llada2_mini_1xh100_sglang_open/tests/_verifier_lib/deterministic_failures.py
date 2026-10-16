#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Tell a DETERMINISTIC pristine-baseline failure apart from a flaky one."""

from __future__ import annotations

import re
from pathlib import Path

_SIGNATURES: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (
        re.compile(r"get_decoding_sched_meta\.cu:\d+"),
        "FlashMLA's decoding scheduler asked for more shared memory than the device allows; the request is "
        "4*(b*5+1) bytes where b is the topk-expanded row count, so it is a function of batch x length",
        "reduce the scored batch/length for this task or fix the kernel; a re-run at the same shape fails identically",
    ),
    (
        re.compile(r"AssertionError: q\.dtype \S+ != k\.dtype \S+"),
        "a kernel was handed a query and a KV cache of different dtypes -- the CuTe flash-attention path asserts "
        "they match, and the KV cache dtype is set by the deployment while the query dtype comes from the model "
        "code, so an edit that changes one and not the other cannot run at all",
        "make the query dtype match the KV cache dtype, or select a backend that accepts the pair; a re-run "
        "executes the same assertion",
    ),
)

_LOG_SUFFIXES = (".log", ".txt", ".err")
_MAX_BYTES = 8 * 1024 * 1024


def deterministic_signature(raw_dir: Path | str | None) -> dict[str, str] | None:
    """Return details of a known deterministic failure found in ``raw_dir``, else None."""
    if raw_dir is None:
        return None
    root = Path(raw_dir)
    if not root.is_dir():
        return None
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in _LOG_SUFFIXES:
            continue
        try:
            size = path.stat().st_size
            with path.open("rb") as fh:
                if size > _MAX_BYTES:
                    fh.seek(size - _MAX_BYTES)
                blob = fh.read(_MAX_BYTES).decode("utf-8", "replace")
        except OSError:
            continue
        for pattern, meaning, remedy in _SIGNATURES:
            match = pattern.search(blob)
            if match:
                return {
                    "signature": match.group(0),
                    "log": path.name,
                    "meaning": meaning,
                    "remedy": remedy,
                }
    return None


deterministic_pristine_signature = deterministic_signature
