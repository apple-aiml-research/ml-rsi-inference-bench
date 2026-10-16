#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The decoding policy, public, in two parts."""
from __future__ import annotations

import statistics

from . import semantic
from .common import completion_tokens
from .recipes import SERVED_NAME


def schedule(srv, cases: list[dict]) -> list[dict]:
    out = []
    for c in cases:
        h, texts = semantic.draws(srv, c, n=4, concurrency=4)
        out.append({"id": c["id"], "hit_rate": sum(h) / len(h), "samples": [t[:80] for t in texts[:2]]})
    return out


def walls(srv, prompts: list[str]) -> tuple[list[float], list]:
    """(per-request walls, completion lengths): one request at a time, thinking disabled, max_tokens 256."""
    out, lengths = [], []
    for p in prompts:
        r = srv.post("/v1/chat/completions", {"model": SERVED_NAME, "max_tokens": 256,
                                              "messages": [{"role": "user", "content": p}],
                                              "chat_template_kwargs": {"enable_thinking": False}})
        out.append(r.get("_wall_s") if r.get("_status") == 200 else float("inf"))
        lengths.append(completion_tokens(r))
    return out, lengths


def ratio(natural: list[float], fixed: list[float]) -> float:
    if not natural or not fixed or any(w == float("inf") for w in natural + fixed):
        return float("inf")
    return statistics.median(natural) / statistics.median(fixed)
