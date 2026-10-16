#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Semantic majority, public."""
from __future__ import annotations

import re

from . import images as IM
from .common import completion_tokens, content, gather, choice
from .recipes import SERVED_NAME

THINKING, MAX_TOKENS, DRAWS, CONCURRENCY, BAR = True, 1024, 16, 4, 0.75


def messages(case: dict) -> list[dict]:
    if case.get("image_seeds"):
        parts = [{"type": "image_url", "image_url": {"url": IM.data_url(IM.image(s))}} for s in case["image_seeds"]]
        parts.append({"type": "text", "text": case["prompt"]})
        return [{"role": "user", "content": parts}]
    return [{"role": "user", "content": case["prompt"]}]


def hit(text: str, answer: str, candidates: list[str]) -> bool:
    low = text.lower()

    def named(word):
        return re.search(rf"\b{re.escape(word.lower())}\b", low) is not None

    if not named(answer):
        return False
    return not (candidates and all(named(c) for c in candidates))


def draws(srv, case: dict, n: int = DRAWS, concurrency: int = CONCURRENCY, max_tokens: int = MAX_TOKENS,
          thinking: bool = THINKING) -> tuple[list[bool], list[str]]:
    body = {"model": SERVED_NAME, "max_tokens": max_tokens, "messages": messages(case),
            "chat_template_kwargs": {"enable_thinking": thinking}}
    outs = gather(lambda _: srv.post("/v1/chat/completions", body), range(n), concurrency=concurrency)
    texts = [content(o) for o in outs]
    return [hit(t, case["answer"], case.get("candidates", [])) for t in texts], texts


def thinking_disabled_diagnostic(srv, case: dict) -> dict:
    body = {"model": SERVED_NAME, "max_tokens": 256, "messages": messages(case), "chat_template_kwargs": {"enable_thinking": False}}
    o = srv.post("/v1/chat/completions", body)
    return {"id": case.get("id"), "status": o.get("_status"), "finish_reason": choice(o).get("finish_reason"),
            "completion_tokens": completion_tokens(o), "content": content(o)[:80]}


def semantic_majority(srv, cases: list[dict]) -> dict:
    rows, ok = [], True
    for c in cases:
        h, texts = draws(srv, c)
        rate = sum(h) / len(h)
        rows.append({"id": c["id"], "hit_rate": rate, "samples": [t[:80] for t in texts[:3]]})
        ok &= rate >= BAR
    diag = thinking_disabled_diagnostic(srv, cases[0]) if cases else None
    return {"rows": rows, "thinking_disabled_diagnostic": diag, "pass": bool(ok)}
