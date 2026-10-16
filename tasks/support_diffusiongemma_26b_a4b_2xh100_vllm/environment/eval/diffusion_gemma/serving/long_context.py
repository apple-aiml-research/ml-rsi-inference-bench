#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Long context, public."""
from __future__ import annotations

from .common import choice, completion_tokens
from .recipes import SERVED_NAME

CONTEXT, NEAR_LIMIT_PROMPT = 262144, 262044


def near_limit(srv, prompt_ids: list[int]) -> dict:
    assert len(prompt_ids) == NEAR_LIMIT_PROMPT, len(prompt_ids)
    r = srv.post("/v1/completions", {"model": SERVED_NAME, "prompt": list(prompt_ids), "max_tokens": CONTEXT - NEAR_LIMIT_PROMPT,
                                     "temperature": 1.0}, timeout=1800)
    n = completion_tokens(r)
    finish = choice(r).get("finish_reason")
    over = srv.post("/v1/completions", {"model": SERVED_NAME, "prompt": list(prompt_ids), "max_tokens": 256, "temperature": 1.0},
                    timeout=1800)
    return {"status": r.get("_status"), "completion_tokens": n, "finish_reason": finish, "wall_s": r.get("_wall_s"),
            "error": (r.get("_error") or "")[:300],
            "over_limit_diagnostic": {"max_tokens": 256, "status": over.get("_status"),
                                      "completion_tokens": completion_tokens(over), "error": (over.get("_error") or "")[:300]},
            "pass": r.get("_status") == 200 and n is not None and n <= 100 and finish in ("length", "stop")}
