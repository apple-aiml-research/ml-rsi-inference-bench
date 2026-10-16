#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The repair observations, public."""
from __future__ import annotations

import json
import re

from .common import choice, completion_tokens, content, gather
from .recipes import SERVED_NAME

THINKING, MAX_TOKENS, DRAWS, CONCURRENCY, BAR = True, 1024, 16, 4, 0.75


def structured_outputs(srv) -> dict:
    schema = {"type": "object", "properties": {"city": {"type": "string"}, "population": {"type": "integer"}},
              "required": ["city", "population"]}
    r = srv.post("/v1/chat/completions", {"model": SERVED_NAME, "max_tokens": 256, "chat_template_kwargs": {"enable_thinking": False},
                                          "messages": [{"role": "user", "content": "Give a city and its population as JSON."}],
                                          "response_format": {"type": "json_schema", "json_schema": {"name": "city", "schema": schema}}})
    st = r.get("_status")
    valid = False
    if st == 200:
        try:
            obj = json.loads(content(r))
            valid = isinstance(obj.get("city"), str) and isinstance(obj.get("population"), int)
        except Exception:  # noqa: BLE001
            valid = False
    refused = st is not None and 400 <= st < 500 and bool(r.get("_error"))
    after = srv.post("/v1/chat/completions", {"model": SERVED_NAME, "max_tokens": 32, "messages": [{"role": "user", "content": "Hello"}],
                                              "chat_template_kwargs": {"enable_thinking": False}})
    return {"status": st, "refused": refused, "valid_json": valid, "after_status": after.get("_status"),
            "pass": (refused or valid) and after.get("_status") == 200}


def long_output(srv, prompt_ids: list[int]) -> dict:
    r = srv.post("/v1/completions", {"model": SERVED_NAME, "prompt": list(prompt_ids), "max_tokens": 1024,
                                     "ignore_eos": True, "temperature": 1.0})
    n = completion_tokens(r)
    return {"status": r.get("_status"), "completion_tokens": n, "pass": r.get("_status") == 200 and n == 1024}


def first_token(srv, cases: list[dict]) -> dict:
    """cases: [{"prompt", "answer"}]."""
    out, ok = [], True
    for c in cases:
        body = {"model": SERVED_NAME, "max_tokens": MAX_TOKENS, "messages": [{"role": "user", "content": c["prompt"]}],
                "chat_template_kwargs": {"enable_thinking": THINKING}}
        res = gather(lambda _: srv.post("/v1/chat/completions", body), range(DRAWS), concurrency=CONCURRENCY)
        texts = [content(x) for x in res]
        words = [re.findall(r"[A-Za-z]+", t)[:1] for t in texts]
        hits = sum(1 for w in words if w and w[0].lower() == c["answer"].lower())
        out.append({"prompt": c["prompt"], "hits": hits, "draws": DRAWS, "samples": [t[:60] for t in texts[:3]],
                    "statuses": sorted({x.get("_status") for x in res}, key=str)})
        ok &= hits >= BAR * DRAWS
    return {"cases": out, "pass": bool(ok)}


def concurrent_logprobs(srv) -> dict:
    bodies = [{"model": SERVED_NAME, "prompt": f"Tell a long story about item {k}.", "max_tokens": 512, "ignore_eos": True,
               "logprobs": 5, "temperature": 1.0} for k in range(36)]
    outs = gather(lambda b: srv.post("/v1/completions", b), bodies, concurrency=36)
    rows = [(o.get("_status"), len(((choice(o).get("logprobs") or {}).get("tokens") or []))) for o in outs]
    return {"results": rows, "pass": all(s == 200 and n == 512 for s, n in rows)}
