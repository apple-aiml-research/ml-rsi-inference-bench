#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""API observations, public: structure and log-probability coherence."""
from __future__ import annotations

import math

from . import images as IM
from .common import choice, completion_tokens, content, gather, tokenizer
from .recipes import SERVED_NAME


def coherent(lp_tokens, lp_tops) -> tuple[int, int]:
    """(positions whose token is not the top-1 of its own top-5, positions whose top-5 mass > 1 + 1e-4)."""
    bad, over = 0, 0
    for t, top in zip(lp_tokens, lp_tops):
        if not top:
            continue
        best = max(top, key=top.get)
        if best != t and top[best] > top.get(t, -math.inf) + 1e-6:
            bad += 1
        if sum(math.exp(v) for v in top.values()) > 1 + 1e-4:
            over += 1
    return bad, over


def api_probes(srv) -> dict:
    rec, ok = {}, True
    text = "The history of the river valley begins"
    ids = tokenizer()(text, add_special_tokens=False)["input_ids"]
    for name, prompt in (("text", text), ("ids", ids)):
        r = srv.post("/v1/completions", {"model": SERVED_NAME, "prompt": prompt, "max_tokens": 64, "logprobs": 5,
                                         "temperature": 1.0, "return_tokens_as_token_ids": True})
        lp = choice(r).get("logprobs") or {}
        toks, tops = lp.get("tokens") or [], lp.get("top_logprobs") or []
        n = completion_tokens(r)
        vals = [v for t in tops for v in (t or {}).values()]
        bad, over = coherent(toks, tops)
        row = {"status": r.get("_status"), "n": n, "positions": len(toks), "not_top1": bad, "mass_over_1": over,
               "bad_values": sum(1 for v in vals if not math.isfinite(v) or v > 0)}
        rec[f"completion_{name}"] = row
        ok &= r.get("_status") == 200 and n == len(toks) and bad == 0 and over == 0 and row["bad_values"] == 0
    st, chunks, _ = srv.stream("/v1/completions", {"model": SERVED_NAME, "prompt": "Rivers carry", "max_tokens": 64,
                                                   "temperature": 1.0})
    rec["stream_completion"] = {"status": st, "chunks": len(chunks)}
    ok &= st == 200 and len(chunks) > 0
    msgs = [{"role": "user", "content": "Name three rivers in Europe."}]
    r = srv.post("/v1/chat/completions", {"model": SERVED_NAME, "messages": msgs, "max_tokens": 128, "return_token_ids": True,
                                          "chat_template_kwargs": {"enable_thinking": False}})
    rec["chat"] = {"status": r.get("_status"), "has_ids": bool(choice(r).get("token_ids"))}
    ok &= r.get("_status") == 200 and rec["chat"]["has_ids"]
    st, chunks, _ = srv.stream("/v1/chat/completions", {"model": SERVED_NAME, "messages": msgs, "max_tokens": 128,
                                                        "chat_template_kwargs": {"enable_thinking": False}})
    rec["chat_stream"] = {"status": st, "chunks": len(chunks)}
    ok &= st == 200 and len(chunks) > 0
    img = IM.data_url(IM.image(7))
    r = srv.post("/v1/chat/completions", {"model": SERVED_NAME, "max_tokens": 128, "chat_template_kwargs": {"enable_thinking": False},
                                          "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": img}},
                                                                                    {"type": "text", "text": "What shapes do you see?"}]}]})
    rec["image_chat"] = {"status": r.get("_status"), "content": content(r)[:200]}
    ok &= r.get("_status") == 200 and bool(rec["image_chat"]["content"])
    bodies = [{"model": SERVED_NAME, "prompt": f"Write one sentence about the number {k}.", "max_tokens": 64, "temperature": 1.0}
              for k in range(8)]
    outs = gather(lambda b: srv.post("/v1/completions", b), bodies, concurrency=8)
    rec["concurrent8"] = [o.get("_status") for o in outs]
    ok &= all(s == 200 for s in rec["concurrent8"])
    rec["pass"] = bool(ok)
    return rec
