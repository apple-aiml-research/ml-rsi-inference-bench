#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""API probes: the served interfaces answer with complete, well-formed results (structure only, no numerics)."""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

from .fidelity import MODEL
from .http import ObservationError, request, stream

N = 16


def _lp(v, where):
    if type(v) not in (int, float) or not math.isfinite(v) or v > 1e-6:
        raise ObservationError(f"{where}: invalid log-probability {v!r}")


def completion(base, prompt, *, model=MODEL, n=N) -> dict:
    body = request(base, "/v1/completions", {"model": model, "prompt": prompt, "max_tokens": n, "temperature": 0.0,
                                              "ignore_eos": True, "logprobs": 5, "prompt_logprobs": 5})
    ch = (body.get("choices") or [None])[0] if isinstance(body, dict) else None
    if not isinstance(ch, dict):
        raise ObservationError("no completion choice")
    lp = ch.get("logprobs") or {}
    toks, scores, tops = lp.get("tokens"), lp.get("token_logprobs"), lp.get("top_logprobs")
    if not all(isinstance(x, list) and len(x) == n for x in (toks, scores, tops)):
        raise ObservationError("completion log-probabilities incomplete")
    for i, (s, top) in enumerate(zip(scores, tops)):
        _lp(s, f"token {i}")
        if not isinstance(top, dict) or len(top) < 5:
            raise ObservationError(f"token {i}: fewer than five top log-probabilities")
        for v in top.values():
            _lp(v, f"token {i} top")
    plp = ch.get("prompt_logprobs")
    n_prompt = len(prompt) if isinstance(prompt, list) else None
    if not isinstance(plp, list) or (n_prompt is not None and len(plp) != n_prompt) or plp[0] is not None:
        raise ObservationError("prompt log-probabilities incomplete")
    for i, entry in enumerate(plp[1:], 1):
        if not isinstance(entry, dict) or not entry:
            raise ObservationError(f"prompt position {i}: no entries")
        for v in entry.values():
            _lp(v.get("logprob") if isinstance(v, dict) else None, f"prompt position {i}")
    return {"tokens": len(toks), "prompt_positions": len(plp), "text": ch.get("text")}


def streamed_completion(base, prompt, *, model=MODEL, n=N) -> dict:
    events = stream(base, "/v1/completions", {"model": model, "prompt": prompt, "max_tokens": n, "temperature": 0.0,
                                              "ignore_eos": True, "logprobs": 1})
    toks = [t for ev in events for ch in ev.get("choices") or [] for t in ((ch.get("logprobs") or {}).get("tokens") or [])]
    if len(toks) != n:
        raise ObservationError(f"streamed completion carried {len(toks)} tokens, expected {n}")
    return {"events": len(events), "tokens": len(toks)}


def chat_probe(base, *, streaming: bool, model=MODEL, n=N) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": "Name three primary colors."}],
            "max_tokens": n, "temperature": 0.0, "chat_template_kwargs": {"thinking_mode": "disabled"}}
    if streaming:
        events = stream(base, "/v1/chat/completions", body)
        text = "".join((ch.get("delta") or {}).get("content") or "" for ev in events for ch in ev.get("choices") or [])
    else:
        resp = request(base, "/v1/chat/completions", body)
        text = ((resp.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    if not text.strip():
        raise ObservationError("chat returned no content")
    return {"streaming": streaming, "chars": len(text)}


def probes(base, *, prompt_ids: list[int], model=MODEL) -> dict:
    out = {"completion_text": completion(base, "The capital of France is", model=model),
           "completion_ids": completion(base, prompt_ids, model=model),
           "completion_stream": streamed_completion(base, prompt_ids, model=model),
           "chat": chat_probe(base, streaming=False, model=model),
           "chat_stream": chat_probe(base, streaming=True, model=model)}
    with ThreadPoolExecutor(max_workers=8) as pool:
        out["concurrent"] = list(pool.map(lambda i: completion(base, prompt_ids[i:] + prompt_ids[:i], model=model),
                                          range(8)))
    return out
