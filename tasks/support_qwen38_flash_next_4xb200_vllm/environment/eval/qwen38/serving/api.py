#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""API observations shared by development and grading: chat, streaming, length termination and
concurrent isolation, each validated structurally before any comparison.

Greedy continuations of the same prompt are compared only at their first divergence (later tokens
have different histories), where each side's token must be among the other's five most likely
tokens (the near-tie rule). The engine is not bitwise deterministic across batches, so byte identity
is not required. A missing, truncated or malformed record fails structurally and is never compared.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

from qwen38.measurement import observations as obs
from .mtp import request, stream_request

TOKENS = 48


def _chat_body(model, prompt, *, max_tokens, stream=False):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0,
            "max_tokens": max_tokens, "seed": 0, "logprobs": True, "top_logprobs": 5,
            "return_tokens_as_token_ids": True}
    if stream:
        body.update(stream=True, stream_options={"include_usage": True})
    return body


def chat(base, model, prompt, *, vocabulary, max_tokens=TOKENS, context="chat"):
    response = request(base, "/v1/chat/completions", _chat_body(model, prompt, max_tokens=max_tokens), timeout=600)
    return obs.chat(response, context, requested=max_tokens, ignore_eos=False, vocabulary=vocabulary)


def chat_stream(base, model, prompt, *, vocabulary, max_tokens=TOKENS, context="chat-stream"):
    payloads = stream_request(base, "/v1/chat/completions",
                              _chat_body(model, prompt, max_tokens=max_tokens, stream=True))
    return obs.stream(payloads, context, kind="chat", requested=max_tokens, ignore_eos=False,
                      vocabulary=vocabulary)


def agreement(a, b) -> dict:
    """First divergence of two greedy records over their common length, with the near-tie rule; a
    shorter record that is not a prefix-consistent stop is a divergence no near tie excuses."""
    n = min(len(a["tokens"]), len(b["tokens"]))
    div = next((i for i in range(n) if a["tokens"][i] != b["tokens"][i]), None)
    if div is None:
        if len(a["tokens"]) == len(b["tokens"]):
            return {"identical": True, "first_divergence": None, "near_tie": True}
        return {"identical": False, "first_divergence": n, "near_tie": False,
                "lengths": [len(a["tokens"]), len(b["tokens"])]}
    return {"identical": False, "first_divergence": div, "tokens": [a["tokens"][div], b["tokens"][div]],
            "near_tie": obs.near_tie(a, b, div)}


def probes(base, model, prompt, *, vocabulary, concurrency=8) -> dict:
    """The four API properties on one prompt; every record is validated before use."""
    first, second = chat(base, model, prompt, vocabulary=vocabulary, context="greedy-1"), \
        chat(base, model, prompt, vocabulary=vocabulary, context="greedy-2")
    repeat = agreement(first, second)
    streamed = chat_stream(base, model, prompt, vocabulary=vocabulary)
    stream_vs = agreement(streamed, first)
    one = chat(base, model, prompt, vocabulary=vocabulary, max_tokens=1, context="length-1")
    length = {"finish_reason": one["finish_reason"], "tokens": len(one["tokens"]),
              "holds": one["finish_reason"] == "length" and len(one["tokens"]) == 1}
    serial = one["tokens"][0]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        rows = list(pool.map(lambda i: chat(base, model, prompt, vocabulary=vocabulary, max_tokens=1,
                                            context=f"concurrent-{i}"), range(concurrency)))
    firsts = [row["tokens"][0] for row in rows]
    isolation = {"serial": serial, "concurrent": firsts, "holds": all(t == serial for t in firsts)}
    return {"schema": "qwen38-api-probes/1",
            "repeat": repeat, "stream": stream_vs, "length": length, "isolation": isolation,
            "holds": bool(repeat["near_tie"] and stream_vs["near_tie"] and length["holds"] and isolation["holds"]),
            "records": {"greedy": [first, second], "stream": streamed, "length": one}}


def completions_example(base, model, token_ids, *, vocabulary, max_tokens=16) -> dict:
    """A text completion (non-streamed and streamed) and prompt probabilities for one prompt."""
    body = {"model": model, "prompt": token_ids, "max_tokens": max_tokens, "temperature": 0.0, "seed": 0,
            "logprobs": 5, "ignore_eos": True, "return_tokens_as_token_ids": True}
    plain = obs.completion(request(base, "/v1/completions", body, timeout=600), "completion",
                           requested=max_tokens, ignore_eos=True, vocabulary=vocabulary, prompt_tokens=len(token_ids))
    payloads = stream_request(base, "/v1/completions", dict(body, stream=True, stream_options={"include_usage": True}))
    streamed = obs.stream(payloads, "completion-stream", kind="completion", requested=max_tokens, ignore_eos=True,
                          vocabulary=vocabulary, prompt_tokens=len(token_ids))
    scored = request(base, "/v1/completions", {"model": model, "prompt": token_ids, "max_tokens": 1,
                                                "temperature": 0.0, "seed": 0, "prompt_logprobs": 0}, timeout=600)
    prompt_scores = obs.prompt_scores(scored, list(token_ids), "prompt-logprobs", vocabulary=vocabulary)
    return {"completion": plain, "stream": streamed, "agreement": agreement(plain, streamed),
            "prompt_logprobs": prompt_scores}
