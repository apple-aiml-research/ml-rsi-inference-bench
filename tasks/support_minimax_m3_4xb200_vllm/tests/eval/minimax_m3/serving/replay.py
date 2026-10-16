#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Generated-continuation replay and prefix-branch replay: decode must agree with a cold teacher-forced replay of
the same history."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

from .fidelity import MODEL, own_token_logprobs
from .http import ObservationError, metric_total, metrics, request


def generate(base: str, prompt: list[int], n: int, *, model: str = MODEL, timeout: float = 900) -> dict:
    body = request(base, "/v1/completions", {
        "model": model, "prompt": prompt, "max_tokens": n, "temperature": 0.0, "ignore_eos": True,
        "logprobs": 0, "return_tokens_as_token_ids": True, "return_token_ids": True}, timeout=timeout)
    choices = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(choices, list) or len(choices) != 1:
        raise ObservationError("expected exactly one completion")
    ch = choices[0]
    ids = ch.get("token_ids")
    lp = ch.get("logprobs") or {}
    tokens, scores = lp.get("tokens"), lp.get("token_logprobs")
    if not isinstance(ids, list) or len(ids) != n or any(type(t) is not int or t < 0 for t in ids):
        raise ObservationError(f"expected {n} generated token ids, got {ids if not isinstance(ids, list) else len(ids)}")
    if not isinstance(tokens, list) or not isinstance(scores, list) or len(tokens) != n or len(scores) != n:
        raise ObservationError("missing or incomplete generated log-probabilities")
    for i, (tok, tid, s) in enumerate(zip(tokens, ids, scores)):
        if tok != f"token_id:{tid}":
            raise ObservationError(f"generated position {i}: token {tok!r} does not name id {tid}")
        if type(s) not in (int, float) or not math.isfinite(s) or s > 1e-6:
            raise ObservationError(f"generated position {i}: invalid log-probability {s!r}")
    return {"token_ids": ids, "logprobs": [float(s) for s in scores],
            "finish_reason": ch.get("finish_reason")}


def reset_prefix_cache(base: str) -> None:
    request(base, "/reset_prefix_cache", {}, timeout=300, method="POST")


def replay(base: str, prompt: list[int], generated: dict, *, model: str = MODEL, timeout: float = 900) -> dict:
    ids = prompt + generated["token_ids"]
    body = request(base, "/v1/completions", {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0.0,
                                              "prompt_logprobs": 0}, timeout=timeout)
    choices = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(choices, list) or len(choices) != 1:
        raise ObservationError("expected exactly one teacher-forced completion")
    served = own_token_logprobs(choices[0].get("prompt_logprobs"), ids, start=len(prompt))
    n = len(generated["token_ids"])
    if len(served) != n:
        raise ObservationError(f"replay returned {len(served)} aligned entries for {n} generated tokens")
    d = [r - g for r, g in zip(served, generated["logprobs"])]
    ks = [math.expm1(min(x, 700.0)) - min(x, 700.0) for x in d]
    return {"replay_logprobs": served, "k3_mean": math.fsum(ks) / n, "max_abs_d": max(abs(x) for x in d)}


def generation_replay(base: str, groups: list[dict], n: int, *, model: str = MODEL) -> list[dict]:
    """``groups``: [{"concurrency": c, "prompts": [{"id", "token_ids"}, ...]}], generated group by group, each
    group's prompts submitted together; every generation completes before the cache reset and the replays."""
    done: list[tuple[dict, dict]] = []
    for g in groups:
        prompts = g["prompts"]
        timeout = 900 if max(len(p["token_ids"]) for p in prompts) < 64_000 else 1800
        if g["concurrency"] == 1:
            outs = [generate(base, p["token_ids"], n, model=model, timeout=timeout) for p in prompts]
        else:
            with ThreadPoolExecutor(max_workers=g["concurrency"]) as pool:
                outs = list(pool.map(lambda p: generate(base, p["token_ids"], n, model=model, timeout=timeout),
                                     prompts))
        done += list(zip(prompts, outs))
    reset_prefix_cache(base)
    rows = []
    for p, gen in done:
        r = replay(base, p["token_ids"], gen, model=model)
        rows.append({"id": p["id"], "prompt_tokens": len(p["token_ids"]), "generated": gen, **r})
    return rows


def prefix_branches(base: str, prefix: list[int], suffixes: dict[str, list[int]], n: int, *,
                    model: str = MODEL, order=("A", "B", "A", "B")) -> dict:
    reset_prefix_cache(base)
    warm = []
    for i, name in enumerate(order):
        before = metric_total(metrics(base), "vllm:prefix_cache_hits")
        gen = generate(base, prefix + suffixes[name], n, model=model)
        after = metric_total(metrics(base), "vllm:prefix_cache_hits")
        warm.append({"request": i + 1, "branch": name, "prefix_cache_hit_tokens": after - before, "generated": gen})
    reset_prefix_cache(base)
    rows = []
    for w in warm:
        prompt = prefix + suffixes[w["branch"]]
        rows.append({**w, **replay(base, prompt, w["generated"], model=model)})
    return {"shared_prefix": len(prefix), "requests": rows}
