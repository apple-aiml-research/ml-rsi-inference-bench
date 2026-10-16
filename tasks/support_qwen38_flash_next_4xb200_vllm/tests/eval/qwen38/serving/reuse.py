#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The ordinary-recipe prefix-reuse observation protocol, on one representative pair.

A prefix is sent and repeated until the repeat reads from the prefix cache; the repeat's hit count
is the deepest cached block boundary before its end. The observed prompt is that boundary plus a few
tail tokens, so the first generated token lies just past a boundary the cache really restores. It is
then generated twice for a fixed length: once warm (reading the cache) and once cold (a distinct
`cache_salt`, which cannot hit). Both records are validated structurally; the comparison reads the
first generated position, the only one whose history is identical on both sides, as k3 of each
side's chosen token under both distributions, and reports the first divergence of the two
continuations with the near-tie rule. Counters are read from `/metrics` (summed over label sets; a
missing or malformed counter is a structural failure).
"""
from __future__ import annotations

import math
import re

from qwen38.measurement import observations as obs
from .mtp import request

HITS = "vllm:prefix_cache_hits"
QUERIES = "vllm:prefix_cache_queries"


def counter(text: str, name: str) -> float:
    wanted = {name, name + "_total"}
    values = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([^\s{]+)(?:\{.*\})?\s+(\S+)(?:\s+\S+)?", line)
        if match and match[1] in wanted:
            try:
                value = float(match[2])
            except ValueError as exc:
                raise obs.ObservationError(f"invalid {name}") from exc
            obs.require(math.isfinite(value) and value >= 0 and value.is_integer(), f"invalid {name}: {value}")
            values.append(value)
        elif re.match(r"(?:" + "|".join(re.escape(key) for key in wanted) + r")(?=[\s{]|$)", line):
            raise obs.ObservationError(f"malformed {name}")
    obs.require(bool(values), f"missing {name}")
    return sum(values)


def cache_reading(base: str) -> dict:
    text = request(base, "/metrics", timeout=60)
    return {"hits": counter(text, HITS), "queries": counter(text, QUERIES)}


def cache_delta(before: dict, after: dict) -> dict:
    result = {key: after[key] - before[key] for key in ("hits", "queries")}
    obs.require(all(v >= 0 for v in result.values()) and result["hits"] <= result["queries"],
                f"invalid isolated cache delta: {result}")
    return {**result, "before": before, "after": after}


def complete(base: str, model: str, prompt, n: int, *, vocabulary: int, salt: str | None = None,
             forced: bool = False, context: str = "completion") -> dict:
    body = {"model": model, "prompt": prompt, "temperature": 0.0, "max_tokens": n, "seed": 0, "logprobs": 5,
            "return_tokens_as_token_ids": True}
    if salt:
        body["cache_salt"] = salt
    if forced:
        body["ignore_eos"] = True
    return obs.completion(request(base, "/v1/completions", body, timeout=600), context, requested=n,
                          ignore_eos=forced, vocabulary=vocabulary,
                          prompt_tokens=len(prompt) if isinstance(prompt, list) else None)


def boundary(base: str, model: str, ids: list[int], tail: int, *, vocabulary: int, repeats: int = 2) -> dict:
    """The deepest boundary the cache restores for `ids`, from a repeat's hits (align-mode registration
    may need a second repeat)."""
    complete(base, model, ids, 1, vocabulary=vocabulary, context="prepare")
    probes = []
    for attempt in range(repeats):
        before = cache_reading(base)
        complete(base, model, ids, 1, vocabulary=vocabulary, context=f"repeat-{attempt}")
        delta = cache_delta(before, cache_reading(base))
        probes.append(delta)
        hit = delta["hits"]
        if hit > 0:
            obs.require(hit + tail <= len(ids), f"a boundary at {hit:g} leaves no tail")
            return {"boundary": int(hit), "prompt": ids[:int(hit) + tail], "probes": probes}
    raise obs.ObservationError(f"no repeat read anything from the prefix cache: {probes}")


def position_zero(warm: dict, cold: dict) -> dict:
    """k3 at the first generated position on each side's chosen token, under both distributions."""
    w0 = dict(warm["top_logprobs"][0]) | {warm["tokens"][0]: warm["token_logprobs"][0]}
    c0 = dict(cold["top_logprobs"][0]) | {cold["tokens"][0]: cold["token_logprobs"][0]}
    terms = [obs.k3(c0[t] - w0[t]) for t in {warm["tokens"][0], cold["tokens"][0]} if t in w0 and t in c0]
    values = [obs.k3_value(t) for t in terms]
    mean = (math.fsum(v / len(values) for v in values) if all(math.isfinite(v) for v in values) else math.inf) \
        if values else None
    return {"terms": terms, "k3": None if mean is None else obs.json_number(mean)}


def pair(base: str, model: str, ids: list[int], *, tail: int, tokens: int, vocabulary: int, salt: str) -> dict:
    """One warm/cold observation of a prefix (token IDs) at the served cache's own boundary."""
    cut = boundary(base, model, ids, tail, vocabulary=vocabulary)
    before = cache_reading(base)
    warm = complete(base, model, cut["prompt"], tokens, vocabulary=vocabulary, forced=True, context="warm")
    warm_cache = cache_delta(before, cache_reading(base))
    cold = complete(base, model, cut["prompt"], tokens, vocabulary=vocabulary, forced=True, salt=salt, context="cold")
    divergence = obs.first_divergence(warm["tokens"], cold["tokens"])
    return {"schema": "qwen38-prefix-reuse-pair/1", "boundary": cut["boundary"], "probes": cut["probes"],
            "prompt_tokens": len(cut["prompt"]), "warm_hits": warm_cache["hits"],
            "warm_read_the_boundary": warm_cache["hits"] == cut["boundary"],
            "position_zero": position_zero(warm, cold), "first_divergence": divergence,
            "near_tie": True if divergence is None else obs.near_tie(warm, cold, divergence),
            "warm": warm, "cold": cold}
