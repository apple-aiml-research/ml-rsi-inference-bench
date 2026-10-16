#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Generated-continuation replay: does decode continue history as a fresh prefill would score it?"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

from .server import flush_cache, generate, input_series, output_series


def generate_group(base_url: str, prompts: list[list[int]], *, new_tokens: int, timeout: float) -> list[dict]:
    """Submit one group of prompts together; returns per prompt its generated IDs, log-probs, cached tokens."""
    def one(p):
        resp = generate(base_url, p, max_new_tokens=new_tokens, timeout=timeout)
        ids, lps = output_series(resp)
        return {"prompt_len": len(p), "generated_ids": ids, "generated_logprobs": lps,
                "cached_tokens": resp["meta_info"].get("cached_tokens")}
    with ThreadPoolExecutor(max_workers=max(1, len(prompts))) as pool:
        return list(pool.map(one, prompts))


def replay_one(base_url: str, prompt: list[int], generated_ids: list[int], generated_logprobs: list[float],
               *, timeout: float) -> dict:
    """Teacher-force prompt + generated and compare the aligned log-probabilities."""
    rec: dict = {"prompt_len": len(prompt), "generated": len(generated_ids)}
    problems = []
    if len(generated_logprobs) != len(generated_ids):
        problems.append("generated ids and log-probabilities differ in length")
    resp = generate(base_url, prompt + generated_ids, max_new_tokens=0, timeout=timeout, logprob_start_len=0)
    ids, lps = input_series(resp)
    n = len(generated_ids)
    if len(ids) != len(prompt) + n:
        problems.append(f"replay returned {len(ids)} positions for {len(prompt) + n}")
    else:
        if ids[-n:] != generated_ids:
            problems.append("replay token IDs differ from the generated tokens")
        tail = lps[-n:]
        for k, (r, g) in enumerate(zip(tail, generated_logprobs)):
            if r is None or not math.isfinite(r) or not math.isfinite(g) or r > 1e-6 or g > 1e-6:
                problems.append(f"position {k}: non-finite or positive log-probability (replay {r}, generated {g})")
                break
    rec["structural_problems"] = problems
    if problems:
        rec["valid"] = False
        return rec
    d = [r - g for r, g in zip(lps[-n:], generated_logprobs)]
    k3 = []
    for x in d:
        try:
            k3.append(math.expm1(x) - x)
        except OverflowError:
            k3.append(math.inf)
    rec.update({"valid": True, "mean_k3": sum(k3) / len(k3), "max_abs_d": max(abs(x) for x in d),
                "d": [round(x, 6) for x in d]})
    return rec


def run(base_url: str, groups: list[list[list[int]]], *, new_tokens: int = 132, timeout: float = 900.0) -> list[dict]:
    """Generate every group (each group submitted together, groups in order), flush, replay every pair."""
    generations = []
    for group in groups:
        for prompt, g in zip(group, generate_group(base_url, group, new_tokens=new_tokens, timeout=timeout)):
            generations.append((prompt, g))
    flush_cache(base_url)
    out = []
    for prompt, g in generations:
        rec = replay_one(base_url, prompt, g["generated_ids"], g["generated_logprobs"], timeout=timeout)
        rec["cached_tokens_at_generation"] = g["cached_tokens"]
        out.append(rec)
    return out
