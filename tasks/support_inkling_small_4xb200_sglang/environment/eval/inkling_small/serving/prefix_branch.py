#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Prefix-branch replay: are continuations after prefix-cache reuse the ones a cold prefill would give?"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from .replay import replay_one
from .server import flush_cache, generate, output_series


def _one(base_url: str, req: dict, new_tokens: int, timeout: float) -> dict:
    resp = generate(base_url, req["ids"], max_new_tokens=new_tokens, timeout=timeout)
    ids, lps = output_series(resp)
    return {"id": req["id"], "prompt_len": len(req["ids"]), "cached_tokens": resp["meta_info"].get("cached_tokens"),
            "generated_ids": ids, "generated_logprobs": lps, "replay": bool(req.get("replay", True))}


def run(base_url: str, waves: list[dict], *, new_tokens: int = 132, timeout: float = 900.0) -> dict:
    """waves: [{"name", "mode": "concurrent"|"sequential", "requests": [{"id", "ids", "replay"}]}]."""
    served = []
    prompts = {}
    for wave in waves:
        reqs = wave["requests"]
        for r in reqs:
            prompts[(wave["name"], r["id"])] = r["ids"]
        if wave["mode"] == "concurrent":
            with ThreadPoolExecutor(max_workers=max(1, len(reqs))) as pool:
                results = list(pool.map(lambda r: _one(base_url, r, new_tokens, timeout), reqs))
        elif wave["mode"] == "sequential":
            results = [_one(base_url, r, new_tokens, timeout) for r in reqs]
        else:
            raise ValueError(f"unknown wave mode {wave['mode']!r}")
        for res in results:
            res["wave"] = wave["name"]
        served += results
    flush_cache(base_url)
    replays = []
    for res in served:
        if not res["replay"]:
            continue
        rec = replay_one(base_url, prompts[(res["wave"], res["id"])], res["generated_ids"], res["generated_logprobs"],
                         timeout=timeout)
        rec.update({"wave": res["wave"], "id": res["id"], "cached_tokens": res["cached_tokens"]})
        replays.append(rec)
    return {"served": [{k: v for k, v in s.items() if k not in ("generated_logprobs",)} for s in served],
            "replays": replays}
