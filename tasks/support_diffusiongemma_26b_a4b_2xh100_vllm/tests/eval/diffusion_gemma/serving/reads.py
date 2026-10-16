#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""One-canvas denoise reads: request construction and statistics (the read interface of reads.md)."""
from __future__ import annotations

import concurrent.futures as cf
import math

from . import images as IM
from .recipes import SERVED_NAME

TOP = 20


def _xargs(seed: list[int], targets: list[int], steps: int) -> dict:
    return {"diffusion_read_only": "1", "diffusion_seed_canvas": ",".join(map(str, seed)),
            "diffusion_max_steps": str(int(steps)), "diffusion_target_token_ids": ",".join(map(str, targets))}


def text_read(prompt_ids: list[int], seed: list[int], targets: list[int], steps: int = 1) -> dict:
    """/v1/completions body for a read on a token-id prompt."""
    return {"model": SERVED_NAME, "prompt": list(prompt_ids), "max_tokens": len(seed), "logprobs": TOP,
            "temperature": 1.0, "return_tokens_as_token_ids": True, "return_token_ids": True,
            "vllm_xargs": _xargs(seed, targets, steps)}


def chat_read(messages: list[dict], seed: list[int], targets: list[int], steps: int = 1,
              mm_processor_kwargs: dict | None = None) -> dict:
    """/v1/chat/completions body for a read on a chat prompt (thinking disabled)."""
    body = {"model": SERVED_NAME, "messages": messages, "max_tokens": len(seed), "logprobs": True,
            "top_logprobs": TOP, "temperature": 1.0, "return_tokens_as_token_ids": True,
            "return_token_ids": True, "chat_template_kwargs": {"enable_thinking": False},
            "vllm_xargs": _xargs(seed, targets, steps)}
    if mm_processor_kwargs:
        body["mm_processor_kwargs"] = mm_processor_kwargs
    return body


def _tid(token: str) -> int | None:
    return int(token.split(":", 1)[1]) if isinstance(token, str) and token.startswith("token_id:") else None


def parse(resp: dict, targets: list[int]) -> dict:
    """Structure and target log-probabilities of a completion or chat read response."""
    out = {"status": resp.get("_status"), "error": (resp.get("_error") or "")[:500], "wall_s": resp.get("_wall_s")}
    ch = (resp.get("choices") or [{}])[0]
    lp = ch.get("logprobs") or {}
    if "content" in lp:  # chat
        items = lp.get("content") or []
        tokens = [_tid(it.get("token")) for it in items]
        tops = [{_tid(t.get("token")): t.get("logprob") for t in (it.get("top_logprobs") or [])} for it in items]
    else:
        tokens = [_tid(t) for t in (lp.get("tokens") or [])]
        tops = [{_tid(k): v for k, v in (t or {}).items()} for t in (lp.get("top_logprobs") or [])]
    L = len(targets)
    problems = []
    if out["status"] != 200:
        problems.append(f"HTTP {out['status']}")
    if len(tokens) != L or len(tops) != L:
        problems.append(f"{len(tokens)} positions, expected {L}")
    target_lp, emitted_missing, target_missing, bad_values = [], 0, 0, 0
    for i in range(min(L, len(tops))):
        top = tops[i]
        if tokens[i] not in top:
            emitted_missing += 1
        v = top.get(targets[i])
        if v is None:
            target_missing += 1
        target_lp.append(v)
        bad_values += sum(1 for x in top.values() if x is None or not math.isfinite(x) or x > 1e-6)
    if emitted_missing:
        problems.append(f"{emitted_missing} positions without the emitted token")
    if target_missing:
        problems.append(f"{target_missing} positions without the target")
    if bad_values:
        problems.append(f"{bad_values} non-finite or positive values")
    out.update(positions=len(tokens), tokens=tokens, target_logprob=target_lp, problems=problems,
               structural=not problems)
    return out


def k3(ref: list[float], served: list[float | None]) -> float:
    """Mean over positions of expm1(ref - served) - (ref - served); a missing value is infinite."""
    total = 0.0
    for r, s in zip(ref, served):
        if s is None:
            return math.inf
        d = r - s
        total += math.expm1(d) - d
    return total / max(len(ref), 1)


def statistics(rd: dict, res: dict) -> dict:
    """The read's statistic over its recorded positions (k3) and over all positions (k3_all)."""
    if not res["structural"]:
        res["k3"] = res["k3_all"] = math.inf
        return res
    n = int(rd.get("stat_positions", len(rd["targets"])))
    res["k3"] = k3(rd["author"][:n], res["target_logprob"][:n])
    res["k3_all"] = k3(rd["author"], res["target_logprob"])
    res["stat_positions"] = n
    return res


def chat_messages(prompt: dict) -> list[dict]:
    """A recorded chat prompt: its images (by seed) and then its text, one user turn."""
    parts = [{"type": "image_url", "image_url": {"url": IM.data_url(IM.image(s))}} for s in prompt.get("image_seeds", [])]
    parts.append({"type": "text", "text": prompt["text"]})
    return [{"role": "user", "content": parts}]


def send(srv, prompt: dict, rd: dict) -> dict:
    """One recorded read: text prompts through /v1/completions, image prompts through chat."""
    if prompt.get("kind") == "image":
        mm = {"max_soft_tokens": prompt["max_soft_tokens"]} if prompt.get("max_soft_tokens") else None
        raw = srv.post("/v1/chat/completions", chat_read(chat_messages(prompt), rd["seed"], rd["targets"], rd["steps"], mm),
                       timeout=1800)
        res = statistics(rd, parse(raw, rd["targets"]))
        if prompt.get("author_ids") is not None:  # a diagnostic: the served rendering against the author's
            res["prompt_ids_match"] = raw.get("prompt_token_ids") == prompt["author_ids"]
        return res
    raw = srv.post("/v1/completions", text_read(prompt["ids"], rd["seed"], rd["targets"], rd["steps"]), timeout=1800)
    return statistics(rd, parse(raw, rd["targets"]))


def run(srv, data: dict, reads: list[dict], concurrency: int = 4) -> list[dict]:
    """Every read of `reads` (entries of data["reads"]) at the given concurrency."""
    with cf.ThreadPoolExecutor(concurrency) as ex:
        results = list(ex.map(lambda rd: send(srv, data["prompts"][rd["prompt"]], rd), reads))
    return [{"id": rd["id"], "band": rd["band"], "k3": res["k3"], "k3_all": res["k3_all"],
             "stat_positions": res.get("stat_positions"), "structural": res["structural"],
             "prompt_ids_match": res.get("prompt_ids_match"), "problems": res["problems"], "wall_s": res["wall_s"]}
            for rd, res in zip(reads, results)]
