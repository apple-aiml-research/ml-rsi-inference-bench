#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Commit-encode replay, public."""
from __future__ import annotations

import hashlib
import json
import math

from ._pairs import cut, pair, prefix_hits, read, seed_of
from .common import choice
from .recipes import SERVED_NAME

GENERATION = "seeded"


def generation_body(P: list[int], generation: str = GENERATION) -> dict:
    """The 768-token generation request; the seed is a sha256 of the prompt ids."""
    body = {"model": SERVED_NAME, "prompt": P, "max_tokens": 768, "ignore_eos": True, "temperature": 1.0,
            "return_token_ids": True}
    if generation == "seeded":
        key = json.dumps([list(P), None, None, None], sort_keys=True)
        body["seed"] = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % (2 ** 31 - 1)
    elif generation == "greedy":
        body["temperature"] = 0.0
    elif generation != "v1":
        raise ValueError(f"unknown generation {generation!r}")
    return body


def cold_read(srv, P: list[int], G: list[int], seed, targets, chunked: bool) -> dict:
    """After /reset_prefix_cache: the read on P + G with P + G's KV written by prefill -- in one pass, or
    (chunked) the way generation writes it: P, then each 256-token canvas of G as its own chunk (reads on
    P and on P + G[:256] leave those blocks in the prefix cache)."""
    if chunked:
        for k in range(0, len(G), 256):
            read(srv, P + G[:k], seed, targets)
    return read(srv, P + G, seed, targets)


def _abs_d(warm: list, cold: list, lo: int, hi: int) -> list[float]:
    """|warm - cold| over positions [lo, hi) of one read's targets."""
    return [abs(w - c) for w, c in zip(warm[lo:hi], cold[lo:hi])]


def _pooled(pool: list[float], complete: bool) -> float:
    """S over the node's pooled positions: the mean of |d|, or inf if any case did not yield its positions."""
    if not complete or not pool:
        return math.inf
    return sum(pool) / len(pool)


def replay(srv, cases: list[dict], T: float | None = None, chunked: bool = True) -> dict:
    """S is the mean |d| pooled over the positions of every case up to and including its first end-of-sequence
    token, and T bounds S (None: recorded only). A case fails on a generation of the wrong length, a reset that
    is not 200, a warm read below the reuse it needs, or a structural failure; S then reads inf and the node
    fails."""
    rows, ok, pool, complete = [], True, [], True
    for c in cases:
        P = c["ids"]
        g = srv.post("/v1/completions", generation_body(P))
        full = choice(g).get("token_ids") or []
        if len(full) != 768:
            rows.append({"id": c["id"], "error": f"generation returned {len(full)} tokens", "pass": False})
            ok = complete = False
            continue
        G, targets = full[:512], full[512:]
        seed = seed_of(targets, 0.5, "replay/" + c["id"])
        h0 = prefix_hits(srv)
        warm = read(srv, P + G, seed, targets)
        hits = prefix_hits(srv) - h0
        reset = srv.reset_prefix_cache()
        cold = cold_read(srv, P, G, seed, targets, chunked)
        upto = cut(targets)
        stat = pair(warm, cold, upto)
        need = len(P) + len(G) - 1280
        structural = warm["structural"] and cold["structural"]
        row = {"id": c["id"], "hits": hits, "need": need, "reset": reset, "k3": stat, "cut": upto}
        if structural:
            w, k = warm["target_logprob"], cold["target_logprob"]
            d = _abs_d(w, k, 0, upto)
            pool += d
            row.update(n=len(d), mean_abs=(sum(d) / len(d) if d else None), warm=w, cold=k)
            complete &= bool(d)
        else:
            row.update(n=0, problems=warm["problems"] + cold["problems"])
            complete = False
        good = reset == 200 and hits >= need and structural
        row["pass"] = good
        rows.append(row)
        ok &= good
    S = _pooled(pool, complete)
    ok &= T is None or S <= T
    return {"rows": rows, "S": S, "n": len(pool), "T": T, "pass": bool(ok)}


LEAK_K, LEAK_HALF, BLOCK = 384, 128, 16


def leak_need(context_len: int) -> int:
    """The prefix-cache hits that exercise the warm leak read: the context's full blocks, less two blocks when the
    context is block-aligned."""
    full = context_len // BLOCK * BLOCK
    return full - 2 * BLOCK if context_len % BLOCK == 0 else full


def _half(warm: list, cold: list, lo: int, hi: int) -> dict:
    d = [w - c for w, c in zip(warm[lo:hi], cold[lo:hi])]
    if not d:
        return {"n": 0, "signed_mean": None, "stat": None}
    m = sum(d) / len(d)
    return {"n": len(d), "signed_mean": m, "stat": abs(m), "mean_abs": sum(abs(x) for x in d) / len(d)}


def leak(srv, cases: list[dict], T: float | None = None, chunked: bool = True) -> dict:
    """The leak read on each case: every case is read and pooled, and T bounds S, the mean |d| over every case's
    statistic positions together (None: recorded only). A case fails on a generation of the wrong length, a reset
    that is not 200, a warm read below the reuse it needs, or a structural failure; S then reads inf and the node
    fails."""
    rows, ok, pool, complete = [], True, [], True
    for c in cases:
        P = c["ids"]
        reset0 = srv.reset_prefix_cache()
        g = srv.post("/v1/completions", generation_body(P))
        full = choice(g).get("token_ids") or []
        if len(full) != 768 or reset0 != 200:
            rows.append({"id": c["id"], "reset": [reset0],
                         "error": f"reset {reset0}; generation returned {len(full)} tokens", "pass": False})
            ok = complete = False
            continue
        context, targets = P + full[:LEAK_K], full[LEAK_K:LEAK_K + 2 * LEAK_HALF]
        seed = seed_of(targets, 1.0, "leak/" + c["id"])
        h0 = prefix_hits(srv)
        warm = read(srv, context, seed, targets)
        hits = prefix_hits(srv) - h0
        reset = srv.reset_prefix_cache()
        cold = cold_read(srv, P, full[:LEAK_K], seed, targets, chunked)
        need, upto = leak_need(len(context)), cut(targets)
        structural = warm["structural"] and cold["structural"]
        row = {"id": c["id"], "context": len(context), "hits": hits, "need": need,
               "exercised": hits >= need, "reset": [reset0, reset], "structural": structural, "cut": upto}
        if structural:
            w, k = warm["target_logprob"], cold["target_logprob"]
            first, rest = _half(w, k, 0, min(LEAK_HALF, upto)), _half(w, k, LEAK_HALF, min(2 * LEAK_HALF, upto))
            d = _abs_d(w, k, 0, min(LEAK_HALF, upto))
            pool += d
            row.update(n=first["n"], signed_mean=first["signed_mean"], stat=first["stat"], third_canvas=rest,
                       warm=w, cold=k)
            complete &= bool(d)
        else:
            row.update(n=0, stat=math.inf, problems=warm["problems"] + cold["problems"])
            complete = False
        good = reset == 200 and row["exercised"] and structural
        row["pass"] = good
        rows.append(row)
        ok &= good
    S = _pooled(pool, complete)
    ok &= T is None or S <= T
    return {"rows": rows, "S": S, "n": len(pool), "T": T, "pass": bool(ok)}
