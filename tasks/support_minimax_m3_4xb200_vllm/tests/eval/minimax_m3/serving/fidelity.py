#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Teacher-forced fidelity: the served model's own-token log-probabilities against independent recordings."""

from __future__ import annotations

import math
import statistics

from .http import ObservationError, request

MODEL = "MiniMaxAI/MiniMax-M3-MXFP8"
KAPPA = 10.0   # the capped mean's per-position cap: k3 = 10 at a log-ratio of about 2.6 nats
FLIP_D = 4.6   # a flip: a position whose log-ratio exceeds 4.6 nats (k3 about 95)


def own_token_logprobs(entries, token_ids, *, start: int = 1) -> list[float]:
    """Own-token log-probabilities of positions ``start..`` from a ``prompt_logprobs`` array."""
    if not isinstance(entries, list) or len(entries) != len(token_ids):
        raise ObservationError(f"prompt_logprobs has {len(entries) if isinstance(entries, list) else 'no'} "
                               f"entries for {len(token_ids)} positions")
    out = []
    for position in range(start, len(token_ids)):
        entry = entries[position]
        own = entry.get(str(token_ids[position])) if isinstance(entry, dict) else None
        value = own.get("logprob") if isinstance(own, dict) else None
        if type(value) not in (int, float) or not math.isfinite(value) or value > 1e-6:
            raise ObservationError(f"position {position}: missing or invalid own-token log-probability {value!r}")
        out.append(float(value))
    return out


def prompt_logprobs(base: str, token_ids: list[int], *, model: str = MODEL, timeout: float = 1800) -> list[float]:
    if len(token_ids) < 2 or any(type(t) is not int or t < 0 for t in token_ids):
        raise ValueError("a teacher-forced input needs at least two valid token IDs")
    body = request(base, "/v1/completions", {"model": model, "prompt": token_ids, "max_tokens": 1,
                                              "temperature": 0.0, "prompt_logprobs": 0}, timeout=timeout)
    choices = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(choices, list) or len(choices) != 1:
        raise ObservationError("expected exactly one teacher-forced completion")
    return own_token_logprobs(choices[0].get("prompt_logprobs"), token_ids)


def k3(lr: float) -> float:
    """``expm1(lr) - lr``; a log-ratio past 700 nats is capped there (already a failure far beyond any bar)."""
    lr = min(lr, 700.0)
    return math.expm1(lr) - lr


def sequence_statistics(seq: dict, served: list[float]) -> dict:
    ref = seq["reference_logprobs"]
    if len(served) != len(ref):
        raise ObservationError(f"{seq['id']}: {len(served)} served positions for {len(ref)} recorded")
    ks = [k3(r - s) for r, s in zip(ref, served)]
    mean = math.fsum(ks) / len(ks)
    without_worst = math.fsum([*ks, -max(ks)]) / (len(ks) - 1) if len(ks) > 1 else mean
    flips = sum(1 for r, s in zip(ref, served) if r - s > FLIP_D)
    return {"id": seq["id"], "band": seq.get("band", len(seq["token_ids"])), "family": seq.get("family"),
            "positions": len(ks), "k3_mean": mean, "k3_mean_without_worst": without_worst,
            "k3_capped": math.fsum(min(k, KAPPA) for k in ks) / len(ks), "flips": flips,
            "flips_per_1k": 1000.0 * flips / len(ks), "k3_max": max(ks),
            "mean_logprob_reference": statistics.fmean(ref), "mean_logprob_served": statistics.fmean(served)}


STATISTICS = ("k3_mean", "k3_mean_without_worst", "k3_capped", "flips_per_1k")


def over_bars(st: dict, statistic, bar) -> dict:
    """{statistic: value} for each named statistic over its bar (``bar`` a number for one statistic, a mapping for a
    tuple of them)."""
    names = (statistic,) if isinstance(statistic, str) else tuple(statistic)
    bars = {statistic: bar} if isinstance(statistic, str) else bar
    return {n: st[n] for n in names if st[n] > bars[n]}


def evaluate(base: str, sequences: list[dict], *, policy: dict | None = None, statistic: str | None = "k3_mean",
             model: str = MODEL, timeout_for=lambda n: 900 if n < 64_000 else 1800) -> dict:
    """Measure every sequence; with ``policy`` ({band: {"T": .., "C": ..}}) apply its verdict: each band's median of
    the per-sequence ``k3_mean`` within ``T``, and each sequence's statistics within ``C``, each re-read once before
    it fails. The per-sequence bar reads ``statistic`` (one of STATISTICS, a tuple of them with C a mapping of bars,
    or None for the band medians alone)."""
    names = () if statistic is None else (statistic,) if isinstance(statistic, str) else tuple(statistic)
    if any(n not in STATISTICS for n in names) or (statistic is not None and not names):
        raise ValueError(f"unknown per-sequence statistic {statistic!r}")
    if not isinstance(statistic, (str, type(None))):
        statistic = list(names)
    if not sequences or len({s["id"] for s in sequences}) != len(sequences):
        raise ValueError("fidelity sequences must be non-empty with unique ids")
    for s in sequences:
        ref = s["reference_logprobs"]
        if len(ref) != len(s["token_ids"]) - 1 or any(
                type(v) not in (int, float) or not math.isfinite(v) or v > 1e-6 for v in ref):
            raise ValueError(f"{s['id']}: invalid recorded reference")
    reads: list[dict] = []

    def read(seq: dict) -> dict:
        served = prompt_logprobs(base, seq["token_ids"], model=model, timeout=timeout_for(len(seq["token_ids"])))
        st = sequence_statistics(seq, served)
        reads.append({"id": seq["id"], "served_logprobs": served, "statistics": st})
        return st

    first = [read(s) for s in sequences]
    bands = sorted({str(st["band"]) for st in first})
    medians = {b: statistics.median(st["k3_mean"] for st in first if str(st["band"]) == b) for b in bands}
    result = {"schema": "infb-m3-fidelity/1", "statistic": statistic, "sequences": first, "band_medians": medians,
              "band_redraws": {}, "sequence_resends": [], "reads": reads, "passed_band_medians": None,
              "passed_sequences": None, "passed": None}
    if policy is None:
        return result
    if set(policy) != set(bands):
        raise ValueError("the policy does not cover exactly the supplied bands")
    by_id = {s["id"]: s for s in sequences}
    for st in first:
        if statistic is None:
            continue
        cap = policy[str(st["band"])]["C"]
        over = over_bars(st, statistic, cap)
        if over:
            again = read(by_id[st["id"]])
            over_again = over_bars(again, statistic, cap)
            if isinstance(statistic, str):
                first_v, second_v = st[statistic], again[statistic]
            else:
                first_v, second_v = {n: st[n] for n in names}, {n: again[n] for n in names}
            result["sequence_resends"].append({"id": st["id"], "band": st["band"], "C": cap, "first": first_v,
                                               "second": second_v, "over_first": over, "over_second": over_again,
                                               "fails": bool(over_again)})
    for b, m in medians.items():
        T = policy[b]["T"]
        if m <= T:
            continue
        again = [read(s) for s in sequences if str(s.get("band", len(s["token_ids"]))) == b]
        second = statistics.median(st["k3_mean"] for st in again)
        result["band_redraws"][b] = {"T": T, "first_median": m, "second_median": second, "fails": second > T}
    result["passed_band_medians"] = not any(r["fails"] for r in result["band_redraws"].values())
    result["passed_sequences"] = not any(r["fails"] for r in result["sequence_resends"])
    result["passed"] = result["passed_band_medians"] and result["passed_sequences"]
    return result
