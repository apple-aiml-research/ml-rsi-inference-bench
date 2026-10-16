#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Teacher-forced fidelity against independent model-author observations.

Common public/graded protocol. Inputs, expected log probabilities and any
calibrated acceptance policy are supplied by the caller. Prompt log
probabilities deliberately use a cold recomputation; this protocol does not
measure prefix restoration.

Every served record is structurally validated before any statistic
(``measurement.observations``): a missing, truncated or invalid score fails
immediately and cannot be repaired by a reread. A sequence's statistic is the
mean over its scored positions of k3 = expm1(r) - r, r = reference - served;
an exponent overflow is a numerical +infinity verdict. With a band policy, a
sequence over its cap is read once more (the bounded numerical reread), and a
band whose median exceeds its threshold is read once more as a whole.
"""

from __future__ import annotations

import math
import statistics

from qwen38.measurement import observations as obs
from .mtp import request


def prompt_logprobs(base, model, token_ids, *, vocabulary, timeout=1800):
    """Own-token scores for positions 1..n-1, validated before they are returned."""
    if len(token_ids) < 2 or any(type(token) is not int or token < 0 for token in token_ids):
        raise ValueError("teacher-forced input needs at least two valid token IDs")
    body = request(base, "/v1/completions", {
        "model": model, "prompt": list(token_ids), "max_tokens": 1,
        "temperature": 0., "seed": 0, "prompt_logprobs": 0}, timeout=timeout)
    return obs.prompt_scores(body, list(token_ids), "teacher-forced", vocabulary=vocabulary)


def per_sequence(seq: dict, served: list[float]) -> dict:
    ref = seq["reference_logprobs"]
    k = obs.k3_mean(ref, served, seq["id"])
    return {
        "id": seq["id"],
        "band": seq.get("band", len(seq["token_ids"])),
        "family": seq.get("family", "example"),
        "positions_reference": len(ref),
        "positions_served": len(served),
        "compared": k["positions"],
        "k3_mean": k["mean"],
        "k3_max": k["max"],
        "overflow": k["overflow"],
        "overflow_positions": k["overflow_positions"],
        "mean_logprob_reference": statistics.fmean(ref),
        "mean_logprob_served": statistics.fmean(served),
    }


def value(row: dict) -> float:
    """A sequence's k3 mean, ``inf`` for an overflow verdict."""
    return math.inf if row["overflow"] else row["k3_mean"]


def band_medians(per_seq: list[dict]) -> dict:
    return {b: statistics.median([value(c) for c in per_seq if c["band"] == b])
            for b in sorted({c["band"] for c in per_seq})}


def _json_medians(medians: dict) -> dict:
    return {str(b): obs.json_number(v) for b, v in medians.items()}


def evaluate(base, model, sequences, *, vocabulary, bands=None):
    """Measure every sequence, with optional caller-supplied band policy.

    A numerical cap/band failure permits one bounded re-reading. Structural
    observation errors fail immediately and cannot be repaired by a redraw.
    Without policy, return diagnostics without a calibrated verdict.
    """
    if not sequences or len({s["id"] for s in sequences}) != len(sequences):
        raise ValueError("fidelity sequences must be nonempty with unique IDs")
    for seq in sequences:
        ref = seq["reference_logprobs"]
        if len(ref) != len(seq["token_ids"]) - 1 or any(
                type(v) not in (int, float) or not math.isfinite(v) or v > obs.LOGPROB_EPSILON for v in ref):
            raise ValueError(f"{seq['id']}: invalid independent reference observations")
    expected_bands = {str(s.get("band", len(s["token_ids"]))) for s in sequences}
    if bands is not None and set(bands) != expected_bands:
        raise ValueError("band policy does not cover exactly the supplied inputs")
    if bands is not None:
        for band, policy in bands.items():
            for name in ("T", "cap"):
                number = policy.get(name)
                if type(number) not in (int, float) or not math.isfinite(number) or number <= 0:
                    raise ValueError(f"{band}: {name} must be finite and positive")
    reads = []

    def read(seq):
        served = prompt_logprobs(base, model, seq["token_ids"], vocabulary=vocabulary)
        row = per_sequence(seq, served)
        reads.append({"id": seq["id"], "served_logprobs": served, "statistics": row})
        return row

    first = [read(seq) for seq in sequences]
    medians = band_medians(first)
    result = {"schema": "qwen38-fidelity/2", "sequences": first, "band_medians": _json_medians(medians),
              "cap_rereads": [], "band_redraws": {}, "reads": reads, "passed": None}
    if bands is None:
        return result
    by_id = {seq["id"]: seq for seq in sequences}
    for row in first:
        cap = bands[str(row["band"])]["cap"]
        if value(row) > cap:
            second = read(by_id[row["id"]])
            result["cap_rereads"].append({"id": row["id"], "band": row["band"], "cap": cap,
                                          "first": obs.json_number(value(row)),
                                          "second": obs.json_number(value(second)),
                                          "counts": value(second) > cap})
    for band, median in medians.items():
        threshold = bands[str(band)]["T"]
        if median <= threshold:
            continue
        again = [read(seq) for seq in sequences if seq.get("band", len(seq["token_ids"])) == band]
        second = statistics.median(value(row) for row in again)
        result["band_redraws"][str(band)] = {"T": threshold, "first_median": obs.json_number(median),
                                             "second_median": obs.json_number(second),
                                             "fails": second > threshold, "sequences": again}
    result["passed"] = not any(row["counts"] for row in result["cap_rereads"]) and not any(
        row["fails"] for row in result["band_redraws"].values())
    return result
