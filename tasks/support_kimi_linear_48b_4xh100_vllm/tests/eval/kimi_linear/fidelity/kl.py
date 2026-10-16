#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The fidelity statistic: SGLang's k3 form, vendored with provenance.

It is imported as a sibling module by `fidelity/check.py`, and through that by
`state_lifecycle/client.py`.

PROVENANCE. `python/sglang/test/kl_test_utils.py::compare_kl_divergence`, sgl-project/sglang, blob
sha `28efcbfa9ab582cb381fcc065c273e4b05015331` (12,689 bytes), read from `main` at `ffac53d779c0`
(2026-09-23). Upstream's body, in full, is:

    kl_divs = []
    for input_logprob, output_logprob in zip(input_logprobs, output_logprobs):
        input_logprob = np.array(input_logprob)
        output_logprob = np.array(output_logprob)
        logr = input_logprob - output_logprob
        kl_approx = (np.exp(logr) - 1) - logr
        kl_divs.append(np.mean(kl_approx))
    avg_kl_div = sum(kl_divs) / len(kl_divs)
    assert avg_kl_div < ACC_THRESHOLDS[model_name]["kl_div"]

That is Schulman's k3 estimator, `(r - 1) - log r` with `r = exp(log p - log q)`: non-negative for every
term, zero only when the two log-probabilities agree, and unbiased for the KL it estimates. The
per-sequence statistic below is upstream's, term for term.

WHAT IS DEPARTED FROM, AND WHY. Upstream scores **one number over every sequence**
(`avg_kl_div = sum(kl_divs) / len(kl_divs)`). This module does not: `band_medians` takes a median
within each band and `scored` reports the **largest** band median, with the whole-set median beside it
and `scored_band` naming which band produced it. A statistic taken over one regime is blind to a defect
confined to another -- `fidelity/examples.json` holds eight short sequences and two long ones, so a
whole-set median is decided by the short band whatever the long band does.

A mean is also the wrong summary here for a second reason: the bands' per-token log-probabilities
differ in scale, and averaging k3 across bands whose entropy differs lets the noisy band set the scale.
Medians per band, compared to per-band thresholds, keep each band's question separate. `per_sequence`
is upstream's arithmetic unchanged.
"""

from __future__ import annotations

import math
from statistics import median


def k3_per_sequence(reference_logprobs, candidate_logprobs) -> float:
    """Upstream's k3 mean for one sequence: mean over tokens of ``(exp(logr) - 1) - logr``.

    `logr = reference - candidate`, matching upstream's `input_logprob - output_logprob`. Pure Python so
    the verifier does not need numpy for this; the arithmetic is identical.

    Two inputs upstream never meets, because upstream compares two healthy engines, are handled here
    because a broken candidate produces them -- an output buffer that is never written hands back
    uninitialised memory:

    * a **non-finite** candidate log-probability (NaN, -inf) makes the sequence's k3 **infinite**. A
      NaN would otherwise propagate into the median and compare False against any threshold, so a
      candidate emitting NaN would pass the gate;
    * a term whose ``exp(logr)`` would overflow a double (``logr`` above ~709, i.e. the candidate puts a
      token the reference finds likely at e^-709 or less) is **infinite** too. Python's ``math.exp``
      raises ``OverflowError`` there rather than returning inf, which would crash the gate instead of
      failing it.
    Both are the maximally wrong answer, and infinity is how that is scored.
    """
    if len(reference_logprobs) != len(candidate_logprobs):
        raise ValueError(
            f"length mismatch: {len(reference_logprobs)} reference vs "
            f"{len(candidate_logprobs)} candidate log-probabilities"
        )
    if not reference_logprobs:
        raise ValueError("no log-probabilities to compare")
    total = 0.0
    for ref, cand in zip(reference_logprobs, candidate_logprobs):
        if not math.isfinite(cand):
            return math.inf
        logr = ref - cand
        if logr > 700.0:
            return math.inf
        total += (math.exp(logr) - 1.0) - logr
    return total / len(reference_logprobs)


def band_medians(per_sequence: dict[str, float], bands: dict[str, list[str]]) -> dict[str, float]:
    """Median k3 within each band. A band with no sequences is omitted rather than scored as zero."""
    out = {}
    for band, ids in bands.items():
        vals = [per_sequence[i] for i in ids if i in per_sequence]
        if vals:
            out[band] = median(vals)
    return out


def scored(per_sequence: dict[str, float], bands: dict[str, list[str]]) -> dict:
    """The full verdict object: the scored statistic, every band median, and the whole-set median."""
    per_band = band_medians(per_sequence, bands)
    if not per_band:
        raise ValueError("no band produced a median; nothing was compared")
    scored_band = max(per_band, key=per_band.__getitem__)
    return {
        "scored_statistic": per_band[scored_band],
        "scored_band": scored_band,
        "band_medians": per_band,
        "whole_set_median": median(per_sequence.values()),
        "n_sequences": len(per_sequence),
    }
