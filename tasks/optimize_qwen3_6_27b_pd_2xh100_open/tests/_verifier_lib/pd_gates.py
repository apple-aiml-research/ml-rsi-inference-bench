#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""ONE verifier-side gate for a PD pair: the TTFT guardrail. NOT part of the agent-visible rig."""

from __future__ import annotations

TTFT_REGRESSION_LIMIT = 1.5

# One token, and it is SCORED: a TTFT regression is the candidate's configuration, not a harness
# fault, so classifying it as a non-result would make the guardrail a free retry instead of a limit.
SCORED_REASONS = ("pd_ttft_regressed",)


def ttft_guardrail(arms: dict[str, dict]) -> list[tuple[str, str]]:
    """Return [(reason_token, message)] if the candidate bought tail ITL with too much TTFT, else [].

    `arms` maps a label to an `extract()` result. The pair must be {"pristine": ..., "candidate": ...} or
    {"aggregated": ..., "pd": ...}; the first key is the reference arm.

    A RATIO against the reference arm, not an absolute bound: a pristine PD deployment already pays this cost,
    and the question is only whether the candidate made it worse.
    """
    labels = list(arms)
    if len(labels) != 2:
        raise ValueError(f"ttft_guardrail needs exactly two arms, got {labels}")
    ref, cand = labels

    t_ref = arms[ref].get("continuation_p95_ttft_ms") or arms[ref].get("continuation_mean_ttft_ms")
    t_cand = arms[cand].get("continuation_p95_ttft_ms") or arms[cand].get("continuation_mean_ttft_ms")
    if not (t_ref and t_cand):
        # Absent rather than good. A missing TTFT key means the guardrail did not run, and saying so is the
        # caller's job -- returning [] here would read as "the trade was acceptable".
        return []
    if t_cand <= t_ref * TTFT_REGRESSION_LIMIT:
        return []
    return [(
        "pd_ttft_regressed",
        f"candidate tail TTFT {t_cand:.0f} ms against {ref} {t_ref:.0f} ms, a factor of "
        f"{t_cand / t_ref:.2f} over the {TTFT_REGRESSION_LIMIT}x limit. Tail ITL can be bought with "
        f"first-token latency, and past this factor the trade is refused rather than scored."
    )]
