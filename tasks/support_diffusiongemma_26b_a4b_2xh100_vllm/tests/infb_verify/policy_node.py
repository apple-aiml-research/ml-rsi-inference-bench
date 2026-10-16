#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""test_denoise_policy, one node in two parts, through the public protocol
(eval/diffusion_gemma/serving/policy.py) with the hidden cases and R_ref.

(i) on tp1_fixed: five policy cases, four draws each; pass iff every case hits in >= 0.75.
(ii) 16 fixed short chat prompts once on tp1 and once on tp1_fixed; pass iff R <= (1 + R_ref) / 2.
Each response's completion length is recorded. Expected on the reference: pass.
"""
from __future__ import annotations

from eval.diffusion_gemma.serving import policy as P

from infb_verify import common

FIXED: dict | None = None


def run_fixed(srv) -> dict:
    cases = common.hidden("policy_cases.json")
    walls, lengths = P.walls(srv, cases["adaptive"])
    return {"schedule": P.schedule(srv, cases["schedule"]), "fixed_walls": walls, "fixed_lengths": lengths}


def verdict(srv, policy) -> dict:
    cases = common.hidden("policy_cases.json")
    fixed = FIXED if FIXED is not None else {"schedule": [], "fixed_walls": [], "fixed_lengths": []}
    natural, lengths = P.walls(srv, cases["adaptive"])
    ok_i = bool(fixed["schedule"]) and all(x["hit_rate"] >= 0.75 for x in fixed["schedule"])
    R = P.ratio(natural, fixed["fixed_walls"])
    R_ref = None if policy is None else policy["R_ref"]
    ok_ii = R < float("inf") and (R_ref is None or R <= (1 + R_ref) / 2)
    return {"part_i": fixed["schedule"], "natural_walls": natural, "fixed_walls": fixed["fixed_walls"], "R": R,
            "R_ref": R_ref, "completion_tokens": {"natural": lengths, "fixed": fixed.get("fixed_lengths")},
            "pass": bool(ok_i and ok_ii)}
