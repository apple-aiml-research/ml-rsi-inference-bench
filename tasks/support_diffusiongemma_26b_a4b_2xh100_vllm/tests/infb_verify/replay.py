#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Reads against the candidate itself: the private inputs and
bars around the public protocols (serving/replay.py, branches.py, mixed.py, long_context.py)."""
from __future__ import annotations

import math

from eval.diffusion_gemma.serving import reads as R
from eval.diffusion_gemma.serving.branches import branches as _branches
from eval.diffusion_gemma.serving.mixed import mixed as _mixed
from eval.diffusion_gemma.serving.replay import GENERATION, leak as _leak, replay as _replay

from infb_verify import common
from infb_verify.serving_nodes import long_context as part1, prose


def _bar(policy, key, deployment):
    return None if policy is None else policy["pairs"][key][deployment]


def replay(srv, deployment: str, policy) -> dict:
    """The node's statistic is S, the mean |d| pooled over the statistic positions of all 8 cases, and T
    bounds S. The record names the generation body's form and the verifier version, so the bar construction reads
    only readings of the form the node grades."""
    rec = _replay(srv, common.hidden("replay_cases.json")["cases"], T=_bar(policy, "replay", deployment))
    rec["generation"], rec["verifier_version"] = GENERATION, common.verifier_version()
    return rec


def leak(srv, deployment: str, policy) -> dict:
    """The stale-phase leak witness (serving/replay.py leak) on the replay node's cases:
    every case is read and pooled, and the statistic is S, the mean |d|
    over every case's statistic positions together, bounded by T from the same construction. In
    calibration mode every case is recorded and nothing is bounded."""
    rec = _leak(srv, common.hidden("replay_cases.json")["cases"], T=_bar(policy, "leak", deployment))
    rec["generation"], rec["verifier_version"] = GENERATION, common.verifier_version()
    return rec


def branches(srv, policy) -> dict:
    spec = common.hidden("branch_cases.json")
    return _branches(srv, spec["cases"], block_size=int(spec.get("block_size", 16)), T=_bar(policy, "branches", "tp1"))


def mixed(srv, deployment: str, policy) -> dict:
    data = common.hidden("reads.json.xz")
    pick = set(common.hidden("mixed_cases.json")["read_ids"])
    reads = [r for r in data["reads"] if r["id"] in pick]
    load = [prose(4096, 300_000 + 5000 * k) for k in range(3)]
    return _mixed(srv, data, reads, load, T=_bar(policy, "mixed", deployment))


def long_context(srv, deployment: str, policy) -> dict:
    """(1) near-limit generation; (2) engine agreement (adapted, weaker, labelled): two reads at
    c = 261,888 (prose and code) with fixed seeds and targets (tests/hidden/engine_series.json.xz)
    against the reference engine's own readings for this deployment, within
    T = max(0.005, 3f), f a frozen dispersion estimate of the reference's own repeat readings. In
    calibration mode the readings are recorded and nothing is compared."""
    rec = {"near_limit": part1(srv)}
    ok = rec["near_limit"]["pass"]
    path = common.HIDDEN / "engine_series.json.xz"
    if path.exists():
        series = common.hidden("engine_series.json.xz")
        T = None if policy is None else policy["engine_agreement"][deployment]
        rows = []
        for r in series["reads"]:
            got = R.parse(srv.post("/v1/completions", R.text_read(series["prompts"][r["prompt"]], r["seed"], r["targets"], 1),
                                   timeout=1800), r["targets"])
            ref = r.get(f"reference_{deployment}")
            n = r["stat_positions"]
            stat = (R.k3(ref[:n], got["target_logprob"][:n]) if got["structural"] else math.inf) if ref else None
            row = {"id": r["id"], "structural": got["structural"], "problems": got["problems"], "k3": stat,
                   "logprobs": got["target_logprob"], "wall_s": got["wall_s"]}
            row["pass"] = got["structural"] and (common.CALIBRATION or (stat is not None and math.isfinite(stat)
                                                                         and (T is None or stat <= T)))
            rows.append(row)
        rec["engine_agreement"] = {"rows": rows, "T": T}
        ok &= all(x["pass"] for x in rows)
    rec["pass"] = bool(ok)
    return rec
