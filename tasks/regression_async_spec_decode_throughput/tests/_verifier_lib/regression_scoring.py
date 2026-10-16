#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Continuous score for the regression family: the optimize contract."""
from __future__ import annotations

import math
import statistics
from typing import Any

CONTRACT = "regression_recovery_v2"


def speedup_reward(ratio: float) -> float:
    """The measured gain, floored at 0. Identical to the optimize family's `speedup_reward`."""
    return max(0.0, float(ratio) - 1.0)


def geomean(values: list[float]) -> float | None:
    """Geometric mean of positive finite values; None when there are none or any is not."""
    if not values:
        return None
    if any((not isinstance(v, (int, float))) or isinstance(v, bool) or not math.isfinite(v) or v <= 0
           for v in values):
        return None
    return math.exp(sum(math.log(v) for v in values) / len(values))


def paired_round_ratios(pristine_runs: object, candidate_runs: object, *,
                        lower_is_better: bool = False) -> list[float] | None:
    """Per-round ratios in the SCORED direction (above 1 = better), or None if unpairable."""
    if not isinstance(pristine_runs, list) or not isinstance(candidate_runs, list):
        return None
    if len(pristine_runs) != len(candidate_runs) or not pristine_runs:
        return None
    out: list[float] = []
    for pv, cv in zip(pristine_runs, candidate_runs):
        if isinstance(pv, bool) or isinstance(cv, bool):
            return None
        if not isinstance(pv, (int, float)) or not isinstance(cv, (int, float)):
            return None
        if not (math.isfinite(pv) and math.isfinite(cv)) or pv <= 0 or cv <= 0:
            return None
        out.append((pv / cv) if lower_is_better else (cv / pv))
    return out


def paired_ratio_geomean(pristine: dict[str, object], candidate: dict[str, object], *,
                         lower_is_better: bool = False) -> tuple[float | None, list[float]]:
    """DIAGNOSTIC: the geometric mean of the per-round paired ratios (the score is the median)."""
    p_runs = pristine.get("runs") if isinstance(pristine, dict) else None
    c_runs = candidate.get("runs") if isinstance(candidate, dict) else None
    paired = paired_round_ratios(p_runs, c_runs, lower_is_better=lower_is_better)
    if not paired:
        return None, []
    value = geomean(paired)
    return (value, paired) if value is not None else (None, [])


def paired_ratio_median(pristine: dict[str, object], candidate: dict[str, object], *,
                        lower_is_better: bool = False) -> tuple[float | None, list[float]]:
    """THE SCORED ESTIMATOR: the median of the per-round paired ratios."""
    p_runs = pristine.get("runs") if isinstance(pristine, dict) else None
    c_runs = candidate.get("runs") if isinstance(candidate, dict) else None
    paired = paired_round_ratios(p_runs, c_runs, lower_is_better=lower_is_better)
    if not paired:
        return None, []
    value = statistics.median(paired)
    return (value, paired) if math.isfinite(value) else (None, [])


def arm_mean(rounds: dict[str, object] | None, arm: str) -> float | None:
    """One arm's aggregate mean off a bench_rounds.json record, finite and positive, else None."""
    if not isinstance(rounds, dict):
        return None
    arms = rounds.get("arms")
    if not isinstance(arms, dict):
        return None
    entry = arms.get(arm)
    if not isinstance(entry, dict):
        return None
    mean = entry.get("mean")
    if isinstance(mean, bool) or not isinstance(mean, (int, float)) or not mean > 0:
        return None
    return float(mean) if math.isfinite(mean) else None


def matrix_workloads(rounds: dict[str, object] | None) -> dict[str, dict] | None:
    """The per-workload section a matrix rounds record carries, or None for a single-shape record."""
    if not isinstance(rounds, dict):
        return None
    wls = rounds.get("workloads")
    if not isinstance(wls, dict) or not wls:
        return None
    return {str(k): v for k, v in wls.items() if isinstance(v, dict)}


def _point_arms(entry: dict) -> tuple[dict, dict]:
    arms = entry.get("arms") if isinstance(entry.get("arms"), dict) else {}
    return (arms.get("pristine") if isinstance(arms.get("pristine"), dict) else {},
            arms.get("candidate") if isinstance(arms.get("candidate"), dict) else {})


def _point_usable(entry: dict, min_per_arm: int) -> tuple[bool, str | None]:
    """`bench_stats.usable_round_arms` restated here so this module stays import-free and pure."""
    pristine, candidate = _point_arms(entry)
    if not pristine:
        return False, "pristine_arm_absent"
    if not candidate:
        return False, "candidate_arm_absent"
    counts = {}
    for arm, e in (("pristine", pristine), ("candidate", candidate)):
        n = e.get("n_ok")
        counts[arm] = int(n) if isinstance(n, int) and not isinstance(n, bool) else 0
    if counts["pristine"] != counts["candidate"]:
        return False, (f"unequal_arm_sample_counts(pristine={counts['pristine']},"
                       f"candidate={counts['candidate']})")
    if counts["pristine"] < min_per_arm:
        return False, f"too_few_rounds_per_arm({counts['pristine']}<{min_per_arm})"
    return True, None


def resolve_matrix_ratio(rounds: dict[str, object], *, min_per_arm: int = 2,
                         default_min_ratio: float = 0.95) -> dict[str, object]:
    """The workload-matrix estimator: geomean over the SCORED points of each point's paired median."""
    wls = matrix_workloads(rounds) or {}
    scored_names = [str(n) for n in (rounds.get("scored_workloads") or [])
                    if isinstance(n, str)] or [n for n, w in wls.items() if w.get("role", "scored") == "scored"]
    guard_names = [str(n) for n in (rounds.get("guard_workloads") or [])
                   if isinstance(n, str)] or [n for n, w in wls.items() if w.get("role") == "guard"]
    anchor = rounds.get("anchor") if isinstance(rounds.get("anchor"), str) else (scored_names[0] if scored_names else None)

    per_point: dict[str, float] = {}
    paired_per_point: dict[str, list[float]] = {}
    unusable: dict[str, str] = {}
    pristine_ok = candidate_ok = True
    for name in scored_names:
        entry = wls.get(name)
        if not isinstance(entry, dict):
            unusable[name] = "point_absent"; pristine_ok = candidate_ok = False
            continue
        ok, why = _point_usable(entry, min_per_arm)
        pristine, candidate = _point_arms(entry)
        lower = bool(entry.get("lower_is_better"))
        ratio, paired = paired_ratio_median(pristine, candidate, lower_is_better=lower)
        if not ok or ratio is None:
            unusable[name] = why or "rounds_unpairable"
            if (why or "").startswith("pristine_arm_absent") or arm_mean({"arms": {"pristine": pristine}}, "pristine") is None:
                pristine_ok = False
            else:
                candidate_ok = False
            continue
        per_point[name] = ratio
        paired_per_point[name] = paired

    guards: dict[str, dict[str, object]] = {}
    unmeasured: list[str] = []
    for name in guard_names:
        entry = wls.get(name) if isinstance(wls.get(name), dict) else {}
        pristine, candidate = _point_arms(entry)
        lower = bool(entry.get("lower_is_better"))
        floor = entry.get("min_ratio")
        floor = float(floor) if isinstance(floor, (int, float)) and not isinstance(floor, bool) else default_min_ratio
        ok_arms, _ = _point_usable(entry, 1)
        ratio, paired = (paired_ratio_median(pristine, candidate, lower_is_better=lower)
                         if ok_arms else (None, []))
        best = max(paired) if paired else None
        verdict = (best >= floor) if best is not None else None
        if verdict is None:
            unmeasured.append(name)
        guards[name] = {"ratio": ratio, "paired_round_ratios": paired, "min_ratio": floor,
                        "lower_is_better": lower, "ok": verdict, "best_round_ratio": best,
                        "rule": "regressed_in_every_round"}
    guards_ok = all(g["ok"] is not False for g in guards.values())

    headline_p = arm_mean({"arms": (wls.get(anchor) or {}).get("arms")}, "pristine") if anchor else None
    headline_c = arm_mean({"arms": (wls.get(anchor) or {}).get("arms")}, "candidate") if anchor else None
    common = {
        "speedup_per_workload": dict(per_point),
        "paired_round_ratios_per_workload": paired_per_point,
        "scored_workloads": list(scored_names),
        "guard_workloads": list(guard_names),
        "anchor_workload": anchor,
        "guard_ratios": guards,
        "guards_ok": guards_ok,
        "guard_unmeasured": unmeasured,
        "unusable_workloads": unusable,
        "worst_workload_speedup": min(per_point.values()) if per_point else None,
        "best_workload_speedup": max(per_point.values()) if per_point else None,
    }
    speedup = geomean(list(per_point.values())) if per_point and not unusable and scored_names else None
    if speedup is not None:
        anchor_paired = paired_per_point.get(anchor or "", [])
        return {
            "speedup": speedup, "paired_round_ratios": anchor_paired,
            "speedup_paired_geomean": geomean(anchor_paired) if anchor_paired else None,
            "baseline_tps": headline_p, "candidate_tps": headline_c,
            "estimator": "matrix_geomean_of_paired_medians", "reason": None,
            "pristine_arm_ok": True, "candidate_arm_ok": True, **common,
        }
    first = next(iter(unusable.items()), (None, "no_scored_workloads"))
    return {
        "speedup": None, "paired_round_ratios": [], "speedup_paired_geomean": None,
        "baseline_tps": None, "candidate_tps": None, "estimator": None,
        "reason": f"workload_{first[0]}:{first[1]}" if first[0] else first[1],
        "pristine_arm_ok": pristine_ok, "candidate_arm_ok": candidate_ok, **common,
    }


def resolve_rounds_ratio(rounds: dict[str, object] | None, *, usable: tuple[bool, str | None],
                         lower_is_better: bool = False, min_per_arm: int = 2) -> dict[str, object]:
    """Both arms and the scored ratio from ONE aggregate, or neither."""
    if matrix_workloads(rounds) is not None:
        return resolve_matrix_ratio(rounds, min_per_arm=min_per_arm)  # type: ignore[arg-type]
    pristine = arm_mean(rounds, "pristine")
    candidate = arm_mean(rounds, "candidate")
    ok, reason = usable
    arms = rounds.get("arms") if isinstance(rounds, dict) else None
    ratio, paired, geo = None, [], None
    if isinstance(arms, dict):
        p_arm, c_arm = arms.get("pristine") or {}, arms.get("candidate") or {}
        ratio, paired = paired_ratio_median(p_arm, c_arm, lower_is_better=lower_is_better)
        geo, _ = paired_ratio_geomean(p_arm, c_arm, lower_is_better=lower_is_better)
    if ok and pristine is not None and candidate is not None and ratio is not None:
        return {
            "speedup": ratio, "paired_round_ratios": paired, "speedup_paired_geomean": geo,
            "baseline_tps": pristine, "candidate_tps": candidate,
            "estimator": "rounds_median_paired", "reason": None,
            "pristine_arm_ok": True, "candidate_arm_ok": True,
        }
    return {
        "speedup": None, "paired_round_ratios": [], "speedup_paired_geomean": None,
        "baseline_tps": None, "candidate_tps": None, "estimator": None,
        "reason": reason or ("rounds_unpairable" if ratio is None else "rounds_incomplete"),
        "pristine_arm_ok": pristine is not None,
        "candidate_arm_ok": candidate is not None,
    }


def classify(*, ratio: float | None, noise_floor: float, pristine_ok: bool, candidate_ok: bool,
             no_patch: bool, tree_undiffable: bool, integrity_ok: bool,
             correctness_measured: bool, correctness_ok: bool,
             pristine_reference_ok: bool = True, guards_ok: bool = True) -> tuple[str | None, str | None]:
    """(reason token, the check that decided it) -- (None, None) when the reward stands."""
    if not pristine_ok:
        return "pristine_failed", "pristine_arm"
    if not pristine_reference_ok:
        return "pristine_failed", "pristine_correctness_reference"
    if tree_undiffable:
        return "static_checks", "tree_undiffable"
    if no_patch:
        return "no_patch", "no_patch"
    if not candidate_ok:
        return "candidate_failed", "candidate_arm"
    if not integrity_ok:
        return "candidate_failed", "workload_integrity"
    if not correctness_measured:
        return "correctness_failed", "correctness_unmeasured"
    if not correctness_ok:
        return "correctness_failed", "correctness"
    if ratio is None or not math.isfinite(ratio):
        return "candidate_failed", "ratio_unavailable"
    if not guards_ok:
        return "guard_regressed", "guard_workload"
    if ratio < noise_floor:
        return "no_speedup", "noise_floor"
    return None, None


def compute_regression_reward(*, reason: str | None, ratio: float | None, noise_floor: float,
                              correctness_ok: bool, oracle_issue: dict[str, object] | None = None,
                              scored_metric: str | None = None,
                              higher_is_better: bool = True,
                              matrix: dict[str, object] | None = None) -> dict[str, Any]:
    """The reward.json contents and the metrics.json sidecar fields, from the verdict."""
    finite = ratio is not None and isinstance(ratio, (int, float)) and math.isfinite(ratio)
    if reason is None and not finite:
        raise ValueError("a standing reward needs a finite ratio; classify the run first")
    if reason is None and ratio < noise_floor:  # type: ignore[operator]
        raise ValueError(f"ratio {ratio} is under the noise floor {noise_floor}; the verdict must be "
                         f"no_speedup, not None")
    rewards: dict[str, float] = {"reward": 0.0, "correctness": 0.0, "performance": 0.0}
    metrics: dict[str, Any] = {
        "scoring_contract": CONTRACT,
        "reason": reason,
        "scored_metric": scored_metric,
        "higher_is_better": higher_is_better,
        "speedup_vs_baseline": float(ratio) if finite else None,
        "noise_floor": noise_floor,
        "above_noise_floor": (bool(ratio >= noise_floor) if finite else None),  # type: ignore[operator]
        "oracle_issue_checks": dict(oracle_issue or {}),
    }
    if matrix:
        for key in ("speedup_per_workload", "worst_workload_speedup", "best_workload_speedup",
                    "scored_workloads", "guard_workloads", "anchor_workload", "guard_ratios",
                    "guards_ok", "guard_unmeasured", "unusable_workloads"):
            if key in matrix:
                metrics[key] = matrix[key]
    if correctness_ok:
        rewards["correctness"] = 1.0
    if reason is None:
        value = speedup_reward(ratio)  # type: ignore[arg-type]
        rewards["reward"] = value
        rewards["performance"] = value
        rewards["speedup_geomean"] = float(ratio)  # type: ignore[arg-type]
        metrics["speedup_geomean"] = float(ratio)  # type: ignore[arg-type]
    elif reason == "no_speedup" and finite:
        rewards["speedup_geomean"] = float(ratio)  # type: ignore[arg-type]
        metrics["speedup_geomean"] = float(ratio)  # type: ignore[arg-type]
    return {"rewards": rewards, "metrics": metrics}
