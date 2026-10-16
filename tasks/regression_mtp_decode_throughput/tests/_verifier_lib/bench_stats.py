#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Bench-round statistics for the regression verifiers: warmup plateau, rotation, aggregation."""

from __future__ import annotations

import math
import statistics


def _spread_of(group: list[float]) -> float:
    """Relative peak-to-peak spread of a group, 0.0 for a group of one."""
    mean = statistics.fmean(group)
    return (max(group) - min(group)) / mean if len(group) > 1 and mean else 0.0


def warmup_plateau_state(
    series: list[float | None],
    min_warm: int = 2,
    tol_floor: float = 0.01,
) -> tuple[bool, list[float], float]:
    """``(is_warm, trailing_plateau, tol)`` from the warmup values so far."""
    vals = [float(v) for v in series if v]
    if not vals:
        return False, [], tol_floor

    plateau = [vals[-1]]
    for value in reversed(vals[:-1]):
        plateau_mean = statistics.fmean(plateau)
        tol = max(_spread_of(plateau), tol_floor)
        if plateau_mean and abs(value - plateau_mean) / plateau_mean <= tol:
            plateau.insert(0, value)
        else:
            break
    tol = max(_spread_of(plateau), tol_floor)
    return len(plateau) > min_warm, plateau, tol


def first_run_off_plateau(
    first_scored: float | None,
    plateau: list[float],
    tol: float,
) -> float | None:
    """Signed relative deviation of the first SCORED bench from the warmup plateau, or ``None``."""
    if not first_scored or not plateau:
        return None
    plateau_mean = statistics.fmean(plateau)
    return (first_scored - plateau_mean) / plateau_mean if plateau_mean else None


def rotated_arms(arms: list[str], round_index: int) -> list[str]:
    """The arm order for a 0-based scored round: ``arms`` left-rotated by ``round_index``."""
    if not arms:
        return []
    offset = round_index % len(arms)
    return arms[offset:] + arms[:offset]


def summarise_arm(values: list[float]) -> dict[str, object]:
    """``{runs, n_ok, mean, stdev}`` for one arm. ``stdev`` is 0.0 for fewer than two samples."""
    clean = [float(v) for v in values if v and math.isfinite(v)]
    return {
        "runs": clean,
        "n_ok": len(clean),
        "mean": statistics.fmean(clean) if clean else None,
        "stdev": statistics.stdev(clean) if len(clean) > 1 else 0.0,
    }


def usable_round_arms(
    rounds: dict[str, object] | None,
    min_per_arm: int = 2,
) -> tuple[bool, str | None]:
    """``(ok, reason)`` -- may this rounds aggregate be scored at all?"""
    if not isinstance(rounds, dict):
        return False, "no_rounds_aggregate"
    arms = rounds.get("arms")
    if not isinstance(arms, dict):
        return False, "rounds_aggregate_has_no_arms"
    counts: dict[str, int] = {}
    for arm in ("pristine", "candidate"):
        entry = arms.get(arm)
        if not isinstance(entry, dict):
            return False, f"{arm}_arm_absent"
        count = entry.get("n_ok")
        counts[arm] = int(count) if isinstance(count, int) else 0
    if counts["pristine"] != counts["candidate"]:
        return False, (
            f"unequal_arm_sample_counts(pristine={counts['pristine']},"
            f"candidate={counts['candidate']})"
        )
    if counts["pristine"] < min_per_arm:
        return False, f"too_few_rounds_per_arm({counts['pristine']}<{min_per_arm})"
    return True, None


def ratio_with_stdev(
    pristine: dict[str, object],
    candidate: dict[str, object],
) -> tuple[float | None, float | None]:
    """``(ratio, ratio_stdev)`` for candidate/pristine, propagating both arms' relative spreads."""
    pristine_mean = pristine.get("mean")
    candidate_mean = candidate.get("mean")
    if not isinstance(pristine_mean, (int, float)) or not pristine_mean:
        return None, None
    if not isinstance(candidate_mean, (int, float)) or not candidate_mean:
        return None, None
    if not (math.isfinite(pristine_mean) and math.isfinite(candidate_mean)):
        return None, None
    ratio = candidate_mean / pristine_mean
    if not math.isfinite(ratio):
        return None, None
    rel_p = (pristine.get("stdev") or 0.0) / pristine_mean
    rel_c = (candidate.get("stdev") or 0.0) / candidate_mean
    stdev = ratio * ((rel_p**2 + rel_c**2) ** 0.5)
    return ratio, (stdev if math.isfinite(stdev) else None)


def median_paired_ratio(
    pristine: dict[str, object],
    candidate: dict[str, object],
) -> tuple[float | None, list[float]]:
    """The MEDIAN of the per-round candidate/pristine ratios, or ``(None, [])`` when unpairable."""
    p_runs = pristine.get("runs") if isinstance(pristine, dict) else None
    c_runs = candidate.get("runs") if isinstance(candidate, dict) else None
    if not isinstance(p_runs, list) or not isinstance(c_runs, list):
        return None, []
    if len(p_runs) != len(c_runs) or not p_runs:
        return None, []
    paired: list[float] = []
    for pv, cv in zip(p_runs, c_runs):
        if isinstance(pv, bool) or isinstance(cv, bool):
            return None, []
        if not isinstance(pv, (int, float)) or not isinstance(cv, (int, float)):
            return None, []
        if not (math.isfinite(pv) and math.isfinite(cv)) or pv <= 0 or cv <= 0:
            return None, []
        paired.append(cv / pv)
    ratio = statistics.median(paired)
    if not math.isfinite(ratio):
        return None, []
    return ratio, paired



MATRIX_POINT_DIR = "wl"


def load_matrix_spec(path) -> dict[str, object]:
    """Parse a task's tests/workloads.json into the points `_bench_rounds.py` benches and reads."""
    import json
    from pathlib import Path as _P
    doc = json.loads(_P(path).read_text(encoding="utf-8"))
    points = {str(w["name"]): dict(w) for w in doc.get("workloads", []) if isinstance(w, dict) and "name" in w}
    anchor = doc.get("primary_workload") or (next(iter(points)) if points else None)
    scored = [str(n) for n in doc.get("scored_workloads") or ([anchor] if anchor else [])]
    guards = [str(n) for n in doc.get("guard_workloads") or []]
    for name in scored + guards:
        if name not in points:
            raise ValueError(f"{path}: workload {name!r} is listed but not declared")
    overlap = set(scored) & set(guards)
    if overlap:
        raise ValueError(f"{path}: {sorted(overlap)} cannot be both scored and guard")
    if anchor and anchor not in scored:
        raise ValueError(f"{path}: the anchor {anchor!r} must be one of the scored points")
    for name, pt in points.items():
        pt.setdefault("role", "scored" if name in scored else ("guard" if name in guards else "unused"))
    return {"anchor": anchor, "scored": scored, "guards": guards, "points": points}


def point_raw_dir(raw_dir, name: str, anchor: str | None):
    """Where a runner writes one point's artefacts: the anchor at the legacy location, the rest under
    `wl/<name>/`, so every existing reader of the legacy files keeps working."""
    from pathlib import Path as _P
    raw = _P(raw_dir)
    return raw if name == anchor else raw / MATRIX_POINT_DIR / name


def read_point_series(spec: dict, raw_dir, reader, *, default_key: str, default_lower: bool):
    """One arm's metric for every declared point, via `reader(dir, metric_key) -> float | None`."""
    out: dict[str, float | None] = {}
    for name, pt in spec["points"].items():
        if pt.get("role") not in ("scored", "guard"):
            continue
        source = pt.get("from") if isinstance(pt.get("from"), str) else name
        key = str(pt.get("metric_key") or default_key)
        out[name] = reader(point_raw_dir(raw_dir, source, spec["anchor"]), key)
    return out


def matrix_section(spec: dict, series: dict[str, dict[str, list[float]]], *,
                   default_key: str, default_lower: bool) -> dict[str, object]:
    """The `workloads` block of a matrix rounds record from per-point, per-arm series."""
    block: dict[str, object] = {}
    for name, pt in spec["points"].items():
        if pt.get("role") not in ("scored", "guard"):
            continue
        arms = series.get(name) or {}
        entry: dict[str, object] = {
            "role": pt["role"],
            "metric_key": str(pt.get("metric_key") or default_key),
            "lower_is_better": bool(pt.get("lower_is_better", default_lower)),
            "arms": {arm: summarise_arm(list(arms.get(arm) or [])) for arm in ("pristine", "candidate")},
        }
        if pt.get("role") == "guard":
            entry["min_ratio"] = float(pt.get("min_ratio", 0.95))
        if isinstance(pt.get("from"), str):
            entry["from"] = pt["from"]
        block[name] = entry
    return block
