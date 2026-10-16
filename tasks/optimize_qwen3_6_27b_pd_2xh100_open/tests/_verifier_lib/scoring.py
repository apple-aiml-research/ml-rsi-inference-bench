#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Reward function for throughput-optimization tasks."""
from __future__ import annotations

import math
from typing import Any


def geomean(values: list[float]) -> float:
    """Geometric mean of positive values. Returns 0.0 if any are non-positive."""
    if not values:
        return 0.0
    if any(v <= 0 for v in values):
        return 0.0
    return math.exp(sum(math.log(v) for v in values) / len(values))


def speedup_reward(speedup: float) -> float:
    """The measured gain, floored at 0. No target, no normalisation, no ceiling."""
    return max(0.0, speedup - 1.0)


def compute_reward(
    *,
    speedup_per_workload: dict[str, float],
    correctness_passed: bool,
    category_gate_kind: str | None = None,
    rejected_categories: set[str] | None = None,
    no_patch: bool = False,
) -> dict[str, Any]:
    """Compute the final reward.json contents AND a sidecar metrics dict."""
    rejected_categories = rejected_categories or set()
    rewards: dict[str, float] = {
        "reward": 0.0,
        "correctness": 0.0,
        "performance": 0.0,
    }
    metrics: dict[str, Any] = {}

    if no_patch:
        metrics["reason"] = "no_patch"
        return {"rewards": rewards, "metrics": metrics}

    if category_gate_kind is not None and category_gate_kind in rejected_categories:
        rewards["correctness"] = 1.0
        metrics["category_gate_kind"] = category_gate_kind
        metrics["reason"] = f"category_gate_rejected_{category_gate_kind}"
        return {"rewards": rewards, "metrics": metrics}

    if not correctness_passed:
        metrics["reason"] = "correctness_failed"
        return {"rewards": rewards, "metrics": metrics}

    rewards["correctness"] = 1.0

    if not speedup_per_workload:
        metrics["reason"] = "no_speedup_measurements"
        return {"rewards": rewards, "metrics": metrics}

    speedup_geomean = geomean(list(speedup_per_workload.values()))
    metrics["speedup_geomean"] = speedup_geomean
    metrics["speedup_per_workload"] = dict(speedup_per_workload)
    metrics["worst_workload_speedup"] = min(speedup_per_workload.values())
    metrics["best_workload_speedup"] = max(speedup_per_workload.values())

    reward_value = speedup_reward(speedup_geomean)
    rewards["reward"] = reward_value
    rewards["performance"] = reward_value
    rewards["speedup_geomean"] = speedup_geomean

    return {"rewards": rewards, "metrics": metrics}
