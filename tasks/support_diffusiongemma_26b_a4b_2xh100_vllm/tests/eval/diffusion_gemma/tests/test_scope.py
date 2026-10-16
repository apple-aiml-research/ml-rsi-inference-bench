#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Self-tests of the public scope and recipes, CPU only."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parents[1]))

from eval.diffusion_gemma.serving import recipes  # noqa: E402


def test_scope_names_both_deployments_and_the_context():
    scope = json.loads((HERE / "scope.json").read_text())
    assert scope["deployments"] == ["tp1", "tp2"]
    assert scope["context_tokens"] == 262144
    assert "--max-model-len" in recipes.BASE_ARGS and recipes.BASE_ARGS[recipes.BASE_ARGS.index("--max-model-len") + 1] == "262144"


def test_fixed_work_configurations_disable_early_stopping():
    assert recipes.GENCFG["tp1_fixed"]["confidence_threshold"] == 0.0
    assert recipes.GENCFG["tp2_fixed"]["confidence_threshold"] == 0.0
    assert "--tensor-parallel-size" in recipes.RECIPES["tp2_fixed"]["args"]


def test_budgets_are_positive_integers():
    budgets = json.loads((HERE / "budgets.json").read_text())
    assert budgets["schema"] == "diffusion-gemma-budgets/1"
    assert all(type(v["cap_s"]) is int and v["cap_s"] > 0 for v in budgets["operations"].values())
