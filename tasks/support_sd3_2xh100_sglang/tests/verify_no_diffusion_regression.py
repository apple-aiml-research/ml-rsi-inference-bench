#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Diagnostic registry-preservation check, outside the scoring profile.

Checks that registry.py still names at least two of five other model families.
This static observation does not establish that those models execute correctly;
runtime regression checks require their own weights and execution evidence.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import GateResult, WORK_DIR, safe_run  # noqa: E402

GATE = "no_diffusion_regression"
MMG = WORK_DIR / "repo" / "python" / "sglang" / "multimodal_gen"

OTHER_MODELS = ["flux", "qwen", "wan", "hunyuan", "cosmos"]


def run() -> GateResult:
    reg = MMG / "registry.py"
    details: dict = {}
    if not reg.exists():
        return GateResult(gate=GATE, passed=False, score=0.0,
                          error=f"registry.py missing at {reg}", details=details)
    txt = reg.read_text(encoding="utf-8", errors="replace").lower()
    present = [m for m in OTHER_MODELS if m in txt]
    details["other_models_still_in_registry"] = present
    details["sd3_registered"] = "stable-diffusion-3" in txt or "stablediffusion3" in txt
    if len(present) < 2:
        return GateResult(gate=GATE, passed=False, score=0.0,
                          error=f"only {present} other families in registry — possible regression",
                          details=details)
    return GateResult(gate=GATE, passed=True, score=1.0, details=details)


if __name__ == "__main__":
    sys.exit(safe_run(run, GATE, fail_exit_code=0))
