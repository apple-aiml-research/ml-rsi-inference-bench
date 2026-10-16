#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""__init__ for the verifier library; keeps tests/_verifier_lib/ a real Python package.

Re-exports the most-used surface for convenience."""
from __future__ import annotations

from .category_gate import classify_diff, classify_repo_diff, is_rejected_by_narrow, render_summary
from .scoring import compute_reward, geomean, speedup_reward

__all__ = [
    "classify_diff",
    "classify_repo_diff",
    "is_rejected_by_narrow",
    "render_summary",
    "compute_reward",
    "geomean",
    "speedup_reward",
]
