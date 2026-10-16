#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The fidelity statistic, computed against an independent implementation of the model."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

EVAL_DIR = Path(os.environ.get("APP_DIR", "/app")) / "eval" / "deepseek_v4_flash"
CORE_NAME = "deepseek_v4_flash_eval_fidelity"


def _core():
    """The shared fidelity core, loaded once by path under one module name (the public prefix-cache module loads its
    sibling under the same name, so both share one instance)."""
    if CORE_NAME in sys.modules:
        return sys.modules[CORE_NAME]
    spec = importlib.util.spec_from_file_location(CORE_NAME, EVAL_DIR / "fidelity" / "fidelity.py")
    if spec is None or spec.loader is None:  # pragma: no cover - a missing file is a verifier fault
        raise RuntimeError(f"cannot load the fidelity core at {EVAL_DIR / 'fidelity' / 'fidelity.py'}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[CORE_NAME] = module
    spec.loader.exec_module(module)
    return module


_c = _core()
StructuralFailure = _c.StructuralFailure
ReferenceFault = _c.ReferenceFault
ServedRequestFailed = _c.ServedRequestFailed
operation_limit_s = _c.operation_limit_s
validate_reference = _c.validate_reference
load_reference = _c.load_reference
parse_input_logprobs = _c.parse_input_logprobs
served_logprobs = _c.served_logprobs
k3 = _c.k3
rescore = _c.rescore
k3_leaving_out = _c.k3_leaving_out
rescore_leaving_out = _c.rescore_leaving_out
median = _c.median
measure_leaving_out = _c.measure_leaving_out
measure = _c.measure


def worst_over_cap(per_sequence: dict[str, float], cap: float) -> list[tuple[str, float]]:
    """The sequences whose k3 exceeds `cap` (the per-sequence cap beside the median), worst first."""
    return sorted(((k, v) for k, v in per_sequence.items() if not v <= cap), key=lambda kv: -kv[1])


def band_of(sequence_id: str) -> str:
    """A sequence's band, its length in tokens: `code-512-05` -> "512", `long-code-32768` -> "32768"."""
    parts = sequence_id.split("-")
    return parts[-1] if sequence_id.startswith("long-") else parts[1]


def band_medians(per_sequence: dict[str, float]) -> dict[str, float]:
    """The median k3 over each band's sequences, bands in length order."""
    by_band: dict[str, list[float]] = {}
    for name, value in per_sequence.items():
        by_band.setdefault(band_of(name), []).append(value)
    return {band: median(values) for band, values in sorted(by_band.items(), key=lambda kv: int(kv[0]))}


def bands_not_below(bands: dict[str, float], bars: dict[str, float]) -> list[tuple[str, float, float]]:
    """The bands whose median is not strictly below the band's bar, as (band, median, bar); NaN never passes."""
    return [(band, value, float(bars[band])) for band, value in bands.items() if not value < float(bars[band])]


def redraw_band(base_url: str, records: list[dict], band: str, *, on_attempt=None, check: str = "fidelity") -> float:
    """One whole re-draw of a band -- every sequence of the band sent again and scored again -- and its median.
    Every answer of the re-draw is checked in full first; a malformed one raises `StructuralFailure`, which is
    decisive for the node, and is recorded as the re-draw's attempt (draw 2), never folded into a median."""
    return median([rescore(base_url, r, draw=2, on_attempt=on_attempt, check=check)["k3"]
                   for r in records if band_of(r["id"]) == band])
