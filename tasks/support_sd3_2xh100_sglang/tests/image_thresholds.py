#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Per-precision image thresholds shared by single-GPU and serving profiles."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
METRICS = ("psnr", "ssim", "clip_img", "clip_txt")
PROFILES = ("bf16", "fp32", "mixed")
TABLE_PATH = HERE / "image_thresholds.json"


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate(table, fixtures):
    if table.get("schema") != "sd3-image-thresholds-v1":
        raise ValueError("Unknown image threshold schema")
    if set(table.get("profiles", {})) != set(PROFILES):
        raise ValueError("Every precision needs its own threshold table")
    expected = {f["id"] for f in fixtures}
    for profile, spec in table["profiles"].items():
        if set(spec.get("fixtures", {})) != expected:
            raise ValueError(f"Incomplete fixture bars: {profile}")
        rows = [spec["defaults"], *spec["fixtures"].values()]
        for bars in rows:
            if set(bars) != set(METRICS):
                raise ValueError(f"Incomplete metric bars: {profile}")
            for key, value in bars.items():
                upper = float("inf") if key == "psnr" else 1.0
                if not _number(value) or not 0 < value <= upper:
                    raise ValueError(f"Invalid bar: {profile}/{key}")
        cfg1 = spec.get("cfg1_psnr")
        if not _number(cfg1) or cfg1 < spec["defaults"]["psnr"]:
            raise ValueError(f"Invalid CFG1 bar: {profile}")
    return table


_FIXTURES = json.loads((HERE / "sd3_golden.json").read_text())["fixtures"]
TABLE = validate(json.loads(TABLE_PATH.read_text()), _FIXTURES)
_CFG = {f["id"]: float(f["cfg"]) for f in _FIXTURES}
REVISION = TABLE["revision"]
SHA256 = hashlib.sha256(TABLE_PATH.read_bytes()).hexdigest()
STATE = "per-precision image thresholds"


def bars_for(fixture, profile):
    """Use one profile's explicit bars; extensions keep that profile's rule.

    Request-derived seed/size variants retain their canonical fixture identity.
    A changed CFG has the reserved extended_request identity and uses that
    profile's defaults (or its CFG1 PSNR), as in the preceding reference contract.
    Unknown IDs and profiles fail; mixed never falls back to BF16.
    """
    if profile not in PROFILES:
        raise ValueError(f"Unknown threshold precision: {profile!r}")
    spec = TABLE["profiles"][profile]
    ident, cfg = fixture.get("id"), fixture.get("cfg")
    if not _number(cfg):
        raise ValueError("Fixture CFG must be finite")
    if ident in spec["fixtures"]:
        if float(cfg) != _CFG[ident]:
            raise ValueError("Canonical fixture CFG changed")
        return dict(spec["fixtures"][ident])
    if ident != "extended_request":
        raise ValueError(f"Unknown threshold fixture: {ident!r}")
    bars = dict(spec["defaults"])
    if abs(float(cfg) - 1.0) < 1e-9:
        bars["psnr"] = spec["cfg1_psnr"]
    return bars


def matches(values, bars):
    """All four finite numerical readings must meet inclusive lower bounds."""
    return all(_number(values.get(m)) and values[m] >= bars[m] for m in METRICS)


def receipt():
    return {"revision": REVISION, "sha256": SHA256, "state": STATE}
