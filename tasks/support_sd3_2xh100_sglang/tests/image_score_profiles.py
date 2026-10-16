#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
'SD3 image score profiles utilities.'
from __future__ import annotations
import hashlib
import json
import math
import os
import sys
from pathlib import Path

PROFILES = ("bf16", "fp32", "mixed")
DEFAULT_PROFILE = "mixed"
METRIC_KEYS = ("image_score", "image_fixtures_passed", "image_fixtures_total")


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_profile(gates: Path, profile: str) -> tuple[dict, dict | None]:
    path = gates / f"image_similarity_full_{profile}.gate.json"
    meta = {"source": path.name, "profile": profile}
    if not path.is_file():
        return {**meta, "status": "missing"}, None
    try:
        raw = path.read_bytes()
        meta["source_sha256"] = hashlib.sha256(raw).hexdigest()
        obj = json.loads(raw)
        if not isinstance(obj, dict) or obj.get("gate") != f"image_similarity_full_{profile}":
            raise ValueError("gate identity mismatch")
        det = obj.get("details")
        if not isinstance(det, dict) or det.get("profile") != profile:
            raise ValueError("profile identity mismatch")
        per = det.get("per")
        if not isinstance(per, list) or not per or any(not isinstance(x, dict) or type(x.get("match")) is not bool for x in per):
            raise ValueError("no complete boolean per-fixture match readings")
        native = isinstance(det.get("native_generation"), dict) and det["native_generation"].get("ok") is True
        n = sum(x["match"] for x in per) if native else 0
        return {**meta, "status": "measured", "native_generation": native,
                "image_fixtures_passed": n, "image_fixtures_total": len(per), "image_score": n / len(per)}, obj
    except (OSError, ValueError, TypeError) as exc:
        return {**meta, "status": "malformed", "error": f"{type(exc).__name__}: {exc}"[:300]}, None


def prepare(verifier: Path) -> dict:
    gates = verifier / "gates"
    by_profile, default = {}, None
    for profile in PROFILES:
        by_profile[profile], obj = read_profile(gates, profile)
        if profile == DEFAULT_PROFILE:
            default = obj
    alias = gates / "image_similarity_full.gate.json"
    if default is None:
        alias.unlink(missing_ok=True)
    else:
        default = dict(default, gate="image_similarity_full")
        default["details"] = dict(default["details"], scored=False, diagnostic=True, compatibility_alias={
            "source": by_profile[DEFAULT_PROFILE]["source"],
            "source_sha256": by_profile[DEFAULT_PROFILE]["source_sha256"],
            "scope": "default-profile image_score only; reward requires all profiles",
        })
        atomic_json(alias, default)
    report = {"schema": "sd3-image-profiles-v1", "image_score_profile": DEFAULT_PROFILE,
              "image_score_scope": "default-profile native fixture fraction",
              "image_scores_by_profile": by_profile,
              "image_profile_set_complete": all(x["status"] == "measured" for x in by_profile.values())}
    atomic_json(verifier / "sd3_image_scores.json", report)
    return report


def annotate(verifier: Path) -> None:
    report = json.loads((verifier / "sd3_image_scores.json").read_text())
    path = verifier / "reward.json"
    reward = json.loads(path.read_text())

    default = report["image_scores_by_profile"][DEFAULT_PROFILE]
    expected = {k: default[k] for k in METRIC_KEYS if k in default}
    actual = {k: reward[k] for k in METRIC_KEYS if k in reward}
    if actual != expected:
        raise ValueError(f"canonical grader image metric differs from its profile: {actual} != {expected}")


    invalid = [k for k, value in reward.items()
               if type(value) not in (int, float)
               or (isinstance(value, float) and not math.isfinite(value))]
    if invalid:
        raise ValueError(f"canonical grader reward has non-numeric/non-finite fields: {invalid}")


if __name__ == "__main__":
    root = Path(os.environ.get("VERIFIER_DIR", os.environ.get("INFB_VERIFIER_DIR", "/logs/verifier")))
    if sys.argv[1:] == ["prepare"]:
        prepare(root)
    elif sys.argv[1:] == ["annotate"]:
        annotate(root)
    else:
        raise SystemExit("usage: image_score_profiles.py prepare|annotate")
