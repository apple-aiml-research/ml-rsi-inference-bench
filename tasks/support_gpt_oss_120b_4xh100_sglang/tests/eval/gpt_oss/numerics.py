#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Public finite-output and error measurement, and the published numerical policy."""

import json
from pathlib import Path

POLICY_PATH = Path(__file__).with_name("numerical-policy.json")


def policy(path=POLICY_PATH) -> dict:
    """The published policy, validated: two operations, each with finite max/RMS limits."""
    data = json.loads(Path(path).read_text())
    if data.get("schema") != "gpt-oss-numerical-policy/1":
        raise ValueError(f"unexpected numerical policy schema in {path}")
    operations = data.get("operations")
    if not isinstance(operations, dict) or set(operations) != {"expert", "attention"}:
        raise ValueError("the numerical policy defines exactly the expert and attention operations")
    for name, limits in operations.items():
        for key in ("max", "rms"):
            value = limits.get(key)
            if type(value) not in (int, float) or not 0 < value < 1:
                raise ValueError(f"{name}.{key} must be a finite positive fraction, not {value!r}")
    return data


def limits(operation: str, path=POLICY_PATH) -> dict:
    values = policy(path)["operations"][operation]
    return {"max": float(values["max"]), "rms": float(values["rms"])}


def error(actual, expected) -> dict:
    import torch
    if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
        raise ValueError("both results must be tensors")
    if not actual.is_floating_point() or not expected.is_floating_point():
        raise ValueError(f"results must have a floating-point dtype, not {actual.dtype} / {expected.dtype}")
    if actual.shape != expected.shape or actual.numel() == 0:
        raise ValueError(f"output shape {actual.shape} differs from {expected.shape}")
    a, e = actual.float(), expected.float().to(actual.device)
    if not torch.isfinite(a).all() or not torch.isfinite(e).all():
        raise ValueError("nonfinite value in finite-domain comparison")
    difference = a - e
    scale = torch.maximum(e.square().mean().sqrt(), e.abs())
    normalized = difference / scale.clamp_min(1e-30)
    return {"count": actual.numel(), "max": normalized.abs().max().item(),
            "rms": normalized.double().square().mean().sqrt().item(),
            "absolute_max": difference.abs().max().item()}


def assess(actual, expected, operation: str, path=POLICY_PATH) -> dict:
    """The measurement and its standing against the published limits for `operation`."""
    bound = limits(operation, path)
    try:
        measured = error(actual, expected)
    except ValueError as exc:
        return {"operation": operation, "limits": bound, "structurally_valid": False,
                "reason": str(exc), "within_policy": False}
    return {"operation": operation, "limits": bound, "structurally_valid": True, **measured,
            "within_policy": measured["max"] <= bound["max"] and measured["rms"] <= bound["rms"]}
