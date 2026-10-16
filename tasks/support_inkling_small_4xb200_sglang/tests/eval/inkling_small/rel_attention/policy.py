#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Numerical policy of the relative-attention operator contract (README.md, 'Numerical policy')."""
from __future__ import annotations

import torch

RTOL = 2.0


def fwd_atol(r: torch.Tensor) -> float:
    r32 = r.to(torch.float32)
    return float(2.0 * (r32 + 0.3 - 0.3 - r32).abs().max())


def evaluate(out: torch.Tensor, r: torch.Tensor, o_bf16: torch.Tensor, rows: torch.Tensor,
             inputs_unchanged: bool) -> dict:
    """Metrics and verdict for one call; `rows` selects the valid output rows."""
    o = out[rows].to(torch.float64)
    ref = r[rows].to(torch.float64)
    pt = o_bf16[rows].to(torch.float64)
    finite_rows = torch.isfinite(o).flatten(1).all(dim=1)
    nonfinite = int((~finite_rows).sum())
    res = {"rows": int(rows.sum()), "nonfinite_rows": nonfinite, "inputs_unchanged": bool(inputs_unchanged)}
    if nonfinite:
        res.update({"max_abs_err": None, "bound": None, "passed": False,
                    "reason": f"{nonfinite} valid output rows are not finite"})
        return res
    err = float((o - ref).abs().max())
    err_pt = float((pt - ref).abs().max())
    atol = fwd_atol(ref)
    bound = RTOL * err_pt + atol
    res.update({"max_abs_err": err, "max_abs_err_bf16_pipeline": err_pt, "fwd_atol": atol, "bound": bound,
                "mean_abs_err": float((o - ref).abs().mean())})
    if not inputs_unchanged:
        res.update({"passed": False, "reason": "an input tensor was modified"})
    elif err > bound:
        res.update({"passed": False, "reason": f"max|o - r| = {err:.6g} > {bound:.6g}"})
    else:
        res.update({"passed": True, "reason": ""})
    return res
