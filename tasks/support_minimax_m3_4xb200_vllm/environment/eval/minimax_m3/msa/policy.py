#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The published numerical policies of the MiniMax Sparse Attention contract (README section "Numerical policies")."""

from __future__ import annotations

import torch

from . import reference as ref

SELECTION_MARGIN = 1e-3
BF16_ABS = 1e-6
FP8_FLOOR = 0.125
FP8_FACTOR = 4.0
EQUAL_LOGIT_FRACTION = 2.0 ** -7


def selection_format_problems(selection: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor) -> list[str]:
    """Format problems of ``[R, 16]`` selections, one string per offending row (first ten reported)."""
    problems: list[str] = []
    sel = selection.to("cpu", torch.int64)
    pos = positions.to("cpu", torch.int64)
    for i in range(sel.shape[0]):
        row = sel[i].tolist()
        if not bool(valid[i]):
            if any(x != -1 for x in row):
                problems.append(f"row {i}: padded row not all -1: {row}")
            continue
        vis = int(pos[i]) // ref.BLOCK + 1
        n = min(ref.TOPK, vis)
        head, tail = row[:n], row[n:]
        if any(x != -1 for x in tail):
            problems.append(f"row {i}: slots after the first {n} must be -1: {row}")
        elif len(set(head)) != n or any(not (0 <= x < vis) for x in head):
            problems.append(f"row {i}: first {n} slots must be distinct blocks in [0, {vis}): {row}")
        elif int(pos[i]) // ref.BLOCK not in head:
            problems.append(f"row {i}: local block {int(pos[i]) // ref.BLOCK} not selected: {row}")
        if len(problems) >= 10:
            break
    return problems


def selection_margin(selection_row: list[int], scores_row: torch.Tensor, position: int) -> float | None:
    """``min_S M_b - (max_U M_b - tol)`` for one valid, well-formed row; None when ``U`` is empty."""
    vis = position // ref.BLOCK + 1
    local = position // ref.BLOCK
    chosen = {b for b in selection_row if b >= 0}
    unselected = [b for b in range(vis) if b not in chosen]
    if not unselected:
        return None
    others = [b for b in chosen if b != local]
    max_u = float(scores_row[unselected].max())
    tol = SELECTION_MARGIN * max(1.0, abs(max_u))
    if not others:
        return float("-inf")
    min_s = float(scores_row[others].min())
    return min_s - (max_u - tol)


def check_selection(selection, positions, valid, row_scores) -> dict:
    """Apply the selection policy. ``row_scores(i)`` returns the FP64 block scores of row ``i``."""
    problems = selection_format_problems(selection, positions, valid)
    margins: list[float] = []
    failures: list[dict] = []
    if not problems:
        sel = selection.to("cpu", torch.int64)
        for i in range(sel.shape[0]):
            if not bool(valid[i]):
                continue
            m = selection_margin(sel[i].tolist(), row_scores(i), int(positions[i]))
            if m is None:
                continue
            margins.append(m)
            if m < 0 and len(failures) < 10:
                failures.append({"row": i, "margin": m, "selection": sel[i].tolist()})
    bad = sum(1 for m in margins if m < 0)
    return {
        "format_problems": problems,
        "rows_with_unselected_visible_blocks": len(margins),
        "margin_violations": bad,
        "min_margin": min(margins) if margins else None,
        "failures": failures,
        "pass": not problems and bad == 0,
    }


def check_attention(out: torch.Tensor, r: torch.Tensor, r_bf16: torch.Tensor | None, fp8: bool,
                    valid: torch.Tensor, equal_rows: torch.Tensor, max_abs_v: torch.Tensor) -> dict:
    """Apply the attention policy to ``[Q, 16, 128]`` outputs over the valid rows."""
    o = out.to(torch.float64)[valid]
    rr = r.to(torch.float64)[valid]
    finite = bool(torch.isfinite(o).all())
    rec: dict = {"valid_rows": int(valid.sum()), "finite": finite}
    if not finite:
        rec["pass"] = False
        return rec
    if o.numel() == 0:
        rec.update({"pass": True, "max_abs_err": 0.0})
        return rec
    if fp8:
        target = rr.to(torch.bfloat16).to(torch.float64)
        err = float((o - target).abs().max())
        floor = float((target - rr).abs().max())
        bound = max(FP8_FLOOR, FP8_FACTOR * floor)
        rec.update({"rule": "fp8", "max_abs_err_vs_bf16_reference": err, "bf16_rounding_of_reference": floor,
                    "bound": bound, "case_pass": err <= bound})
    else:
        err = float((o - rr).abs().max())
        env = float((r_bf16.to(torch.float64)[valid] - rr).abs().max())
        bound = 2.0 * env + BF16_ABS
        rec.update({"rule": "bf16", "max_abs_err": err, "bf16_operand_envelope": env, "bound": bound,
                    "case_pass": err <= bound})
    eq = equal_rows[valid]
    if bool(eq.any()):
        lim = EQUAL_LOGIT_FRACTION * max_abs_v.to(torch.float64)[valid][:, None].expand_as(eq)
        dev = (o - rr).abs().amax(dim=-1)
        worst = float((dev - lim)[eq].max())
        rec.update({"equal_logit_row_heads": int(eq.sum()), "equal_logit_worst_excess": worst,
                    "equal_logit_pass": worst <= 0})
    else:
        rec["equal_logit_pass"] = True
    rec["pass"] = bool(rec["case_pass"] and rec["equal_logit_pass"])
    return rec
