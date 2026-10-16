#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Expected outputs for captured native calls, from each request's logical history.

`schedule.run` returns, per runner step, the batch order, query offsets, computed-token counts and
input tokens, and per region one captured call per step. This module walks those steps in order,
keeps each request's committed logical history (rejected speculative rows are dropped), and asks
the comparator for the expected output of every measured and probe call.
"""
from __future__ import annotations

import torch

from . import comparator


class History:
    """One request's committed rows for one region."""

    def __init__(self):
        self.parts = {}
        self.tokens = []

    def extend(self, values: dict, tokens: list[int], count: int):
        for key, value in values.items():
            prior = self.parts.get(key)
            piece = value[:count]
            self.parts[key] = piece if prior is None else torch.cat((prior, piece))
        self.tokens.extend(tokens[:count])

    def with_rows(self, values: dict) -> dict:
        return {k: (v if k not in self.parts else torch.cat((self.parts[k], v))) for k, v in values.items()}


def steps(result: dict, capture) -> list[dict]:
    """The per-step row layout joined with this region's captured call for that step."""
    layout = result["layout"]
    calls = capture.calls
    if len(calls) != len(layout):
        raise AssertionError(f"{len(calls)} captured region calls for {len(layout)} runner steps: "
                             "the declared region did not run exactly once per step")
    out = []
    for index, (row, call) in enumerate(zip(layout, calls)):
        hidden = call["hidden"]
        qsl = row["query_start_loc"]
        if hidden.shape[0] < qsl[-1]:
            raise AssertionError(f"step {index}: region input has {hidden.shape[0]} rows for {qsl[-1]} tokens")
        out.append({"index": index, "phase": row["phase"], "call": call, "req_ids": row["req_ids"],
                    "qsl": qsl, "computed": row["num_computed_tokens"], "input_ids": row["input_ids"]})
    return out


def committed(result: dict, step: dict, req_id: str, rows: int, spec_step: dict) -> int:
    """Rows of this step that stay in the request's history."""
    executed = result["executed"][step["index"]]["requests"].get(req_id)
    if executed is None:
        raise AssertionError(f"step {step['index']}: {req_id} was not scheduled")
    # A joining prefill in a speculative step keeps all its rows; only speculative verification drops rows.
    if rows > 1 and step["phase"] in ("measured", "probe") and result.get("mode") == "spec" \
            and executed.get("speculative", True):
        accepted = result["accepted"][req_id][spec_step[req_id]]
        spec_step[req_id] += 1
        return accepted + 1
    return rows


def expected(result: dict, capture, case: dict, ckpt: comparator.Checkpoint) -> list[dict]:
    """Expected outputs of the measured and probe calls of one region in one case."""
    component = capture.region["component"]
    layer = capture.entry["layer"]
    model = comparator.QSA(ckpt, layer) if component == "qsa" else comparator.PLE(ckpt, layer)
    histories: dict[str, History] = {}
    spec_step: dict[str, int] = {}
    out = []
    for step in steps(result, capture):
        call, qsl = step["call"], step["qsl"]
        hidden = call["hidden"]
        expected_rows = None if step["phase"] == "history" else torch.empty_like(call["output"][:qsl[-1]])
        margins, searches = [], []
        for b, req_id in enumerate(step["req_ids"]):
            a, z = qsl[b], qsl[b + 1]
            history = histories.setdefault(req_id, History())
            spec_step.setdefault(req_id, 0)
            positions = torch.arange(step["computed"][b], step["computed"][b] + (z - a), device=hidden.device)
            if len(history.tokens) != step["computed"][b]:
                raise AssertionError(f"{req_id}: logical history has {len(history.tokens)} rows, the runner "
                                     f"reports {step['computed'][b]} computed tokens")
            tokens = step["input_ids"][a:z]
            if component == "qsa":
                rows = model.rows(hidden[a:z], positions)
                if step["phase"] != "history":
                    context = history.with_rows(rows)
                    y, m = model.outputs(hidden[a:z], positions, context)
                    margins.extend(m)
                    # Rows whose own-selection output disagrees: search the admissible selections.
                    native = call["output"][a:z].float()
                    bound = comparator.QSA_ALLOWANCE["atol"] + comparator.QSA_ALLOWANCE["rtol"] * y.float().abs()
                    for j in ((native - y.float()).abs() > bound).any(-1).nonzero().flatten().tolist():
                        found, info = model.admissible(hidden[a + j:a + j + 1], int(positions[j]), context,
                                                       native[j])
                        searches.append({"request": req_id, "row": a + j, **info, "agreed": found is not None})
                        if found is not None:
                            y[j] = found.to(y.dtype)
                    expected_rows[a:z] = y.to(expected_rows.dtype)
            else:
                context = ([model.eos] * model.context_len + history.tokens)[-model.context_len:]
                gated, conv_in = model.rows(hidden[a:z], context, tokens)
                rows = {"conv": conv_in}
                if step["phase"] != "history":
                    prior = history.parts.get("conv")
                    width = model.width
                    if prior is None:
                        prior = torch.zeros(width, conv_in.shape[1], dtype=conv_in.dtype, device=conv_in.device)
                    elif prior.shape[0] < width:
                        prior = torch.cat((torch.zeros(width - prior.shape[0], conv_in.shape[1], dtype=prior.dtype,
                                                       device=prior.device), prior))
                    y = model.outputs(gated, conv_in, prior[-width:])
                    if "ple_residual" in capture.region["encloses"]:
                        y = hidden[a:z] + y.to(hidden.dtype)
                    expected_rows[a:z] = y.to(expected_rows.dtype)
            keep = committed(result, step, req_id, z - a, spec_step)
            history.extend(rows, tokens, keep)
            if component == "ple" and history.parts["conv"].shape[0] > model.width:
                history.parts["conv"] = history.parts["conv"][-model.width:]
        if step["phase"] != "history":
            out.append({"index": step["index"], "phase": step["phase"], "expected": expected_rows,
                        "selection_searches": searches,
                        "selection_rows": len(margins),
                        "rows_within_score_atol": sum(1 for x in margins if x <= 0.05),
                        "min_margin": min(margins) if margins else None})
    return out


def allowance(component: str) -> dict:
    return comparator.QSA_ALLOWANCE if component == "qsa" else comparator.PLE_ALLOWANCE
