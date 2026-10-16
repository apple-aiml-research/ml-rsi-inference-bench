#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The mask witness (the correctness rules of README.md): exact probes of visibility.

A probe makes one tested key dominate one row. Probe j owns one (row, kv head) and a probe axis a
of that head: the row's query is 5 * u_a (on its head's first query head) and the tested key
carries 4 * u_a, a logit of 20; every other key has zeros on the probe axes. Several rows may test
one key on one head (the position tables test an image span's first position from three rows):
the key then carries each of their axes, and its value is one marker (entries +-2, distinct per
tested key and head). If the key is visible, the row's output equals the marker within 1e-3
relative (its softmax weight is at least 1 - 262,144 * e^-20); if it is invisible, the output's
norm is below 0.1 of the marker's.

The contract admits two readings of the canvas window on sliding layers, used throughout: W - 1
shared by the canvas ("w1") or W ("w"), W being the model's sliding window (`reference.py`).
`expected` gives each reading's visibility of a probe.
"""
from __future__ import annotations

import torch

from .reference import visible_mask

QUERY_SCALE, KEY_SCALE = 5.0, 4.0
VISIBLE_REL, INVISIBLE_FRAC = 1e-3, 0.1


def place(probes: list[tuple[int, int, int]], num_kv_heads: int) -> list[tuple[int, int]]:
    """(kv head, axis) per probe: no two probes share a (sequence, row, head); a key may be shared."""
    rows_used, per_head, out = set(), [0] * num_kv_heads, []
    for j, (r, i, p) in enumerate(probes):
        for dh in range(num_kv_heads):
            h = (j + dh) % num_kv_heads
            if (r, i, h) not in rows_used:
                break
        else:
            raise ValueError(f"probe {(r, i, p)} cannot be placed: every kv head already uses its row")
        rows_used.add((r, i, h))
        out.append((h, per_head[h]))
        per_head[h] += 1
        if per_head[h] > 64:
            raise ValueError("more than 64 probes on one kv head")
    return out


def observe(o_row: torch.Tensor, marker: torch.Tensor) -> tuple[str, float]:
    mk = marker.double()
    got = o_row.double()
    rel = float((got - mk).norm() / mk.norm())
    if rel <= VISIBLE_REL:
        return "visible", rel
    if float(got.norm()) < INVISIBLE_FRAC * float(mk.norm()):
        return "invisible", rel
    return "ambiguous", rel


def expected(kind: str, n: int, m: int, spans, layer_kind: str, row: int, key: int, reading: str) -> str:
    mask = visible_mask(kind, n, m, spans, layer_kind, reading)
    return "visible" if bool(mask[row, key]) else "invisible"


def verdict(table: list[dict]) -> dict:
    """Pass iff every probe outcome fits W - 1 throughout, or W throughout."""
    fits = [rd for rd in ("w1", "w") if all(t["observed"] == t["expected"][rd] for t in table)]
    return {"probes": len(table), "fits": fits, "ambiguous": sum(t["observed"] == "ambiguous" for t in table),
            "pass": bool(fits)}
