#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The numerical policies of the `attend` contract (the correctness rules of README.md)."""
from __future__ import annotations

import torch

from .reference import _gather, visible_mask

EQUAL_TOL = 2.0 ** -7


def accuracy(o: torch.Tensor, r: torch.Tensor, o_bf16: torch.Tensor) -> dict:
    """The any-case rule over one call's valid rows."""
    err = float((o.double() - r.double()).abs().max()) if o.numel() else 0.0
    bound = 2.0 * float((o_bf16.double() - r.double()).abs().max()) + 1e-6 if o.numel() else 1e-6
    finite = bool(torch.isfinite(o).all())
    return {"max_err": err, "bound": bound, "finite": finite, "pass": finite and err <= bound}


def equal_logits(o, k_cache, v_cache, q, block_table, seq_lens, query_start_loc, encode, image_ranges,
                 layer_kind: str, reading: str = "w1") -> dict:
    """The equal-logits rule for rows whose q is identically zero."""
    worst_ratio, rows, ok = 0.0, 0, True
    hk = v_cache.shape[2]
    hq = q.shape[1]
    group = hq // hk
    qsl = query_start_loc.tolist()
    for r in range(len(qsl) - 1):
        a, b = qsl[r], qsl[r + 1]
        m, n = b - a, int(seq_lens[r])
        if m == 0:
            continue
        kind = "encode" if int(encode[r]) else "denoise"
        spans = [(int(s), int(e)) for s, e in image_ranges[r].tolist() if e > s] if image_ranges is not None else []
        mask = visible_mask(kind, n, m, spans, layer_kind, reading, device=q.device)
        v = _gather(v_cache, block_table[r], n).double()
        zero_rows = (q[a:b].abs().amax(dim=(1, 2)) == 0).nonzero().flatten().tolist()
        for i in zero_rows:
            vis = mask[i]
            for h in range(hq):
                vv = v[vis, h // group]
                mean = vv.mean(dim=0)
                tol = EQUAL_TOL * float(vv.abs().max())
                dev = float((o[a + i, h].double() - mean).abs().max())
                worst_ratio = max(worst_ratio, dev / tol if tol > 0 else (0.0 if dev == 0 else float("inf")))
                ok &= dev <= tol
            rows += 1
    return {"rows": rows, "worst_ratio": worst_ratio, "pass": ok}
