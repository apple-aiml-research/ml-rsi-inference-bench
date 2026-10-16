#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Prefix branches, public."""
from __future__ import annotations

from ._pairs import cut, pair, prefix_hits, read


def branches(srv, cases: list[dict], block_size: int = 16, T: float | None = None) -> dict:
    rows, ok = [], True
    for c in cases:
        srv.reset_prefix_cache()
        order = [("A", c["A"]), ("B", c["B"]), ("A", c["A"]), ("B", c["B"])]
        warm, hits = [], []
        for name, suf in order:
            h0 = prefix_hits(srv)
            warm.append(read(srv, c["prefix"] + suf, c[f"seed_{name}"], c[f"targets_{name}"]))
            hits.append(prefix_hits(srv) - h0)
        need = block_size * (len(c["prefix"]) // block_size) - 1280
        reuse_ok = all(h >= need for h in hits[1:])
        srv.reset_prefix_cache()
        stats = []
        for k, (name, suf) in enumerate(order):
            cold = read(srv, c["prefix"] + suf, c[f"seed_{name}"], c[f"targets_{name}"])
            stats.append(pair(warm[k], cold, cut(c[f"targets_{name}"])))
        good = reuse_ok and all(s < float("inf") for s in stats) and (T is None or all(s <= T for s in stats))
        rows.append({"id": c["id"], "hits": hits, "need": need, "k3": stats, "pass": good})
        ok &= good
    return {"rows": rows, "T": T, "pass": bool(ok)}
