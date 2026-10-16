#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The `attend` timing protocol (the timing rule of README.md), public and identical for both arms."""
from __future__ import annotations

import statistics
import time

import torch

from .check import Inputs

B = 8192
CONTEXTS = {"encode": (4096, 65536, 262144 - B), "denoise": (4096, 65536, 261888), "mixed": (4096, 65536, 261888)}
LAYERS = {"sliding": 25, "full": 5}


def case(family: str, L: int, layer_kind: str, seed: int = 7) -> dict:
    if family == "encode":
        seqs = [{"kind": "encode", "start": L, "m": B, "spans": []}]
    elif family == "denoise":
        seqs = [{"kind": "denoise", "c": L, "m": 256} for _ in range(4)]
    else:
        seqs = [{"kind": "encode", "start": L, "m": 256, "spans": []}] + [{"kind": "denoise", "c": L, "m": 256} for _ in range(3)]
    return {"id": f"time/{family}/{L}/{layer_kind}", "layer_kind": layer_kind, "tp": 1, "values": "rand", "seed": seed,
            "sequences": seqs}


def time_calls(x: Inputs, n: int) -> tuple[float, float]:
    """(per-call GPU seconds, per-call wall seconds) over n back-to-back calls."""
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    start.record()
    for _ in range(n):
        x.call()
    end.record()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) / n
    return start.elapsed_time(end) / 1000.0 / n, wall


def measure(impl, family: str, L: int, layer_kind: str, n: int | None = None, device=None) -> dict:
    device = device or torch.device("cuda")
    x = Inputs(case(family, L, layer_kind), impl, device)
    for _ in range(10):
        x.call()
    torch.cuda.synchronize()
    if n is None:  # size N so a draw lasts at least 100 ms
        g, _ = time_calls(x, 3)
        n = max(1, int(0.1 / max(g, 1e-6)) + 1)
    draws = [time_calls(x, n) for _ in range(3)]
    return {"family": family, "L": L, "layer_kind": layer_kind, "n": n,
            "gpu_s": statistics.median(d[0] for d in draws), "wall_s": statistics.median(d[1] for d in draws),
            "draws": draws}


def collection(impl, ns: dict | None = None) -> dict:
    """Every workload of the three families, both layer kinds; the combined statistic per context."""
    out: dict = {}
    for family, contexts in CONTEXTS.items():
        for L in contexts:
            per = {lk: measure(impl, family, L, lk, (ns or {}).get(f"{family}/{L}/{lk}")) for lk in LAYERS}
            out[f"{family}/{L}"] = {"per_layer": per,
                                    "gpu_s": sum(LAYERS[lk] * per[lk]["gpu_s"] for lk in LAYERS),
                                    "wall_s": sum(LAYERS[lk] * per[lk]["wall_s"] for lk in LAYERS)}
            torch.cuda.empty_cache()
    return out
