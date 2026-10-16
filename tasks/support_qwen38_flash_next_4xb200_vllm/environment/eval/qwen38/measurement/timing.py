#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The component timing protocol, shared by development and grading.

For a case, a *draw* restores the logical state the measured work starts from, then executes the
complete measured sequence of native region calls: prefill eagerly (one synchronized call), decode
and speculative sequences as one captured CUDA graph over all of their steps. Warmups, graph capture
and state restoration are outside timing; every measured draw is validated afterwards (outside
timing) and a correctness failure is terminal for the case, never a reason to draw again.

A *matrix* measures every case of a recipe once, in the fixed case order, with `WARMUPS` warmups and
`DRAWS` timed draws per case. A collection runs `MATRICES` complete matrices. Statistic, per case and
matrix: each rank's median synchronized wall time over its draws, then the maximum over ranks. Per
case: the median of its matrix statistics. CUDA-event time is recorded as additional evidence; it does
not replace synchronized wall time. Allocation tracing never runs inside timing draws.
"""
from __future__ import annotations

import math
import statistics
import time

WARMUPS = 3
DRAWS = 9
MATRICES = 3


class TimingError(RuntimeError):
    """A measured draw could not be executed or validated."""


def _barrier():
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def draws(sequence, restore, validate, *, execution: str, warmups: int = WARMUPS, samples: int = DRAWS) -> dict:
    """Time one case once (one matrix entry) on this rank.

    `sequence()` executes the complete measured native calls and returns their outputs;
    `restore()` restores the starting state rows; `validate(outputs)` raises when an output or the
    state it left is wrong (it runs after the draw's clocks stop)."""
    import torch
    if execution not in ("eager", "cuda_graph"):
        raise ValueError("execution is eager (prefill) or cuda_graph (decode/speculative)")
    if type(samples) is not int or samples < 3 or type(warmups) is not int or warmups < 1:
        raise ValueError("at least one warmup and three draws are required")
    for _ in range(warmups):
        restore()
        sequence()
    torch.cuda.synchronize()
    graph = outputs = None
    if execution == "cuda_graph":
        restore()
        torch.cuda.synchronize()
        _barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = sequence()
        torch.cuda.synchronize()
    rows = []
    for index in range(samples):
        restore()
        torch.cuda.synchronize()
        _barrier()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        begun = time.perf_counter_ns()
        if graph is not None:
            graph.replay()
            result = outputs
        else:
            result = sequence()
        end.record()
        torch.cuda.synchronize()
        wall = (time.perf_counter_ns() - begun) / 1e9
        device = start.elapsed_time(end) / 1000
        if not (math.isfinite(wall) and wall > 0 and math.isfinite(device) and device > 0):
            raise TimingError("invalid timing observation")
        check = validate(result)
        rows.append({"draw": index, "elapsed_s": wall, "device_s": device, "validation": check})
    del graph
    return {"execution": execution, "warmups": warmups, "samples": samples, "draws": rows,
            "median_elapsed_s": statistics.median(r["elapsed_s"] for r in rows),
            "median_device_s": statistics.median(r["device_s"] for r in rows)}


def matrix_statistic(ranks: list[dict]) -> float:
    """Per case and matrix: the maximum over ranks of each rank's median synchronized wall time."""
    if len(ranks) != 4 or {r["rank"] for r in ranks} != {0, 1, 2, 3}:
        raise TimingError("a matrix statistic needs one record from each of the four ranks")
    medians = []
    for rank in ranks:
        values = [d["elapsed_s"] for d in rank["draws"]]
        if len(values) != rank["samples"] or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                                                 for v in values):
            raise TimingError(f"rank {rank['rank']}: missing or invalid draws")
        medians.append(statistics.median(values))
    return max(medians)


def case_statistic(matrices: list[list[dict]]) -> dict:
    """Per case: the median of its matrix statistics (every scheduled matrix is used)."""
    if len(matrices) != MATRICES:
        raise TimingError(f"a case statistic needs {MATRICES} complete matrices, got {len(matrices)}")
    values = [matrix_statistic(ranks) for ranks in matrices]
    return {"matrix_statistics_s": values, "statistic_s": statistics.median(values)}


def ratio(reference: dict, candidate: dict) -> float:
    """Reference time over candidate time for one case (higher is faster)."""
    return reference["statistic_s"] / candidate["statistic_s"]
