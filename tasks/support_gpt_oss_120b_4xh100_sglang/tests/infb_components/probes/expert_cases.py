#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Private complete-expert correctness cases on the native TP4 operation.

Run under torchrun as `python -m infb_components.probes.expert_cases`. Each case
is admitted into the operation domain before any candidate call; an inadmissible
case is an author-data fault (exit 4), never a candidate result. Rank records are
raw measurements; the private gates apply the published expert policy.
"""

import argparse
from dataclasses import replace
import json
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

from gpt_oss.expert_math import AuthorDataFault, admit, checkpoint_weights, reference
from gpt_oss.expert_timing import build, load_binding
from gpt_oss.native_experts import capture_graph, initialize
from gpt_oss.numerics import error

from infb_components.probes.weights import diagnostic_weights

AUTHOR_DATA_FAULT = 4


def inputs(count, hidden, experts, family, seed):
    """Correctness routes: the timing families and a concentrated family."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(count, hidden, generator=g, device="cuda", dtype=torch.bfloat16)
    if family == "concentrated":
        ids = torch.tensor([0, 1, experts-2, experts-1], device="cuda").expand(count, -1).contiguous()
    elif family == "balanced":
        ids = (torch.arange(count * 4, device="cuda").reshape(count, 4) + seed) % experts
    elif family == "skewed":
        ids = torch.stack((torch.zeros(count, device="cuda", dtype=torch.int64),
                           1 + torch.arange(count, device="cuda") % (experts-3),
                           torch.full((count,), experts-2, device="cuda", dtype=torch.int64),
                           torch.full((count,), experts-1, device="cuda", dtype=torch.int64)), dim=-1)
    else:
        raise ValueError(family)
    weights = torch.randn(count, 4, generator=g, device="cuda").softmax(-1)
    return x, ids, weights


def compared(actual, expected):
    """A structural failure is recorded as such, never converted into an error figure."""
    try:
        return {"structurally_valid": True, **error(actual, expected)}
    except ValueError as exc:
        return {"structurally_valid": False, "reason": str(exc)}


def measurement(op, x, ids, probabilities, repeats, graph_mode, expert_count):
    for _ in range(3):
        op(x, ids, probabilities)
    torch.cuda.synchronize()
    graph = None
    capture_workspace = 0
    capture_seconds = 0
    if graph_mode:
        torch.cuda.reset_peak_memory_stats()
        capture_baseline = torch.cuda.memory_allocated()
        capture_start = time.monotonic()
        graph = torch.cuda.CUDAGraph()
        with capture_graph(graph):
            result = op(x, ids, probabilities)
        torch.cuda.synchronize()
        capture_workspace = torch.cuda.max_memory_allocated() - capture_baseline
        capture_seconds = time.monotonic() - capture_start
        def call():
            graph.replay()
            return result
    else:
        def call():
            return op(x, ids, probabilities)
    dist.barrier()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        result = call()
    end.record()
    end.synchronize()
    milliseconds = torch.tensor(start.elapsed_time(end) / repeats, device="cuda")
    dist.all_reduce(milliseconds, op=dist.ReduceOp.MAX)
    transient = torch.cuda.max_memory_allocated() - baseline
    # Same captured graph, changed inputs/routes: a stale output must fail.
    before = result.detach().clone()
    x.neg_()
    ids.add_(17).remainder_(expert_count)
    probabilities.copy_(probabilities.roll(1, dims=-1))
    changed = call().detach().clone()
    torch.cuda.synchronize()
    return before, changed, {"slowest_rank_ms": milliseconds.item(),
                             "peak_transient_bytes": transient,
                             "graph_capture_workspace_bytes": capture_workspace,
                             "graph_capture_seconds": capture_seconds,
                             "graph": graph_mode, "repeats": repeats}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=271828)
    p.add_argument("--counts", default="1,7,64,200,257")
    p.add_argument("--families", default="balanced,concentrated,skewed")
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--binding", help="Equivalent native binding as module:class")
    p.add_argument("--disable-custom-all-reduce", action="store_true")
    p.add_argument("--weight-family", default="generated",
                   choices=("generated", "negative_gate", "positive_clamps", "plus_one", "tp_blocks", "bias"))
    p.add_argument("--checkpoint-layer", type=int)
    a = p.parse_args()
    rank = initialize(custom_all_reduce=not a.disable_custom_all_reduce)
    torch.backends.cuda.matmul.allow_tf32 = False
    hidden = intermediate = 2880
    experts = 128
    a.output.mkdir(parents=True, exist_ok=True)
    record_path = a.output / f"rank-{rank}.json"
    weights = (checkpoint_weights("/model", a.checkpoint_layer) if a.checkpoint_layer is not None
               else diagnostic_weights(a.weight_family, hidden, intermediate, experts, a.seed))
    result = {"schema": "gpt-oss-expert-measurement/2", "rank": rank, "seed": a.seed,
              "weight_family": a.weight_family, "checkpoint_layer": a.checkpoint_layer, "rows": []}
    families = a.families.split(",")
    counts = list(map(int, a.counts.split(",")))
    # Admit every case, including its changed-input replay, before the candidate runs.
    try:
        admissions = []
        for family in families:
            for count in counts:
                x, ids, probabilities = inputs(count, hidden, experts, family, a.seed + count)
                admissions.append(admit(x, ids, probabilities, weights))
                changed = (x.neg(), ids.add(17).remainder(experts), probabilities.roll(1, dims=-1))
                admissions.append(admit(*changed, weights))
        result["admissions"] = len(admissions)
    except AuthorDataFault as exc:
        result["author_data_fault"] = str(exc)
        record_path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"[expert_cases] author-data fault: {exc}", flush=True)
        dist.barrier()
        dist.destroy_process_group()
        sys.exit(AUTHOR_DATA_FAULT)
    op, prepared = build(load_binding(a.binding), weights)
    result.update(prepared)
    for family in families:
        for count in counts:
            x, ids, probabilities = inputs(count, hidden, experts, family, a.seed + count)
            expected = reference(x, ids, probabilities, weights) if rank == 0 else None
            for graph_mode in (False, True):
                original_x, original_i, original_p = x.clone(), ids.clone(), probabilities.clone()
                output, changed, timing = measurement(op, x, ids, probabilities, a.repeats, graph_mode, experts)
                row = {"family": family, "count": count, **timing}
                if rank == 0:
                    row["error"] = compared(output, expected)
                    row["changed_error"] = compared(changed, reference(x, ids, probabilities, weights))
                result["rows"].append(row)
                record_path.write_text(json.dumps(result, indent=2) + "\n")
                x.copy_(original_x); ids.copy_(original_i); probabilities.copy_(original_p)
                dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    with torch.inference_mode():
        main()
