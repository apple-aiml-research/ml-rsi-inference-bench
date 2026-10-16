#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""A small checkpoint-backed expert example: the native TP4 result against the mathematics.

Run with four GPUs:
    torchrun --standalone --nproc-per-node=4 -m gpt_oss.expert_probe --output /tmp/expert-example

It loads one released layer through the bound native implementation, routes eight
BF16 tokens with that layer's own router, and compares the complete TP4 result in
eager and CUDA-graph execution with `expert_math.reference` under the published
expert policy (`numerical-policy.json`). The record states what this sample shows;
it is not a task acceptance verdict.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist

from .expert_math import admit, example, reference
from .expert_timing import load_binding, memory
from .native_experts import capture_graph, implementation_identity, initialize, live_storages
from .numerics import assess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint-layer", type=int, default=0)
    p.add_argument("--tokens", type=int, default=8)
    p.add_argument("--seed", type=int, default=271828)
    p.add_argument("--binding", help="Equivalent native binding as module:class")
    a = p.parse_args()
    rank = initialize()
    torch.backends.cuda.matmul.allow_tf32 = False
    weights, x, ids, probabilities = example(layer=a.checkpoint_layer, tokens=a.tokens, seed=a.seed)
    admission = admit(x, ids, probabilities, weights)
    x, ids, probabilities = x.cuda(), ids.cuda(), probabilities.cuda()
    before = memory()
    op = load_binding(a.binding)(weights)
    after = memory()
    record = {"schema": "gpt-oss-expert-example/1", "rank": rank, "checkpoint_layer": a.checkpoint_layer,
              "tokens": a.tokens, "seed": a.seed, "admission": admission,
              "identity": implementation_identity(op.module), "persistent": live_storages(op.module),
              "allocator_live_delta": after["allocated"] - before["allocated"], "rows": []}
    expected = reference(x, ids, probabilities, weights) if rank == 0 else None
    for mode in ("eager", "graph"):
        for _ in range(3):
            op(x, ids, probabilities)
        torch.cuda.synchronize()
        if mode == "graph":
            graph = torch.cuda.CUDAGraph()
            with capture_graph(graph):
                result = op(x, ids, probabilities)
            graph.replay()
        else:
            result = op(x, ids, probabilities)
        torch.cuda.synchronize()
        if rank == 0:
            record["rows"].append({"mode": mode, **assess(result.detach().clone(), expected, "expert")})
        dist.barrier()
    if rank == 0:
        record["sample_correctness"] = ("pass" if all(r["within_policy"] for r in record["rows"])
                                        else "fail")
        record["task_acceptance"] = "not_determined"
        a.output.mkdir(parents=True, exist_ok=True)
        (a.output / "example.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({k: record[k] for k in ("sample_correctness", "task_acceptance")} |
                         {"rows": [{k: r.get(k) for k in ("mode", "max", "rms", "within_policy")}
                                   for r in record["rows"]]}))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    with torch.inference_mode():
        main()
