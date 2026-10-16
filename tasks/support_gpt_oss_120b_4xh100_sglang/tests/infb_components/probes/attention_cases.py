#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Private attention cases: causal, full and released-window attention with sinks.

`python -m infb_components.probes.attention_cases --output <file> [--binding m:C]`.
Every case is admitted into the operation domain first (exit 4 otherwise); rows
are raw measurements and the private gate applies the published attention policy.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from gpt_oss.attention import AuthorDataFault, admit, case, causal, load_binding, reference
from gpt_oss.numerics import error

AUTHOR_DATA_FAULT = 4
CASES = (([0], [257]), ([127], [8]), ([128, 129, 511], [8, 5, 7]), ([4096, 127], [8, 3]))
WINDOWS = (-1, 127)


def compared(actual, expected):
    try:
        return {"structurally_valid": True, **error(actual, expected)}
    except ValueError as exc:
        return {"structurally_valid": False, "reason": str(exc)}


def measure(binding, seed=314159, device="cuda"):
    generator = torch.Generator(device=device).manual_seed(seed)
    rows = []
    for window in WINDOWS:
        for prefixes, counts in CASES:
            q, keys, values, masks, positions, sinks, target = case(prefixes, counts, window, generator, device)
            admit(q, keys, values, prefixes, counts, sinks, window)
            op = binding(q, keys, values, masks, positions, prefixes, counts, sinks, window)
            for _ in range(3):
                op()
            torch.cuda.synchronize()
            rows.append({"window": window, "prefixes": prefixes, "counts": counts,
                         "mode": "eager", **compared(op(), target)})
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_out = op()
            graph.replay()
            rows.append({"window": window, "prefixes": prefixes, "counts": counts,
                         "mode": "graph", **compared(graph_out, target)})
            # Change input contents in place, then replay the same graph.
            q.neg_()
            admit(q, keys, values, prefixes, counts, sinks, window)
            start = 0
            changed = []
            for k, v, mask, prefix, count in zip(keys, values, masks, prefixes, counts):
                position = positions[start:start+count]
                kp = torch.cat((torch.arange(prefix, device=device), position))
                changed.append(reference(q[start:start+count], k, v, mask, position, kp,
                                         sinks, window))
                start += count
            graph.replay()
            rows.append({"window": window, "prefixes": prefixes, "counts": counts,
                         "mode": "changed-replay", **compared(graph_out, torch.cat(changed))})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", default=314159, type=int)
    parser.add_argument("--binding", help="Equivalent native attention binding as module:class")
    args = parser.parse_args()
    record = {"schema": "gpt-oss-attention-measurement/3", "seed": args.seed}
    try:
        record["rows"] = measure(load_binding(args.binding), seed=args.seed)
    except AuthorDataFault as exc:
        record["author_data_fault"] = str(exc)
        args.output.write_text(json.dumps(record, indent=2) + "\n")
        print(f"[attention_cases] author-data fault: {exc}", flush=True)
        sys.exit(AUTHOR_DATA_FAULT)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"cases": len(record["rows"]),
                      "structurally_invalid": sum(not r["structurally_valid"] for r in record["rows"])}))


if __name__ == "__main__":
    main()
