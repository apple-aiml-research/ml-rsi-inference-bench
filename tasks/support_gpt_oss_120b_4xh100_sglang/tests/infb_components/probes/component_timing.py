#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Private driver of the public complete-expert timing, as the paired runner invokes it.

`python -m infb_components.probes.component_timing` under torchrun. It measures with
the trusted public primitives (`gpt_oss.expert_timing.build/measure/draw`) and adds
only private observations: the binding witness on the first block, and an
identity-only mode for when timing evidence is incomplete. `--cells` selects extra
cells beyond the declared fourteen; the graded runner never passes it.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist

from gpt_oss.expert_math import checkpoint_weights
from gpt_oss.expert_timing import CELLS, build, load_binding, measure
from gpt_oss.native_experts import initialize
from gpt_oss.timing_inputs import FAMILIES, inputs

from infb_components.probes.witness import binding_witness


def cells(text):
    if not text:
        return CELLS
    result = []
    for item in text.split(','):
        count, mode = item.split(':')
        if mode not in ('graph', 'eager') or not 1 <= int(count) <= 4096:
            raise SystemExit(f'invalid cell {item!r}')
        result.append((int(count), mode == 'graph'))
    return tuple(result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint-layer', type=int, default=0)
    p.add_argument('--seed', type=int, default=271828)
    p.add_argument('--repeats', type=int, default=128)
    p.add_argument('--binding')
    p.add_argument('--profile', action='store_true')
    p.add_argument('--check', action='store_true')
    p.add_argument('--witness', action='store_true')
    p.add_argument('--identity-only', action='store_true')
    p.add_argument('--cells', help='extra cells to measure, e.g. 1:graph,65:eager')
    a = p.parse_args()
    if a.repeats < 8 or a.repeats % 8:
        p.error('repeats must cover complete cycles of eight rotating inputs')
    rank = initialize(custom_all_reduce=True)
    weights = checkpoint_weights('/model', a.checkpoint_layer)
    op, prepared = build(load_binding(a.binding), weights)
    a.output.mkdir(parents=True, exist_ok=True)
    if a.identity_only:
        # One bounded invocation of the same prepared operation: it establishes which
        # native source runs on every rank, and supplies no timing or numerical evidence.
        x, ids, probabilities = inputs(8, 2880, 128, 'balanced', a.seed)
        result = op(x, ids, probabilities)
        torch.cuda.synchronize()
        record = {'schema': 'gpt-oss-expert-identity/1', 'rank': rank, 'seed': a.seed,
                  'checkpoint_layer': a.checkpoint_layer, **prepared,
                  'invocation': {'tokens': 8, 'output_shape': list(result.shape),
                                 'output_dtype': str(result.dtype)}}
        (a.output / f'rank-{rank}.json').write_text(json.dumps(record, indent=2) + '\n')
        dist.barrier()
        dist.destroy_process_group()
        return
    record = {'schema': 'gpt-oss-expert-timing/2', 'rank': rank, 'seed': a.seed,
              'checkpoint_layer': a.checkpoint_layer, **prepared, 'rows': []}
    if a.witness:
        record['binding_witness'] = binding_witness(op, *inputs(8, 2880, 128, 'balanced', a.seed))
    measure(op, weights, record, a.output, seed=a.seed, repeats=a.repeats, cells=cells(a.cells),
            families=FAMILIES, profile=a.profile, check=a.check, custom=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    with torch.inference_mode():
        main()
