#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Complete native TP4 expert timings with rotating inputs and expert working sets.

Each invocation is one draw. The private paired runner fixes five alternating
reference/candidate draws; this command contains no speed acceptance threshold.
Compilation and graph capture are reported separately from steady-state events.
Resident storage and workspace are reported with their baselines, so the
accounting in `component-contract.md` can be recomputed from the record.
"""
import argparse
from collections import Counter
import gc
import importlib
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist

from .expert_math import checkpoint_weights, reference
from .native_experts import (NativeExperts, capture_graph, collective, implementation_identity,
                             initialize, live_storages)
from .numerics import assess
from .timing_inputs import FAMILIES, inputs

CELLS = ((1, True), (8, True), (64, True), (192, True),
         (256, False), (1024, False), (4096, False))


def memory():
    """A synchronized reading: allocator live and reserved bytes, and device bytes in use."""
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {'allocated': torch.cuda.memory_allocated(), 'reserved': torch.cuda.memory_reserved(),
            'device_used': total - free}


def draw(op, count, family, graph_mode, seed, repeats, profile, check_weights=None):
    slots = [inputs(count, 2880, 128, family, seed+17*i) for i in range(8)]
    for i, (_, ids, _) in enumerate(slots):
        ids.add_(17*i).remainder_(128)
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    before = memory()
    baseline = before['allocated']
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    for slot in slots:
        op(*slot)
    torch.cuda.synchronize()
    compile_seconds = time.monotonic()-start
    graphs, outputs = [], []
    first_graph = None
    start = time.monotonic()
    if graph_mode:
        for slot in slots:
            graph = torch.cuda.CUDAGraph()
            with capture_graph(graph):
                output = op(*slot)
            graphs.append(graph)
            outputs.append(output)
            if first_graph is None:
                torch.cuda.synchronize()
                reading = memory()
                first_graph = {'setup_peak': torch.cuda.max_memory_allocated()-baseline,
                               'retained': reading['allocated']-baseline,
                               'external': ((reading['device_used']-before['device_used'])
                                            - (reading['reserved']-before['reserved'])),
                               'memory': reading}
    torch.cuda.synchronize()
    capture_seconds = time.monotonic()-start
    setup_peak = torch.cuda.max_memory_allocated()-baseline
    def call(i):
        if graph_mode:
            graphs[i % len(graphs)].replay()
            return outputs[i % len(outputs)]
        return op(*slots[i % len(slots)])
    for i in range(8):
        call(i)
    torch.cuda.synchronize()
    dist.barrier()
    steady = memory()
    steady_baseline = steady['allocated']
    torch.cuda.reset_peak_memory_stats()
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start_event.record()
    for i in range(repeats):
        output = call(i)
    end_event.record()
    end_event.synchronize()
    milliseconds = torch.tensor(start_event.elapsed_time(end_event)/repeats, device='cuda')
    dist.all_reduce(milliseconds, op=dist.ReduceOp.MAX)
    steady_peak = torch.cuda.max_memory_allocated()-steady_baseline
    after = memory()
    record = {'count': count, 'family': family, 'graph': graph_mode, 'repeats': repeats,
              'rotating_slots': len(slots), 'slowest_rank_ms': float(milliseconds.item()),
              'compile_warmup_seconds': compile_seconds, 'capture_seconds': capture_seconds,
              'setup_peak_workspace_bytes': first_graph['setup_peak'] if graph_mode else setup_peak,
              'retained_workspace_bytes': first_graph['retained'] if graph_mode else steady_baseline-baseline,
              'steady_peak_workspace_bytes': steady_peak,
              'rotation': {'graphs': len(graphs), 'setup_peak_bytes': setup_peak,
                           'retained_bytes': steady_baseline-baseline,
                           'external_bytes': ((after['device_used']-before['device_used'])
                                              - (after['reserved']-before['reserved']))},
              'external_retained_bytes': first_graph['external'] if graph_mode else (
                  (after['device_used']-before['device_used']) - (after['reserved']-before['reserved'])),
              'memory': {'before': before, 'steady': steady, 'after': after,
                         **({'first_graph': first_graph['memory']} if graph_mode else {})},
              'device_live_workspace_bytes': after['device_used']-before['device_used']}
    if profile is not None:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as trace:
            call(0)
            torch.cuda.synchronize()
        trace.export_chrome_trace(str(profile))
        record['kernels'] = dict(Counter(event.name for event in trace.events()
                                if event.device_type == torch.autograd.DeviceType.CUDA))
        record['trace'] = str(profile)
    if check_weights is not None:
        record['checked_slots'] = []
        for index in (0, 7):
            actual = call(index).detach().clone()
            torch.cuda.synchronize()
            if dist.get_rank() == 0:
                expected = reference(*slots[index], check_weights)
                record['checked_slots'].append({'slot': index,
                                                'error': assess(actual, expected, 'expert')})
            dist.barrier()
    return record


def load_binding(name=None):
    if not name:
        return NativeExperts
    module, attribute = name.split(':')
    return getattr(importlib.import_module(module), attribute)


def prepare(binding, layer, model_path='/model'):
    """Build the bound native operation on one checkpoint layer, with its resident accounting."""
    weights = checkpoint_weights(model_path, layer)
    op, record = build(binding, weights)
    return op, weights, record


def build(binding, weights):
    """Construct `binding(weights)` and record its resident storage with the baselines:
    reachable tensor ranges, allocator live/reserved deltas and device use."""
    before = memory()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.monotonic()
    op = binding(weights)
    gc.collect()
    after = memory()
    persistent = live_storages(op.module)
    record = {'identity': implementation_identity(op.module), 'persistent': persistent,
              'preparation_seconds': time.monotonic()-t0,
              'allocator_live_delta': after['allocated']-before['allocated'],
              'allocator_reserved_delta': after['reserved']-before['reserved'],
              'device_used_delta': after['device_used']-before['device_used'],
              'preparation_peak_bytes': torch.cuda.max_memory_allocated()-before['allocated'],
              'memory': {'before': before, 'after': after}}
    return op, record


def measure(op, weights, record, output, *, seed, repeats, cells=CELLS, families=FAMILIES,
            profile=False, check=False, custom=True):
    """Every requested cell, rank record rewritten after each so a partial draw is visible."""
    rank = dist.get_rank()
    output.mkdir(parents=True, exist_ok=True)
    for count, graph_mode in cells:
        for family in families:
            trace = output / f'rank-{rank}-{count}-{family}.trace.json' if profile else None
            with collective(custom=custom):
                row = draw(op, count, family, graph_mode, seed+count, repeats, trace,
                           weights if check else None)
            row['serving_mode'] = 'DECODE' if graph_mode else 'EXTEND'
            row['custom_all_reduce'] = custom
            record['rows'].append(row)
            (output / f'rank-{rank}.json').write_text(json.dumps(record, indent=2)+'\n')
            gc.collect()
            torch.cuda.synchronize()
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint-layer', type=int, default=0)
    p.add_argument('--seed', type=int, default=271828)
    p.add_argument('--repeats', type=int, default=128)
    p.add_argument('--counts', help='Select a subset of the declared counts')
    p.add_argument('--binding', help='Equivalent native binding as module:class')
    p.add_argument('--disable-custom-all-reduce', action='store_true')
    p.add_argument('--profile', action='store_true', help='Untimed CUDA execution traces after each draw')
    p.add_argument('--check', action='store_true',
                   help='Compare two rotating slots per cell with the independent FP32 mathematics')
    a = p.parse_args()
    if a.repeats < 8 or a.repeats % 8:
        p.error('repeats must cover complete cycles of eight rotating inputs')
    rank = initialize(custom_all_reduce=not a.disable_custom_all_reduce)
    op, weights, prepared = prepare(load_binding(a.binding), a.checkpoint_layer)
    record = {'schema': 'gpt-oss-expert-timing/2', 'rank': rank, 'seed': a.seed,
              'checkpoint_layer': a.checkpoint_layer, **prepared, 'rows': []}
    counts = set(map(int, a.counts.split(','))) if a.counts else None
    cells = [cell for cell in CELLS if counts is None or cell[0] in counts]
    measure(op, weights, record, a.output, seed=a.seed, repeats=a.repeats, cells=cells,
            profile=a.profile, check=a.check, custom=not a.disable_custom_all_reduce)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    with torch.inference_mode():
        main()
