#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""The private native-metadata schedules (specification.json `metadata`), on the supplied primitives.

Every call takes its attention metadata from the loaded runner's own builder for the layer (the supplied
`layer_adapter.native_metadata`) and is compared with the independent mathematics; every state slot a call
does not own must stay bit-identical. Families:

* mixed validity: fresh and continued rows together, nonadjacent slots holding finite nonzero stale state;
* slot turnover: valid history in slots that are then released and handed to fresh requests in another
  order, beside a still-live continuation; only the reused owners reset;
* branch copies: a prefix state copied into two nonadjacent slots, different suffixes, each branch revisited
  after the other advanced; the source checkpoint stays unchanged;
* replay tail: CUDA graphs of one layer call captured on the builder's capture entry at the fixed extents and
  replayed for changing live populations after the builder stages each step, as the runner replays a full
  decode graph (every builder staged at every step, the zero-live step on the null block as the runner's dummy
  forward); the graph extent may exceed the builder's own padded extent for the population;
* checkpoint splits: a sequence continued across segments, its state copied to a new slot at every segment
  boundary and continued from there, against the continuous call.

Mixed validity, turnover and branch run on every loaded layer; replay populations and splits on a cyclic
layer counter (seed-major), together covering every layer. Each worker writes its record to
`<out>/rank-dp<d>-tp<t>-<device uuid>.json` and returns it; under data parallelism the RPC's return carries
only the first engine's workers, so the files are the coverage evidence.
"""

import json
import os
from pathlib import Path
import random
import socket

import torch

import layer_adapter as adapter
from mathematics import error, layer_reference
from schedules import SPEC, capture_extent, schedule

SCHEMA = 'infb-kda-native-metadata/1'


def _turnover_calls(spec):
    first = spec['first']
    calls = [{'lengths': first['lengths'], 'computed': first['computed'], 'slots': first['slots']}]
    second = spec['second']
    calls.append({'lengths': second['lengths'], 'computed': second['computed'], 'slots': second['slots']})
    calls.append({'lengths': second['lengths'],
                  'computed': [c + n for c, n in zip(second['computed'], second['lengths'])],
                  'slots': second['slots']})
    return calls


def _branch_calls(spec):
    prefix, (a, b) = spec['prefix'], spec['branches']
    return [
        {'lengths': [prefix['length']], 'computed': [0], 'slots': [prefix['slot']]},
        {'lengths': [a['suffix']], 'computed': [prefix['length']], 'slots': [a['slot']],
         'copies': [[prefix['slot'], a['slot']], [prefix['slot'], b['slot']]]},
        {'lengths': [b['suffix']], 'computed': [prefix['length']], 'slots': [b['slot']]},
        {'lengths': [1], 'computed': [prefix['length'] + a['suffix']], 'slots': [a['slot']]},
        {'lengths': [1], 'computed': [prefix['length'] + b['suffix']], 'slots': [b['slot']]},
    ]


def checkpoint_split(supplied, runner, layer, segments, config, seed):
    """One sequence (valid history in slot 1) as one continuous call, then as its segments with the state copied
    to a new slot at each boundary and continued there; every segment and the final state against the
    mathematics, the split against the continuous call, and each earlier checkpoint left unchanged."""
    history = SPEC['history_tokens']
    slots = [1 + 2 * i for i in range(len(segments))]
    region = max(slots) + 4
    builder = adapter.metadata_builder(runner, layer)[0]
    snapshot = adapter.BuilderSnapshot(builder)
    saved = [state[:region].cpu() for state in layer.states]
    gen = torch.Generator(device='cuda').manual_seed(seed + 1)
    hidden = torch.randn((sum(segments), layer.weights['q_proj'].shape[1]), generator=gen,
                         device='cuda', dtype=torch.bfloat16) * 0.3
    try:
        supplied._seed_states(layer, region, seed)
        initial = [s[:region].clone() for s in layer.states]
        whole = supplied.Call(runner, layer, [sum(segments)], [history], [slots[0]])
        want, expected, _ = layer_reference(hidden, layer.weights, initial, whole.starts, whole.indices,
                                            whole.valid, eps=layer.eps)
        want = supplied._tp_sum(want)
        continuous = torch.empty((whole.extent, want.shape[1]), device='cuda', dtype=want.dtype)
        layer.invoke(whole.padded(hidden), continuous, whole.metadata, config, whole.positions)
        continuous = continuous[:whole.live]
        continuous_state = [s[slots[0]].clone() for s in layer.states]
        for state, value in zip(layer.states, initial):
            state[:region].copy_(value)
        split, offset, records = torch.empty_like(want), 0, []
        reference = [s.clone() for s in initial]
        for k, length in enumerate(segments):
            slot = slots[k]
            if k:
                for state, ref in zip(layer.states, reference):
                    state[slot].copy_(state[slots[k - 1]])
                    ref[slot].copy_(ref[slots[k - 1]])
            call = supplied.Call(runner, layer, [length], [history + offset], [slot])
            part_want, part_state, _ = layer_reference(hidden[offset:offset + length], layer.weights, reference,
                                                       call.starts, call.indices, call.valid, eps=layer.eps)
            part_want = supplied._tp_sum(part_want)
            before = [s[:region].clone() for s in layer.states]
            piece = torch.empty((call.extent, want.shape[1]), device='cuda', dtype=want.dtype)
            layer.invoke(call.padded(hidden[offset:offset + length]), piece, call.metadata, config, call.positions)
            torch.cuda.synchronize()
            split[offset:offset + length] = piece[:length]
            others = torch.ones(region, dtype=torch.bool, device='cuda')
            others[slot] = False
            records.append({'segment': k, 'length': length, 'computed': history + offset, 'slot': slot,
                            'extent': call.extent, 'output': error(piece[:length], part_want),
                            'states': [error(s[slot], r[slot]) for s, r in zip(layer.states, part_state)],
                            'untouched_exact': all(torch.equal(s[:region][others], b[others])
                                                   for s, b in zip(layer.states, before))})
            reference = part_state
            offset += length
        last = slots[-1]
        return {'segments': segments, 'history_tokens': history, 'slots': slots, 'segment_calls': records,
                'continuous_output': error(continuous, want), 'split_output': error(split, want),
                'split_vs_continuous': error(split, continuous),
                'continuous_states': [error(c, e[slots[0]]) for c, e in zip(continuous_state, expected)],
                'final_states': [error(s[last], e[slots[0]]) for s, e in zip(layer.states, expected)],
                'state_agreement': [error(s[last], c) for s, c in zip(layer.states, continuous_state)]}
    finally:
        for s, original in zip(layer.states, saved):
            s[:region].copy_(original)
        snapshot.restore()


class Replay:
    """The replay-tail family: captured graphs of single loaded-layer calls, replayed as the runner replays."""

    def __init__(self, supplied, runner, layers, config):
        self.supplied, self.runner, self.layers, self.config = supplied, runner, layers, config
        self.builders = {}
        for layer in layers:
            builder = adapter.metadata_builder(runner, layer)[0]
            self.builders.setdefault(id(builder), (builder, layer))
        self.graphs, self.events = {}, []
        self.pool = torch.cuda.graph_pool_handle()

    def storage(self):
        return {f'group{i}': {name: value[0] for name, value in adapter.builder_identity(builder)['buffers'].items()}
                for i, (builder, _) in enumerate(self.builders.values())}

    def capture(self, layer_index, extent):
        from vllm.distributed.parallel_state import graph_capture
        layer = self.layers[layer_index]
        width_in = layer.weights['q_proj'].shape[1]
        static = {'hidden': torch.zeros((extent, width_in), device='cuda', dtype=torch.bfloat16),
                  'output': torch.zeros((extent, layer.weights['out_proj'].shape[0]), device='cuda',
                                        dtype=torch.bfloat16),
                  'positions': torch.zeros(extent, device='cuda', dtype=torch.int64)}
        metadata, built = adapter.native_metadata(self.runner, layer, [1] * extent, [0] * extent, [0] * extent,
                                                  capture=True)
        if built != extent:
            raise AssertionError(f'the builder captured {built} rows for a {extent}-row graph')
        # As the runner: one eager warm-up, then the capture; dummy rows use only the null block (slot 0).
        layer.invoke(static['hidden'], static['output'], metadata, self.config, static['positions'])
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with graph_capture(device=static['hidden'].device) as context:
            with torch.cuda.graph(graph, pool=self.pool, stream=context.stream):
                layer.invoke(static['hidden'], static['output'], metadata, self.config, static['positions'])
        torch.cuda.synchronize()
        self.graphs[(layer_index, extent)] = {'graph': graph, **static}
        self.events.append({'event': 'capture', 'layer_index': layer_index, 'extent': extent,
                            'builder_storage': self.storage(),
                            'static_storage': {k: v.data_ptr() for k, v in static.items()}})

    def stage(self, computed, slots, dummy_rows=None):
        """Build every builder's metadata for this step, as the runner does; returns {builder id: extent}."""
        extents = {}
        for key, (builder, layer) in self.builders.items():
            if dummy_rows is not None:
                _, extent = adapter.native_metadata(self.runner, layer, [1] * dummy_rows, [0] * dummy_rows,
                                                    [0] * dummy_rows, capture=True)
            else:
                _, extent = adapter.native_metadata(self.runner, layer, [1] * len(slots), computed, slots)
            extents[key] = extent
        return extents

    def population(self, layer_index, population, rng, gen, reference, region):
        layer = self.layers[layer_index]
        extent = capture_extent(population)
        static = self.graphs[(layer_index, extent)]
        slots = rng.sample(range(1, region - 3), population)
        computed = [0 if rng.random() < SPEC['replay']['fresh_fraction'] else rng.randint(1, 4096)
                    for _ in range(population)]
        dummy = population == 0
        extents = self.stage(computed, slots, dummy_rows=extent if dummy else None)
        builder_extent = extents[id(adapter.metadata_builder(self.runner, layer)[0])]
        width_in = static['hidden'].shape[1]
        static['hidden'].copy_(torch.randn((extent, width_in), generator=gen, device='cuda',
                                           dtype=torch.bfloat16) * 0.3)
        static['positions'].zero_()
        if population:
            static['positions'][:population] = torch.tensor(computed, device='cuda')
        static['output'].fill_(float('nan'))
        before = [s[:region].clone() for s in layer.states]
        owned = torch.tensor(slots if not dummy else [0], dtype=torch.int64, device='cuda')
        record = {'event': 'replay', 'layer_index': layer_index, 'population': population, 'graph_extent': extent,
                  'builder_extent': builder_extent, 'dummy': dummy, 'slots': slots, 'computed': computed,
                  'builder_storage': self.storage()}
        if population:
            valid = torch.tensor([c > 0 for c in computed], dtype=torch.bool, device='cuda')
            starts = list(range(population + 1))
            want, expected, _ = layer_reference(static['hidden'][:population], layer.weights, reference,
                                                starts, owned, valid, eps=layer.eps)
            want = self.supplied._tp_sum(want)
        static['graph'].replay()
        torch.cuda.synchronize()
        untouched = torch.ones(region, dtype=torch.bool, device='cuda')
        untouched[owned] = False
        record['untouched_exact'] = all(torch.equal(s[:region][untouched], b[untouched])
                                        for s, b in zip(layer.states, before))
        if population:
            record['output'] = error(static['output'][:population], want)
            record['states'] = [error(s[owned], e[owned]) for s, e in zip(layer.states, expected)]
            reference = expected
        else:
            # The null block's content after a dummy forward is no request's state; the mathematics keeps
            # whatever the cache holds there.
            for ref, s in zip(reference, layer.states):
                ref[0].copy_(s[0])
        self.events.append(record)
        return record, reference


def _identity(runner):
    from vllm.distributed.parallel_state import get_tp_group
    parallel = runner.vllm_config.parallel_config
    uuid = getattr(torch.cuda.get_device_properties(runner.device), 'uuid', None)
    return {'dp_rank': parallel.data_parallel_rank, 'dp_size': parallel.data_parallel_size,
            'tp_rank': get_tp_group().rank_in_group, 'tp_size': get_tp_group().world_size,
            'device_uuid': f'GPU-{uuid}' if uuid is not None else None, 'host': socket.gethostname(),
            'pid': os.getpid()}


def run(worker, seed_unused, supplied, families='all', out=None):
    """Run the schedule on this worker; `families` is 'all' or a comma list (e.g. 'replay' on DP replicas)."""
    runner, config = worker.model_runner, worker.vllm_config
    layers = adapter.discover(runner.model, supplied._root(runner.model))
    wanted = None if families == 'all' else set(families.split(','))
    record = {'schema': SCHEMA, 'identity': _identity(runner), 'rank': worker.rank, 'families': families,
              'layers': [layer.identity() for layer in layers],
              'builders': [adapter.builder_identity(adapter.metadata_builder(runner, layer)[0]) for layer in layers],
              'cases': []}
    cases = [c for c in schedule(len(layers)) if wanted is None or c[0] in wanted]
    for family, name, seed, layer_index, params in cases:
        if family in ('replay',):
            continue
        layer = layers[layer_index]
        entry = {'family': family, 'id': name, 'seed': seed, 'layer_index': layer_index, 'layer': layer.name}
        if family == 'mixed_validity':
            entry['calls'] = supplied._case(runner, layer, params['lengths'], params['computed'], config, seed,
                                            SPEC['mixed_validity_steps'], params['slots'])
        elif family == 'turnover':
            entry['calls'] = supplied._schedule(runner, layer, _turnover_calls(params), config, seed)
        elif family == 'branch':
            entry['calls'] = supplied._schedule(runner, layer, _branch_calls(params), config, seed)
        else:
            entry['split'] = checkpoint_split(supplied, runner, layer, params['segments'], config, seed)
        record['cases'].append(entry)
    replays = [c for c in cases if c[0] == 'replay']
    if replays:
        record['replay'] = replay_family(supplied, runner, layers, config, replays)
    if out:
        directory = Path(out)
        directory.mkdir(parents=True, exist_ok=True)
        identity = record['identity']
        name = 'rank-dp{dp_rank}-tp{tp_rank}-{device_uuid}.json'.format(**identity)
        path = directory / name
        if path.exists():
            path = directory / f'{path.stem}-dup-{os.getpid()}.json'
        path.write_text(json.dumps(record) + '\n')
        record['path'] = str(path)
    return record


def replay_family(supplied, runner, layers, config, replays):
    """Capture every used layer's graphs first, then run the populations in order. A population seeds its own
    layer's region right before it is staged: KDA layers of different cache groups share physical state
    storage (the hybrid allocator's shared tensors), so no layer's content survives another layer's case."""
    region = SPEC['replay']['region']
    used = sorted({layer_index for _, _, _, layer_index, _ in replays})
    saved = {i: [s[:region].cpu() for s in layers[i].states] for i in used}
    replay = Replay(supplied, runner, layers, config)
    snapshots = [adapter.BuilderSnapshot(builder) for builder, _ in replay.builders.values()]
    results = []
    try:
        for i in used:
            for extent in SPEC['replay']['capture']:
                replay.capture(i, extent)
        for family, name, seed, layer_index, params in replays:
            layer = layers[layer_index]
            supplied._seed_states(layer, region, seed)
            reference = [s[:region].clone() for s in layer.states]
            rng = random.Random(seed)
            gen = torch.Generator(device='cuda').manual_seed(seed)
            result, _ = replay.population(layer_index, params['population'], rng, gen, reference, region)
            results.append({'id': name, 'seed': seed, 'layer': layer.name, **result})
    finally:
        for i, values in saved.items():
            for s, original in zip(layers[i].states, values):
                s[:region].copy_(original)
        for snapshot in snapshots:
            snapshot.restore()
        replay.graphs.clear()
    return {'populations': results, 'captures': [e for e in replay.events if e['event'] == 'capture']}
