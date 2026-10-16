#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""The private loaded-layer correctness schedule, composed from the supplied primitives.

`specification.json` `mathematics.manifest` lists the cases by id, in schedule order: single sequences at every
listed length with zero and valid nonzero history (a fresh length-one row included), the packed sequences and
repeated decode; each names its rows' lengths and computed counts (valid history is `history_tokens` earlier
tokens, a fresh row has none), its steps, its seed offset and the loaded layer it runs on. The base seed is the
caller's. Each case runs the supplied worker's `_case` (one loaded layer against the independent mathematics,
attention metadata from the loaded runner's own builder for that layer, its touched cache region and the
builder's staging buffers restored); the split cases compare continuous and split prefill of one sequence.
"""

import json
from pathlib import Path

import torch

import layer_adapter as adapter
from mathematics import error, layer_reference

SPEC = json.loads(Path(__file__).with_name('specification.json').read_text())['mathematics']


def manifest():
    """The trusted case manifest (`mathematics.manifest`), checked against the schedule it encodes."""
    cases = SPEC['manifest']
    history = SPEC['history_tokens']
    expected = [(f'single-{n}-{"valid" if valid else "fresh"}', [n], valid, 1)
                for n in SPEC['single_sequence_lengths'] for valid in (False, True) if n != 1 or valid]
    expected += [('packed-' + '-'.join(map(str, lengths)) + '-valid', list(lengths), True, 1)
                 for lengths in SPEC['packed_sequences']]
    decode = SPEC['repeated_decode']
    expected += [(f'decode-{decode["batch"]}x{decode["steps"]}-valid', [1] * decode['batch'], True, decode['steps'])]
    expected += [('single-1-fresh', [1], False, 1)]
    got = [(c['id'], c['lengths'], c['valid'], c['steps']) for c in cases]
    if got != expected:
        raise AssertionError(f'mathematics manifest does not encode the schedule: {got} != {expected}')
    for i, case in enumerate(cases):
        if (case['computed'] != [history if case['valid'] else 0] * len(case['lengths'])
                or case['seed_offset'] != i * 73 or case['layer'] != i % 20):
            raise AssertionError(f'mathematics manifest entry {case["id"]} is inconsistent')
    return cases


def split_case(supplied, runner, layer, segments, config, seed):
    """Native continuous vs split prefill + valid decode of one sequence continuing valid history in slot 1,
    both against the independent mathematics; metadata from the layer's own builder for every call."""
    history = SPEC['history_tokens']
    region = 5
    builder = adapter.metadata_builder(runner, layer)[0]
    snapshot = adapter.BuilderSnapshot(builder)
    saved = [state[:region].cpu() for state in layer.states]
    gen = torch.Generator(device='cuda').manual_seed(seed + 1)
    hidden = torch.randn((sum(segments), layer.weights['q_proj'].shape[1]), generator=gen,
                         device='cuda', dtype=torch.bfloat16) * 0.3
    try:
        supplied._seed_states(layer, region, seed)
        initial = [s[:region].clone() for s in layer.states]
        whole = supplied.Call(runner, layer, [sum(segments)], [history], [1])
        want, expected_state, _ = layer_reference(hidden, layer.weights, initial, whole.starts,
                                                   whole.indices, whole.valid, eps=layer.eps)
        want = supplied._tp_sum(want)
        continuous = torch.empty((whole.extent, want.shape[1]), device='cuda', dtype=want.dtype)
        layer.invoke(whole.padded(hidden), continuous, whole.metadata, config, whole.positions)
        continuous = continuous[:whole.live]
        continuous_state = [s[:region].clone() for s in layer.states]
        for state, value in zip(layer.states, initial):
            state[:region].copy_(value)
        split = torch.empty_like(want)
        offset, extents = 0, []
        for length in segments:
            call = supplied.Call(runner, layer, [length], [history + offset], [1])
            piece = torch.empty((call.extent, want.shape[1]), device='cuda', dtype=want.dtype)
            layer.invoke(call.padded(hidden[offset:offset + length]), piece, call.metadata, config, call.positions)
            split[offset:offset + length] = piece[:length]
            extents.append(call.extent)
            offset += length
        indices = whole.indices
        return {'segments': segments, 'history_tokens': history, 'extents': extents,
                'continuous_output': error(continuous, want),
                'split_output': error(split, want), 'split_vs_continuous': error(split, continuous),
                'continuous_states': [error(s[indices], e[indices]) for s, e in zip(continuous_state, expected_state)],
                'split_states': [error(s[indices], e[indices]) for s, e in zip(layer.states, expected_state)],
                'state_agreement': [error(s[indices], c[indices]) for s, c in zip(layer.states, continuous_state)],
                'untouched_exact': all(torch.equal(s[:region][[0, 2, 3, 4]], c[[0, 2, 3, 4]])
                                       for s, c in zip(layer.states, initial))}
    finally:
        for s, original in zip(layer.states, saved):
            s[:region].copy_(original)
        snapshot.restore()


def run(worker, seed, supplied):
    runner = worker.model_runner
    layers = adapter.discover(runner.model, supplied._root(runner.model))
    records = []
    for case in manifest():
        layer = layers[case['layer']]
        measurements = supplied._case(runner, layer, case['lengths'], case['computed'], worker.vllm_config,
                                      seed + case['seed_offset'], case['steps'])
        records.append({'id': case['id'], 'layer': layer.name, 'layer_index': case['layer'],
                        'lengths': case['lengths'], 'computed': case['computed'], 'valid': case['valid'],
                        'steps': case['steps'], 'measurements': measurements})
    split = [{'id': 'split-' + '-'.join(map(str, segments)), 'layer': layers[i].name, 'layer_index': i,
              **split_case(supplied, runner, layers[i], segments, worker.vllm_config, seed + 3000 + i)}
             for i, segments in enumerate(SPEC['split_prefill'])]
    return {'rank': worker.rank, 'layers': [layer.identity() for layer in layers],
            'builders': [adapter.builder_identity(adapter.metadata_builder(runner, layer)[0]) for layer in layers],
            'cases': records, 'split_cases': split}
