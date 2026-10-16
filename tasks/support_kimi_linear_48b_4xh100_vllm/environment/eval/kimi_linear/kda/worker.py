#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Worker RPCs for the loaded native KDA layer: inspection, a small sample and complete-layer timing.

`launch.py` loads `worker.KDAWorker` as the server's worker extension; `client.py` calls its RPCs
through `/collective_rpc` on all four TP ranks. The shared primitives here run one loaded layer on
controlled inputs and states and compare it with the independent mathematics; the parameters are in
protocol.json. Every controlled call takes its attention metadata from the loaded runner's own
metadata builder for that layer (`layer_adapter.native_metadata`), built from the call's rows: query
lengths, computed-token counts and state slots. A row with no computed tokens is a fresh request.
"""

import json
import statistics
import time
from pathlib import Path

import torch

import layer_adapter as adapter
from mathematics import error, layer_reference

PROTOCOL = json.loads(Path(__file__).with_name('protocol.json').read_text())


def _root(model):
    import inspect
    source = Path(inspect.getsourcefile(type(model))).resolve()
    return str(next(p for p in source.parents if (p / 'vllm').is_dir()))


def _seed_states(layer, slots, seed):
    gen = torch.Generator(device='cuda').manual_seed(seed)
    for state in layer.states:
        state[:slots].copy_(torch.randn(state[:slots].shape, generator=gen, device='cuda', dtype=state.dtype) * 0.1)


def _tp_sum(x):
    from vllm.distributed import get_tp_group
    # An independent FP32 sum of the local BF16 projection results, then one cast.
    # Reducing directly in BF16 would add transport-order rounding to the expected value.
    total = x.float()
    torch.distributed.all_reduce(total, group=get_tp_group().device_group)
    return total.to(x.dtype)


def _starts(lengths):
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    return starts


class Call:
    """One controlled call's rows in the runner's native order: metadata from the layer's own builder,
    the padded extent, absolute positions, and the state slots and validity the mathematics uses."""

    def __init__(self, runner, layer, lengths, computed, slots, capture=False):
        builder = adapter.metadata_builder(runner, layer)[0]
        self.order = adapter.native_order(builder, lengths)
        self.lengths = [lengths[i] for i in self.order]
        self.computed = [computed[i] for i in self.order]
        self.slots = [slots[i] for i in self.order]
        self.metadata, self.extent = adapter.native_metadata(runner, layer, self.lengths, self.computed,
                                                             self.slots, capture=capture)
        self.live = sum(self.lengths)
        self.starts = _starts(self.lengths)
        self.indices = torch.tensor(self.slots, dtype=torch.int64, device='cuda')
        self.valid = torch.tensor([c > 0 for c in self.computed], dtype=torch.bool, device='cuda')
        self.positions = adapter.positions(self.lengths, self.computed, self.extent, 'cuda')

    def native_rows(self, packed, lengths):
        """A packed per-token tensor in the caller's row order, rearranged into this call's native order."""
        parts = torch.split(packed, lengths)
        return torch.cat([parts[i] for i in self.order]) if parts else packed

    def padded(self, rows):
        """Live rows followed by zero padding up to the call's extent."""
        if rows.shape[0] == self.extent:
            return rows
        out = rows.new_zeros((self.extent, *rows.shape[1:]))
        out[:rows.shape[0]] = rows
        return out


def _attention_probe(layer):
    """The ordinary boundary exposes raw attention as the input to normalization. A fused enclosing
    boundary can declare null; its complete output and state are then the compared operation."""
    probe_path = layer.config.get('attention_probe', 'o_norm')
    observed, hook = [], None
    if probe_path is not None:
        probe = adapter.resolve(layer.module, probe_path)
        hook = probe.register_forward_pre_hook(lambda module, args: observed.append(args[0].detach().clone()))
    return probe_path, observed, hook


def _schedule(runner, layer, calls, config, seed):
    """A schedule of controlled calls of one loaded layer, each against the independent mathematics.

    Every call names its rows: `lengths` (new tokens), `computed` (earlier tokens; none means a fresh request)
    and `slots`, and may first copy whole states between slots (`copies`: [source, target] pairs, applied
    to the cache and to the mathematics alike, as a restored checkpoint). The mathematics starts a fresh row
    from zero state, whatever its slot holds, and carries every slot's state from call to call. Each call's
    output, core attention and four states are compared; every state slot the call does not own must stay
    bit-identical. The touched cache region and the builder's staging buffers are saved and restored exactly."""
    region = max(max(call['slots']) for call in calls) + 4
    builder = adapter.metadata_builder(runner, layer)[0]
    snapshot = adapter.BuilderSnapshot(builder)
    saved = [state[:region].cpu() for state in layer.states]
    gen = torch.Generator(device='cuda').manual_seed(seed + 1)
    records = []
    try:
        _seed_states(layer, region, seed)
        reference_states = [s[:region].clone() for s in layer.states]
        for step, spec in enumerate(calls):
            lengths = list(spec['lengths'])
            for source, target in spec.get('copies', []):
                for state, reference in zip(layer.states, reference_states):
                    state[target].copy_(state[source])
                    reference[target].copy_(reference[source])
            call = Call(runner, layer, lengths, spec['computed'], spec['slots'])
            packed = torch.randn((sum(lengths), layer.weights['q_proj'].shape[1]),
                                 generator=gen, device='cuda', dtype=torch.bfloat16) * 0.3
            hidden = call.padded(call.native_rows(packed, lengths))
            before = [s[:region].clone() for s in layer.states]
            want, state, attention = layer_reference(hidden[:call.live], layer.weights, reference_states,
                                                     call.starts, call.indices, call.valid, eps=layer.eps)
            want = _tp_sum(want)
            got = torch.full((call.extent, want.shape[1]), float('nan'), device='cuda', dtype=want.dtype)
            probe_path, observed, hook = _attention_probe(layer)
            try:
                layer.invoke(hidden, got, call.metadata, config, call.positions)
            finally:
                if hook is not None:
                    hook.remove()
            torch.cuda.synchronize()
            untouched = torch.ones(region, dtype=torch.bool, device='cuda')
            untouched[call.indices] = False
            records.append({'step': step, 'lengths': call.lengths, 'computed': call.computed, 'slots': call.slots,
                            'copies': spec.get('copies', []), 'native_order': call.order, 'extent': call.extent,
                            'input_shape': list(hidden.shape), 'output_shape': list(got.shape),
                            'output': error(got[:call.live], want),
                            'attention': (error(observed[0].reshape(-1, *attention.shape[1:])[:call.live], attention)
                                          if observed else None),
                            'attention_observed': probe_path is None or len(observed) == 1,
                            'states': [error(s[call.indices], ref[call.indices]) for s, ref in zip(layer.states, state)],
                            'untouched_exact': all(torch.equal(s[:region][untouched], b[untouched])
                                                   for s, b in zip(layer.states, before))})
            reference_states = state
    finally:
        for s, original in zip(layer.states, saved):
            s[:region].copy_(original)
        snapshot.restore()
    return records


def _case(runner, layer, lengths, computed, config, seed, steps=1, slots=None):
    """One loaded layer on packed rows (`lengths` new tokens each, after `computed` earlier ones) for `steps`
    consecutive calls; the rows keep their slots (default 1, 3, 5, ...) and every later call continues every
    row from what it computed. See `_schedule`."""
    slots = list(slots or [2 * i + 1 for i in range(len(lengths))])
    calls, computed = [], list(computed)
    for _ in range(steps):
        calls.append({'lengths': list(lengths), 'computed': list(computed), 'slots': slots})
        computed = [c + n for c, n in zip(computed, lengths)]
    return _schedule(runner, layer, calls, config, seed)


def _validate_live(runner, layer, hidden, call, config):
    region = max(call.slots) + 3
    before = [s[:region].clone() for s in layer.states]
    want, expected, _ = layer_reference(hidden[:call.live], layer.weights, before, call.starts,
                                        call.indices, call.valid, eps=layer.eps)
    want = _tp_sum(want)
    output = torch.full((call.extent, want.shape[1]), float('nan'), device='cuda', dtype=want.dtype)
    layer.invoke(hidden, output, call.metadata, config, call.positions)
    unused = torch.ones(region, dtype=torch.bool, device='cuda')
    unused[call.indices] = False
    return {'layer': layer.name, 'output': error(output[:call.live], want),
            'states': [error(s[call.indices], ref[call.indices]) for s, ref in zip(layer.states, expected)],
            'untouched_exact': all(torch.equal(s[:region][unused], old[unused])
                                   for s, old in zip(layer.states, before))}


class KDAWorker:
    def infb_kda_inspect(self):
        model = self.model_runner.model
        layers = adapter.discover(model, _root(model))
        builders = [adapter.metadata_builder(self.model_runner, layer) for layer in layers]
        return {'rank': self.rank, 'layers': [layer.identity() for layer in layers],
                'builders': [{'layer': layer.name, 'kv_cache_group': group, **adapter.builder_identity(builder)}
                             for layer, (builder, group, _) in zip(layers, builders)],
                'device': torch.cuda.get_device_name(), 'torch': str(torch.__version__)}

    @torch.inference_mode()
    def infb_kda_sample(self, seed=str(PROTOCOL['sample']['seed_default']), case='all'):
        """The public sample (protocol.json `sample`) on the first discovered loaded layer; `case` names one."""
        layers = adapter.discover(self.model_runner.model, _root(self.model_runner.model))
        layer = layers[0]
        cases = []
        for i, spec in enumerate(PROTOCOL['sample']['cases']):
            if case not in ('all', spec['name']):
                continue
            measurements = _case(self.model_runner, layer, spec['lengths'], spec['computed'], self.vllm_config,
                                 int(seed) + i * 73, spec['steps'])
            cases.append({'name': spec['name'], 'layer': layer.name, 'lengths': spec['lengths'],
                          'computed': spec['computed'], 'steps': spec['steps'], 'measurements': measurements})
        if not cases:
            raise ValueError(f'no sample case {case!r}')
        return {'rank': self.rank, 'layers': [layer.identity() for layer in layers],
                'builder': adapter.builder_identity(adapter.metadata_builder(self.model_runner, layer)[0]),
                'sample_layer': layer.name, 'state_shapes': [list(s.shape[1:]) for s in layer.states],
                'cases': cases}

    @torch.inference_mode()
    def infb_kda_timing(self, seed='42817', draws='5', repeats='10', family='all'):
        """Complete loaded layer, all 20 layer/state working sets; no tracing in this RPC.

        The measurement parameters are in protocol.json. State is reset symmetrically
        before every draw; required gather/scatter/copies stay inside invoke. Every row
        continues valid history (protocol.json `timing.history_tokens` computed tokens).
        """
        runner = self.model_runner
        layers = adapter.discover(runner.model, _root(runner.model))
        history = PROTOCOL['timing']['history_tokens']
        families = [('prefill', [n]) for n in (128, 512, 2048)]
        families += [('prefill_packed', [2048] * 4)]
        families += [('decode', [1] * n) for n in (1, 3, 16, 64)]
        if family != 'all':
            families = [families[int(family)]]
        builders = {id(adapter.metadata_builder(runner, layer)[0]): adapter.metadata_builder(runner, layer)[0]
                    for layer in layers}
        snapshots = [adapter.BuilderSnapshot(b) for b in builders.values()]
        records = []
        try:
            for name, lengths in families:
                slots = list(range(1, len(lengths) + 1))
                region = len(lengths) + 3
                saved = [[s[:region].cpu() for s in layer.states] for layer in layers]
                # Each layer's metadata from its own group's builder; layers of one group share it.
                calls = {}
                for layer in layers:
                    key = id(adapter.metadata_builder(runner, layer)[0])
                    if key not in calls:
                        calls[key] = Call(runner, layer, lengths, [history] * len(lengths), slots)
                layer_calls = [calls[id(adapter.metadata_builder(runner, layer)[0])] for layer in layers]
                extent = layer_calls[0].extent
                gen = torch.Generator(device='cuda').manual_seed(int(seed))
                # Layer weights and recurrent caches are distinct. The controlled hidden
                # input and output scratch are reused sequentially, as serving reuses its
                # activation workspace; 20 full 8192-token input/output pairs would add
                # several GiB beside the loaded model and manufacture an OOM requirement.
                hidden = layer_calls[0].padded(torch.randn((sum(lengths), layers[0].weights['q_proj'].shape[1]),
                                                           generator=gen, device='cuda', dtype=torch.bfloat16) * 0.3)
                output = torch.empty_like(hidden)

                def call():
                    for layer, c in zip(layers, layer_calls):
                        layer.invoke(hidden, output, c.metadata, self.vllm_config, c.positions)

                try:
                    for layer in layers:
                        _seed_states(layer, region, int(seed))
                    before_checks = [_validate_live(runner, layers[i], hidden, layer_calls[i], self.vllm_config)
                                     for i in (0, 19)]
                    for _ in range(3):
                        call()
                    torch.cuda.synchronize()
                    graph = None
                    if name == 'decode':
                        # Match serving: graph launch overhead amortized over 20 distinct KDA layers.
                        from vllm.distributed.parallel_state import graph_capture
                        with graph_capture(device=hidden.device) as capture:
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph, stream=capture.stream):
                                call()
                    run = graph.replay if graph is not None else call
                    measured = []
                    for _ in range(int(draws)):
                        for layer in layers:
                            _seed_states(layer, region, int(seed))
                        torch.cuda.synchronize()
                        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        t0 = time.perf_counter()
                        begin.record()
                        for _ in range(int(repeats)):
                            run()
                        end.record()
                        torch.cuda.synchronize()
                        wall = time.perf_counter() - t0
                        measured.append({'wall_s': wall / int(repeats) / 20,
                                         'device_s': begin.elapsed_time(end) / 1000 / int(repeats) / 20})
                    records.append({'family': name, 'lengths': lengths, 'layers': 20, 'extent': extent,
                                    'history_tokens': history,
                                    'mode': 'cuda_graph' if graph else 'eager', 'draws': measured,
                                    'wall_median_s': statistics.median(d['wall_s'] for d in measured),
                                    'before_checks': before_checks,
                                    'after_checks': [_validate_live(runner, layers[i], hidden, layer_calls[i],
                                                                    self.vllm_config) for i in (0, 19)],
                                    'outputs_finite': bool(torch.isfinite(output[:layer_calls[0].live]).all())})
                finally:
                    for layer, original in zip(layers, saved):
                        for state, value in zip(layer.states, original):
                            state[:region].copy_(value)
        finally:
            for snapshot in snapshots:
                snapshot.restore()
        return {'rank': self.rank, 'layers': [layer.identity() for layer in layers], 'families': records}
