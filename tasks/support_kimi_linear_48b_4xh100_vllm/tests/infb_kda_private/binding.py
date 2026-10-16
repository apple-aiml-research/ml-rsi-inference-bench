#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Separate serving evidence pass. Never imported in a component timing process.

Wrap each mapped layer's state-consumption call (`lifecycle.consume_entry`, by default the inner `_forward`
that the layer's opaque custom op runs) before graph capture; the compiled outer `forward` is never
wrapped. Device counters and intervention scalars are captured in the graph, so replay is observable without
Python callbacks. The rows, their state indices and query offsets are read through the documented mapping
paths of the layer's native metadata, the consumed output through `consume_output_argument` (or the
returned value) with its token axis moved first for observation. The client also uses vLLM's release-era
profiler. Trace presence alone is not the acceptance rule: perturbing a mapped output must change serving;
perturbing its final state must change a later token while leaving the first token unchanged.
"""

import functools
import inspect
from pathlib import Path

import torch

import layer_adapter as adapter


def source_root(model):
    source = Path(inspect.getsourcefile(type(model))).resolve()
    return str(next(p for p in source.parents if (p / 'vllm').is_dir()))


class Binding:
    def __init__(self, model):
        self.layers = adapter.discover(model, source_root(model))
        # [layer, output multiplier/state addition]. These are verifier-owned tensors.
        self.control = torch.zeros((20, 2), device='cuda', dtype=torch.float32)
        self.control[:, 0] = 1
        # prefill, graph/eager decode, mixed; columns are calls/tokens/output/state evidence.
        self.stats = torch.zeros((20, 3, 6), device='cuda', dtype=torch.float32)
        self.events = []
        for i, layer in enumerate(self.layers):
            self._wrap(i, layer)

    def _wrap(self, i, layer):
        original = layer.consume
        signature = inspect.signature(original)
        lifecycle = layer.lifecycle
        axis = lifecycle['consume_token_axis']

        @functools.wraps(original)
        def observed(*args, **kwargs):
            from vllm.forward_context import get_forward_context
            context = get_forward_context()
            meta = context.attn_metadata
            if meta is None:
                return original(*args, **kwargs)
            meta = meta[layer.prefix]
            indices = layer.metadata_view(meta, 'state_indices')
            rows = int(indices.shape[0])
            capturing = torch.cuda.is_current_stream_capturing()
            if capturing:
                # Only uniform decode is captured: one token per row; padded rows carry PAD indices.
                mode, actual, prefill_rows, decode_rows = 1, None, 0, rows
            else:
                starts = layer.metadata_view(meta, 'query_start_loc')[:rows + 1].tolist()
                lengths = [b - a for a, b in zip(starts[:-1], starts[1:])]
                prefill_rows = sum(n > 1 for n in lengths)
                decode_rows = sum(n == 1 for n in lengths)
                mode = 2 if prefill_rows and decode_rows else (0 if prefill_rows else 1)
                actual = starts[-1]
            # Decode metadata retains one index per padded graph row; its negative
            # sentinels are updated at replay even though Python counts are captured.
            indices = indices.long()
            live = indices >= 0
            safe = indices.clamp_min(0)
            state = layer.states[3]
            before = state[:, :, :4, :4].index_select(0, safe).clone()
            with torch.profiler.record_function(f'infb.kda.native.layer.{i}.mode.{mode}'):
                result = original(*args, **kwargs)
                if lifecycle['consume_returns_output']:
                    output = result
                else:
                    output = signature.bind(*args, **kwargs).arguments[lifecycle['consume_output_argument']]
                # The consumed output with its token axis first, for observation and the output intervention.
                consumed = output.movedim(axis, 0)
                if actual is None:
                    actual = int(consumed.shape[0])
                # Both effects live inside captured graphs. Zero state intervention and
                # unit output scale are algebraic identities in baseline evidence runs.
                consumed.mul_(self.control[i, 0])
                addition = torch.ones((safe.numel(), 8, 128), device=state.device)
                addition.mul_(live[:, None, None]).mul_(self.control[i, 1])
                state[:, :, 0, :].index_add_(0, safe, addition)
                after = state[:, :, :4, :4].index_select(0, safe)
                mask = live[:, None, None, None]
                # Graph capture includes padding rows. Their storage need not be
                # initialized, and NaN * 0 is still NaN. Exclude inactive rows with
                # where(), rather than letting padding contaminate valid evidence.
                before_live = torch.where(mask, before, 0)
                after_live = torch.where(mask, after, 0)
                observed_output = consumed[:actual].float().flatten(1)
                tokens = consumed.new_full((), actual, dtype=torch.float32)
                if mode == 1:
                    observed_output = torch.where(live[:actual, None], observed_output, 0)
                    tokens = live.sum().float()
                values = torch.stack((
                    consumed.new_ones((), dtype=torch.float32),
                    tokens, observed_output.abs().mean(),
                    before_live.abs().sum(), after_live.abs().sum(),
                    (after_live - before_live).abs().sum(),
                ))
                self.stats[i, mode].add_(values)
            if len(self.events) < 4096:
                self.events.append({'layer': i, 'mode': mode, 'tokens': actual, 'rows': rows,
                                    'prefills': prefill_rows, 'decodes': decode_rows, 'capturing': capturing})
            return result

        setattr(layer.consume_owner, layer.consume_name, observed)

    def set_control(self, layer=-1, output_scale=1.0, state_add=0.0, reset=True):
        self.control.zero_()
        self.control[:, 0] = 1
        if layer < 0:
            self.control[:, 0] = output_scale
            self.control[:, 1] = state_add
        else:
            self.control[layer, 0] = output_scale
            self.control[layer, 1] = state_add
        if reset:
            self.stats.zero_()
            self.events.clear()
        torch.cuda.synchronize()

    def report(self):
        torch.cuda.synchronize()
        return {'layers': [layer.identity() for layer in self.layers],
                'stats': self.stats.cpu().tolist(), 'events': self.events,
                'control': self.control.cpu().tolist()}


def install_startup_hook():
    """Called when the verifier worker extension is loaded, before load_model/capture."""
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    original = GPUModelRunner.load_model

    @functools.wraps(original)
    def load_and_instrument(runner, *args, **kwargs):
        result = original(runner, *args, **kwargs)
        runner.infb_kda_binding = Binding(runner.model)
        return result

    GPUModelRunner.load_model = load_and_instrument
