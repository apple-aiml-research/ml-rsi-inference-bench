#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Read-only semantic mapping onto a loaded native layer; candidate computation stays in /app.

The optional /app/kda-verification.json changes paths and argument names, not expected
arithmetic. Tensor selectors support fused parameter slices and transposed state layouts.
Attention metadata for a controlled call comes from the loaded runner's own metadata builder
for the mapped layer, built from the call's facts (query offsets, computed-token counts,
sequence lengths, state blocks); nothing here constructs a backend's metadata class.
"""

import inspect
import json
from pathlib import Path

import torch

DEFAULT_WEIGHTS = {
    'q_proj': 'q_proj.weight', 'k_proj': 'k_proj.weight', 'v_proj': 'v_proj.weight',
    'q_conv': 'q_conv1d.weight', 'k_conv': 'k_conv1d.weight', 'v_conv': 'v_conv1d.weight',
    'decay_a': 'f_a_proj.weight', 'decay_b': 'f_b_proj.weight', 'a_log': 'A_log',
    'dt_bias': 'dt_bias', 'beta_proj': 'b_proj.weight', 'gate_a': 'g_a_proj.weight',
    'gate_b': 'g_b_proj.weight', 'norm': 'o_norm.weight', 'out_proj': 'o_proj.weight',
}
#: Where the mapped layer consumes its persistent state, and how to read that consumption. Paths are
#: relative to the mapped layer (`consume_entry`) or to the layer's native attention metadata (the rest).
DEFAULT_LIFECYCLE = {
    'consume_entry': '_forward',
    'state_indices': 'non_spec_state_indices_tensor',
    'query_start_loc': 'non_spec_query_start_loc',
    'valid_state': 'has_initial_state',
    'consume_output_argument': 'core_attn_out',
    'consume_token_axis': 1,
    'consume_returns_output': False,
}


def resolve(obj, path):
    for part in path.split('.'):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def tensor(obj, spec):
    if isinstance(spec, str):
        return resolve(obj, spec)
    out = resolve(obj, spec['path'])
    for selection in spec.get('slices', []):
        dim, start, end = selection
        out = out.narrow(dim, start, end - start)
    if 'transpose' in spec:
        out = out.transpose(*spec['transpose'])
    return out


def discover(model, root='/app'):
    path = Path(root) / 'kda-verification.json'
    config = json.loads(path.read_text()) if path.exists() else {}
    modules = dict(model.named_modules())
    names = config.get('layers')
    if names is None:
        names = [name for name, module in modules.items()
                 if getattr(module, 'mamba_type', None) == 'linear_attention']
    if len(names) != 20 or len(set(names)) != 20:
        raise AssertionError(f'expected the checkpoint\'s 20 native KDA layers, got {names}')
    return [Layer(name, modules[name], config) for name in names]


def _source_of(obj):
    import hashlib
    try:
        source = inspect.getsourcefile(inspect.unwrap(obj))
    except (TypeError, OSError):
        source = None
    path = Path(source).resolve() if source else None
    return {'path': str(path) if path else None,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest() if path else None}


class Layer:
    def __init__(self, name, module, config):
        self.name, self.module, self.config = name, module, config
        self.entry_name = config.get('entry', 'forward')
        self.positions_argument = config.get('positions_argument', 'positions')
        self.prefix = resolve(module, config.get('metadata_key', 'prefix'))
        self.weights = {k: tensor(module, v) for k, v in
                        {**DEFAULT_WEIGHTS, **config.get('weights', {})}.items()}
        self.eps = float(resolve(module, config.get('epsilon', 'o_norm.eps')))
        self.state_paths = config.get('states', ['kv_cache.0.0', 'kv_cache.0.1', 'kv_cache.0.2', 'kv_cache.0.3'])
        if len(self.state_paths) != 4:
            raise AssertionError('KDA needs three convolution states and one recurrent state')
        self.lifecycle = {**DEFAULT_LIFECYCLE, **config.get('lifecycle', {})}
        unknown = set(self.lifecycle) - set(DEFAULT_LIFECYCLE)
        if unknown:
            raise AssertionError(f'unknown lifecycle mapping keys {sorted(unknown)}')
        owner, _, self.consume_name = self.lifecycle['consume_entry'].rpartition('.')
        self.consume_owner = resolve(module, owner) if owner else module

    @property
    def entry(self):
        return getattr(self.module, self.entry_name)

    @property
    def consume(self):
        """The callable that consumes the persistent state: the mapped inner boundary (may equal `entry`)."""
        return getattr(self.consume_owner, self.consume_name)

    @property
    def states(self):
        states = [tensor(self.module, spec) for spec in self.state_paths]
        if tuple(states[3].shape[1:]) != (8, 128, 128):
            raise AssertionError(f'canonical TP4 recurrent state: {states[3].shape}')
        return states

    def metadata_view(self, metadata, key):
        """A path of the lifecycle mapping read from this layer's native metadata (None when mapped to null)."""
        path = self.lifecycle[key]
        return None if path is None else resolve(metadata, path)

    def identity(self):
        entry_source, consume_source = _source_of(self.entry), _source_of(self.consume)
        return {'layer': self.name, 'entry': self.entry_name,
                'source': entry_source['path'], 'source_sha256': entry_source['sha256'],
                'consume_entry': self.lifecycle['consume_entry'],
                'consume_source': consume_source['path'], 'consume_source_sha256': consume_source['sha256'],
                'consume_callable_module': getattr(self.consume, '__module__', None),
                'module_source': _source_of(type(self.module)),
                'callable_module': getattr(self.entry, '__module__', None),
                'state_shapes': [list(s.shape) for s in self.states],
                'state_strides': [list(s.stride()) for s in self.states]}

    def invoke(self, hidden, output, metadata, vllm_config, positions=None):
        """One call of the complete mapped layer under its native metadata.

        `positions` are the absolute per-token positions of the call (each row continues at its
        computed-token count); they are passed when the mapping names a positions argument. Under data
        parallelism the call is local to this replica: every replica is given this call's token count,
        so entering the forward context exchanges nothing with another replica."""
        from vllm.forward_context import set_forward_context
        args = {self.config.get('hidden_argument', 'hidden_states'): hidden}
        if not self.config.get('returns_output', False):
            args[self.config.get('output_argument', 'output')] = output
        if self.positions_argument is not None:
            if positions is None:
                raise ValueError('the mapping names a positions argument; a call must supply absolute positions')
            args[self.positions_argument] = positions
        replicas = vllm_config.parallel_config.data_parallel_size
        across = torch.full((replicas,), hidden.shape[0], dtype=torch.int32) if replicas > 1 else None
        with set_forward_context({self.prefix: metadata}, vllm_config, num_tokens=hidden.shape[0],
                                 num_tokens_across_dp=across):
            result = self.entry(**args)
        if self.config.get('returns_output', False):
            output.copy_(result)


# ── native metadata for controlled calls ────────────────────────────────────────────────────────


def metadata_builder(runner, layer):
    """The metadata builder of the loaded runner's attention group that holds `layer`, and that group's
    KV-cache group index and specification: the builder serving actually uses for this layer."""
    for group_id, groups in enumerate(runner.attn_groups):
        for group in groups:
            if layer.prefix in group.layer_names:
                return group.get_metadata_builder(), group_id, group.kv_cache_spec
    raise AssertionError(f'no attention group of the loaded runner holds {layer.prefix}')


def builder_identity(builder):
    cls = type(builder)
    return {'class': f'{cls.__module__}.{cls.__qualname__}', 'source': _source_of(cls),
            'reorder_batch_threshold': getattr(builder, 'reorder_batch_threshold', None),
            'buffers': {name: [value.data_ptr(), list(value.shape)] for name, value in sorted(vars(builder).items())
                        if isinstance(value, torch.Tensor)}}


class BuilderSnapshot:
    """The builder's persistent tensors (staging buffers that captured graphs read), saved and restored in
    place: the same storage, the same addresses, the same contents afterwards."""

    def __init__(self, builder):
        self.saved = {name: (value, value.clone()) for name, value in vars(builder).items()
                      if isinstance(value, torch.Tensor)}

    def restore(self):
        for value, saved in self.saved.values():
            value.copy_(saved)


def native_order(builder, query_lens):
    """The order the native runner gives these rows: requests whose query length is at most the builder's
    reorder threshold first, each part in its original order (the runner's decode/prefill split)."""
    threshold = getattr(builder, 'reorder_batch_threshold', None)
    rows = list(range(len(query_lens)))
    if threshold is None:
        return rows
    return [i for i in rows if query_lens[i] <= threshold] + [i for i in rows if query_lens[i] > threshold]


def common_metadata(query_lens, computed, slots, spec, device):
    """The runner's per-batch metadata for these rows, already in native order.

    Each row's state block sits in its block table at the position of its latest computed token
    ((sequence length - 1) // block size), every other entry null (0); no row writes attention KV."""
    from vllm.v1.attention.backends.utils import CommonAttentionMetadata
    starts = [0]
    for n in query_lens:
        starts.append(starts[-1] + n)
    seq_lens = [c + n for c, n in zip(computed, query_lens)]
    columns = [max(s - 1, 0) // spec.block_size for s in seq_lens]
    table = torch.zeros((len(query_lens), max(columns, default=0) + 1), dtype=torch.int32)
    for row, (column, slot) in enumerate(zip(columns, slots)):
        table[row, column] = slot
    query_start_loc_cpu = torch.tensor(starts, dtype=torch.int32)
    seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int32)
    return CommonAttentionMetadata(
        query_start_loc=query_start_loc_cpu.to(device), query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens_cpu.to(device), seq_lens_cpu=seq_lens_cpu,
        num_computed_tokens_cpu=torch.tensor(computed, dtype=torch.int32),
        num_reqs=len(query_lens), num_actual_tokens=starts[-1],
        max_query_len=max(query_lens, default=0), max_seq_len=max(seq_lens, default=0),
        block_table_tensor=table.to(device),
        slot_mapping=torch.full((starts[-1],), -1, dtype=torch.int64, device=device),
        causal=True,
    )


def native_metadata(runner, layer, query_lens, computed, slots, capture=False):
    """Metadata from the runner's own builder for `layer`, for rows already in native order.

    Returns (metadata, extent): the number of tokens the call carries, at least the live tokens; a
    builder that pads a decode batch (for a full CUDA graph) reports more state rows than live ones, and
    the call is then padded to them, as the runner pads its inputs. `capture` uses the builder's
    graph-capture entry instead of its normal one."""
    builder, _, spec = metadata_builder(runner, layer)
    common = common_metadata(query_lens, computed, slots, spec, runner.device)
    if capture:
        metadata = builder.build_for_cudagraph_capture(common)
    else:
        metadata = builder.build(common_prefix_len=0, common_attn_metadata=common)
    live = sum(query_lens)
    extent = live
    indices = layer.metadata_view(metadata, 'state_indices')
    if query_lens and max(query_lens) <= 1 and indices is not None:
        extent = max(live, int(indices.shape[0]))
    return metadata, extent


def positions(query_lens, computed, extent, device):
    """Absolute positions of each row's tokens: a row continues at its computed-token count; padding is 0."""
    values = [c + i for n, c in zip(query_lens, computed) for i in range(n)]
    values += [0] * (extent - len(values))
    return torch.tensor(values, dtype=torch.int64, device=device)
