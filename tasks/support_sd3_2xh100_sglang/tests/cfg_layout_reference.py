#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Independent diffusers CFG layout control; never imports the candidate engine.

Diffusers presents [negative rows, positive rows] to the transformer and owns
the CFG combination and scheduler. This adapter only changes how those rows
reach the SAME transformer: concatenated, or two full-batch branch forwards.
The returned order stays negative then positive. No thresholds live here.
"""
from contextlib import contextmanager


LAYOUTS = ('concat', 'split_positive_first', 'split_negative_first')
ROW_FIELDS = ('hidden_states', 'encoder_hidden_states', 'pooled_projections')


@contextmanager
def reference_cfg_layout(pipeline, guidance_scale, layout='concat'):
    """Temporarily adapt the independent pipeline; always restore its module.

Each split invokes the original module through __call__, so its actual pre/post
hooks witness both forwards. Weights and model identity are never copied. Only
the SD3 base-model call contract is accepted; unknown batched conditions fail.
"""
    import torch

    if layout not in LAYOUTS:
        raise ValueError(f'Unsupported reference CFG layout: {layout}')
    original = pipeline.transformer
    if layout == 'concat' or guidance_scale <= 1:
        yield original
        return

    class SplitCFG(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(super().__getattr__('model'), name)

        def forward(self, *args, **kwargs):
            if args:
                raise ValueError('Reference SD3 call must name its boundary tensors')
            allowed = set(ROW_FIELDS) | {'timestep', 'return_dict', 'joint_attention_kwargs', 'skip_layers'}
            if set(kwargs) - allowed:
                raise ValueError(f'Unreviewed reference arguments: {sorted(set(kwargs) - allowed)}')
            if kwargs.get('return_dict', True) is not False:
                raise ValueError('Reference expects return_dict=False')
            x = kwargs.get('hidden_states')
            if not isinstance(x, torch.Tensor) or x.ndim < 1 or x.shape[0] < 2 or x.shape[0] % 2:
                raise ValueError('CFG reference needs equal negative and positive row counts')
            rows = x.shape[0]
            for name in ROW_FIELDS:
                value = kwargs.get(name)
                if not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[0] != rows:
                    raise ValueError(f'{name} does not match the complete CFG batch')
            time = kwargs.get('timestep')
            if not isinstance(time, torch.Tensor) or (time.ndim > 0 and time.shape[0] not in (1, rows)):
                raise ValueError('timestep must be scalar, singleton or cover every row')
            attention = kwargs.get('joint_attention_kwargs')
            if attention and (set(attention) != {'scale'} or not isinstance(attention['scale'], (int, float))):
                raise ValueError('Unreviewed batched joint-attention conditions')
            half = rows // 2
            order = (1, 0) if layout == 'split_positive_first' else (0, 1)
            results = {}
            for branch in order:
                sl = slice(branch * half, (branch + 1) * half)
                call = dict(kwargs)
                for name in ROW_FIELDS:
                    call[name] = kwargs[name][sl]
                if time.ndim > 0 and time.shape[0] == rows:
                    call['timestep'] = time[sl]
                result = self.model(**call)
                if not isinstance(result, tuple) or len(result) != 1:
                    raise ValueError('Unexpected independent SD3 transformer result')
                if not isinstance(result[0], torch.Tensor) or result[0].shape != call['hidden_states'].shape:
                    raise ValueError('Independent SD3 transformer returned an incomplete branch')
                results[branch] = result[0]
            return (torch.cat((results[0], results[1]), dim=0),)

    pipeline.transformer = SplitCFG(original)
    try:
        yield original
    finally:
        pipeline.transformer = original
