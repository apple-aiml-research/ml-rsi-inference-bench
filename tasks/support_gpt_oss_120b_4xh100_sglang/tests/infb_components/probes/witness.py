#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Private binding witness: the named native module must be invoked and must matter.

Outside all timed draws, the verifier observes native calls into the bound module
and replaces its returned contribution with zeros on every rank, then records the
change at the complete-operation boundary. Merely naming an unused module fails.
"""

import torch


def binding_witness(op, x, ids, probabilities):
    calls = []
    def entered(module, args, kwargs):
        calls.append(1)
    handle = op.module.register_forward_pre_hook(entered, with_kwargs=True)
    try:
        ordinary = op(x, ids, probabilities).detach().clone()
        ordinary_calls = len(calls)
        calls.clear()
        def suppress(module, args, kwargs, output):
            if not isinstance(output, torch.Tensor):
                raise TypeError('native contribution must be a tensor at the binding boundary')
            return output * 0
        perturb = op.module.register_forward_hook(suppress, with_kwargs=True)
        try:
            suppressed = op(x, ids, probabilities).detach().clone()
        finally:
            perturb.remove()
        torch.cuda.synchronize()
        difference = (ordinary.float()-suppressed.float()).abs()
        return {'ordinary_native_calls': ordinary_calls,
                'suppressed_native_calls': len(calls),
                'finite': bool(torch.isfinite(difference).all().item()),
                'max_output_change': float(difference.max().item()),
                'changed_elements': int((difference != 0).sum().item())}
    finally:
        handle.remove()
