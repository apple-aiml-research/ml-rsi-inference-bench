#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Small standalone KDA prefill, continuation and packed-state examples; model loading is not required.

The optional adapter is a user-chosen module:function in the submitted source tree.
It receives semantic tensors and returns (attention_output, complete_state_buffer).
This development adapter is not used by the grader, which checks the loaded native layer.
"""

import argparse
import importlib
import json
from pathlib import Path
import time

import torch

from assessment import error_statistics, evidence_problems, numerical_problems, worst_errors
from mathematics import error, packed_recurrence


@torch.inference_mode()
def run(adapter=None, seed=37129):
    gen = torch.Generator(device='cuda').manual_seed(seed)

    def random(shape, dtype=torch.bfloat16):
        return torch.randn(shape, generator=gen, dtype=dtype, device='cuda')

    families = [('prefill', [8], False, 1),
                ('prefill', [8], True, 2),
                ('prefill', [5, 11], False, 2),
                ('decode', [1, 1], True, 3)]
    records = []
    for mode, lengths, initial, steps in families:
        slots = len(lengths) + 1
        states = random((slots, 8, 128, 128), torch.float32) * 0.1
        expected = states.clone()
        indices = torch.arange(len(lengths), device='cuda', dtype=torch.int32)
        valid = torch.full((len(lengths),), initial, device='cuda', dtype=torch.bool)
        untouched = torch.ones(slots, dtype=torch.bool, device='cuda')
        untouched[indices] = False
        starts = [0]
        for length in lengths:
            starts.append(starts[-1] + length)
        offsets = torch.tensor(starts, device='cuda', dtype=torch.int32)
        for step in range(steps):
            shape = (starts[-1], 8, 128)
            q, k, v = [random(shape) for _ in range(3)]
            g = -random(shape, torch.float32).abs() * 0.1
            beta = random(shape[:-1], torch.float32).sigmoid()
            selected = expected[indices].clone()
            selected[~valid] = 0
            want, next_selected = packed_recurrence(q, k, v, g, beta, starts, selected)
            expected[indices] = next_selected
            record = {'mode': mode, 'lengths': lengths, 'initial': initial, 'step': step}
            if adapter is None:
                if not bool(torch.isfinite(want).all() and torch.isfinite(expected).all()):
                    raise AssertionError('nonfinite mathematical reference')
                record.update(reference_only=True, output_shape=[1, *want.shape],
                              state_shape=list(expected.shape))
            else:
                before = states.clone()
                got, final = adapter(mode=mode, q=q[None].clone(), k=k[None].clone(),
                                     v=v[None].clone(), g=g[None].clone(), beta=beta[None].clone(),
                                     states=states.clone(), state_indices=indices.clone(),
                                     has_initial_state=valid.clone(), cu_seqlens=offsets.clone())
                if final.shape != expected.shape:
                    raise AssertionError(f'complete state shape {final.shape} != {expected.shape}')
                if got.shape != want[None].shape:
                    raise AssertionError(f'attention shape {got.shape} != {want[None].shape}')
                record.update(attention=error(got, want[None]), state=error(final[indices], expected[indices]),
                              untouched_exact=bool(torch.equal(final[untouched], before[untouched])))
                states = final.clone()
            records.append(record)
            valid.fill_(True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', help='native-source module:function with the semantic signature in the README')
    parser.add_argument('--seed', type=int, default=37129)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    record = {'schema': 'infb-kda-development/1', 'seed': args.seed,
              'reference_only': args.adapter is None, 'completed': False,
              'sample_correctness': 'not_evaluated', 'task_acceptance': 'not_determined'}
    rc = 1
    try:
        adapter = None
        if args.adapter is not None:
            from runtime import initialize_triton_allocator
            initialize_triton_allocator()
            module, separator, name = args.adapter.partition(':')
            if not separator:
                raise ValueError('--adapter must be module:function')
            adapter = getattr(importlib.import_module(module), name)
        record.update(cases=run(adapter, args.seed), completed=True)
        if adapter is not None:
            # This adapter returns core attention, before normalization/projection/TP.
            record['problems'] = evidence_problems(record['cases'])
            record['evidence_complete'] = not record['problems']
            policy = json.loads(Path(__file__).with_name('numerical-policy.json').read_text())
            record['numerical_problems'] = numerical_problems(record['cases'], policy)
            record['problems'].extend(record['numerical_problems'])
            record['sample_correctness'] = ('unavailable' if not record['evidence_complete']
                                            else 'fail' if record['numerical_problems'] else 'pass')
            record['worst'] = worst_errors(error_statistics(record['cases']))
            rc = 0 if not record['problems'] else 1
        else:
            record['verdict'] = 'reference execution only'
            rc = 0
    except Exception as exc:
        record['error'] = repr(exc)
        raise
    finally:
        record['elapsed_s'] = time.monotonic() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + '\n')
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
