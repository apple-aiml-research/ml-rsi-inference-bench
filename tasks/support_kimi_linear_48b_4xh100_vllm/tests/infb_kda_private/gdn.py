#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Independent output/state checks for existing GDN interfaces, on the base, reference and candidate.

The private GDN preservation cases (specification.json `gdn`). Uses the serial delta recurrence with a scalar decay
broadcast across key channels, from the trusted copy of the supplied kda/. It imports only the
existing GDN entry points; it imports no optimized KDA implementation. verdicts.gdn_problems reads
its record.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import torch

# The supplied mathematics, from the trusted copy beside this package (the grader's /tests/eval).
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / 'eval' / 'kimi_linear' / 'kda'))

from mathematics import error, normalize_qk  # noqa: E402
from recurrence import naive_recurrent_kda  # noqa: E402


def reference(q, k, v, g, beta, states, starts, normalized):
    q, k = (normalize_qk(q), normalize_qk(k)) if normalized else (q, k)
    result, final = [], []
    for i, (start, end) in enumerate(zip(starts[:-1], starts[1:])):
        out, state = naive_recurrent_kda(
            q[:, start:end], k[:, start:end], v[:, start:end],
            g[:, start:end, :, None].expand(-1, -1, -1, 128), beta[:, start:end],
            initial_state=states[i:i + 1], output_final_state=True,
        )
        result.append(out)
        final.append(state)
    return torch.cat(result, dim=1), torch.cat(final)


@torch.inference_mode()
def run(seed=17647):
    from runtime import initialize_triton_allocator
    initialize_triton_allocator()
    from vllm.model_executor.layers.fla.ops.chunk import chunk_gated_delta_rule
    from vllm.model_executor.layers.fla.ops.fused_recurrent import fused_recurrent_gated_delta_rule
    gen = torch.Generator(device='cuda').manual_seed(seed)

    def random(shape, dtype=torch.bfloat16):
        return torch.randn(shape, generator=gen, device='cuda', dtype=dtype)

    records = []
    for lengths in ([63], [64], [65], [127], [128], [129], [512], [63, 65, 129]):
        starts = [0]
        for length in lengths:
            starts.append(starts[-1] + length)
        shape = (1, starts[-1], 8, 128)
        q, k, v = [random(shape) for _ in range(3)]
        g = -random(shape[:-1], torch.float32).abs() * 0.1
        beta = random(shape[:-1], torch.float32).sigmoid()
        offsets = torch.tensor(starts, dtype=torch.int32, device='cuda')
        for valid in (False, True):
            initial = random((len(lengths), 8, 128, 128), torch.float32) * 0.1
            if not valid:
                initial.zero_()
            want, state = reference(q, k, v, g, beta, initial, starts, True)
            got, final = chunk_gated_delta_rule(q, k, v, g, beta, initial_state=initial,
                                               output_final_state=True, cu_seqlens=offsets,
                                               use_qk_l2norm_in_kernel=True)
            records.append({'family': 'chunk_packed', 'lengths': lengths, 'initial': valid,
                            'output': error(got, want), 'state': error(final, state)})
        # Ordinary two-sequence call exercises the documented non-packed interface too.
        if len(lengths) == 1:
            q2, k2, v2 = [x.expand(2, -1, -1, -1).contiguous() for x in (q, k, v)]
            g2, b2 = [x.expand(2, -1, -1).contiguous() for x in (g, beta)]
            initial = random((2, 8, 128, 128), torch.float32) * 0.1
            normalized_q, normalized_k = normalize_qk(q2), normalize_qk(k2)
            want, state = naive_recurrent_kda(normalized_q, normalized_k, v2,
                                              g2[..., None].expand_as(q2), b2,
                                              initial_state=initial, output_final_state=True)
            got, final = chunk_gated_delta_rule(q2, k2, v2, g2, b2, initial_state=initial,
                                               output_final_state=True, use_qk_l2norm_in_kernel=True)
            records.append({'family': 'chunk_ordinary', 'lengths': lengths * 2,
                            'output': error(got, want), 'state': error(final, state)})
    for batch in (1, 3, 16, 64):
        slots = batch * 2 + 3
        # The engine's state buffer is contiguous within each [H,K,V] block, but slot
        # mapping is sparse and nontrivial. Unused slots must stay bit-identical.
        states = random((slots, 8, 128, 128), torch.float32) * 0.1
        indices = torch.arange(1, 2 * batch + 1, 2, dtype=torch.int32, device='cuda')
        unused = torch.ones(slots, dtype=torch.bool, device='cuda')
        unused[indices] = False
        starts = list(range(batch + 1))
        offsets = torch.tensor(starts, dtype=torch.int32, device='cuda')
        expected = states.clone()
        for step in range(8):
            q, k, v = [random((1, batch, 8, 128)) for _ in range(3)]
            g = -random((1, batch, 8), torch.float32).abs() * 0.1
            beta = random((1, batch, 8), torch.float32).sigmoid()
            want, state = reference(q, k, v, g, beta, expected[indices], starts, True)
            expected[indices] = state
            before = states[unused].clone()
            got, _ = fused_recurrent_gated_delta_rule(
                q, k, v, g, beta, initial_state=states, inplace_final_state=True,
                cu_seqlens=offsets, ssm_state_indices=indices, use_qk_l2norm_in_kernel=True,
            )
            records.append({'family': 'decode_continuation', 'batch': batch, 'step': step,
                            'output': error(got, want), 'state': error(states[indices], state),
                            'untouched_exact': torch.equal(states[unused], before)})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=17647)
    args = parser.parse_args()
    started = time.time()
    record = {'schema': 'infb-gdn/1', 'seed': args.seed}
    try:
        record.update(cases=run(args.seed), completed=True)
    except Exception as exc:
        record.update(completed=False, error=repr(exc))
        raise
    finally:
        record['elapsed_s'] = time.time() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
