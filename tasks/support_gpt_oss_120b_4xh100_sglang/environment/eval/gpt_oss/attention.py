#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Independent causal/window/sink mathematics and a native attention example.

Q/K are already rotary-position encoded. A sink contributes to the softmax
denominator and has zero value. Positions are ordinary consecutive token
positions. The callable accepts semantic tensors; it need
not use the default SGLang function's layout internally.

    python -m gpt_oss.attention --output /tmp/attention.json

runs one representative case against `reference` under the published attention
policy (`numerical-policy.json`); its record is not a task acceptance verdict.
"""

import argparse
import importlib
import json
from pathlib import Path

import torch

from .numerics import assess

QUERY_HEADS, KV_HEADS, HEAD_WIDTH = 16, 2, 64
# Keys within 127 positions before the query: the released 128-token window.
RELEASE_WINDOW = 127
MAX_QUERY_TOKENS, MAX_HISTORY = 512, 16384


class AuthorDataFault(ValueError):
    """An attention case outside the admitted operation domain."""


def causal(count: int, prefix: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(prefix, prefix + count, device=device)
    allowed = torch.arange(prefix + count, device=device)[None, :] <= positions[:, None]
    return allowed, positions


def reference(q, k, v, allowed, query_positions, key_positions, sinks, window):
    """FP32 grouped-query attention; no candidate operation is used."""
    repeat = q.shape[1] // k.shape[1]
    key = k.float().repeat_interleave(repeat, dim=1)
    value = v.float().repeat_interleave(repeat, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), key) * q.shape[-1] ** -0.5
    mask = allowed.clone()
    if window >= 0:
        mask &= query_positions[:, None] <= key_positions[None, :] + window
    scores.masked_fill_(~mask[None], -torch.inf)
    sink_scores = sinks.float()[:, None, None].expand(-1, q.shape[0], 1)
    probabilities = torch.cat((scores, sink_scores), dim=-1).softmax(dim=-1)[..., :-1]
    return torch.einsum("hqk,khd->qhd", probabilities, value)


def admit(q, keys, values, prefixes, counts, sinks, window):
    """The admitted attention domain: TP4 head dimensions, finite BF16 Q/K/V with
    consecutive causal positions, full (-1) or the released sliding window, finite
    FP32 sink scores, 1-512 query tokens and at most 16,384 positions of history
    (query included) per sequence. Raises `AuthorDataFault`."""
    if window not in (-1, RELEASE_WINDOW):
        raise AuthorDataFault(f"window must be full (-1) or the released {RELEASE_WINDOW}, not {window}")
    if q.dtype != torch.bfloat16 or q.ndim != 3 or tuple(q.shape[1:]) != (QUERY_HEADS, HEAD_WIDTH):
        raise AuthorDataFault(f"queries must be BF16 [n, {QUERY_HEADS}, {HEAD_WIDTH}], "
                              f"not {q.dtype} {tuple(q.shape)}")
    if not 1 <= q.shape[0] <= MAX_QUERY_TOKENS or q.shape[0] != sum(counts):
        raise AuthorDataFault(f"one invocation has 1-{MAX_QUERY_TOKENS} query tokens matching its counts")
    if not torch.isfinite(q).all():
        raise AuthorDataFault("queries must be finite")
    if not len(keys) == len(values) == len(prefixes) == len(counts):
        raise AuthorDataFault("one key/value history per sequence")
    for k, v, prefix, count in zip(keys, values, prefixes, counts):
        if prefix < 0 or count < 1 or prefix + count > MAX_HISTORY:
            raise AuthorDataFault(f"history {prefix}+{count} is outside 1-{MAX_HISTORY} positions")
        for name, t in (("keys", k), ("values", v)):
            if t.dtype != torch.bfloat16 or tuple(t.shape) != (prefix + count, KV_HEADS, HEAD_WIDTH):
                raise AuthorDataFault(f"{name} must be BF16 [{prefix + count}, {KV_HEADS}, {HEAD_WIDTH}]")
            if not torch.isfinite(t).all():
                raise AuthorDataFault(f"{name} must be finite")
    if sinks.dtype != torch.float32 or tuple(sinks.shape) != (QUERY_HEADS,) or not torch.isfinite(sinks).all():
        raise AuthorDataFault("sink scores must be finite FP32, one per query head")
    return {"query_tokens": int(q.shape[0]), "sequences": len(prefixes),
            "longest_history": max(p + c for p, c in zip(prefixes, counts))}


def case(prefixes, counts, window, generator, device="cuda"):
    """One invocation's semantic inputs and its independent FP32 result."""
    q = torch.randn(sum(counts), QUERY_HEADS, HEAD_WIDTH, generator=generator, device=device,
                    dtype=torch.bfloat16)
    keys, values, masks, positions, expected = [], [], [], [], []
    sinks = torch.linspace(-1, 3, QUERY_HEADS, device=device)
    start = 0
    for prefix, count in zip(prefixes, counts):
        k = torch.randn(prefix + count, KV_HEADS, HEAD_WIDTH, generator=generator,
                        device=device, dtype=torch.bfloat16)
        v = torch.randn(k.shape, generator=generator, device=device, dtype=torch.bfloat16)
        mask, position = causal(count, prefix, device)
        key_positions = torch.cat((torch.arange(prefix, device=device), position))
        expected.append(reference(q[start:start+count], k, v, mask, position,
                                  key_positions, sinks, window))
        keys.append(k); values.append(v); masks.append(mask); positions.append(position)
        start += count
    return q, keys, values, masks, torch.cat(positions), sinks, torch.cat(expected)


class NativeExtend:
    """Adapter to the native attention called by the default Triton backend.

    An alternate engine binding can replace this adapter without changing the
    independent reference, input schedule or measurement code.
    """
    def __init__(self, q, k, v, masks, positions, prefixes, counts, sinks, window):
        from sglang.srt.layers.attention.triton_ops.extend_attention import extend_attention_fwd
        self.fn = extend_attention_fwd
        self.q, self.sinks, self.window = q, sinks, window
        self.out = torch.empty_like(q)
        self.k_extend = torch.cat([a[p:] for a, p in zip(k, prefixes)])
        self.v_extend = torch.cat([a[p:] for a, p in zip(v, prefixes)])
        self.qo = torch.tensor([0, *torch.tensor(counts).cumsum(0).tolist()],
                               dtype=torch.int32, device=q.device)
        offsets = [max(0, p - window) if window >= 0 else 0 for p in prefixes]
        self.k_buffer = torch.cat([a[o:p] for a, o, p in zip(k, offsets, prefixes)])
        self.v_buffer = torch.cat([a[o:p] for a, o, p in zip(v, offsets, prefixes)])
        lengths = [p - o for p, o in zip(prefixes, offsets)]
        self.kv = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()],
                               dtype=torch.int32, device=q.device)
        self.indices = torch.arange(sum(lengths), dtype=torch.int32, device=q.device)
        self.max_count = max(counts)
        self.prefixes, self.offsets_host = prefixes, offsets

    def __call__(self):
        self.fn(self.q, self.k_extend, self.v_extend, self.out, self.k_buffer,
                self.v_buffer, self.qo, self.kv, self.indices, None, True,
                None, self.max_count, sliding_window_size=self.window, sk=self.sinks)
        return self.out


def load_binding(name=None):
    if not name:
        return NativeExtend
    module, attribute = name.split(':')
    return getattr(importlib.import_module(module), attribute)


def example(binding=NativeExtend, seed=314159, device="cuda"):
    """One representative case, a history crossing the released window, for both layer
    types (full and sliding), in eager and CUDA-graph execution."""
    generator = torch.Generator(device=device).manual_seed(seed)
    rows = []
    prefixes, counts = [129], [8]
    for window in (-1, RELEASE_WINDOW):
        q, keys, values, masks, positions, sinks, target = case(prefixes, counts, window, generator, device)
        admission = admit(q, keys, values, prefixes, counts, sinks, window)
        op = binding(q, keys, values, masks, positions, prefixes, counts, sinks, window)
        for _ in range(3):
            op()
        torch.cuda.synchronize()
        rows.append({"window": window, "prefixes": prefixes, "counts": counts, "mode": "eager",
                     "admission": admission, **assess(op().clone(), target, "attention")})
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out = op()
        graph.replay()
        torch.cuda.synchronize()
        rows.append({"window": window, "prefixes": prefixes, "counts": counts, "mode": "graph",
                     "admission": admission, **assess(graph_out.clone(), target, "attention")})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", default=314159, type=int)
    parser.add_argument("--binding", help="Equivalent native attention binding as module:class")
    args = parser.parse_args()
    rows = example(binding=load_binding(args.binding), seed=args.seed)
    record = {"schema": "gpt-oss-attention-example/1", "seed": args.seed, "rows": rows,
              "sample_correctness": "pass" if all(r["within_policy"] for r in rows) else "fail",
              "task_acceptance": "not_determined"}
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"cases": len(rows), "sample_correctness": record["sample_correctness"],
                      "task_acceptance": record["task_acceptance"],
                      "max_error": max(r.get("max", float("inf")) for r in rows),
                      "rms_error": max(r.get("rms", float("inf")) for r in rows)}))


if __name__ == "__main__":
    main()
