# Relative-attention operator contract

The model's attention is causal softmax attention with a learned, per-token, per-query-head bias
on the query-key distance. This directory specifies the operator through which that attention is
graded: one callable, committed at `/app/inkling_rel_attention.py`, which must launch the same GPU
computation the served model runs for the same layer kind and call shape.

Provenance: the law below is the "Ragged Relative Bias" contract of the `vllm-project/tml-fa4`
README (the section's file has sha256
`b560214ef398461653b556edc6be2a302cac63d8c08336faf7b396c2649d4f31`), which is also the model
author's definition (`/public/reference-impl/modeling_inkling.py`). The paged key form and the
unreferenced-input clause are this task's extension to the engine's paged KV cache. The tml-fa4
code is not supplied.

## Entry point

```python
def relative_attention(q, k, v, rel_bias, cu_seqlens_q, max_seqlen_q, *,
                       cu_seqlens_k=None, max_seqlen_k=None,
                       page_table=None, seqused_k=None,
                       causal=True, window_size=(None, None),
                       softmax_scale, out=None) -> torch.Tensor
```

| Argument | Contract |
|---|---|
| `q` | BF16 `[total_q, 8, 128]`: this rank's eight query heads, after the model's QK normalization and, on full layers, its log-scaling |
| `k`, `v` | Ragged form: BF16 `[total_k, 2, 128]`, rows ordered by `cu_seqlens_k`. Paged form: BF16 `[num_pages, 128, 2, 128]`; key position `j` of request `b` is at page `page_table[b, j // 128]`, slot `j % 128` |
| `rel_bias` | BF16 `[total_q, 8, E]`, `E` in {512, 1,024}, the same flattened order as `q`; on full layers already multiplied by the model's log-scaling |
| `cu_seqlens_q` | Contiguous int32 CUDA `[B + 1]`, `cu_seqlens_q[B] = total_q`; `max_seqlen_q` is its exact maximum length |
| Ragged keys | `cu_seqlens_k` int32 `[B + 1]` and `max_seqlen_k`, with `k_len_b >= q_len_b` |
| Paged keys | `page_table` int32 `[B, max_pages]` (every entry in `[0, num_pages)`), `seqused_k` int32 `[B]` with `q_len_b <= seqused_k[b]`, and `max_seqlen_k = max(seqused_k)`. The queries' own keys are already in the cache |
| `causal`, `window_size` | Always `causal=True`. `window_size=(511, 0)` with `E = 512` (sliding-window layers) or `(None, None)` with `E = 1,024` (full layers) |
| `softmax_scale` | `1/128` (0.0078125) |
| `out` | Optional BF16 `[total_q, 8, 128]`; if given, the result is written into it and that tensor is returned |
| Result | BF16 `[total_q, 8, 128]` |

Domain: `1 <= B <= 64`; `1 <= q_len_b <= 16,384`; `k_len_b <= 262,144`. Other head counts, head
dimensions, extents, dtypes, non-causal calls and an LSE output are outside the contract.

Any GPU language, algorithm, fusion and internal layout is allowed. A layout the serving path does
not use natively may be converted inside the call; the conversion is part of the call's cost.

## Semantics

For request `b` with `q_len_b` queries and `k_len_b` keys (`k_len_b = seqused_k[b]` in the paged
form), query `i` sits at key position `p = k_len_b - q_len_b + i`.

- With `W = 511` if `window_size = (511, 0)` and `W` unbounded otherwise, query `i` attends to the
  keys `j` with `0 <= p - j <= W`.
- For query head `h` (0-7) and its KV head `floor(h/4)`, with `d = p - j`:

```
logit[i, j, h] = softmax_scale * <q[t, h], k[j, floor(h/4)]> + (rel_bias[t, h, d] if d < E else 0)
out[t, h]      = sum_j softmax_j(logit[i, ., h]) * v[j, floor(h/4)]        (t = cu_seqlens_q[b] + i)
```

The bias is indexed by query head and added to the already-scaled logit; keys at distance `E` or
more receive no bias. `q` and `rel_bias` arrive with the model's full-layer log-scaling
`tau(p) = 1 + 0.1 * ln(max((p + 1)/128,000, 1))` already applied; `tau` is not an operator input.

**Unreferenced inputs.** The result must not depend on:

1. paged slots at positions `>= seqused_k[b]` in request `b`'s last page, which hold arbitrary finite values;
2. page-table entries at indices `>= ceil(seqused_k[b]/128)`, which point at arbitrary pages;
3. with `window_size = (511, 0)`, pages all of whose 128 positions lie below `k_len_b - q_len_b - 511`;
4. pool pages no request references.

The whole pages of items 2-4 may hold any values, including NaN and +-inf: the engine recycles freed
sliding-window pages, and the page table can still point at them. The call must not modify any
input tensor.

## Numerical policy

Per call, over its valid rows, with `r` the FP64 reference computed from the stored BF16 values and
`o_bf16` the same attention evaluated with the operand precisions of FlashAttention-4's own test
pipeline (`S = bf16(q . bf16(k * scale))` with masked entries -inf, then `S <- bf16(S + bias)`,
`P = bf16(softmax(S))`, `O = bf16(P . V)`), and `eps = 2 * max|fl32(r) + 0.3 - 0.3 - fl32(r)|`
evaluated in FP32, a call passes iff:

- every valid output row is finite;
- `max|o - r| <= 2 * max|o_bf16 - r| + eps` (FlashAttention-4's forward test rule at
  `fa4-v4.0.0.beta15`, `rtol = 2` with `fwd_atol`, its FP32 reference replaced by FP64);
- every input tensor is bitwise unchanged.

`reference.py` implements `r` (and an FP32 variant) and `o_bf16`; `policy.py` implements the rule.

## Checking your implementation

```sh
python /app/eval/inkling_small/run.py rel-check --example paged-decode-local --output /tmp/rel.json
python /app/eval/inkling_small/rel_attention/check.py --output /tmp/rel-all.json   # every public example
```

`check.py` imports the committed `/app/inkling_rel_attention.py`, runs it on explicit cases,
validates the result's structure first (a BF16 tensor of the right shape on the inputs' device),
checks the inputs are unchanged, applies the policy and prints every metric. The public examples
(`../examples/rel_cases.json`, values from public seeds) cover both layer kinds, ragged prefill
with and without a prefix, paged decode and extend, and one sliding-window decode case whose
unreferenced pages hold NaN. They explain the contract; they do not exhaust it, and passing them
establishes only those cases.

## Efficiency

The operator is also timed through this entry point, for prefill and decode on both layer kinds;
each must be an efficient GPU implementation. The protocol is `measure.py`, identical for grading:

| Family | Workload (paged form) |
|---|---|
| `prefill_full`, `prefill_local` | one request, `q_len` 8,192 at positions 122,880-131,071 (`seqused_k` 131,072) |
| `decode_full`, `decode_local` | 16 requests, one query each, `seqused_k` 65,536 |

One B200, no server resident; 10 untimed calls, then 3 draws of 20 (prefill) or 100 (decode)
back-to-back calls; per-call GPU time from CUDA events around each draw's loop and per-call
synchronized wall time; the statistic is the median of the 3 draws. JIT compilation, warm-up and
allocation fall outside the statistic but inside the invocation limit (`../limits.json`).

```sh
python /app/eval/inkling_small/run.py rel-measure --family decode_full --output /tmp/t.json
```

## Attribution

`binding.py` records which GPU kernels the entry point launches: one ragged prefill call, one paged
prefill call and one paged decode call per layer kind under the PyTorch profiler, giving four
identity sets (`prefill_full`, `prefill_local`, `decode_full`, `decode_local`: the kernel names
launched inside the calls, minus memcpy/memset and PyTorch copy/fill kernels). Grading observes the
served model on all four ranks with the same profiler (prefill requests and graph-replayed decode)
and requires those computations there. Identity comes from observation, so renamed or differently
fused kernels are fine; a correct operator the served model does not run is not.

```sh
python /app/eval/inkling_small/run.py binding --output /tmp/ids.json
```
