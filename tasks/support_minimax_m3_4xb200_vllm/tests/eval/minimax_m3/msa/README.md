# MiniMax Sparse Attention: the operator contract

MiniMax-M3's sparse layers (3-59) use MiniMax Sparse Attention (MSA): an index branch scores 128-token key blocks and
selects 16 of them per query and GQA group, and block-sparse attention attends only to the selected blocks. The
mathematics is defined in sections 2-3 of the MSA paper (`/public/msa/MiniMaxSparseAttention.pdf`) and by the
model-author implementation (`/public/reference-impl/modeling_minimax_m3_vl.py`, the `MiniMaxM3VLIndexer` and the
attention mask it builds); `reference.py` beside this file states the four operators in plain PyTorch.

The implementation is committed in `/app`. It exposes the operators through `/app/minimax_m3_msa.py`, and the served
model must run the same GPU computation.

## Entry points

`/app/minimax_m3_msa.py` exposes exactly:

| Callable | Arguments → result |
|---|---|
| `create_cache(num_pages: int, kv_dtype: torch.dtype, device) -> object` | One layer's per-rank main and index caches for `num_pages` 128-token pages. `kv_dtype` is `torch.bfloat16` or `torch.float8_e4m3fn`; the index cache is BF16 in both deployments |
| `cache_views(cache) -> (k, v, index_k)` | Three tensors of shape `[num_pages, 128, 128]` (K and V in `kv_dtype`, index keys BF16) that alias the storage the four operators read: a write through a view is what the operators see |
| `select_prefill(cache, index_q, block_table, query_start_loc, seq_lens, context_lens) -> int32 [T, 16]` | Block selection for a variable-length prefill batch |
| `select_decode(cache, index_q, block_table, seq_lens) -> int32 [R, 16]` | Block selection, one query per request |
| `attend_prefill(cache, q, selection, block_table, query_start_loc, seq_lens, context_lens) -> bf16 [T, 16, 128]` | Block-sparse attention for a prefill batch |
| `attend_decode(cache, q, selection, block_table, seq_lens) -> bf16 [R, 16, 128]` | Block-sparse attention, one query per request |

Inputs are one tensor-parallel rank's (TP4: one GQA group per rank): `index_q` BF16 `[T or R, 128]` and `q` BF16
`[T or R, 16, 128]` are post-norm, post-RoPE queries of the rank's GQA group; the caches hold post-norm, post-RoPE keys.
`block_table` int32 `[R, max_pages]` maps logical block `b` of request `r` to a physical page. `query_start_loc` int32
`[R+1]`; `context_lens` int32 `[R]` are the tokens before this chunk; `seq_lens` int32 `[R]` include the chunk's
queries, whose keys are already in the cache. Prefill query `i` of request `r` sits at position `context_lens[r] + i`; a
decode query at `seq_lens[r] − 1`. A decode row with `seq_lens[r] = 0` is graph padding. FP8 caches hold
`float8_e4m3fn` values read with unit scale (`--kv-cache-dtype fp8`; the checkpoint carries no KV scales). The attention
scale is `1/√128`. The entry points must not modify the caches.

Any GPU language, algorithm, fusion and internal layout is allowed. A layout that cannot be expressed as strided views
may convert inside the entry points; that conversion is part of the measured time.

**Selection format.** For a query at position `p` with `V = ⌊p/128⌋ + 1` visible blocks, the first `n = min(16, V)`
slots hold distinct logical block indices in `[0, V)`, in any order, including `⌊p/128⌋`; the remaining slots are −1.
Padded decode rows are all −1. Semantics (paper §3.1, the checkpoint's `sparse_attention_config`): the block score is the
maximum over the block's causally visible tokens of `index_q · index_k`; the query's own block is always selected;
there is no forced first block; the 16 query heads of the group share the selection.

## Numerical policies

They define admissible implementations (`policy.py`).

**Selection**, per valid row: the format above holds; and with `M_b` the block score computed exactly (FP64) from the
stored BF16 values, `S` the selected blocks other than the local one and `U` the visible unselected blocks, if `U` is
non-empty then `min_{b∈S} M_b ≥ max_{b∈U} M_b − 10⁻³ · max(1, |max_{b∈U} M_b|)`. This admits any tie order and FP32
summation order.

**Attention**, per case, over the attended set `A = {j ≤ p : ⌊j/128⌋ selected}` with the stored values (E4M3 read with
unit scale) and the FP64 reference `r`:

| KV cache | Rule |
|---|---|
| BF16 | `max|o − r| ≤ 2 · max|o_bf16 − r| + 10⁻⁶`, where `o_bf16` evaluates the same reference with BF16 operands: BF16 score matmul, softmax in FP32 rounded to BF16, BF16 P·V matmul |
| FP8 E4M3 | `max|o − bf16(r)| ≤ max(0.125, 4 · max|bf16(r) − r|)` |
| Either, equal logits | When every logit of a row is equal, `|o − r| ≤ 2⁻⁷ · max_A|v|` elementwise: the output is the mean of the attended values up to output rounding |

Every valid row is finite. Selections passed to the attention operators are duplicate-free (the format).

## Checking an implementation

```sh
python /app/eval/minimax_m3/run.py msa-check --example prefill-bf16 --output /tmp/msa.json
```

`msa-check` installs each case's cache contents through `create_cache`/`cache_views`, calls the entry point, validates
structure first (type, device, dtype, shape, finite valid rows, caches unmodified), applies the policies above against
the FP64 reference and reports every metric. Examples (`examples/msa_cases.json`, N(0,1) inputs from public seeds, both
KV dtypes): prefill requests `(context, queries)` = `(0, 300)` and `(2500, 129)`; decode `seq_lens`
`[1, 129, 2049, 5000, 0]`. Each example runs the selection operator and then the attention operator on the reference
selection of the same inputs. The examples explain the contract; they do not exhaust it, and passing them says only that
those checks passed.

## Efficiency of the operators

The operators must be efficient GPU implementations. They are measured through the contract (`msa-measure`,
`measure.py`), one B200, no model server resident:

| Phase | Families | Workload |
|---|---|---|
| Prefill | `select_prefill`, `attend_prefill_bf16`, `attend_prefill_fp8` | one request, 8,192 queries at positions 122,880–131,071 (1,024 visible blocks at the last query) |
| Decode | `select_decode`, `attend_decode_bf16`, `attend_decode_fp8` | 16 requests, one query each at position 65,535 |

Protocol, identical for every implementation measured: 10 untimed calls, then 3 draws of 20 (prefill) or 100 (decode)
back-to-back calls; per-call GPU time from CUDA events around each draw's loop and per-call synchronized wall time; the
statistic is the median of the 3 draws, for each clock. The attention families receive the reference selection of the
same inputs. Compilation, warm-up and allocation count toward the invocation's time limit, not toward the statistic.
Each family is judged on its own; a fast family does not compensate for a slow one.

```sh
python /app/eval/minimax_m3/run.py msa-measure --family attend_decode_fp8 --output /tmp/t.json
```

## Context scaling

Attention time per query must not grow with context length beyond the selected blocks (`msa-sparsity`,
`sparsity.py`), BF16 KV cache, GPU-time medians by the protocol above, with the reference selections (16 blocks per
query):

- `R_dec = t(attend_decode, 16 requests at 262,144 tokens) / t(attend_decode, 16 requests at 16,384 tokens)`, one query
  per request at the last position;
- `R_pre = t(attend_prefill, 8,192 queries ending at 262,144) / t(attend_prefill, 8,192 queries ending at 32,768)`.

And directly, with a poisoned cache: after a case is installed, every page no query of the case selects is
overwritten with NaN in K and V (FP8: the byte `0x7F`), and the case is called once. The attention operators read
only the selected blocks, so the call must still meet the case's correctness policy; arithmetic that touches an
unselected key or value returns NaN or changed values. Geometries (`sparsity.POISON`): decode at 16 requests of
16,384, 65,536 (BF16 and FP8) and 262,144 tokens; prefill of 256 queries at the end of a 131,072-token request
(BF16 and FP8).

```sh
python /app/eval/minimax_m3/run.py msa-sparsity --output /tmp/s.json [--poison]
```

## What the attribution protocol observes

1. **Component side.** One call of each entry point is profiled with the PyTorch profiler (CUDA activity). The entry
   point's identity set is the set of GPU kernel names launched inside the call, minus memory copies and fills
   (`Memcpy`/`Memset` activity and PyTorch copy and fill kernels); the attention sets are kept separately for BF16 and FP8.
2. **Serving side.** With the server's torch profiler armed, prompts are served between `/start_profile` and
   `/stop_profile`, and each of the four ranks' traces is read.
3. **On every rank**, each of the four entry points must have at least one identity-set kernel in the trace (the
   attention sets of the deployment's KV dtype), the `attend_decode` computation must run on the decode steps of every
   sparse layer, and CUDA graphs must be replayed during decode. A static import, a registration or a kernel name
   printed in a log is not evidence. Identity comes from observation, never from fixed names: a renamed or differently
   fused implementation is attributed by what it launches.

```sh
python /app/eval/minimax_m3/run.py binding --traces <directory of rank traces> --output /tmp/b.json
```

## Limits

Each component correctness, timing or scaling invocation has 1,800 seconds from dispatch to record, including
compilation and warm-up (`../limits.json`).
