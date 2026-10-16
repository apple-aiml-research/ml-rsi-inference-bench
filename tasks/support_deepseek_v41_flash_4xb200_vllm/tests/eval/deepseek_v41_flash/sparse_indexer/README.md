# The two-level sparse indexer: dependency operations, numerical contract and component measurement

DeepSeek-V4.1-Flash selects, for every query, the compressed history positions its sparse attention reads, in two
levels. The candidate-source indexer (layer 20) scores the visible compressed positions, gives each 8-position
block its highest score, always keeps the block holding the newest visible position, and publishes up to 2,048
blocks as a **candidate pool**. The candidate-consuming indexers (layers 24, 28, 32 and 36, compression ratio 1 in
this checkpoint) share layer 20's index keys and that pool; each computes **its own** scores with its own queries
and head weights, selects its top 512 positions **within the pool**, and native attention uses those selections.
The released `inference/` source in the checkpoint is the model's own statement of these semantics.

The work described here is part of the task: the four MXFP4 sparse MQA-logits operations of the supplied editable
dependency, and their use by native prefill, decode and target verification. Correctness and component efficiency
are required separately for prefill (packed) and for decode (paged), in addition to the whole-serving requirements
of the task.

## The supplied dependency

`/app/third_party/deepgemm_v41_sparse/` is the source of the runtime published as
`vllm.third_party.deep_gemm_v41_sparse`: `csrc/` (the operations' host code, the JIT runtime and the narrow
extension `csrc/python_api_sparse.cpp`), `deep_gemm/include/` (the JIT kernel headers), the DeepJIT and CUTLASS
headers under `third-party/`, and generic quantization utilities in `deep_gemm/utils/math.py`. It is built by
`/app/tools/build_deepgemm_v41_sparse.py`, which `../build.sh` runs after the engine's editable build; the build
writes the extension, the package's JIT include tree and `build-manifest.json` next to
`/app/vllm/third_party/deep_gemm_v41_sparse/__init__.py`. Rebuild after changing any file of the tree, the
package initializer or the builder. The package is separate from the engine's ordinary DeepGEMM provider
(`vllm.third_party.deep_gemm`); its configuration environment variables use the prefix `DSV41_SPARSE_` (the JIT
cache is `DSV41_SPARSE_JIT_CACHE_DIR`). Importing it creates no CUDA context.

As supplied, the four operations below are declared and registered with their argument names and defaults, and
each raises `NotImplementedError`. DeepSelect's top-k (`vllm._deepselect_C`, with
`vllm/model_executor/layers/indexer_topk.py`) is supplied and built with the engine;
`AttentionConfig.indexer_sparse_logits` is declared.

## Required dependency operations

```python
get_sparse_mqa_logits_metadata(cu_seq_len_k_start, cu_seq_len_k_end, num_kv_tokens,
                               sparse_kv_block_indices, qk_dtype, sparse_block_kv,
                               use_unaligned_ks=False) -> metadata
get_paged_sparse_mqa_logits_metadata(context_lens, block_table, indices, page_kv,
                                     sparse_kv_block_indices, qk_dtype, sparse_block_kv) -> metadata
fp8_fp4_sparse_mqa_logits(q, kv, weights, metadata, num_max_sparse_blocks, sparse_block_kv,
                          use_unaligned_ks=False) -> logits
fp8_fp4_paged_sparse_mqa_logits(q, kv_cache, weights, metadata, num_max_sparse_blocks,
                                sparse_block_kv) -> logits
```

For this model `qk_dtype` is `torch.int8` (packed MXFP4) and `sparse_block_kv` is 8. `q` is `(q_values, q_scales)`;
the packed call takes `kv = (k_values, k_scales)` and the paged call the fused cache. The metadata is opaque: any
bytes, shape, schedule or language, created fresh for changed inputs. The logits are BF16 `[rows,
num_max_sparse_blocks * 8]`; only valid candidate columns are meaningful. Packed rows: `cu_seq_len_k_start/end`
give each row's absolute `[ks, ke)` in the K workspace, and in unaligned mode block id `i` addresses the eight
positions starting at `i * 8 + ks % 8`; column `j * 8 + o` then scores request-local position
`(sparse_kv_block_indices[r, j] - ks // 8) * 8 + o`. Block ids in a row are sorted ascending, and a row with fewer
valid blocks repeats a legal block in the unused slots. The metadata and compute calls take the same
`use_unaligned_ks`. Paged rows: blocks are context-relative (starts are zero), `context_lens` is each row's own
length, `block_table` holds each row's physical page ids (rows of one request repeat that request's page-table row)
and `indices` maps each row to its request; the rows of one request are consecutive. Implementations may adapt
these exports over another kernel, but diagnostic calls and native serving must exercise the same underlying
computation.

## Geometry and numerical semantics

| | |
|---|---|
| Heads, head dimension | 32, 128 |
| Candidate block, pool, top-k | 8 positions, 2,048 blocks, 512 positions |
| Visible compressed positions per query | 0–32,768 |
| Q, K | MXFP4: E2M1 values packed two per byte (low nibble first), UE8M0 scales |

Packed Q is int8 `[R, 32, 64]` with scales int32 `[R, 32]`; packed K is int8 `[N, 64]` with scales int32 `[N]`;
paged Q is `[R, 1, 32, 64]` with scales `[R, 1, 32]`. The low three bits of a nibble select the magnitude
`[0, 0.5, 1, 1.5, 2, 3, 4, 6]` and bit 3 is the sign. Each int32 scale word holds four bytes, low byte first; byte
`i` scales coordinates `[32 i, 32 i + 32)` by `2 ** (byte - 127)`. Weights are BF16 `[R, 32]` with a contiguous inner
dimension; the fused query step must produce them as the FP32 head weights rounded **once** to BF16. The score is

```
score[r, t] = sum_h float(weight[r, h]) * max(sum_d dequant(q[r, h, d]) * dequant(k[t, d]), 0)
```

The paged cache is uint8 with logical shape `[physical_pages, page_kv, 1, 68]`: each page stores all `page_kv * 64`
value bytes and then `page_kv * 4` scale bytes (not interleaved 68-byte records); the page stride may exceed that
and is a multiple of 512 bytes. Inputs within this contract dequantize to finite values and give finite scores.
An empty batch returns empty outputs without a GPU launch; an empty row selects nothing (all -1) without reading a
key.

## Selection semantics

A consumer row's selectable positions are the positions of its pool's valid blocks that are visible to it
(`8 b + o` below its visible length); pool padding (-1) and invisible positions are excluded, and pool order is
arbitrary. The consumer's selection is its 512 highest-scoring selectable positions (all of them when fewer),
written as request-local int32 positions, distinct, with -1 padding; any choice among equal scores is valid. Below
16,385 visible positions every block fits in the pool and a consumer may select from the whole visible history;
beyond that the pool restricts it. Speculative target verification gives each query row its own visible length
(one row per verified token, up to six rows per request), and a request's rejected draft positions never become
visible. Selections must follow each step's current lengths, pools, page tables and request slots: metadata may be
shared by the four consumers within a step and must be rebuilt when its inputs change, including in replayed CUDA
graphs.

## Comparison rules

These define the admissible arithmetic; evaluation applies them to held-out inputs within the scope above.

- Structure is exact: output shape and dtype, request mapping, membership in the selectable positions,
  distinctness, the number of valid entries and -1 padding.
- Logits: per row, the absolute error against the FP32 score divided by that row's largest absolute FP32 score
  over valid candidates must have a 99th percentile below 0.01 and a maximum below 0.02 (an all-zero reference row
  requires zero scores). Finite inputs require finite valid outputs; padding values are ignored and never selected.
- Top-k over known BF16 logits: the multiset of selected scores and its cardinality must match exactly.
- Selections whose scores are not materialized: with `delta = 0.02 * max_abs(reference scores)` for that row and
  consumer, no excluded selectable position may score more than `2 * delta` above the smallest selected score.
- The source pool: with `delta` over all of a row's visible source scores, after keeping the newest block, no
  unselected block may exceed the lowest selected non-forced block by more than `2 * delta` in block-maximum
  score; if every visible block fits, the pool holds all of them.

## The component-measurement interface

`/app/vllm/model_executor/layers/sparse_indexer_eval.py` declares two entry points (supplied as stubs):

```python
prefill(q, q_scale, k_quant, k_scale, weights, k_start, k_end, candidate_blocks, out, *, workspace=None)
decode(q, q_scale, kv_cache, weights, context_lens, block_table, row_indices, candidate_blocks, out, *,
       workspace=None)
```

Each call is one step of the four consumers over one shared K (packed workspace or paged cache) and one shared
pool: Q, scales and weights carry a leading consumer dimension of four (`q` int8 `[4, R, 32, 64]` or
`[4, R, 1, 32, 64]`, `q_scale` int32 `[4, R, 32]` or `[4, R, 1, 32]`, `weights` BF16 `[4, R, 32]`);
`k_start`, `k_end`, `context_lens` and `row_indices` are int32 `[R]`; `candidate_blocks` is int32 `[R, 2048]`;
`block_table` is int32 `[R, ceil(max_visible_length / page_kv)]`, padded with unused entries, with each request's
page-table row repeated for each of its rows; `out` is int32 `[4, R, 512]`. Q, scales, K and weights are
contiguous; candidate and page-table views may have a strided outer dimension with contiguous rows. Paged rows may
arrive with requests interleaved or reordered between steps; the entry point may regroup them and must write `out`
in the caller's row order. The returned workspace holds reusable allocations only: every call is a fresh step and
must honor its current tensor values. The entry points call the implementation native serving uses and do only
the argument adaptation measurement needs; conversion to another internal layout belongs inside them.

The component domain is packed batches through 512 rows and paged batches through 64 rows, including six-row
request groups, at 0–32,768 visible positions. Native serving must also handle the recipe's actual scheduler
batches, by chunking or a larger kernel invocation; the component ranges do not permit rejecting a legal native
batch.

## Measurement

`measure.py` times one call of an entry point as the work unit: candidate and metadata preparation, the four
consumers' scores, selection and remapping, and all adaptation and scratch work, from quantized inputs to the four
selection buffers. It warms each case with ten work units, fixes a repetition count by doubling until a pilot
block reaches 200 ms of synchronized wall time (at most 2,048), then records three blocks, each by wall time around
device synchronization and by CUDA events on the execution stream, rotating four preallocated input sets between
work units. It prints continuous per-work-unit wall and GPU times; it declares no pass. Evaluation measures the
same work unit in the same way on held-out workloads of this domain, against the reference implementation on the
same GPU, and checks the selections the measured work units write under the comparison rules above.

```sh
python /app/eval/deepseek_v41_flash/sparse_indexer/measure.py --phase prefill \
  --cases /app/eval/deepseek_v41_flash/sparse_indexer/examples.json --output /tmp/sparse-measure-prefill.json
python /app/eval/deepseek_v41_flash/sparse_indexer/measure.py --phase decode \
  --cases /app/eval/deepseek_v41_flash/sparse_indexer/examples.json --output /tmp/sparse-measure-decode.json
```

One measurement invocation is bounded at 1,800 s (the operation limits in `../README.md`).

## Examples and the checking script

`examples.json` holds four public records: a packed case with four rows at 513 visible positions and a strided
candidate view, a paged case with three rows grouped 1 + 2 at `page_kv` 64 and unequal visible lengths (both with
their expected selections), and the two measurement setups used above (packed, 128 rows at 20,480; paged, 16 rows
at 20,480). `contract.py` holds the typed protocol and loads the implementation, `reference.py` the FP32
mathematics of this document, and

```sh
python /app/eval/deepseek_v41_flash/sparse_indexer/check.py --cases /app/eval/deepseek_v41_flash/sparse_indexer/examples.json
```

runs the small examples through the four operations and the two entry points and reports every discrepancy with
its continuous errors. A clean report establishes only the checks it performed on these inputs.

## Optional binding description

Evaluation observes the source indexer, the consumer indexers and the native FlashMLA calls of the served model
directly. An implementation that reorganizes those boundaries may describe its equivalents by providing
`vllm.model_executor.layers.sparse_indexer_audit.get_bindings(model)` with the read-only schema of `audit.py`. The
description is optional; it maps actual calls to views of their inputs and outputs and computes nothing itself.
