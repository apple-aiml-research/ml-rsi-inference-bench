# DeepSeek-V4-Flash kernel interfaces

This task includes implementing five GPU operations of DeepSeek-V4-Flash's attention path: sparse attention in a BF16
component form and in the packed mixed-precision form the deployment serves, C4/C128 KV compression with its
persistent state, the post-compression normalization and rotary transform, and the two cache-store formats. Their
optimized implementations are not supplied. This file specifies the callable interfaces, their mathematics, the packed
formats, the supported domain, the numerical tolerances, the component workloads and the clocks. `check.py` beside it
implements the comparison and timing procedure the grader uses and runs the fixed examples in `examples/`.

Equivalent internal layouts, fusion, thin adapters and CUDA, Triton or TileLang implementations are accepted through
the callables below; an adapter's conversion and dispatch work is part of the measured operation. The packed attention
and the compression/cache callables must perform the computation native serving uses: evaluation calls them directly
with its own inputs and also observes serving (see "Execution attribution"). No private class, pool type or kernel name
is required.

## The callables

| Callable | Where |
|---|---|
| `flash_mla.flash_mla_sparse_fwd` | the vendored FlashMLA package, `/app/third_party/flash_mla` |
| `flash_mla.flash_mla_with_kvcache` (sparse use) | the same |
| `sglang.jit_kernel.deepseek_v4.compress_forward` | the engine, with its existing `compress_plan` |
| `sglang.jit_kernel.deepseek_v4.compress_fused_norm_rope_inplace` | the engine |
| `sglang.jit_kernel.deepseek_v4.fused_store_cache` | the engine |

The signatures, argument meanings, dtypes and return layouts are the existing ones in the starting tree: FlashMLA's
`flash_mla/flash_mla_interface.py` and SGLang's `python/sglang/jit_kernel/deepseek_v4.py`. The sections below state
what each computes.

### BF16 sparse attention: `flash_mla_sparse_fwd(q, kv, indices, sm_scale, d_v=512, attn_sink=None, topk_length=None)`

- `q` `[s_q, 64, 512]` BF16; `kv` `[s_kv, 1, 512]` BF16, one KV head shared by the 64 query heads; keys and values are
  the same 512 coordinates (`d_v = 512`).
- `indices` `[s_q, 1, K]` int32: the selected keys of each query. An invalid index is `-1` (the pinned interface also
  treats an index `>= s_kv` as invalid). Order is free; a repeated index contributes once per occurrence.
- `topk_length` `[s_q]` int32, optional: only the first `topk_length[i]` entries of row `i` participate.
- `attn_sink` `[64]` FP32, optional: one logit per query head in the softmax denominator.
- Returns `(output [s_q, 64, 512] BF16, max_logits [s_q, 64] FP32, lse [s_q, 64] FP32)`.

### Packed sparse attention: `flash_mla_with_kvcache(q, k_cache, None, None, 512, sched, None, softmax_scale, False, True, indices=..., attn_sink=..., extra_k_cache=..., extra_indices_in_kvcache=..., topk_length=..., extra_topk_length=...)`

- `q` `[b, s_q, 64, 512]` BF16 (`s_q` is 1 for decode, the number of verified tokens for speculative verification).
- `k_cache` a MODEL1 pool (below), viewed `[num_pages, page_size, 1, 584]` over its padded page stride.
- `indices` `[b, s_q, K_main]` int32: physical token indices into the main pool, `page * page_size + offset`; `-1` is
  invalid. `topk_length` `[b]` int32: the valid length of every query row of request `b`; entries at or past it are
  ignored whatever they hold.
- `extra_k_cache`, `extra_indices_in_kvcache` `[b, s_q, K_extra]`, `extra_topk_length` `[b]`: an optional second pool
  with its own page size, indices and lengths. Main and extra keys enter the same softmax.
- `sched` comes from `flash_mla.get_mla_metadata()`; one may be reused only while the shapes and the length tensors'
  values are unchanged.
- Returns `(output [b, s_q, 64, 512] BF16, lse [b, 64, s_q] FP32)`.

### The attention both compute

For one query row and head, over the ordered multiset of valid selected keys `k_i` (both pools for the packed form):

```text
z_i = softmax_scale * <q, k_i>
m   = max(sink, max_i z_i)                      (sink = -inf when absent)
o   = sum_i exp(z_i - m) * k_i / (sum_i exp(z_i - m) + exp(sink - m))
lse = log(sum_i exp(z_i))                       (natural log; the sink excluded)
max_logits = max_i z_i                          (the sink excluded)
```

An empty valid selection returns `o = 0`, `lse = +inf` and `max_logits = -inf`. These sentinels are part of the
interface. For the packed form the key values are the decoded MODEL1 entries: all 512 coordinates, the dequantized
448 and the 64 BF16 rotary coordinates.

### MODEL1, the packed K/V format

- One logical entry has 512 coordinates: 448 E4M3 values in seven groups of 64, then 64 BF16 rotary values. Each
  group of 64 has one E8M0 scale (a power of two, byte `e` meaning `2 ** (e - 127)`); a value is its E4M3 number times
  its group's scale. All 512 coordinates are the key and the value.
- A page holds `page_size` entries. Within a page, the `page_size` data rows of 576 bytes (448 E4M3 bytes, then the 64
  BF16 values as 128 bytes) come first, then the `page_size` scale rows of 8 bytes (seven E8M0 bytes and one byte with
  no meaning). The data rows and the scale rows are not interleaved: 584 bytes per entry is not a record size.
- The page stride is `ceil(584 * page_size / 576) * 576` bytes; the 4-D view does not change that stride.
- Physical index `t` is page `t // page_size`, offset `t % page_size`.

The deployment's pools: the uncompressed (SWA) pool has pages of 256 entries; the C4 compressed pool 64; the C128
compressed pool 2. The target's C4 and C128 layers attend to the SWA pool (main) plus their compressed pool (extra);
its uncompressed layers and the draft model attend to the SWA pool alone.

### The indexer cache format

The C4 indexer's compressed keys: 128 E4M3 values per entry and one FP32 scale per entry. Within a page, the
`page_size` rows of 128 bytes come first, then the `page_size` FP32 scales (4 bytes each): `132 * page_size` bytes per
page. A value is its E4M3 number times the entry's scale.

### Compression: `compress_forward(state, x, ape, indices, plan, extra, head_dim=D, compress_ratio=r)`

The two compressors fold groups of `r` tokens into one entry. `x` holds one row per token of the call (a ragged batch
of requests, request after request), FP32:

- **C4** (`r = 4`, `D = 512` for the attention cache, `D = 128` for the indexer): a row is
  `[kv_overlap, kv, score_overlap, score]`, each `D` wide; `ape` is `[8, D]`. At zero-based position `p` with
  `(p + 1) % 4 == 0`, the entry combines the overlap channels of tokens `p-7 .. p-4` and the fresh channels of tokens
  `p-3 .. p`; before the first full group (`p < 7`) the overlap part is absent. `ape` rows 0-3 bias the overlap tokens'
  scores, rows 4-7 the fresh tokens'. For each coordinate independently the eight (or four) `score + ape` values are
  softmax-normalized and weight the corresponding `kv` values.
- **C128** (`r = 128`, `D = 512`): a row is `[kv, score]`; `ape` is `[128, D]`. At `(p + 1) % 128 == 0` the entry is,
  per coordinate, the softmax of the 128 `score + ape` values of tokens `p-127 .. p` (`ape` row `j` biasing token
  `p-127+j`) weighting their `kv` values.

The output has one row per token (`[num_tokens, D]`); only the rows at the compression events above carry values:
in a prefill/extension call the row of the token at position `p` with `(p + 1) % r == 0`, in a one-token decode call
the row of a request whose length `L` (including this token) satisfies `L % r == 0`. Other rows are unspecified.

**The persistent state.** `state` is the compressor's state pool, FP32 `[windows, r, 4 * D]` (C4) or
`[windows, r, 2 * D]` (C128). A request's tokens are addressed through its SWA pages: position `pos` belongs to the
window `(page(pos) * ring + pos % ring) // r`, `page(pos)` the physical SWA page holding `pos`, `ring` 16 for C4 and
256 for C128 (the speculative deployment's ring sizes), negative positions clamped to 0. The window contents are the
implementation's own: evaluation never compares them, and a window starts as whatever the pool held (the engine does
not clear it). A call may change only the windows of its own requests' histories.

`indices` `[b]` int32 is each request's write window (the window of `((seq_len - 1) // r) * r`). `extra` is, for a C4
prefill/extension, `[b, 4]` = (load-overlap window, load window, write-overlap window, write position) with the load
windows those of the group holding the request's last token before the call and of the group before it (the write
windows likewise for its last token after the call); for a C4 decode `[b, 1]` = the write-overlap window; for a C128
prefill/extension `[b]` = the load window; for a C128 decode none. `plan` is `compress_plan(r, num_tokens, seq_lens,
extend_lens, device)` (`extend_lens` `None` for a decode). `check.py`'s `prefill_metadata` and `decode_metadata`
compute exactly these from the request's pages and lengths.

The position conventions of the three modes:

| Mode | First position | Exclusive end |
|---|---:|---:|
| One-token decode, engine length `L` including the token | `L-1` | `L` |
| Speculative verification of `D` tokens after an accepted history of length `L` | `L` | `L+D` |
| Extension ending at `L` with `E` tokens | `L-E` | `L` |

Every emission depends only on the request's accepted logical history. After a verification step in which only the
first `a` of the `D` verified tokens are accepted, the next call continues from length `L+a`, and its emissions must
equal those of the history without the rejected tokens. A one-shot extension, the same tokens in several extensions,
and extension followed by decodes all give the same emissions.

### `compress_fused_norm_rope_inplace(kv, weight, eps, freq_cis, plan)`

In place, on the rows `plan` marks as compression events: RMS normalization with the supplied `weight` `[D]` and
`eps`, `y = x * rsqrt(mean(x^2) + eps) * weight`, then the rotary transform of the last 64 coordinates as interleaved
pairs `(y[2j], y[2j+1])`, rotated by `freq_cis[pos, j]` (complex, `[max_pos, 32]`). `pos` is the compressed group's
start: `p + 1 - r` for the event at position `p` (`L - r` for a decode at length `L`). Other rows are unspecified.

### `fused_store_cache(input, cache, indices, page_size=..., type="flashmla" | "indexer")`

Writes rows into a MODEL1 pool (`type="flashmla"`, `input` `[n, 512]` FP32) or an indexer pool (`type="indexer"`,
`input` `[n, 128]` BF16) at physical destinations `indices` `[n]` int32. Destination 0 is the pool's reserved scratch
slot (rows with no live destination are directed there); its contents have no meaning. Destinations are never
negative. Every other byte of the pool keeps its value.

The quantization, per group of 64 (MODEL1) or of 128 (indexer):

```text
raw   = max(1e-4, max |x|) / 448
MODEL1:  scale = the smallest power of two >= raw   (stored as its E8M0 byte)
indexer: scale = raw                                 (stored as FP32)
stored value = E4M3(x / scale), round to nearest even
```

MODEL1's 64 rotary coordinates are stored as BF16 (round to nearest even). The scale row's eighth byte has no meaning.
This rule is not FlashMLA's `tests/quant.py`, which divides by 448 before applying the 1e-4 floor. The indexer's FP32
scale and quotient may use any equivalent FP32 division lowering (`x / scale` or `x * (1 / scale)`): its scale is
accepted within one ULP of the formula, and an E4M3 value whose unrounded quotient lies at a rounding tie (within a
relative `2 ** -21` of the midpoint between two E4M3 values) may take either neighbour. MODEL1's power-of-two scale makes
its quotient exact.

## Supported domain

| Operation | Domain |
|---|---|
| BF16 attention | query counts through 4,096; selected lists through 512 entries; KV pools through 8,192 entries; optional valid lengths and sink |
| Packed attention | `s_q` 1 and 4; main capacity 128 in the SWA pool (pages 256); C4 extra selections through 512 (pages 64); C128 extra selections as long as the compressed history (pages 2; 2,048 entries at 262,144 tokens); full and partial valid lengths; `-1` holes |
| Compression | C4 D512, C4 D128, C128 D512; extensions, decodes and verification steps in any mix; batches of requests of different lengths |
| Store | both formats at their deployment page sizes; destinations crossing pages; the scratch slot |

NaN inputs, positive physical addresses outside a pool, negative store destinations and compression modes other than
these are outside the domain.

## Numerical tolerances

For finite elements `abs_err = |answer - reference|` and `rel_err = abs_err / (|reference| + 1e-6)`, compared inclusively.
An attention element passes when EITHER bound holds. For an output tensor the symmetric cosine difference
`1 - 2 * sum(a * r) / (sum(a^2) + sum(r^2))` is computed in FP64 over the case, and is 0 when `sum(r^2) < 1e-12`; the
elementwise rule applies to every element regardless. Shapes, dtypes and the specified sentinels (`+inf` LSE, `-inf`
max logits of an empty selection) must match exactly before any finite comparison; any other non-finite value fails.
Each case is judged alone; nothing is pooled across cases or requests.

| Output | Acceptance |
|---|---|
| BF16 attention output | every element `abs_err <= 8e-4` or `rel_err <= 3.01/128`; cosine difference `<= 7e-6` |
| BF16 max logits, LSE | every element `abs_err <= 1e-6` or `rel_err <= 2.01/65536`; sentinels exact |
| Packed attention output | every element `abs_err <= 1e-3` or `rel_err <= 2.01/128`; cosine difference `<= 5e-6` |
| Packed LSE | every element `abs_err <= 1e-6` or `rel_err <= 8.01/65536`; sentinels exact |
| Compression and post-norm/RoPE values | every valid element `abs_err <= 5e-3 + 5e-3 * |reference|`, all finite |
| MODEL1 store | exact destinations, E4M3 bytes, E8M0 bytes and BF16 rotary bits on inputs without rounding ties; unrelated bytes unchanged |
| Indexer store | exact destinations and E4M3 bytes (either neighbour at a rounding tie); the FP32 scale within one ULP of the formula; unrelated bytes unchanged |

A store that follows a computed intermediate is checked against the quantization of that intermediate as supplied,
while the intermediate itself must meet the compression tolerance. `check.py`'s `compare` implements this table.

## Component efficiency

Each family below is measured on one B200 with no model server resident, in a worker process of the arm's own (one
per phase), from the arm's own build and JIT caches, against the human reference implementation measured the same way
on the same device. For each family two
latencies are formed -- synchronized wall time and CUDA-event GPU time -- and both are compared; a fast family does
not compensate for a slow one. The pass threshold of that comparison is not published.

| Phase | Families |
|---|---|
| Prefill | BF16 attention, Q 4,096 / KV 8,192 / K 512; packed main-only, main+C4 and main+C128 with B 4,096, `s_q` 1; C4 D512, C4 D128 and C128 D512 over 4,096 projected tokens of one request |
| Decode | packed main-only, main+C4 and main+C128 with B 64, `s_q` 1 (the engine's flattened 16 x 4 verification rows); C4 D512, C4 D128 and C128 D512 over 16 requests, each from a 4,096-token accepted prefix, one cycle = 128 advancing one-token steps |

Packed families use main capacity 128 and extra capacity 512, the deployment's page sizes, 8,192 logical entries per
present pool, full valid lengths, finite non-constant sinks and per-row selections. A compression invocation includes
its normalization/RoPE and cache store.

The procedure (`check.Measure`): ten untimed warm-up invocations, then three draws; a draw is 100 invocations for
attention, ten 4,096-token invocations for compression prefill, ten complete 128-step cycles for compression decode.
Before every invocation the logical input state is restored outside the timed interval, identically in both arms.
A draw's latency is the mean over its invocations of each clock; a family's latency is the median of its three draws.
Prefill is measured eagerly. Decode captures one invocation (or one 128-step cycle) in a CUDA graph after warm-up and
times its replays; a capture failure is a failed measurement. Warm-up, JIT and allocations count toward the operation's
limit, not toward the latency. Required conversions, packing, dispatch and device metadata updates are part of the
invocation.

`python check.py --tree /app --mode measure --workload prefill` (or `decode`) runs these families on inputs from fixed
public seeds; evaluation uses its own inputs of the same shapes.

## Build

The vendored FlashMLA source at `/app/third_party/flash_mla` is part of the committed submission. `build.py` is the
offline build both arms use: it runs the tree's editable engine install, clears stale vendor products and runs
`setup.py build_ext --inplace` in `third_party/flash_mla` (`FLASH_MLA_DISABLE_SM90=1`, `MAX_JOBS=8`,
`NVCC_THREADS=4`), then checks that `flash_mla` and its extension import from the tree. The generic CUTLASS it builds
against is at `/opt/infb-deps/flashmla-cutlass` (linked as `csrc/cutlass`). `python build.py --tree /app` is the
evaluation's build step.

## Execution attribution

`python check.py --tree /app --profile-map /tmp/component-map.json` runs one invocation of every workload family under
the PyTorch profiler and records, per role, the kernels the public calls launched by base name (the name without
template arguments or parameters, `(anonymous namespace)` qualifiers kept), PyTorch's own and the CUDA libraries'
generic kernels (the `at::native::`, `at::cuda::`, `c10::`, `cub::` and `thrust::` namespaces) left out. Evaluation builds this map from
the tree's own component runs and requires, in ordinary warm serving on every tensor-parallel rank, that the target's
replayed verification graphs run kernels of every role the map attributes -- packed attention, C4 compression (D512
and D128) and C128 compression -- and that the draft model runs the packed attention. `../tests/served_recipe.py
--component-map /tmp/component-map.json` reports the served attention against the map. Specializations of the same
kernels, other tilings and fused kernels are expected; a kernel of a supplied library the public call composes counts
as that call's.

## Clocks

| Operation | Limit |
|---|---:|
| The build (`build.py`), dispatch through usable imports | 3,600 s |
| One component family measurement, including setup, compilation, warm-up, repetitions and readback | 1,800 s |
| One component correctness node, every case | 1,800 s |

`../limits.json` carries these (`kernel_build`, `invocation`, `component_check`) with the other operation limits.

## Examples

| Example | What it illustrates |
|---|---|
| `bf16` | Q 16, selected capacity 128: BF16 output and statistics under the tolerances |
| `packed` | B 2, `s_q` 1; main-only, main+C4 and main+C128 at the deployment's page sizes |
| `c4-core`, `c4-indexer`, `c128` | 256 projected tokens: an extension, then one-token continuation; the valid emitted rows after normalization/RoPE, and the stored cache bytes against the store rule applied to those rows |

```sh
python /app/eval/deepseek_v4_flash/kernels/check.py --tree /app --example packed --mode correctness
python /app/eval/deepseek_v4_flash/kernels/check.py --tree /app --example c128 --mode measure
```

An example reports every discrepancy and whether the example conforms. Passing the examples, or any timing, does not
establish acceptance: evaluation uses held-out inputs, cases and combinations within the domain above.
