# The `attend` operator contract

The model's paged attention is exposed for measurement through `/app/diffusion_gemma_attend.py`,
which defines exactly three callables:

| Callable | Arguments → result |
|---|---|
| `create_cache(layer_kind: str, tp_size: int, capacity_tokens: int, device) -> object` | One layer's per-rank K and V caches with room for at least `capacity_tokens` tokens. `layer_kind` is `"sliding"` or `"full"`; `tp_size` (1 or 2) selects the per-rank head counts |
| `cache_views(cache) -> (k, v)` | Two BF16 tensors `[num_blocks, block_size, num_kv_heads, head_dim]` that alias the storage `attend` reads, so a write through a view is what `attend` sees. `block_size` is the implementation's choice |
| `attend(cache, q, block_table, seq_lens, query_start_loc, encode, image_ranges, layer_kind) -> bf16 [T, num_q_heads, head_dim]` | Attention for a batch of R sequences, each in its own phase |

Per-rank shapes are those of the served model; the scale is 1.0 and there is no logit softcap
inside attention.

| Layer kind | TP1 heads (q / kv) | TP2 heads (q / kv) | Head dim |
|---|---|---|---|
| Sliding | 16 / 8 | 8 / 4 | 256 |
| Full | 16 / 2 | 8 / 1 | 512 |

Full layers have no value projection: their values are the key projection normalized without
scale and without rotary embedding, as the model-author implementation defines them, and the
operator receives keys and values as the model stores them.

## Inputs

| Input | Type and meaning |
|---|---|
| `q` | BF16 `[T, num_q_heads, head_dim]`: this rank's queries after normalization and rotary embedding |
| `block_table` | int32 `[R, max_blocks]`: logical block `b` of sequence `r` → physical block |
| `seq_lens` | int32 `[R]`: keys of sequence `r`, including this call's query rows, whose K and V are already in the cache |
| `query_start_loc` | int32 `[R+1]`: query row `i` of sequence `r` sits at absolute position `seq_lens[r] − q_len[r] + i` |
| `encode` | int32 `[R]`: 1 = encode phase, 0 = denoise phase |
| `image_ranges` | int32 `[R, M, 2]`: absolute half-open spans `[start, end)` of image tokens, padded with `(0, 0)`. Only encode sequences carry spans, and each span lies wholly within that call's query rows |
| `layer_kind` | `"sliding"` (window W = 1,024) or `"full"` |

A sequence with no query rows is allowed and produces nothing. The entry points must not modify
the caches. Any GPU language, algorithm, fusion and internal layout is allowed; a layout that
cannot be expressed as strided views may convert inside the entry points, and conversion time is
timed. `attend` launches the same GPU computation the served model runs for these phases. Calls
are eager (never captured in a CUDA graph by the checker).

## Semantics

For a sequence with `n` keys and `m` query rows, `q` is a query row's position and `p` a key's.

- **Encode.** Row `q` sees keys `p ≤ q`; on sliding layers it also requires `p > q − W`. If `q`
  lies in an image span `[s, e)` of the sequence, row `q` also sees every `p` in `[s, e)`: a span
  is bidirectional within itself whatever the window, including spans longer than W.
- **Denoise.** The query rows are the canvas, positions `[c, n)` with committed length
  `c = n − m` and `m ≤ 256` (`m < 256` only for a canvas clamped at the context limit). Every
  canvas row sees every canvas position. Committed positions on full layers: all of `[0, c)`. On
  sliding layers: one window shared by every row of the canvas, clipped at 0 — either
  `[c − (W − 1), c)` (the author's default cache path) or `[c − W, c)` (the author's static-cache
  path); an implementation uses one of the two throughout.
- **Values** are weighted by the softmax, over the visible keys, of `q·k` at scale 1.0.

The model-author implementation under `../reference/model_author/` remains the definition;
`reference.py` states the semantics executably (`visible_mask`, `attend_reference`).

## Correctness

Every case is evaluated over its visible set. The FP64 reference `r` is computed from the stored
BF16 values (`policy.py`).

| Case | Rule |
|---|---|
| Any | `max|o − r| ≤ 2 · max|o_bf16 − r| + 10⁻⁶`, where `o_bf16` evaluates the same reference with BF16 operands (BF16 score matmul, FP32 softmax rounded to BF16, BF16 P·V) |
| Equal logits (`q = 0` in a row) | `|o − r| ≤ 2⁻⁷ · max_visible |v|` elementwise |

Every valid row must be finite.

**The mask witness** (`witness.py`) makes one tested key dominate one row. Probe directions are
coordinate axes of the head: a row probing position `p` has query `5·u` and the key carries `4·u`
(a logit of 20); every other key has zeros on the probe axes; the key's value is a marker vector
(entries ±2). If `p` is visible the row's output equals the marker within 10⁻³ relative; if it is
invisible the output norm is below 0.1 of the marker's. Several rows may test one key on one kv
head; the key then carries each of their axes and one marker. The probe outcomes pass iff they fit
the canvas window `W − 1` throughout, or `W` throughout — the two admissible readings above.

`check.py` runs a case through the contract (random, equal-logit and probe values; a seeded
permuted block table; ±64 sentinels in blocks the case does not reference; a GPU checksum of both
views before and after, which must not change). The public cases are in
`../examples/attend_cases.json`:

    python /app/eval/diffusion_gemma/run.py attend-check --example denoise-1030 --output /tmp/attend.json

## Timing (`measure.py`)

One H100, TP1 shapes, random values, no image ranges. Three families, each at three committed
contexts, both layer kinds timed separately; the statistic is `25·t_sliding + 5·t_full` (the model's
layer counts):

| Family | Workload |
|---|---|
| encode | one sequence: a chunk of B = 8,192 rows after committed L ∈ {4,096, 65,536, 262,144 − B} |
| denoise | four sequences, each a 256-row canvas after committed L ∈ {4,096, 65,536, 261,888} |
| mixed | one commit pass (256 encode rows) and three canvases, all after committed L |

Protocol: GPU 0 with no server resident; 10 untimed calls; 3 draws of N back-to-back calls (N sized
so a draw lasts at least 100 ms); per-call GPU time from CUDA events around the loop and per-call
synchronized wall time; the statistic is the median of the 3 draws. JIT, warm-up and allocation
count toward the invocation limit (`../budgets.json`), not the statistic.

    python /app/eval/diffusion_gemma/run.py attend-measure --family mixed --output /tmp/t.json

## Attribution (`binding.py`, `../serving/binding.py`)

One call per family and layer kind runs under the torch profiler; a family's identity set is the
set of kernel names its call launches, minus memcpy/memset, PyTorch copy/fill kernels and generic
ATen kernels (`at::native::`). The served model, run with an idle torch profiler, is then observed
in two windows: generation (one 6,144-token prompt, then three concurrent 2,000-token prompts, 256
tokens each) and one denoising read at c = 4,096. On every rank, each family's identity set must
appear in the first window, the most frequent kernel of the denoise set at least 30 × 4 times,
CUDA graph launches must occur, and the second window must contain the denoise set's kernels at
least 30 times.
