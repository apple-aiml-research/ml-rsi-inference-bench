# Support Inkling-Small-NVFP4 in SGLang

Implement native support for `thinkingmachines/Inkling-Small-NVFP4` in the SGLang checkout at
`/app`, using the checkpoint at `/model/thinkingmachines/Inkling-Small-NVFP4` on four NVIDIA
B200 GPUs. Support the two launches of the registered test
`/app/eval/inkling_small/tests/test_inkling_small_nvfp4.py`: the production recipe and its
deterministic-inference variant, text only.

The starting tree supplies the engine's ordinary dependencies at their pinned versions,
including FlashAttention-4 `4.0.0b15`, Helion `0.2.6` and Numba `0.65.1`. Everything
specific to this model is part of the task, including its relative attention (causal
attention with a learned per-token, per-head bias on the query-key distance, for its
sliding-window and full layers, over ragged and paged KV caches, for prefill and decode) and
the model-specific GPU computation around it (short convolutions and their cached state,
attention pre-processing, expert routing and tensor-parallel reductions). No implementation of
the relative-attention extension or of these model-specific computations is supplied.

All required behavior, relative-attention correctness and efficiency, and the applicable
existing engine tests must pass together for one committed implementation, in both launches
where they apply. Equivalent algorithms, GPU languages, fusions, helper names and cache
layouts are allowed.

## Required behavior

- Load the checkpoint and register its declared architecture natively; the engine's
  Transformers installation remains at 5.12.1 and does not include this model. Serve with the
  launches' tensor parallelism, NVFP4 execution, 128-token KV pages, hybrid sliding-window and
  full KV cache with prefix caching, chunked prefill and CUDA graphs for decode.
- Implement the model's attention as the model defines it. For query position p and key
  position j ≤ p, with d = p − j, the logit of query head h is ⟨q, k⟩/128 plus b[h, d] when
  d < E, where b is the bias the model computes for that query token from its relative
  projection, and ⟨q, k⟩/128 alone when d ≥ E. Sliding-window layers attend to keys with
  d ≤ 511 and have E = 512; full layers attend to every earlier key and have E = 1,024. On full
  layers the query and the bias are multiplied by τ(p) = 1 + 0.1·ln(max((p + 1)/128,000, 1)).
  The model-author implementation under `/public/reference-impl/` defines the rest of the
  model.
- Expose the relative attention through the operator contract `/app/inkling_rel_attention.py`
  specified in `/app/eval/inkling_small/rel_attention/README.md`, within its numerical
  tolerance. The served model must run the same GPU computation on every rank.
- Serve SGLang's `/generate` and `/v1/completions` interfaces from text and from token IDs,
  streaming and non-streaming, with greedy generation, input and output token
  log-probabilities, and concurrent requests, for requests of up to 262,144 tokens.
- Preserve model quality and numerical agreement with independent model-author recordings
  through prefill, chunked prefill, decode, prefix reuse and concurrent requests, at contexts
  up to 32,768 tokens, and agreement with the reference engine's own recorded log-probabilities
  at 262,144 tokens. Quality is evaluated with the registered test's GSM8K protocol.
- Agreement is measured per sequence as the mean per-token k3 between the reference and the
  served log-probabilities, leaving out a fixed number of its largest terms, together with a
  bound on the fraction of tokens that are outliers: tokens whose served probability is more
  than e^3 (about 20) times below the reference's. The model's top-6-of-256 expert routing makes
  a few tokens' log-probabilities change by several nats under any change of arithmetic; the
  trimmed count and the outlier bound absorb that, while an error that affects more than a few
  tokens still counts. At 262,144 tokens the reference is the upstream implementation's own
  production series, and its bounds were set from the same implementation under its
  deterministic-inference arithmetic, since the production reference repeats itself exactly.
- Under the deterministic-inference launch, prefill and decode must score every token
  identically, including after prefix-cache hits, as the registered test's deterministic
  class checks. Preserve the applicable existing engine tests.
- Repeating a request under the deterministic-inference launch must return the same tokens
  and log-probabilities whatever ran before it, including when it restores a cached prefix.

## Efficiency

Efficient production serving is part of acceptance. The production launch's required
execution paths must be active during ordinary serving. Per-user decode speed and concurrent
output throughput are evaluated with the measurement protocol in
`/app/eval/inkling_small/serving/measure.py` on natural-text prompts of 4,096 input and 512
output tokens, greedy with EOS ignored, at concurrency 1 and 16, against the upstream
implementation served at the same recipe with its optional fused execution paths and its
sheared-bias attention kernel switched off. The relative attention is also timed through its
operator contract at the workloads in its README, for prefill and decode on both layer kinds;
each must be an efficient GPU implementation. The operator comparison is with
FlashAttention-4's generic score-modification path carrying this bias for full-layer decode and
for prefill, and with the upstream implementation's own kernel for sliding-window decode, whose
short 512-key calls that generic path serves faster than the upstream kernel does. Inadequate
performance can fail acceptance even with correct outputs. Achieved performance is recorded
with the result.

## Development resources

`python /app/eval/inkling_small/run.py --help` lists the public protocols: the registered
test, served configuration, API, fidelity, generation replay, prefix branches, conv-state
restore, efficiency, and relative-attention correctness, timing and kernel attribution. The
public and grading copies have identical source; grading supplies its own inputs and
acceptance policy.
`/app/eval/inkling_small/README.md` documents commands, inputs, outputs and resource limits.

The checkpoint's configuration, model card, tokenizer and chat template are at `/model`. The
model-author implementation (Transformers v5.16.1 `modeling_inkling.py` and
`configuration_inkling.py`) is readable under `/public/reference-impl/` and is not installed.
Public fidelity examples with model-author log-probabilities, GSM8K data, LongBench prompts
for the deterministic test and efficiency prompts are under `/app/eval/inkling_small/`.

A serving launch has 3,600 seconds to become healthy; other verifier workloads have the
limits in `/app/eval/inkling_small/limits.json`. `/app/.timer/remaining_secs` reports the
number of seconds remaining for this run.

## Scope and submission

This task covers the pinned NVFP4 checkpoint's text model with tensor parallelism 4 on B200 at
the two registered launches. Speculative decoding (including the checkpoint's MTP layers),
chat completions with reasoning and tool-call parsing, images and audio, LoRA, FP8 or MXFP8
KV caches, expert, pipeline, data and context parallelism, disaggregated serving, hierarchical
caching, other checkpoints and precisions, requests beyond 262,144 tokens and other hardware
are outside scope. The machine is offline apart from the coding harness; the starting engine
and dependency versions remain fixed.

Commit your changes in `/app`. The verifier installs the committed final HEAD's `python/`
package with the editable install documented in the README, in a fresh environment. The
prebuilt `sglang-kernel` package is used as supplied and `/app/sgl-kernel` is not built; GPU
code compiled at run time is compiled during the launch. Uncommitted files, caches and running
servers are not submitted. Grading restores the evaluation materials and engine tests before
importing them; editing them does not change acceptance.
