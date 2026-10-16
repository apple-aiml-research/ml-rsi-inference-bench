# Support MiniMax-M3-MXFP8 in vLLM

Implement native support for `MiniMaxAI/MiniMax-M3-MXFP8` in `/app`, using the checkpoint
at `/model` on four NVIDIA B200 GPUs. Support the two deployments in
`/app/eval/minimax_m3/serving/recipes.py`: a BF16 KV cache and an FP8 KV cache, text only.

The starting tree supplies ordinary engine dependencies and three generic kernel
extensions the model uses: the FP32 router GEMM at the model's router shape, the clamped
SwiGLU activation with gate scaling and up-projection bias, and a fused all-reduce with
Gemma RMSNorm. Everything else is part of the task, including the model's MiniMax Sparse
Attention (MSA) GPU computation (index-branch block scoring and top-16 block selection,
and block-sparse attention over the selected blocks, for prefill and decode, with BF16
and FP8 KV caches) and its attention pre-processing (per-head QK normalization, partial
rotary embedding and cache insertion). No optimized implementation of these is supplied.

All required behavior, component correctness, efficiency and applicable engine
regressions must pass for one committed implementation, in both deployments. Equivalent
algorithms, languages, fusion, helper names and cache layouts are allowed.

## Required behavior

- Load the checkpoint and register its declared architectures natively, parsing its
  configuration without executing code from the checkpoint repository. Use TP4 without
  expert parallelism, 128-token KV blocks, the recipe's dense-layer attention backend,
  default compilation with CUDA graphs, chunked prefill, prefix caching and a configured
  context of 262,144 tokens.
- Serve text completions and chat completions, streaming and non-streaming, greedy
  generation, token and prompt log-probabilities, and concurrent requests. Chat
  completions follow the checkpoint's chat template, including its three thinking modes,
  and return reasoning and tool calls through the `minimax_m3` reasoning and tool-call
  parsers, streaming included.
- Implement MSA as the model defines it: sections 2–3 of the MSA paper in `/public/msa/`,
  the model-author implementation readable under `/public/reference-impl/`, and the plain
  PyTorch statement of its four operators in `/app/eval/minimax_m3/msa/reference.py`.
  Expose them through the operator contract `/app/minimax_m3_msa.py` specified in
  `/app/eval/minimax_m3/msa/README.md`, within the numerical tolerances stated there. The
  served model must run the same GPU computation on every rank.
- Preserve model quality and numerical agreement with independent model-author
  recordings through prefill, chunked prefill, decode, prefix reuse and concurrent
  requests, at contexts up to 262,144 tokens, in both deployments. Quality is evaluated
  with the supplied GSM8K chat protocol. Preserve the applicable existing engine tests.

## Efficiency

Efficient production serving is part of acceptance. The deployment's required execution
paths must be active during ordinary serving. Per-user decode speed and concurrent output
throughput are evaluated with `/public/efficiency/client.py` on natural-text prompts of
4,096 input and 512 output tokens, greedy with EOS ignored, at concurrency 1 and 16, in
both deployments. MSA's operators are also measured through the contract at the
long-context workloads in its README: prefill and decode selection and attention must each
be efficient GPU implementations, and attention time per query must not grow with context
length beyond the selected blocks. Inadequate performance can fail even with correct
outputs. Achieved performance is recorded with the result.

## Development resources

`python /app/eval/minimax_m3/run.py --help` lists the public protocols: served
configuration, API, chat and tool calls, fidelity, generation replay, prefix branches,
accuracy, and MSA correctness, timing, context scaling and kernel attribution. The public
and grading copies have identical source; grading supplies its own inputs and acceptance
policy. `/app/eval/minimax_m3/README.md` documents commands, inputs, outputs and resource
limits.

The checkpoint config, model card, tokenizer and chat template are at `/model`. The
engine's Transformers installation remains at 5.5.3 and does not include this model; do
not upgrade dependencies. Public fidelity examples with model-author log-probabilities,
the GSM8K data and efficiency prompts are under `/public`. `/public/build.sh` rebuilds
CUDA, C++ and Rust changes; Python edits use the editable installation.

A serving launch has 3,600 seconds to become healthy and a rebuild 7,200 seconds; other
verifier workloads have the limits in `/app/eval/minimax_m3/limits.json`. The episode's
remaining time is `/app/.timer/remaining_secs`.

## Scope and submission

This task covers the pinned MXFP8 checkpoint's text model on TP4 B200 with BF16 and FP8
KV caches. Images and video, other checkpoints and precisions, speculative decoding,
expert, pipeline and data parallelism, disaggregated serving, KV connectors, CPU offload,
contexts beyond 262,144 tokens and other hardware are outside scope. The machine is
offline apart from the coding harness. The starting engine and dependency versions remain
fixed.

Commit your changes in `/app`. The verifier rebuilds and serves the committed final HEAD
in a fresh environment. Uncommitted files, caches and running servers are not submitted.
Grading restores trusted evaluation code and tests before importing them; editing those
resources does not change acceptance.
