# Support DiffusionGemma in vLLM

Implement native support for `google/diffusiongemma-26B-A4B-it` in `/app`, using the
checkpoint at `/model` on two NVIDIA H100 GPUs. Support the two deployments in
`/app/eval/diffusion_gemma/serving/recipes.py`: one GPU, which is the model's published
serving recipe, and tensor parallelism across both GPUs.

DiffusionGemma generates text by block diffusion. An encoding pass attends causally over
the prompt and everything already committed. A 256-token canvas is then denoised over
several steps; it attends bidirectionally within itself and to the committed context. An
accepted canvas is committed by an encoding pass. The sliding-window layers use a
1,024-token window, and image tokens attend bidirectionally within their image.

The starting tree supplies ordinary engine dependencies. They include the sources of the
engine's FlashAttention fork at its pinned revision, under `/app/third_party/vllm-flash-attn`,
editable and built by the documented rebuild. At this revision, neither those kernels nor
the engine's Triton attention supports a batch in which each sequence has its own attention
phase. Everything else is part of the task, and no implementation of it is supplied. This
includes:

- the paged attention GPU computation for batches that mix encoding and denoising
  sequences, for both of the model's attention head sizes;
- the model's generation, self-conditioning and sampler in the engine.

All required behavior, component correctness, efficiency and applicable engine regressions
must pass for one committed implementation, in both deployments. Equivalent algorithms,
languages, fusion, helper names and cache layouts are allowed.

## Required behavior

- Load the checkpoint and register its declared architecture natively, parsing its
  configuration without executing code from the checkpoint repository. Use the recipes'
  compilation (TorchInductor's deterministic mode with combo kernels off, the
  `--compilation-config` of `recipes.py`) with CUDA graphs, chunked prefill, prefix caching,
  asynchronous scheduling at the engine default, four concurrent sequences with preemption,
  and a configured context of 262,144 tokens.
- Generate as the model defines it:
  - encoding, denoising and commit;
  - self-conditioning;
  - the sampler, temperature schedule and stopping rule configured by the checkpoint's
    `generation_config.json`, which a deployment may override, and a request's `temperature`
    (0 decodes greedily) and `seed`.

  Serve images through the model's vision tower.
- Implement the model's attention as the model defines it, using two sources:
  - the model-author implementation, readable under
    `/app/eval/diffusion_gemma/reference/model_author/`;
  - the plain PyTorch statement in `/app/eval/diffusion_gemma/attend/reference.py`.

  Expose it through the operator contract `/app/diffusion_gemma_attend.py` specified in
  `/app/eval/diffusion_gemma/attend/README.md`, within the numerical tolerances stated there.
  The served model must run the same GPU computation on every rank.
- Serve text completions and chat completions, streaming and non-streaming, with token
  log-probabilities and concurrent requests. Chat completions follow the checkpoint's chat
  template, including thinking. They return reasoning and tool calls through the `gemma4`
  reasoning and tool-call parsers, streaming included; a tool call emitted inside an unclosed
  thought block must be served. A structured-output request is either honored or refused with
  a client error.
- Serve the denoising reads specified in `/app/eval/diffusion_gemma/serving/reads.md`.
- Preserve model quality and numerical agreement with independent model-author recordings,
  in both deployments, at contexts up to 262,144 tokens. This covers chunked prefill,
  denoising, commit, prefix reuse, images, mixed batches and concurrent requests. Quality is
  evaluated with the supplied GSM8K protocol. Preserve the applicable existing engine tests.

## Efficiency

Efficient production serving is part of acceptance. The deployment's required execution
paths must be active during ordinary serving.

Per-user decode speed and concurrent output throughput are evaluated with
`/public/efficiency/client.py` in both deployments:

- natural-text prompts of 4,096 input and 1,024 output tokens, with EOS ignored;
- concurrency 1 and 4;
- every canvas denoised for the full 48 steps (the fixed-work configurations in
  `recipes.py`).

The attention computation is also measured through the contract, at the workloads in its
README: encoding chunks, denoising canvases and mixed batches at committed contexts up to
262,144 tokens.

Inadequate performance can fail even with correct outputs. Achieved performance is
recorded with the result.

## Development resources

`python /app/eval/diffusion_gemma/run.py --help` lists the public protocols:

- served configuration, API, chat and tool calls;
- denoising reads, generation replay, prefix branches, mixed batches and preemption;
- decoding policy, long context and accuracy;
- the attention contract's correctness, timing and kernel attribution.

The public and grading copies have identical source; grading supplies its own inputs and
acceptance policy. `/app/eval/diffusion_gemma/README.md` documents commands, inputs,
outputs and resource limits.

Other resources:

- `/model` holds the checkpoint's configuration, generation configuration, model card,
  tokenizer, chat template and processor configuration.
- The engine's Transformers installation remains at 5.5.3 and does not include this model;
  do not upgrade dependencies.
- Public read examples with model-author log-probabilities are under
  `/app/eval/diffusion_gemma/examples/`. The GSM8K data and efficiency prompts are under
  `/public`.
- `Qwen/Qwen2.5-0.5B-Instruct`, at `/models/qwen2.5-0.5b-instruct`, is available for checking
  that other models still serve.
- `/app/eval/diffusion_gemma/build.sh` rebuilds CUDA and C++ changes, including the
  FlashAttention fork's compiled kernels. Python edits, including the fork's CuTe DSL
  sources, take effect through the editable installation. The rebuild is incremental from
  the CMake build directory `/app/build/temp`, which the image carries; a change to
  `setup.py`, `pyproject.toml` or `requirements/` re-runs the editable installation, a full
  compile of about 45 minutes that the install clock does not cover.

A serving launch has 1,800 seconds to become healthy and a rebuild has 7,200 seconds.
Other verifier workloads have the limits in `/app/eval/diffusion_gemma/budgets.json`. The
episode's remaining time is in `/app/.timer/remaining_secs`.

## Scope and submission

This task covers the pinned checkpoint, with text and images, at TP1 and TP2 on H100 with a
BF16 KV cache. The following are outside scope: video; other checkpoints and precisions;
speculative decoding; expert, pipeline and data parallelism; disaggregated serving; KV
connectors; CPU offload; request-level sampling parameters other than `temperature` and `seed`; prompt
log-probabilities; contexts beyond 262,144 tokens; and other hardware. The machine is offline
apart from the coding harness. The starting engine and dependency versions remain fixed.

Commit your changes in `/app`. The verifier rebuilds and serves the committed final HEAD
in a fresh environment. Uncommitted files, caches and running servers are not submitted.
Grading restores trusted evaluation code and tests before importing them; editing those
resources does not change acceptance.
