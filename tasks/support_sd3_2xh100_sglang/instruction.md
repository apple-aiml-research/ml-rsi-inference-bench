# Support Stable Diffusion 3 in SGLang

Implement native support for `stabilityai/stable-diffusion-3-medium-diffusers`
in `/app/repo`, using the checkpoint at `/app/model` on two NVIDIA H100 80 GB
GPUs. Support the one-GPU and two-GPU deployments below.

The starting revision, `9e6d1c066ec3445b534eef6ccdcd752f1bddae1f`, supplies
ordinary engine dependencies and diffusion-serving infrastructure but cannot
load SD3 through `multimodal_gen`. Integrate its CLIP-L, CLIP-G, T5, MMDiT,
scheduler and 16-channel VAE through SGLang's configuration, registry,
component loaders and pipeline stages. Equivalent helper names and file
organization are allowed.

## Required behavior

- Load and generate through
  `sglang.multimodal_gen.DiffGenerator.from_pretrained(...).generate(...)`,
  preserving the result interface. Default loading without precision overrides
  must produce finite, valid images.
- Preserve SD3's conditioning and sampling: CLIP penultimate hidden states and
  projected pooled embeddings, T5 final hidden states, joint attention,
  scheduler, and VAE scaling and shift. Preserve checkpoint token layouts and
  projections; pad or truncate T5 embeddings to 256 tokens. Omitted
  `negative_prompt` uses `" "`; an explicitly empty negative prompt is valid. Use both
  CFG branches only when `guidance_scale > 1`, otherwise only the positive branch.
- Run the requested denoising steps unless cancelled, timed out or failed with
  a reported error. Different prompts at the same seed must change conditioning
  and images. Repeated requests may change prompts, seeds, steps, guidance and
  image sizes without leaking state.
- Accept prompt lists and `num_outputs_per_prompt=n`. Each prompt's output `i`
  uses `seed+i` and agrees with standalone generation. Distinct requests sharing
  a seed retain separate ownership.
- Support `request_id`, `timeout_s` and `gen.abort(request_id)`. Cancel queued
  requests before execution and running requests promptly, returning no image;
  any raised exception must identify cancellation or timeout. Invalid requests
  raise informative exceptions before execution; execution failures raise with
  the underlying message. Keep the engine usable for other in-flight and subsequent requests.
- Use `runtime/layers/attention` in every MMDiT block. Support default
  lower-precision FlashAttention and `attention_backend="torch_sdpa"` with
  correct joint attention.
- With default VAE tiling enabled, decode large latents through overlapping
  tiles and seam blending, preserving independent decode correctness without
  introducing seams relative to untiled decoding.
- Support `dit_layerwise_offload=True`: keep block weights on CPU, transfer a
  small working window to GPU, and materially reduce peak GPU memory while
  preserving images and repeat requests.
- Preserve Flux text encoding and shared component-loader selection.

## Precision and deployments

Support `dit_precision`, `vae_precision` and `text_encoder_precisions`
(CLIP-L, CLIP-G, T5 order), accepting `bf16`, `fp16` and `fp32`. CLI equivalents
are `--dit-precision`, `--vae-precision` and `--text-encoder-precisions`.

Core model, component and image correctness are required in three profiles:
all-BF16, all-FP32 with TF32 disabled, and mixed component precisions. Requested
precision applies to weights and main GEMM, convolution and attention operands.
Higher-precision accumulation, outputs, normalization, softmax, reductions,
elementwise operations and scheduler arithmetic are allowed. Joint conditioning
uses dtype promotion; initial noise uses its dtype. FlashAttention must support
lower-precision DiT execution, and SDPA FP32; reject unsupported precision/backend
combinations explicitly.

Deployment controls are `from_pretrained` arguments:

| Deployment | Required configuration and behavior |
| --- | --- |
| One GPU | `num_gpus=1`; all applicable behavior above. |
| TP2 | `num_gpus=2, tp_size=2`; shard DiT weights and computation across both ranks. |
| SP2 | `num_gpus=2, tp_size=1, sp_degree=2, ulysses_degree=2`; partition image tokens, preserve joint context, and support token counts not divisible by two, with and without CFG. Guidance alone must not allocate another CFG-parallel group. |
| CLIP parallelism | Under SP2, each rank computes a genuine share of both CLIP encoders, matching dense-reference outputs. |
| CFG parallelism | `num_gpus=2, enable_cfg_parallel=True`; distribute and combine positive/negative branches correctly; also support unguided requests. |
| Dynamic batching | On one GPU and TP2, merge compatible concurrent requests into shared DiT forwards, returning each caller's results. Support `batching_max_size` and `batching_delay_ms`, also exposed as `--batching-max-size` and `--batching-delay-ms`. |

TP2 and SP2 must preserve correctness, request isolation, cancellation and error
recovery. Dynamic batching must preserve conditioning, per-request RNG, multiple
outputs and replies; cancelling one member must preserve others' outputs.
Keep incompatible configurations separate; default `batching_max_size=1`
preserves unbatched behavior. Parallelism, batching, attention and offload must
perform real work whose results contribute to returned images.

Serving deployments use fixed representative precisions with matching
references; every precision/switch combination is not required. FP8, LoRA,
their combinations (including adapter-change batching fences), and batching
with SP or CFG-parallel groups are outside scope. Explicitly refuse unsupported
topologies.

## Correctness and scoring

Components are compared with independent references on common inputs, possibly
evaluated in FP32 with fixed lower-precision tolerances. Images are compared
with Diffusers using the same checkpoint, component precisions and generation
settings. Batching and parallelism keep the corresponding standalone reference;
reducing resolution or steps does not satisfy acceptance.

Image metrics are PSNR, SSIM, CLIP image cosine and CLIP image–prompt cosine.
The base lower bounds are:

| Precision | PSNR (dB) | SSIM | CLIP image | CLIP image–prompt |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 16.8 | 0.710 | 0.958 | 0.255 |
| FP32 | 26.5 | 0.725 | 0.960 | 0.252 |
| mixed | 18.0 | 0.725 | 0.960 | 0.257 |

Fixed case-specific bars may differ. Every image must pass all applicable bars;
averaging cannot hide a failure. Serving combinations use the bars of their
matching precision. Bit-exact reference agreement is not required.

Binary reward is 1 only when all required checks, including existing-model
regressions, pass for one committed implementation. Partial score is the
fraction passed; failed, missing or skipped checks receive no credit. There
is no speedup target.

Reports also show three views of the same checks: base image quality in each
precision, the functionality of the TP2, SP2 and batching deployments, and
end-to-end deployment image quality. A base image-quality failure alone does
not establish a parallelism or batching defect. Functional credit requires
execution and numerical-correctness evidence; matching another output of the
same implementation is not sufficient. These views change neither the binary
reward nor the required checks, and they are not combined into a new score.

## Development and submission

Public model/interface notes and a runnable API example are in
`/app/public/sd3_arch_notes.md` and `/app/public/sglang_integration_notes.md`.
`python /app/public/check_candidate.py` checks syntax only; exercise loading,
generation and serving through the public API. Keep the `__main__` guard in
scripts because workers use `spawn`. `INFB_MODEL_PATH` may also identify the
staged checkpoint. Dependencies remain fixed and the machine is offline apart
from the coding harness; do not download models or install runtime dependencies.

Text encoders, DiT and VAE must be SGLang-owned runtime classes. Complete
Diffusers/Transformers models, pipelines and loaders are prohibited, including
wrappers, renamed copies, subclasses and Diffusers `from_pretrained` component
loading. Layer-level building blocks (`JointTransformerBlock`, `PatchEmbed`,
normalization) and VAE output containers are allowed within the native
implementation. Do not hardcode reference images, latents or input-specific
answers, or access private evaluation files.

You have eight hours; `/app/.timer/remaining_secs` reports time remaining.
Validate and commit changes in `/app/repo` before the deadline. Only committed
code is submitted; uncommitted files, cached outputs and running servers are
not submission artifacts.
