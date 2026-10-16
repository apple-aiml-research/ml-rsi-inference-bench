# Stable Diffusion 3 — architecture notes

> Read-only context. What you need to know about SD3 to integrate it.
> Reference checkpoint: `stabilityai/stable-diffusion-3-medium-diffusers`
> (weights are available locally at `/app/model`, also exposed through
> `INFB_MODEL_PATH` when configured).

## High level

SD3 is a **text-to-image** latent diffusion model built on **MMDiT**
(Multimodal Diffusion Transformer). It denoises image latents in a
16-channel VAE latent space, conditioned on text, using a
flow-matching schedule. Output is a single image (`num_frames=1`).

## Components

1. **Three text encoders** (this is the SD3-specific wrinkle):
   - **CLIP-L** (`text_encoder`) — pooled + sequence embeds.
   - **CLIP-G / OpenCLIP bigG** (`text_encoder_2`) — pooled + sequence embeds.
   - **T5-XXL** (`text_encoder_3`) — sequence embeds only.
   The two CLIP **pooled** embeddings are concatenated into a pooled
   conditioning vector; the CLIP **sequence** embeddings are concatenated
   and padded/projected to align with the **T5** sequence embeddings.
   CLIP sequence conditioning uses the penultimate hidden states and
   projected pooled embeddings; T5 uses its final hidden states.
   Classifier-free guidance uses a negative prompt (default `" "`) when
   guidance is greater than one; otherwise it uses only the positive branch.

2. **MMDiT transformer** (`SD3Transformer2DModel`): joint-attention
   blocks that attend over concatenated [text tokens ; image patch
   tokens]. Key arch knobs (from the diffusers config): `sample_size`,
   `patch_size`, `in_channels` (=16), `num_layers`,
   `attention_head_dim`, `num_attention_heads`, `joint_attention_dim`,
   `caption_projection_dim`, `pooled_projection_dim`. Layer-level
   Diffusers building blocks may be used as described in the task;
   complete library models and model loaders may not replace native
   runtime components. Joint attention must execute through SGLang's
   attention layer and contribute to the result.

3. **VAE** (16-channel): encodes/decodes between pixel and latent space.
   Mind the **scaling factor / shift factor** (SD3 uses non-trivial
   `scaling_factor` and `shift_factor`) — getting these wrong corrupts
   colors/brightness even when everything else is correct.

## Sampling contract (what determinism hinges on)

`generate(sampling_params_kwargs=dict(...))` — relevant fields
(`SamplingParams` base + SD3 defaults):

| field | SD3 default | note |
|---|---|---|
| `prompt` | — | text prompt |
| `negative_prompt` | `" "` | single space (valid non-empty for CFG) |
| `seed` | `42` | seeds the initial latent noise → **the determinism knob** |
| `num_inference_steps` | `50` | honor an explicit request override |
| `guidance_scale` | `7.0` | CFG strength |
| `guidance_rescale` | `0.0` | |
| `num_frames` | `1` | image (not video) |
| `height` / `width` | model default (1024) | |
| `return_frames` | `False` | set `True` to get the image tensor back |
| `generator_device` | `None` | which device the RNG generator lives on |

**Determinism**: at a fixed `(prompt, seed, steps, guidance_scale,
height, width, dtype)` the output is reproducible on the same hardware,
but need not be bit-exact across GPU or dtype. Independent Diffusers
references use matching checkpoint, component precisions and generation
settings. Image correctness uses the metrics and tolerances described
in the task. Batching and parallelism preserve the corresponding
single-request image target.
