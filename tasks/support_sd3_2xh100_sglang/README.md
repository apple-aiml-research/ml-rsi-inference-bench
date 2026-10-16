# Stable Diffusion 3 support in SGLang

Implement native SD3 text-to-image generation with BF16, FP32 and mixed precision on two H100 GPUs. The participant requirements are in [instruction.md](instruction.md).

The task evaluates image quality, native component execution, request lifecycle, TP2, SP2, CFG/CLIP parallelism, offload, VAE tiling and dynamic batching. FP8, LoRA and their dependent combinations are outside the scoring profile.

All 47 fail-to-pass checks and both regression checks must pass for binary success. Partial credit is the fraction of the 49 checks passed. Image-quality, serving-functionality and end-to-end reports are also available; they are not combined into an additional weighted score.

## Prepare model assets

On the host, accept the SD3 model access and license terms on Hugging Face and authenticate with an account that has access. Download the pinned SD3 checkpoint and CLIP scorer before starting a trial:

```bash
uv run --with huggingface-hub==1.28.0 hf auth login
uv run --with huggingface-hub==1.28.0 python \
  tasks/support_sd3_2xh100_sglang/environment/scripts/model_assets.py fetch \
  --root /absolute/path/sd3-assets
```

`environment/model-assets.lock.json` pins SD3 to `ea42f8cef0f178587cf766dc8129abd379c90671` and CLIP to `32bd64288804d66eefd0ccbe215aa642df71cc41`, including file sizes and SHA-256 hashes. Preparation verifies each file and resumes only files with matching hashes. Allow at least 55 GiB of free disk space for the download cache and prepared assets; identical weight aliases are hard-linked in the prepared directory.

To use existing copies, replace `fetch` with `stage` and add `--sd3-source /path/to/sd3` and `--clip-source /path/to/clip`. The same file checks apply. Credentials remain on the host; do not mount them into the task container.

## Run with Docker

```bash
uv run pier run -p tasks/support_sd3_2xh100_sglang --env docker --agent oracle \
  --mounts-json "$(python3 tasks/support_sd3_2xh100_sglang/environment/scripts/model_assets.py \
    mounts --root /absolute/path/sd3-assets)"
```

The command mounts the prepared assets read-only at `/opt/infb/models`. The image resolves SD3 from `/opt/infb/models/sd3` and the CLIP scorer from `/opt/infb/models/clip`; both load offline. The healthcheck verifies all required files before starting the agent timer. Missing or changed assets fail setup.

The task requests two H100 GPUs, 64 GiB of RAM and **8 GiB of private shared memory** through `shm_size_mb = 8192`. Use the NVIDIA Container Toolkit on the Docker host. If launching the image outside Pier, set `--shm-size=8g` and provide the same read-only asset mount; host IPC is not required.

- `environment/`: environment setup and public supporting materials.
- `tests/`: verifier, independent references and scoring.
- `solution/`: complete reference patch and checked installer.
- `pre_artifacts.sh`: candidate patch capture.
