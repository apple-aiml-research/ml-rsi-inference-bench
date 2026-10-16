# Lock provenance

- resolved: 2026-09-21T10:50:24Z inside nvidia/cuda:13.0.3-devel-ubuntu24.04 (x86_64, Python 3.12.3)
- resolver: uv 0.9.28 (pip compile, --generate-hashes)
- inputs at vllm 687db597440bf2b0cc65404e5f8cc238632127e0:
  - requirements/cuda.txt        sha256 1bcca767ba1e8de38aacc19755e20bc2fca2da77e1b0b25ef68be021d96ebfc1
  - requirements/common.txt      sha256 aa482a5d01a5e71158bca06f3772fc6d9bceaf68ab47e96fb7f45f46fc87eb1b   (pulled in by cuda.txt's -r)
  - requirements/build/cuda.txt  sha256 3aa681ee629ec4b90a76d01a506c0bdf991c2a4a985303f2b882a600c4774b52
- additions (extra.in): pytest, tblib, transformers==5.12.1, torch/torchvision/torchaudio +cu130
- indexes: https://pypi.org/simple, https://download.pytorch.org/whl/cu130,
           https://flashinfer.ai/whl/cu130, https://flashinfer.ai/whl (unsafe-best-match)
- evalscope: 1.10.0 from PyPI, resolved separately for its own venv
- tools: huggingface-hub==1.32.0

## Added by hand: `requirements-flashinfer-jit-cache.lock` (2026-09-24)

One wheel, not produced by `resolve.sh`: `flashinfer-jit-cache==0.6.18+cu130`, with the sha256 the public index
`https://flashinfer.ai/whl/cu130/flashinfer-jit-cache/` publishes
(`428a47a554ade93c30a818e142b781df58582bf056bab94611fb4c906cc366bf`; the wheel is hosted on FlashInfer's own
GitHub release `v0.6.18`, 1,016,181,643 bytes, 906 pre-compiled modules including `fused_moe_trtllm_sm100`). It is
the same release as `flashinfer-python` above and is installed `--no-deps` in its own layer. It was added by hand
because re-running the resolver today would move unrelated pins that the measured image was built with. The
wheel is needed for the pre-compiled `fused_moe_trtllm_sm100` module the runtime loads.
