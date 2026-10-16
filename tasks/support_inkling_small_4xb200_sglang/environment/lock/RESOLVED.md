# Lock provenance

- resolved: 2026-09-29T16:56:50Z inside nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04 (x86_64, Python 3.12.3)
- resolver: uv 0.9.28 (pip compile, --generate-hashes, --exclude-newer 2026-07-20T05:57:37Z for PyPI)
- input: python/pyproject.toml at sglang 02236fa38cb065660a0b4156aab283a3807a5290 (the landing squash: the base's pins plus helion==0.2.6 and numba==0.65.1), sha256 428bb9ce7b9fe91c4d272b404d5c180f478a8039adfde1d3998c3a5cc4f5da85
- additions (extra.in): pytest, setuptools, setuptools-rust, setuptools-scm, wheel, torch 2.11.0 / torchvision 0.26.0 / torchaudio 2.11.0 +cu130
- overrides: none
- indexes: https://pypi.org/simple, https://download.pytorch.org/whl/cu130 (unsafe-best-match); flashinfer: https://flashinfer.ai/whl/cu130, https://flashinfer.ai/whl
- asserted pins: nvidia-cutlass-dsl==4.5.2 flash-attn-4==4.0.0b15 transformers==5.12.1 quack-kernels==0.5.0 helion==0.2.6 numba==0.65.1 apache-tvm-ffi==0.1.11 flashinfer-python==0.6.14 sglang-kernel==0.4.5 tilelang==0.1.11 torch==2.11.0+cu130 kernels==0.14.1
- tools: huggingface-hub==1.24.0
