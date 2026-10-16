# Lock provenance

- resolved: 2026-09-21T01:18:15Z inside nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04 (x86_64, Python 3.12.3)
- resolver: uv 0.9.28 (pip compile, --generate-hashes)
- input: python/pyproject.toml at sglang 35870d55aca7c6912aa4c785f583da4acac04938 (the REFERENCE: it adds tilelang==0.1.8 and pins apache-tvm-ffi==0.1.9, and the image supplies that delta), sha256 d4837571992a51e030144a49d79fc0991e1b55dbb778f44915080b8331cb5dd3
- additions (extra.in): pytest, setuptools, setuptools-rust, setuptools-scm, wheel, torch/torchvision/torchaudio +cu130, cuda-python==13.2.0, kernels>=0.12.0,<0.13 (transformers 5.6.0's own declared range; SGLang's pyproject leaves kernels unconstrained)
- overrides: none (this recipe installs no DeepEP, so upstream's nvidia-nccl-cu13 force-pin does not apply)
- indexes: https://pypi.org/simple, https://download.pytorch.org/whl/cu130 (unsafe-best-match); flashinfer: https://flashinfer.ai/whl/cu130, https://flashinfer.ai/whl
- tools: huggingface-hub==1.32.0
