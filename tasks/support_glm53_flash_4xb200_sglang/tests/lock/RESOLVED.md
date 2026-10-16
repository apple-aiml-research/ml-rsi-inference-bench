# Lock provenance

- resolved: 2026-09-19T23:05:48Z inside nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04 (x86_64, Python 3.12.3)
- resolver: uv 0.9.28 (pip compile, --generate-hashes)
- input: python/pyproject.toml at sglang a9944aec01e281f5c12c9651f981a53bdece6989, sha256 79a8240d85e310935bb64995290efeeb1e69c1f61f466b19882c08ac544482be
- additions (extra.in): pytest, setuptools, setuptools-rust, setuptools-scm, wheel, torch/torchvision/torchaudio +cu130
- override: nvidia-nccl-cu13==2.30.7 (upstream docker/Dockerfile forces it for DeepEP)
- indexes: https://pypi.org/simple, https://download.pytorch.org/whl/cu130 (unsafe-best-match); flashinfer: https://flashinfer.ai/whl/cu130, https://flashinfer.ai/whl
- tools: huggingface-hub==1.32.0
