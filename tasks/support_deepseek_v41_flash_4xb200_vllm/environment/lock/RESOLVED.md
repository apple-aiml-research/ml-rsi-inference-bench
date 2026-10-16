# Lock provenance

- resolved: 2026-09-24T05:10:14Z inside nvidia/cuda:13.0.3-devel-ubuntu24.04 (x86_64, Python 3.12.3)
- resolver: uv 0.9.28 (pip compile, --generate-hashes)
- inputs at vllm e83d57d8ae14e240e2779c37db58b6dacd21bab2:
  - requirements/cuda.txt        sha256 dd0f01d0c1d1c69248938821fc19549190afb22826136099b3bb10d0fea44a00
  - requirements/common.txt      sha256 74a932cfe0469864247509838b68484bb6e6b978cb87df7ccebf75112e289e9d   (pulled in by cuda.txt's -r)
  - requirements/build/cuda.txt  sha256 3aa681ee629ec4b90a76d01a506c0bdf991c2a4a985303f2b882a600c4774b52
- additions (extra.in): pytest, tblib, torch/torchvision/torchaudio +cu130
- indexes: https://pypi.org/simple, https://download.pytorch.org/whl/cu130,
           https://flashinfer.ai/whl/cu130, https://flashinfer.ai/whl (unsafe-best-match)
- tools: huggingface-hub==1.32.0
