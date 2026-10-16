# Lock provenance

- resolved: 2026-09-29T14:47:05Z inside nvidia/cuda:13.0.3-devel-ubuntu24.04 (x86_64, Python 3.12.3)
- resolver: uv 0.9.28 (pip compile, --generate-hashes)
- inputs at vllm a3195fab7b1227e75403fef83891e961e69228a6 (the base commit, committed 2026-06-15T16:37:52Z):
  - requirements/test/cuda.txt   sha256 b71f874a9f7941ad61b352bdd4ec5d9f23b783f7cc810565b0c87855c902cf8e   (every '==' pin, 316 packages)
  - requirements/cuda.txt        sha256 b15cdeba2ca41715bf4376b076d1236ea55034f771af171e4198cffceda97ad7
  - requirements/common.txt      sha256 3797a405bce48097f492dac32ee64e72fc600711d84c19aaf6ab02274ff87235   (pulled in by cuda.txt's -r)
  - requirements/build/cuda.txt  sha256 f60e5120e13b7e66e10f01293d46692bfd502e1457bafd824c5bcce7822751bd
- --exclude-newer 2026-06-15T16:37:52Z; transformers==5.5.3 restated
- indexes: https://pypi.org/simple, https://download.pytorch.org/whl/cu130, https://flashinfer.ai/whl/cu130 (unsafe-best-match)
- tools: huggingface-hub==1.10.2
- flashinfer-jit-cache: 0.6.12 from https://flashinfer.ai/whl/cu130 (--no-deps)
