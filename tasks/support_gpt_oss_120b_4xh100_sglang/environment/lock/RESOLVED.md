# Lock provenance

- resolved: 2026-09-23T11:43:52Z inside nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04 (x86_64, Python 3.12.3)
- resolver: uv 0.9.28 (pip compile, --generate-hashes)
- input: python/pyproject.toml at sglang 556e4143f0ba8e480a73fc0e08f05d7ea83880cc, extra 'srt', sha256 fd2b335c3b1ea170924684a06ba49c65e0916d76935890188feba501ddc498ed
- additions (extra.in): pytest, setuptools, wheel, pandas, openai>=1.99.1 and <2 (the landing era; see extra.in), openai-harmony==0.0.3
- overrides: torch==2.8.0+cu126, torchaudio==2.8.0+cu126, torchvision==0.23.0+cu126, sgl-kernel==0.3.2 (the reference's stack, local versions included; see overrides.txt)
- indexes: https://pypi.org/simple, https://download.pytorch.org/whl/cu126 (unsafe-best-match)
- exclude-newer: 2025-08-10T00:00:00Z (the landing era; see the header for why this is load-bearing)
- exclude-newer-package: torch, torchaudio, torchvision at 2030-01-01T00:00:00Z -- i.e. unbounded, because download.pytorch.org publishes no upload dates and all three are pinned to an exact version with its local version
- tools: huggingface-hub==1.32.0 (NOT era-bounded; see the note above tools.in)

## The pins this resolve produced, for the record

- torch==2.8.0+cu126 \
- torchvision==0.23.0+cu126 \
- torchaudio==2.8.0+cu126 \
- triton==3.4.0 \
- sgl-kernel==0.3.2 \
- flashinfer-python==0.2.10 \
- transformers==4.55.0 \
- tokenizers==0.21.4 \
- numpy==2.3.2 \
- pandas==2.3.1 \
- openai==1.99.6 \
- openai-harmony==0.0.3 \
- xgrammar==0.1.22 \
- torchao==0.9.0+cu126 \
- safetensors==0.6.2 \
- huggingface-hub==0.34.4 \
- pytest==8.4.1 \
