#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
set -euo pipefail
BASE_SHA=a9944aec01e281f5c12c9651f981a53bdece6989
UV_VERSION=0.9.28
OUT=${OUT:-/out}

apt-get update -qq && apt-get install -y -qq --no-install-recommends python3.12-venv curl ca-certificates >/dev/null
python3.12 -m venv /tmp/uvenv
/tmp/uvenv/bin/pip install -q "uv==${UV_VERSION}"
UV=/tmp/uvenv/bin/uv
PY=/tmp/uvenv/bin/python

mkdir -p /tmp/src "$OUT" && cd /tmp/src
curl -fsSL -o pyproject.toml "https://raw.githubusercontent.com/sgl-project/sglang/${BASE_SHA}/python/pyproject.toml"
PYPROJECT_SHA=$(sha256sum pyproject.toml | cut -d' ' -f1)

cat > extra.in <<'EOF'
# Additions for the task image, locked together with SGLang's own dependency set:
# the verifier's test runner, and the build backend of the editable install (no build isolation).
pytest
setuptools>=61.0
setuptools-rust>=1.11
setuptools-scm>=8.0
wheel
# The CUDA 13 builds of the torch family from the PyTorch index (upstream's docker/Dockerfile
# installs with --extra-index-url https://download.pytorch.org/whl/cu130).
torch==2.13.0+cu130
torchvision==0.28.0+cu130
torchaudio==2.11.0+cu130
EOF
cat > overrides.txt <<'EOF'
# torch 2.13 pins nvidia-nccl-cu13==2.29.7; DeepEP needs 2.30.7 at runtime, and upstream's image
# force-reinstalls it after the dependency solve (docker/Dockerfile at the base commit).
nvidia-nccl-cu13==2.30.7
EOF

"$UV" pip compile pyproject.toml extra.in --override overrides.txt --generate-hashes \
  --python "$PY" \
  --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu130 \
  --index-strategy unsafe-best-match \
  -o "$OUT/requirements.lock"

cat > flashinfer.in <<'EOF'
flashinfer-cubin==0.6.18
flashinfer-jit-cache==0.6.18
EOF
"$UV" pip compile flashinfer.in --no-deps --generate-hashes \
  --python "$PY" \
  --index-url https://flashinfer.ai/whl/cu130 --extra-index-url https://flashinfer.ai/whl \
  --index-strategy unsafe-best-match \
  -o "$OUT/requirements-flashinfer.lock"

HFV=$(grep -oE '^huggingface[-_]hub==[0-9A-Za-z.+]+' "$OUT/requirements.lock" | head -1)
printf '%s\n' "$HFV" > tools.in
"$UV" pip compile tools.in --generate-hashes --python "$PY" --index-url https://pypi.org/simple \
  -o "$OUT/requirements-tools.lock"

{
  echo "# Lock provenance"
  echo
  echo "- resolved: $(date -u +%Y-%m-%dT%H:%M:%SZ) inside nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04 (x86_64, $($PY --version))"
  echo "- resolver: uv ${UV_VERSION} (pip compile, --generate-hashes)"
  echo "- input: python/pyproject.toml at sglang ${BASE_SHA}, sha256 ${PYPROJECT_SHA}"
  echo "- additions (extra.in): pytest, setuptools, setuptools-rust, setuptools-scm, wheel, torch/torchvision/torchaudio +cu130"
  echo "- override: nvidia-nccl-cu13==2.30.7 (upstream docker/Dockerfile forces it for DeepEP)"
  echo "- indexes: https://pypi.org/simple, https://download.pytorch.org/whl/cu130 (unsafe-best-match); flashinfer: https://flashinfer.ai/whl/cu130, https://flashinfer.ai/whl"
  echo "- tools: ${HFV}"
} > "$OUT/RESOLVED.md"
echo "resolved: $(wc -l < "$OUT/requirements.lock") lines in requirements.lock"
