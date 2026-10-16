#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Resolve the hash-locked wheel sets environment/Dockerfile installs, inside the pinned CUDA base
# image (the target platform and interpreter), from public indexes only. Run once at authoring:
#
#   docker run --rm -v <this dir>:/out -v <this file>:/resolve.sh \
#     docker.io/nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04@sha256:8b2705ea... bash /resolve.sh
#
# Writes requirements.lock (SGLang's dependency set plus pytest and the editable install's build
# backend), requirements-flashinfer.lock (FlashInfer's prebuilt cubins and JIT cache from
# flashinfer.ai) and requirements-tools.lock (huggingface_hub for the checkpoint download), plus
# RESOLVED.md recording the inputs.
#
# The pyproject is fetched at the REFERENCE commit, not at the task's base. The reference change
# is the one change in this task that moves SGLang's dependency set: it adds `tilelang==0.1.8`
# and pins `apache-tvm-ffi==0.1.9`. That dependency delta is supplied in the image, and the
# instruction tells the agent those libraries are pre-installed, so the image must carry the
# reference's dependency set. Nothing else about the base tree changes: the agent's checkout is
# base' and its pyproject is the base one.
set -euo pipefail
REF_SHA=35870d55aca7c6912aa4c785f583da4acac04938
UV_VERSION=0.9.28
OUT=${OUT:-/out}

apt-get update -qq && apt-get install -y -qq --no-install-recommends python3.12-venv curl ca-certificates >/dev/null
python3.12 -m venv /tmp/uvenv
/tmp/uvenv/bin/pip install -q "uv==${UV_VERSION}"
UV=/tmp/uvenv/bin/uv
PY=/tmp/uvenv/bin/python

mkdir -p /tmp/src "$OUT" && cd /tmp/src
curl -fsSL -o pyproject.toml "https://raw.githubusercontent.com/sgl-project/sglang/${REF_SHA}/python/pyproject.toml"
PYPROJECT_SHA=$(sha256sum pyproject.toml | cut -d' ' -f1)

cat > extra.in <<'EOF'
# Additions for the task image, locked together with SGLang's own dependency set:
# the verifier's test runner, and the build backend of the editable install (no build isolation).
pytest
setuptools>=61.0
setuptools-rust>=1.10
setuptools-scm>=8.0
wheel
# The CUDA 13 builds of the torch family from the PyTorch index. torch and torchaudio are pinned by
# the pyproject (2.11.0); torchvision is not, and scripts/ci/cuda/ci_install_dependency.sh at the
# reference reinstalls all three from the CU_VERSION index (default cu130), which is what these
# three lines do at resolve time instead.
torch==2.11.0+cu130
torchvision==0.26.0+cu130
torchaudio==2.11.0+cu130
# The pyproject asks for cuda-python>=13.0; upstream's docker/Dockerfile pins 13.2.0 on a CUDA 13
# image, so the lock pins the same.
cuda-python==13.2.0
# SGLang's pyproject lists `kernels` unconstrained, so an unconstrained resolve takes the newest
# release -- and a newer `kernels` refuses the call transformers 5.6.0 itself makes at import time
# (`LayerRepository(...)` with neither revision nor version: "ValueError: Either a revision or a
# version must be specified.", observed on a CUDA 13 build of this image). The constraint below is
# transformers 5.6.0's own declared range for its `integrations`/`kernels` extras, read from its
# PyPI metadata, so this pins the pair upstream ships rather than a version of ours.
kernels>=0.12.0,<0.13
EOF

"$UV" pip compile pyproject.toml extra.in --generate-hashes \
  --python "$PY" \
  --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu130 \
  --index-strategy unsafe-best-match \
  -o "$OUT/requirements.lock"

cat > flashinfer.in <<'EOF'
flashinfer-cubin==0.6.8.post1
flashinfer-jit-cache==0.6.8.post1
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
  echo "- resolved: $(date -u +%Y-%m-%dT%H:%M:%SZ) inside nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04 (x86_64, $($PY --version))"
  echo "- resolver: uv ${UV_VERSION} (pip compile, --generate-hashes)"
  echo "- input: python/pyproject.toml at sglang ${REF_SHA} (the REFERENCE: it adds tilelang==0.1.8 and pins apache-tvm-ffi==0.1.9, and the image supplies that delta), sha256 ${PYPROJECT_SHA}"
  echo "- additions (extra.in): pytest, setuptools, setuptools-rust, setuptools-scm, wheel, torch/torchvision/torchaudio +cu130, cuda-python==13.2.0, kernels>=0.12.0,<0.13 (transformers 5.6.0's own declared range; SGLang's pyproject leaves kernels unconstrained)"
  echo "- overrides: none (this recipe installs no DeepEP, so upstream's nvidia-nccl-cu13 force-pin does not apply)"
  echo "- indexes: https://pypi.org/simple, https://download.pytorch.org/whl/cu130 (unsafe-best-match); flashinfer: https://flashinfer.ai/whl/cu130, https://flashinfer.ai/whl"
  echo "- tools: ${HFV}"
} > "$OUT/RESOLVED.md"
echo "resolved: $(wc -l < "$OUT/requirements.lock") lines in requirements.lock"
