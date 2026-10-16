#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Resolve the hash-locked wheel sets environment/Dockerfile installs, inside the pinned CUDA base
# image (the target platform and interpreter), from public indexes only. Run once at authoring:
#
#   podman run --rm -v <this dir>:/out -v <this file>:/resolve.sh \
#     docker.io/nvidia/cuda:13.0.3-devel-ubuntu24.04@sha256:b7ae301d... bash /resolve.sh
#
# Writes requirements.lock (vLLM's dependency set at the base commit -- requirements/cuda.txt,
# which includes common.txt, plus requirements/build/cuda.txt for the no-build-isolation editable
# install -- with pytest and tblib, which the vendored upstream test's conftest chain imports),
# requirements-evalscope.lock (EvalScope 1.10.0, the evaluator the upstream acceptance test calls,
# in its own venv so the engine environment is untouched) and requirements-tools.lock
# (huggingface_hub for the checkpoint download), plus RESOLVED.md recording the inputs.
#
# Not written here: requirements-flashinfer-jit-cache.lock, one wheel added by hand with its public
# index hash (see its own header and RESOLVED.md). A re-run of this script must carry that
# section of RESOLVED.md forward, since the script rewrites the file.
set -euo pipefail
BASE_SHA=687db597440bf2b0cc65404e5f8cc238632127e0
UV_VERSION=0.9.28
OUT=${OUT:-/out}

apt-get update -qq && apt-get install -y -qq --no-install-recommends python3.12-venv curl ca-certificates >/dev/null
python3.12 -m venv /tmp/uvenv
/tmp/uvenv/bin/pip install -q "uv==${UV_VERSION}"
UV=/tmp/uvenv/bin/uv
PY=/tmp/uvenv/bin/python

mkdir -p /tmp/src/requirements/build "$OUT" && cd /tmp/src
for f in requirements/cuda.txt requirements/common.txt requirements/build/cuda.txt; do
  curl -fsSL -o "$f" "https://raw.githubusercontent.com/vllm-project/vllm/${BASE_SHA}/${f}"
done
CUDA_SHA=$(sha256sum requirements/cuda.txt | cut -d' ' -f1)
COMMON_SHA=$(sha256sum requirements/common.txt | cut -d' ' -f1)
BUILD_SHA=$(sha256sum requirements/build/cuda.txt | cut -d' ' -f1)

cat > extra.in <<'EOF'
# Additions for the task image, locked together with vLLM's own dependency set.
# pytest runs the vendored upstream acceptance test; tblib is imported by tests/conftest.py,
# which pytest loads for any module under tests/.
pytest
tblib
# Transformers is pinned at 5.12.1. requirements/common.txt asks only for
# >= 5.10.4, and an unconstrained resolve takes 5.17.0 -- which is the release that carries the
# model-author qwen4_exp implementation. Shipping that in the engine environment would put the
# reference implementation in the agent's own interpreter, would stop the unsolved base failing
# with the config error the task is defined by, and would conflate the engine environment with the
# fidelity reference, which is a DIFFERENT, newer Transformers in its own authoring venv.
transformers==5.12.1
# The CUDA 13 builds of the torch family from the PyTorch index. requirements/cuda.txt pins the
# plain versions and upstream's docker/Dockerfile installs them with
# --extra-index-url https://download.pytorch.org/whl/cu130; naming the local version here stops
# the resolver preferring the PyPI build under --index-strategy unsafe-best-match.
torch==2.13.0+cu130
torchvision==0.28.0+cu130
torchaudio==2.11.0+cu130
EOF

"$UV" pip compile requirements/cuda.txt requirements/build/cuda.txt extra.in --generate-hashes \
  --python "$PY" \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  --extra-index-url https://flashinfer.ai/whl/cu130 \
  --extra-index-url https://flashinfer.ai/whl \
  --index-strategy unsafe-best-match \
  -o "$OUT/requirements.lock"

cat > evalscope.in <<'EOF'
evalscope==1.10.0
EOF
"$UV" pip compile evalscope.in --generate-hashes --python "$PY" \
  --index-url https://pypi.org/simple -o "$OUT/requirements-evalscope.lock"

HFV=$(grep -oE '^huggingface[-_]hub==[0-9A-Za-z.+]+' "$OUT/requirements.lock" | head -1)
printf '%s\nhf_transfer\n' "$HFV" > tools.in
"$UV" pip compile tools.in --generate-hashes --python "$PY" --index-url https://pypi.org/simple \
  -o "$OUT/requirements-tools.lock"

{
  echo "# Lock provenance"
  echo
  echo "- resolved: $(date -u +%Y-%m-%dT%H:%M:%SZ) inside nvidia/cuda:13.0.3-devel-ubuntu24.04 (x86_64, $($PY --version))"
  echo "- resolver: uv ${UV_VERSION} (pip compile, --generate-hashes)"
  echo "- inputs at vllm ${BASE_SHA}:"
  echo "  - requirements/cuda.txt        sha256 ${CUDA_SHA}"
  echo "  - requirements/common.txt      sha256 ${COMMON_SHA}   (pulled in by cuda.txt's -r)"
  echo "  - requirements/build/cuda.txt  sha256 ${BUILD_SHA}"
  echo "- additions (extra.in): pytest, tblib, transformers==5.12.1, torch/torchvision/torchaudio +cu130"
  echo "- indexes: https://pypi.org/simple, https://download.pytorch.org/whl/cu130,"
  echo "           https://flashinfer.ai/whl/cu130, https://flashinfer.ai/whl (unsafe-best-match)"
  echo "- evalscope: 1.10.0 from PyPI, resolved separately for its own venv"
  echo "- tools: ${HFV}"
} > "$OUT/RESOLVED.md"
echo "resolved: $(wc -l < "$OUT/requirements.lock") lines in requirements.lock, $(wc -l < "$OUT/requirements-evalscope.lock") in requirements-evalscope.lock"
