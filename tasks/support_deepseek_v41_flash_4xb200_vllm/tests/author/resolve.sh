#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Resolve the hash-locked wheel set of the model-author runtime (/opt/model-author/venv), inside the pinned CUDA base
# image, from public indexes only. Run once at authoring, like ../lock/resolve.sh:
#
#   docker run --rm -v <this dir>:/out -v <this file>:/resolve.sh -v <../lock>:/engine-lock:ro \
#     docker.io/nvidia/cuda:13.0.3-devel-ubuntu24.04@sha256:b7ae301d... bash /resolve.sh
#
# The inputs are the author implementation's own requirements (inference/requirements.txt at the checkpoint revision:
# torch, transformers, tokenizers, safetensors, numpy, sympy, Pillow, tilelang, tqdm) at the versions of the author run
# that recorded this task's reference log-probabilities (the fidelity data's provenance: torch 2.13.0+cu130,
# TileLang 0.1.12 with apache-tvm-ffi 0.1.11 -- not the 0.1.8 the file names, which returns corrupt values on B200 --
# numpy 2.5.3, safetensors 0.8.0, tokenizers 0.23.2, transformers 5.17.0). Every other package is held to the version
# the engine's own lock resolved (../lock/requirements.lock), so the author environment shares the engine's recorded
# CUDA/PyTorch stack; every transitive version and hash is captured. Writes requirements.lock and RESOLVED.md.
set -euo pipefail
UV_VERSION=0.9.28
OUT=${OUT:-/out}
apt-get update -qq && apt-get install -y -qq --no-install-recommends python3.12-venv curl ca-certificates >/dev/null
python3.12 -m venv /tmp/uvenv
/tmp/uvenv/bin/pip install -q "uv==${UV_VERSION}"
UV=/tmp/uvenv/bin/uv
PY=/tmp/uvenv/bin/python
mkdir -p /tmp/src && cd /tmp/src
cat > author.in <<'EOT'
# The model-author implementation's requirements (inference/requirements.txt), at the recorded author run's versions.
torch==2.13.0+cu130
transformers==5.17.0
tokenizers==0.23.2
safetensors==0.8.0
numpy==2.5.3
sympy
Pillow
tilelang==0.1.12
apache-tvm-ffi==0.1.11
tqdm
EOT
# Every other package at the engine lock's version (numpy is the recorded author run's own, above).
grep -oE '^[A-Za-z0-9_.-]+==[^ ]+' /engine-lock/requirements.lock | grep -viE '^numpy==' > constraints.txt
"$UV" pip compile author.in --constraint constraints.txt --generate-hashes \
  --python "$PY" \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  --index-strategy unsafe-best-match \
  -o "$OUT/requirements.lock"
{
  echo "# The model-author runtime's lock, resolved $(date -u +%FT%TZ)"
  echo
  echo "uv ${UV_VERSION}; python $($PY -c 'import platform; print(platform.python_version())'); inputs: author.in (below) with every other"
  echo "package constrained to ../lock/requirements.lock (sha256 $(sha256sum /engine-lock/requirements.lock | cut -c1-64))."
  echo
  echo '```'
  cat author.in
  echo '```'
  echo
  echo "Resolved $(grep -cE '^[A-Za-z0-9_.-]+==' "$OUT/requirements.lock") packages; requirements.lock sha256 $(sha256sum "$OUT/requirements.lock" | cut -c1-64)."
} > "$OUT/RESOLVED.md"
cat "$OUT/RESOLVED.md"
