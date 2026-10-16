#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Resolve the hash-locked wheel sets environment/Dockerfile installs, inside the pinned CUDA base
# image (the target platform and interpreter), from public indexes only. Run once at authoring:
#
#   podman run --rm --network host -v <this dir>:/out -v <this file>:/resolve.sh \
#     docker.io/nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04@sha256:8b2705ea... bash /resolve.sh
#
# Writes requirements.lock (SGLang's dependency set plus pytest and the editable install's build
# backend), requirements-flashinfer.lock (FlashInfer's prebuilt cubins and JIT cache from
# flashinfer.ai) and requirements-tools.lock (huggingface_hub for the checkpoint download), plus
# RESOLVED.md recording the inputs.
#
# The pyproject is fetched at the landing squash. The Inkling landing's only dependency
# change against the base is `helion==0.2.6` and `numba==0.65.1` (git diff 829e9ce9..02236fa3 --
# python/pyproject.toml; its other hunk is the unbuilt `sglang.srt.multimodal._core` Rust extension).
# Both libraries are supplied in the image and the instruction says so; the
# agent's checkout is the base and its pyproject is the base one. The resolution is made as of the
# landing's merge (`--exclude-newer`): the human integrator's state.
set -euo pipefail
REF_SHA=02236fa38cb065660a0b4156aab283a3807a5290
EXCLUDE_NEWER=2026-07-20T05:57:37Z
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
# the pyproject (2.11.0); torchvision is not, and upstream's docker/Dockerfile at the base reinstalls
# all three from the cu130 index at torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0, which is
# what these three lines do at resolve time instead.
torch==2.11.0+cu130
torchvision==0.26.0+cu130
torchaudio==2.11.0+cu130
EOF

# --exclude-newer applies to indexes that publish upload times (PyPI). The PyTorch index publishes
# none, and the torch family is pinned exactly above, so no later build of theirs can enter.
"$UV" pip compile pyproject.toml extra.in --generate-hashes \
  --python "$PY" \
  --exclude-newer "$EXCLUDE_NEWER" \
  --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu130 \
  --index-strategy unsafe-best-match \
  -o "$OUT/requirements.lock"

# FlashInfer's cubins (CUDA-version agnostic, https://flashinfer.ai/whl) and its cu130 JIT cache, at the
# pyproject's flashinfer_python version, as upstream's docker/Dockerfile installs them.
cat > flashinfer.in <<'EOF'
flashinfer-cubin==0.6.14
flashinfer-jit-cache==0.6.14
EOF
"$UV" pip compile flashinfer.in --no-deps --generate-hashes \
  --python "$PY" \
  --index-url https://flashinfer.ai/whl/cu130 --extra-index-url https://flashinfer.ai/whl \
  --index-strategy unsafe-best-match \
  -o "$OUT/requirements-flashinfer.lock"

HFV=$(grep -oE '^huggingface[-_]hub==[0-9A-Za-z.+]+' "$OUT/requirements.lock" | head -1)
printf '%s\n' "$HFV" > tools.in
"$UV" pip compile tools.in --generate-hashes --python "$PY" --exclude-newer "$EXCLUDE_NEWER" \
  --index-url https://pypi.org/simple \
  -o "$OUT/requirements-tools.lock"

pin() { grep -oiE "^$1==[0-9A-Za-z.+]+" "$OUT/requirements.lock" | head -1; }
{
  echo "# Lock provenance"
  echo
  echo "- resolved: $(date -u +%Y-%m-%dT%H:%M:%SZ) inside nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04 (x86_64, $($PY --version))"
  echo "- resolver: uv ${UV_VERSION} (pip compile, --generate-hashes, --exclude-newer ${EXCLUDE_NEWER} for PyPI)"
  echo "- input: python/pyproject.toml at sglang ${REF_SHA} (the landing squash: the base's pins plus helion==0.2.6 and numba==0.65.1), sha256 ${PYPROJECT_SHA}"
  echo "- additions (extra.in): pytest, setuptools, setuptools-rust, setuptools-scm, wheel, torch 2.11.0 / torchvision 0.26.0 / torchaudio 2.11.0 +cu130"
  echo "- overrides: none"
  echo "- indexes: https://pypi.org/simple, https://download.pytorch.org/whl/cu130 (unsafe-best-match); flashinfer: https://flashinfer.ai/whl/cu130, https://flashinfer.ai/whl"
  echo "- asserted pins: $(pin nvidia-cutlass-dsl) $(pin flash-attn-4) $(pin transformers) $(pin quack-kernels) $(pin helion) $(pin numba) $(pin apache-tvm-ffi) $(pin flashinfer-python) $(pin sglang-kernel) $(pin tilelang) $(pin torch) $(pin kernels)"
  echo "- tools: ${HFV}"
} > "$OUT/RESOLVED.md"
echo "resolved: $(wc -l < "$OUT/requirements.lock") lines in requirements.lock"
