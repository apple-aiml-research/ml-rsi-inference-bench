#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Resolve the hash-locked wheel sets environment/Dockerfile installs, inside the pinned CUDA base image (the target
# platform and interpreter), from public indexes only. Run once at authoring:
#
#   podman run --rm -v <this dir>:/out -v <this file>:/resolve.sh \
#     docker.io/nvidia/cuda:13.0.3-devel-ubuntu24.04@sha256:b7ae301d... bash /resolve.sh
#
# requirements.lock: vLLM's dependency set at the base commit. Every package requirements/test/cuda.txt pins (the
#   base's own compiled CI environment: torch 2.11.0+cu130, triton 3.6.0, transformers 5.5.3, jsonschema, pytest,
#   tblib and the rest) enters at exactly that pin; the remainder of requirements/cuda.txt (with common.txt) and
#   requirements/build/cuda.txt, which the no-build-isolation editable install needs, is resolved with
#   --exclude-newer at the base commit's time, so nothing published after the starting point is chosen.
#   transformers==5.5.3 is restated explicitly: 5.12.0 and later carry this model.
# requirements-tools.lock: huggingface_hub (the engine lock's own version) for the checkpoint fetch.
# requirements-flashinfer-jit-cache.lock: flashinfer-jit-cache at the engine's FlashInfer release, from its
#   CUDA 13 index, in its own layer (the base's docker/Dockerfile installs the same wheel the same way).
# RESOLVED.md records the inputs and their digests.
set -euo pipefail
BASE_SHA=a3195fab7b1227e75403fef83891e961e69228a6
EXCLUDE_NEWER=2026-06-15T16:37:52Z
UV_VERSION=0.9.28
OUT=${OUT:-/out}

apt-get update -qq && apt-get install -y -qq --no-install-recommends python3.12-venv curl ca-certificates >/dev/null
python3.12 -m venv /tmp/uvenv
/tmp/uvenv/bin/pip install -q "uv==${UV_VERSION}"
UV=/tmp/uvenv/bin/uv
PY=/tmp/uvenv/bin/python

mkdir -p /tmp/src/requirements/build /tmp/src/requirements/test "$OUT" && cd /tmp/src
for f in requirements/cuda.txt requirements/common.txt requirements/build/cuda.txt requirements/test/cuda.txt; do
  curl -fsSL -o "$f" "https://raw.githubusercontent.com/vllm-project/vllm/${BASE_SHA}/${f}"
done
sha() { sha256sum "$1" | cut -d' ' -f1; }

# The CI pins as hard requirements (so every one of them is installed at its pin), without their `# via` comments.
grep -E '^[A-Za-z0-9_.-]+(\[[A-Za-z0-9_,.-]+\])?==' requirements/test/cuda.txt > ci-pins.in
cat > extra.in <<'EOF'
# Restated: requirements/common.txt asks only for >= 5.5.3, and 5.12.0 and later carry the model this task asks for.
transformers==5.5.3
EOF

"$UV" pip compile ci-pins.in requirements/cuda.txt requirements/build/cuda.txt extra.in --generate-hashes \
  --python "$PY" \
  --exclude-newer "$EXCLUDE_NEWER" \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  --extra-index-url https://flashinfer.ai/whl/cu130 \
  --index-strategy unsafe-best-match \
  -o "$OUT/requirements.lock"

HFV=$(grep -oE '^huggingface[-_]hub==[0-9A-Za-z.+]+' "$OUT/requirements.lock" | head -1)
printf '%s\n' "$HFV" > tools.in
"$UV" pip compile tools.in --generate-hashes --python "$PY" --exclude-newer "$EXCLUDE_NEWER" \
  --index-url https://pypi.org/simple -o "$OUT/requirements-tools.lock"

FIV=$(grep -oE '^flashinfer-python==[0-9A-Za-z.+]+' "$OUT/requirements.lock" | head -1 | cut -d= -f3)
printf 'flashinfer-jit-cache==%s\n' "$FIV" > jit.in
"$UV" pip compile jit.in --generate-hashes --python "$PY" --no-deps \
  --index-url https://flashinfer.ai/whl/cu130 -o "$OUT/requirements-flashinfer-jit-cache.lock"

{
  echo "# Lock provenance"
  echo
  echo "- resolved: $(date -u +%Y-%m-%dT%H:%M:%SZ) inside nvidia/cuda:13.0.3-devel-ubuntu24.04 (x86_64, $($PY --version))"
  echo "- resolver: uv ${UV_VERSION} (pip compile, --generate-hashes)"
  echo "- inputs at vllm ${BASE_SHA} (the base commit, committed ${EXCLUDE_NEWER}):"
  echo "  - requirements/test/cuda.txt   sha256 $(sha requirements/test/cuda.txt)   (every '==' pin, $(wc -l < ci-pins.in) packages)"
  echo "  - requirements/cuda.txt        sha256 $(sha requirements/cuda.txt)"
  echo "  - requirements/common.txt      sha256 $(sha requirements/common.txt)   (pulled in by cuda.txt's -r)"
  echo "  - requirements/build/cuda.txt  sha256 $(sha requirements/build/cuda.txt)"
  echo "- --exclude-newer ${EXCLUDE_NEWER}; transformers==5.5.3 restated"
  echo "- indexes: https://pypi.org/simple, https://download.pytorch.org/whl/cu130, https://flashinfer.ai/whl/cu130 (unsafe-best-match)"
  echo "- tools: ${HFV}"
  echo "- flashinfer-jit-cache: ${FIV} from https://flashinfer.ai/whl/cu130 (--no-deps)"
} > "$OUT/RESOLVED.md"
echo "resolved: $(grep -c '==' "$OUT/requirements.lock") pins in requirements.lock"
