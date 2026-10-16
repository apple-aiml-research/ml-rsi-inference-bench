#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Resolve the hash-locked wheel sets environment/Dockerfile installs, inside the pinned CUDA base
# image (the target platform and interpreter), from public indexes only. Run once at authoring:
#
#   podman run --rm -v <this dir>:/out -v <this file>:/resolve.sh \
#     docker.io/nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04@sha256:0f825061... bash /resolve.sh
#
# Writes requirements.lock (SGLang's own dependency set at the base commit, with the REFERENCE's
# torch and sgl-kernel pins overriding the base pyproject's, plus what the vendored acceptance test
# and the verifier need) and requirements-tools.lock (huggingface_hub for the checkpoint download),
# plus RESOLVED.md recording the inputs. The pyproject is fetched from GitHub at the base commit.
set -euo pipefail
BASE_SHA=556e4143f0ba8e480a73fc0e08f05d7ea83880cc
UV_VERSION=0.9.28
# The resolve is held to the LANDING ERA, and this is the load-bearing knob rather than a nicety.
# SGLang's pyproject leaves most of its dependencies unconstrained, so an unbounded resolve mixes an
# August-2025 engine with 2026 releases of its own dependency tree -- and those releases have moved
# on. Measured on a build attempt: `compressed-tensors`, which the base lists
# with no constraint at all, resolved to 0.19.0, which requires `torch>=2.10.0`, and the wheel
# install died with ResolutionImpossible against this task's torch 2.8.0. uv's --override had hidden
# it, because an override discards EVERY requirement on the overridden package, including other
# packages' constraints on it; pip then enforces them and refuses.
EXCLUDE_NEWER=${EXCLUDE_NEWER:-2025-08-10T00:00:00Z}
EXEMPT_DATE=2030-01-01T00:00:00Z
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
# Additions for the task image, locked together with SGLang's own dependency set.
#
# The verifier's runner and the editable install's build backend (installed with
# --no-build-isolation, so the backend has to be in the image). The base's build-system is plain
# setuptools -- no setuptools-rust, unlike the later SGLang bases -- so no Rust toolchain and no
# protoc are needed anywhere in this recipe.
pytest
setuptools>=61.0
wheel

# What the VENDORED acceptance test imports, stated rather than inherited: the GPQA evaluator reads
# its CSV with pandas and talks to the server through the OpenAI client.
pandas
# openai: the floor is sgl-project/sglang#8837's declared `openai>=1.99.1`, and the CEILING is this
# recipe's, with a reason. The client is not incidental here -- it is what the frozen test speaks through
# (`OpenAI(base_url=..., http_client=LargerHttpxClient())`, `openai.BadRequestError`,
# `reasoning_effort=` passed as a body field) and what sgl-project/sglang#8837's serving path imports at
# start-up (`openai.types.responses`). Both are August-2025 code against the 1.x client; an unconstrained
# resolve takes 3.19.0, two major versions past the landing, whose constructor and
# `types.responses` layout are not what either was written against. The instruction promises "the
# dependency stack the maintainers used the day the model landed", so the resolve is held to that
# era. 1.99.1 shipped 2025-08-08, two days after the reference.
openai>=1.99.1,<2
openai-harmony==0.0.3
EOF

cat > overrides.txt <<'EOF'
# The base tree's pyproject pins torch==2.7.1 and sgl-kernel==0.2.8, and the reference needs
# neither: sgl-project/sglang#8824's own PR body tells users "Additional install for gpt-oss:
# torch==2.8.0 (cu126 test index), sgl-kernel==0.3.2", and sgl-project/sglang#8836 bumped exactly
# those two in the tree three PRs after the base -- a commit this task deliberately does not carry
# (it is one of the seven unrelated PRs interleaved with the landing commits). sgl-kernel 0.3.2 is
# also what VENDORS OpenAI's triton_kernels package with MXFP4 matmul_ogs, which the closure rule
# supplies rather than asks for. The instruction states this stack, so the agent is not asked to
# discover it.
#
# The LOCAL VERSIONS are part of the pin, not decoration. download.pytorch.org publishes no upload
# timestamps, so --exclude-newer drops its files; without `+cu126` here the resolve silently took
# torch 2.8.0 from PyPI -- which bundles cu128 -- beside torchvision 0.23.0+cu126 from the PyTorch
# index, an incoherent pair, and sgl-kernel 0.3.2 was built against cu126. Naming the local version
# makes the cu126 wheel the only candidate, so the resolve either produces the reference's stack or
# fails loudly instead of substituting a different CUDA.
torch==2.8.0+cu126
torchaudio==2.8.0+cu126
torchvision==0.23.0+cu126
sgl-kernel==0.3.2
EOF

"$UV" pip compile pyproject.toml extra.in --extra srt --override overrides.txt --generate-hashes \
  --python "$PY" --exclude-newer "$EXCLUDE_NEWER" \
  --exclude-newer-package "torch=$EXEMPT_DATE" \
  --exclude-newer-package "torchaudio=$EXEMPT_DATE" \
  --exclude-newer-package "torchvision=$EXEMPT_DATE" \
  --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu126 \
  --index-strategy unsafe-best-match \
  -o "$OUT/requirements.lock"

# The TOOLS venv is deliberately NOT held to the landing era, and it is the one place that
# distinction is worth stating. /opt/infb/tools exists only so this recipe can download and hash the
# checkpoint at build time: it is not on PATH, the engine never imports from it, and nothing the
# verifier measures can see it. The era bound exists to keep the ENGINE's stack the one the
# maintainers had; applying it to the build's own downloader would buy nothing and would cost the
# newer Hub client's transfer speed. Keeping the two independent also keeps the checkpoint layers
# cacheable when the engine lock is re-resolved, since this file is copied before them.
printf '%s\n' huggingface_hub > tools.in
"$UV" pip compile tools.in --generate-hashes --python "$PY" \
  --index-url https://pypi.org/simple -o "$OUT/requirements-tools.lock"
HFV=$(grep -oE '^huggingface[-_]hub==[0-9A-Za-z.+]+' "$OUT/requirements-tools.lock" | head -1)

{
  echo "# Lock provenance"
  echo
  echo "- resolved: $(date -u +%Y-%m-%dT%H:%M:%SZ) inside nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04 (x86_64, $($PY --version))"
  echo "- resolver: uv ${UV_VERSION} (pip compile, --generate-hashes)"
  echo "- input: python/pyproject.toml at sglang ${BASE_SHA}, extra 'srt', sha256 ${PYPROJECT_SHA}"
  echo "- additions (extra.in): pytest, setuptools, wheel, pandas, openai>=1.99.1 and <2 (the landing era; see extra.in), openai-harmony==0.0.3"
  echo "- overrides: torch==2.8.0+cu126, torchaudio==2.8.0+cu126, torchvision==0.23.0+cu126, sgl-kernel==0.3.2 (the reference's stack, local versions included; see overrides.txt)"
  echo "- indexes: https://pypi.org/simple, https://download.pytorch.org/whl/cu126 (unsafe-best-match)"
  echo "- exclude-newer: ${EXCLUDE_NEWER} (the landing era; see the header for why this is load-bearing)"
  echo "- exclude-newer-package: torch, torchaudio, torchvision at ${EXEMPT_DATE} -- i.e. unbounded, because download.pytorch.org publishes no upload dates and all three are pinned to an exact version with its local version"
  echo "- tools: ${HFV} (NOT era-bounded; see the note above tools.in)"
  echo
  echo "## The pins this resolve produced, for the record"
  echo
  for p in torch torchvision torchaudio triton sgl-kernel flashinfer-python transformers tokenizers \
           numpy pandas openai openai-harmony xgrammar torchao safetensors huggingface-hub pytest; do
    v=$(grep -iE "^${p}==" "$OUT/requirements.lock" | head -1)
    [ -n "$v" ] && echo "- ${v}"
  done
} > "$OUT/RESOLVED.md"
echo "resolved: $(wc -l < "$OUT/requirements.lock") lines in requirements.lock"
