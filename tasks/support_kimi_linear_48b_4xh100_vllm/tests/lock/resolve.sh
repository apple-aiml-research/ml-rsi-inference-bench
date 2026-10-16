#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Resolve this task's dependency set ONCE, inside the pinned CUDA base image, and emit fully
# hash-pinned locks. The recipe then installs them with `--require-hashes --no-deps`, so the image's
# dependency set is a decision taken here and recorded, not whatever an index served on the day of a
# build.
#
#   podman run --rm --network host -v "$PWD:/lock" -v <vllm requirements tree>:/src:ro \
#     docker.io/nvidia/cuda:12.9.1-devel-ubuntu24.04@sha256:020bc241a628776338f4d4053fed4c38f6f7f3d7eb5919fecb8de313bb8ba47c \
#     bash /lock/resolve.sh
#
# Provenance and every decision below are in RESOLVED.md. This file stays in the build context as
# provenance and is NEVER copied into the image: only the .lock files enter it (the answer-address
# rule -- a resolver script is a natural place for a reference commit to hide).
set -euo pipefail

# The instant the resolution is taken at; it is what keeps transformers on the 4.x line (D-LOCK-2).
EXCLUDE_NEWER=${EXCLUDE_NEWER:-2025-10-31T00:00:00Z}
PY=${PY:-3.12}
SRC=${SRC:-/src}
OUT=${OUT:-/lock/requirements.lock}
OUT_FI=${OUT_FI:-/lock/requirements-flashinfer.lock}

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends -qq ca-certificates curl python3.12-full python3.12-dev >/dev/null
curl -fsSL --retry 5 https://astral.sh/uv/0.9.28/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh >/dev/null
uv --version

TORCH_CHANNEL=https://download.pytorch.org/whl/cu129
# The artifact the vendor's own PEP 503 index (flashinfer.ai/whl/flashinfer-python/) links to: the
# project's public GitHub release asset. Read from that index page rather than guessed.
FLASHINFER_WHEEL=https://github.com/flashinfer-ai/flashinfer/releases/download/v0.4.1/flashinfer_python-0.4.1-py3-none-any.whl

INDEXES=(
  --index-url https://pypi.org/simple
  --extra-index-url https://download.pytorch.org/whl/cu129
  --extra-index-url https://flashinfer.ai/whl
)

# ---------------------------------------------------------------------------- pass 1: the engine set
#
# D-LOCK-1: flashinfer is resolved SEPARATELY, in pass 2, and is therefore stripped from the engine
# input here. The reason is measured, not stylistic: `flashinfer-python==0.4.1` (which
# requirements/cuda.txt pins at the base commit) declares `apache-tvm-ffi==0.1.0b15`, and that
# pre-release **no longer exists on PyPI** -- its earliest surviving release is the stable 0.1.0 of
# 2025-10-21 (PyPI JSON API, read 2026-09-23). So the engine's own declared set is unsatisfiable by
# any resolver today, at any --prerelease setting, and a single-pass resolve fails with
# "there is no version of apache-tvm-ffi==0.1.0b15". Pass 2 pins the wheel and the surviving
# apache-tvm-ffi together with --no-deps, which is exactly how the recipe installs them.
#
# The strip is done in a WRITABLE COPY of the whole requirements directory, not in /tmp: cuda.txt's
# first line is a relative `-r common.txt`, which uv resolves against the including file's own
# directory, so a stripped copy somewhere else cannot find it.
mkdir -p /tmp/req
cp "${SRC}"/requirements/*.txt /tmp/req/
sed -i '/^flashinfer-python==/d' /tmp/req/cuda.txt
diff -u "${SRC}/requirements/cuda.txt" /tmp/req/cuda.txt || true

cat > /tmp/requirements.in <<IN
-r /tmp/req/cuda.txt
-r /tmp/req/build.txt

# The vendored upstream L0 test's conftest chain: tests/conftest.py imports tblib, and pytest runs
# it. Everything else that chain imports (numpy, torch, huggingface_hub, PIL, transformers) is
# already in the engine's own set. Read in the source, not assumed.
pytest
tblib

# D-LOCK-2: transformers is pinned, not merely bounded. requirements/common.txt asks for ">= 4.56.0"
# with no upper bound, and two things depend on the exact line:
#   - the engine reads tokenizer internals that the 5.x line removed (measured, not assumed);
#   - no RELEASED transformers through v5.2.0 carries a native models/kimi_linear, but the
#     development head DOES (GitHub contents API, 2026-09-23: v4.57.1, v4.58.0, v4.59.0, v5.0.0 and
#     v5.2.0 all 404; main 200). A resolve that ever reached such a release would put a working
#     implementation of the very model this task asks for on an interpreter's path.
# The pin and environment/checks/no_reference_impl.py are the two halves of that guard.
transformers==4.57.1

# D-LOCK-3: the torch triple comes in as DIRECT WHEEL URLS from the cu129 channel, not as version
# specifiers. Two measurements forced this, both taken here on 2026-09-23:
#   - with only upstream's `torch==2.9.0` to go on, uv mixed the channels: PyPI's plain torch 2.9.0
#     and torchvision 0.24.0 (which bundle the cu128 runtime) beside the cu129 channel's
#     torchaudio 2.9.0+cu129. This image's toolkit is CUDA 12.9.1 and the engine links torch's
#     bundled runtime, so a cu128 torch here is a mismatch waiting for a runtime error.
#   - pinning `torch==2.9.0+cu129` instead FAILS under --exclude-newer: that index serves no upload
#     dates, so the cutoff drops its files and uv reports "there is no version of torch==2.9.0+cu129".
#     Without the cutoff the same pin resolves in 2 ms. Isolated with a three-way probe.
# A direct URL is exempt from the cutoff (the pin already fixes the artifact, which is what the
# cutoff exists to do), --generate-hashes still hashes it, and `torch==2.9.0` from
# requirements/cuda.txt is satisfied by 2.9.0+cu129 under PEP 440, so nothing else has to change.
torch @ ${TORCH_CHANNEL}/torch-2.9.0%2Bcu129-cp312-cp312-manylinux_2_28_x86_64.whl
torchvision @ ${TORCH_CHANNEL}/torchvision-0.24.0%2Bcu129-cp312-cp312-manylinux_2_28_x86_64.whl
torchaudio @ ${TORCH_CHANNEL}/torchaudio-2.9.0%2Bcu129-cp312-cp312-manylinux_2_28_x86_64.whl
IN

uv pip compile /tmp/requirements.in \
  --output-file "$OUT" \
  --python-version "$PY" \
  --generate-hashes --emit-index-url --no-header \
  --exclude-newer "$EXCLUDE_NEWER" \
  --index-strategy unsafe-best-match \
  "${INDEXES[@]}"

# ------------------------------------------------------------------- pass 2: flashinfer, dependency-free
#
# `--no-deps` resolves only what is written here, so flashinfer's withdrawn pre-release requirement is
# never consulted. Both lines are hash-generated from the indexes like every other line; neither is a
# hand-edited hash. apache-tvm-ffi 0.1.0 is the surviving release of the line 0.1.0b15 belonged to,
# and whether flashinfer 0.4.1 actually imports and serves against it is a MEASUREMENT taken in the
# image smoke and again when the oracle serves.
#
# D-LOCK-4: the flashinfer wheel is a DIRECT URL, for the same reason as the torch triple and proven on
# a second index. Resolved by NAME under --exclude-newer, uv could not see flashinfer.ai's wheel -- that
# index serves no upload dates either -- and fell back to PyPI's SDIST: the first lock recorded
# fd120f05..., which is the tar.gz's digest, where the wheel's is 5b44d28e... . An sdist here is fatal
# rather than merely slower, and fatal in a way `--no-deps` cannot prevent: pip BUILDS it, and build
# requirements are not suppressed by --no-deps, so the build resolved `apache-tvm-ffi==0.1.0b15` and
# failed with "No matching distribution found" -- the withdrawn pre-release arriving by the one door
# left open.
# A direct URL to the pure-Python wheel means pip installs an artifact and builds nothing.
cat > /tmp/flashinfer.in <<IN
flashinfer-python @ ${FLASHINFER_WHEEL}
apache-tvm-ffi==0.1.0
IN

# --no-cache, because uv caches a resolution by package name and version: after the first, sdist-based
# resolve it handed back the SDIST's digest for a direct WHEEL url. Measured, and the generated digest is
# checked against an independent download of the bytes (5b44d28e..., 6,849,339 B).
uv pip compile /tmp/flashinfer.in \
  --output-file "$OUT_FI" \
  --python-version "$PY" \
  --generate-hashes --emit-index-url --no-header --no-deps --no-cache \
  --exclude-newer "$EXCLUDE_NEWER" \
  --index-strategy unsafe-best-match \
  "${INDEXES[@]}"

printf '\nengine lock: %s packages\n' "$(grep -cE '^[a-zA-Z0-9]' "$OUT")"
grep -E '^(torch|torchvision|torchaudio|transformers|tokenizers|triton|numba|ray|safetensors|numpy|setuptools|pytest|tblib|huggingface-hub|aiohttp|compressed-tensors)==' "$OUT" || true
printf '\nflashinfer lock:\n'; grep -E '^[a-zA-Z0-9]' "$OUT_FI" || true

# ----------------------------------------------------------- pass 3: the recipe's own tooling venv
#
# Only what THIS RECIPE runs: huggingface_hub, for the checkpoint fetch. It is a separate interpreter
# from the engine's so that the fetch tooling can never move the engine's resolved set, and it is
# resolved here rather than pinned by hand so its own transitive set is hashed too.
cat > /tmp/tools.in <<IN
huggingface-hub==0.36.0
IN

uv pip compile /tmp/tools.in \
  --output-file "${OUT_TOOLS:-/lock/requirements-tools.lock}" \
  --python-version "$PY" \
  --generate-hashes --emit-index-url --no-header \
  --exclude-newer "$EXCLUDE_NEWER" \
  --index-url https://pypi.org/simple

printf '\ntools lock: %s packages\n' "$(grep -cE '^[a-zA-Z0-9]' "${OUT_TOOLS:-/lock/requirements-tools.lock}")"
