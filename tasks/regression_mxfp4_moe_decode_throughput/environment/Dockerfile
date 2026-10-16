#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        curl \
        g++ \
        gcc \
        git \
        libnuma1 \
        make \
        ninja-build \
        patch \
        rsync \
 && rm -rf /var/lib/apt/lists/*

RUN set -eu; \
    src=""; \
    for cand in /usr/local/bin/python3.12 /usr/bin/python3.12; do \
        [ -x "$cand" ] && { src="$cand"; break; }; \
    done; \
    [ -n "$src" ] || { echo "no python3.12 in this base image" >&2; exit 1; }; \
    [ "$src" = /usr/bin/python3.12 ] || ln -sf "$src" /usr/bin/python3.12; \
    ln -sf "$src" /usr/bin/python3; \
    ln -sf "$src" /usr/local/bin/python3; \
    ln -sf "$src" /usr/local/bin/python; \
    python3 --version && python --version

ENV INFB_VENV=/opt/venv
ENV INFB_VLLM_PYTHON=${INFB_VENV}/bin/python
# Where the build-time artefacts live once the engine Python is gone. Read by
# scripts/stage_engine_extensions.sh, which both this build and prepare_target_env.sh call.
ENV INFB_VLLM_ENGINE_EXT=/opt/inference-bench/engine-ext

ARG ENGINE_COMMIT=c8dc4d2d1f1ab32626f43a04accc67917078b22d
ARG ENGINE_WHEEL_VERSION=0.5.7
# LOAD-BEARING, AND THE SINGLE MOST FRAGILE NUMBER IN THIS IMAGE. See the assertion below.
ARG SGL_KERNEL_VERSION=0.3.20

RUN python3.12 -m venv --copies "${INFB_VENV}" \
 && ( "${INFB_VENV}/bin/pip" install --upgrade pip setuptools wheel \
   && "${INFB_VENV}/bin/pip" install "sglang==${ENGINE_WHEEL_VERSION}" \
   )

RUN set -eu; \
    "${INFB_VENV}/bin/python" -c "import torch; print('[engine] torch', torch.__version__, 'built for CUDA', torch.version.cuda)"; \
    v="$("${INFB_VENV}/bin/python" -c 'import importlib.metadata as m; print(m.version("sglang"))')"; \
    [ "$v" = "${ENGINE_WHEEL_VERSION}" ] \
      || { echo "FATAL: installed sglang is $v, expected ${ENGINE_WHEEL_VERSION}" >&2; exit 8; }; \
    "${INFB_VENV}/bin/python" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('sgl_kernel') else 1)" \
      || { echo "FATAL: sgl_kernel is not installed; this engine version keeps every compiled kernel there" >&2; exit 8; }; \
    k="$("${INFB_VENV}/bin/python" -c 'import importlib.metadata as m; print(m.version("sgl-kernel"))')"; \
    [ "$k" = "${SGL_KERNEL_VERSION}" ] \
      || { echo "FATAL: sgl-kernel is $k, expected ${SGL_KERNEL_VERSION} -- see the triton_kernels assertion below" >&2; exit 8; }; \
    echo "[engine] sglang $v with sgl-kernel $k"

RUN set -eu; \
    "${INFB_VENV}/bin/python" -c "import importlib.util,sys; \
        spec=importlib.util.find_spec('triton_kernels'); \
        sys.exit(0 if spec else 1); " \
      || { echo "FATAL: the scored path is gated on triton_kernels and it is absent" >&2; exit 8; }; \
    "${INFB_VENV}/bin/python" -c "import importlib.util; print('[triton-kernels] resolves to', importlib.util.find_spec('triton_kernels').origin)"

COPY scripts/stage_engine_extensions.sh /opt/inference-bench/stage_engine_extensions.sh
RUN bash /opt/inference-bench/stage_engine_extensions.sh harvest "${INFB_VENV}"

# GATED ON CONTENT, NOT ON THE TARBALL'S BYTES. GitHub's codeload archives are not guaranteed
# byte-stable over time, so pinning the archive digest would turn a cosmetic upstream change into an
# unbuildable image. What is pinned is transport-independent: the sha256 of the file the fix touches.
ARG ENGINE_SRC_SHA256=8a835ab71a5d4ea9
RUN set -eu; \
    mkdir -p /opt/engine-src; \
    if ! curl -fsSL -o /tmp/engine-src.tar.gz \
         "https://codeload.github.com/sgl-project/sglang/tar.gz/${ENGINE_COMMIT}"; then \
        echo "[engine-src] codeload unavailable; falling back to the git protocol" >&2; \
        git init -q /tmp/engine-git; \
        git -C /tmp/engine-git remote add origin https://github.com/sgl-project/sglang.git; \
        git -C /tmp/engine-git fetch -q --depth 1 origin "${ENGINE_COMMIT}"; \
        git -C /tmp/engine-git checkout -q FETCH_HEAD; \
        rm -rf /tmp/engine-git/.git; \
        tar -czf /tmp/engine-src.tar.gz -C /tmp --transform 's|^engine-git|sglang-src|' engine-git; \
        rm -rf /tmp/engine-git; \
    fi; \
    tar -xzf /tmp/engine-src.tar.gz --strip-components=1 -C /opt/engine-src; \
    rm -f /tmp/engine-src.tar.gz; \
    [ -f /opt/engine-src/python/pyproject.toml ] && [ -d /opt/engine-src/python/sglang ] \
      || { echo "FATAL: /opt/engine-src is not an SGLang tree" >&2; exit 7; }; \
    got="$(sha256sum /opt/engine-src/python/sglang/srt/layers/quantization/mxfp4.py | cut -c1-16)"; \
    [ "$got" = "${ENGINE_SRC_SHA256}" ] \
      || { echo "FATAL: staged tree is not the pinned one: got $got" >&2; exit 7; }; \
    find /opt/engine-src -maxdepth 4 -name .git -exec rm -rf {} + 2>/dev/null || true; \
    echo "[engine-src] pinned tree at /opt/engine-src ($(find /opt/engine-src -name '*.py' -type f | wc -l) *.py), content verified"

ENV INFB_SGLANG_SOURCE_DIR=/opt/engine-src
ENV INFB_VLLM_SOURCE_DIR=/opt/engine-src

# ShareGPT corpus for `sglang.bench_serving --dataset-name random`, baked in so the bench never
# downloads it at run time. Revision and sha256 pinned: every run samples the same prompts.
ARG SHAREGPT_REVISION=192ab2185289094fc556ec8ce5ce1e8e587154ca
ARG SHAREGPT_SHA256=35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4
RUN set -eu; \
    corpus=/opt/infb/datasets/ShareGPT_V3_unfiltered_cleaned_split.json; \
    mkdir -p "$(dirname "$corpus")"; \
    curl -fsSL --retry 5 --retry-all-errors -o "$corpus" \
      "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/${SHAREGPT_REVISION}/ShareGPT_V3_unfiltered_cleaned_split.json"; \
    echo "${SHAREGPT_SHA256}  $corpus" | sha256sum -c - >/dev/null; \
    chmod 0444 "$corpus"

WORKDIR /app
COPY deidentify_engine_version.py /tmp/deidentify_engine_version.py
RUN /opt/venv/bin/python /tmp/deidentify_engine_version.py --package sglang --scan \
    && rm -f /tmp/deidentify_engine_version.py

# COPY . /app HERE WOULD BE A LEAK: the build context is this directory, so it would copy this
# Dockerfile INTO the agent workspace -- and this file names the upstream commit, whose CHILD IS THE
# FIX. Copying `public` explicitly is exactly equivalent for everything the task needs.
COPY public /app/public

COPY scripts /app/scripts

RUN set -eu; \
    found="$(find / -xdev -path '*/sglang/srt/managers/*' -name 'scheduler.py' \
                  -not -path '/proc/*' -not -path '/opt/engine-src/*' 2>/dev/null || true)"; \
    if [ -n "${found}" ]; then \
        echo "FATAL: a SECOND engine tree survives in this image, beside /opt/engine-src:" >&2; \
        echo "${found}" >&2; exit 9; \
    fi; \
    [ -f /opt/engine-src/python/sglang/srt/managers/scheduler.py ] \
      || { echo "FATAL: /opt/engine-src is not the engine tree it claims to be -- the exclusion above" >&2; \
           echo "  would then be hiding nothing and this check would pass vacuously." >&2; exit 9; }; \
    PYTHONPATH= "${INFB_VENV}/bin/python" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('sgl_kernel') else 1)" \
      || { echo "FATAL: the eviction took the compiled kernels with it" >&2; exit 9; }; \
    PYTHONPATH= "${INFB_VENV}/bin/python" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('triton_kernels') else 1)" \
      || { echo "FATAL: triton_kernels is gone after the eviction; the scored path would fall back silently" >&2; exit 9; }; \
    site="$("${INFB_VENV}/bin/python" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"; \
    scan="${INFB_VLLM_ENGINE_EXT} /opt/engine-src"; \
    for d in "${site}"/sglang-*.dist-info; do [ -e "$d" ] && scan="${scan} $d"; done; \
    if grep -rqa 'c8dc4d2d' ${scan} 2>/dev/null; then \
        echo "FATAL: something the agent can reach still names the pinned upstream commit" >&2; \
        grep -rla 'c8dc4d2d' ${scan} >&2 || true; exit 9; \
    fi; \
    echo "[engine-ext] whole-image check: one engine Python tree, kernels and triton_kernels survive the eviction, nothing names the pin"

# Normalise the engine tree's mtimes: `git archive` stamps every file with the pinned commit's
# timestamp, so `ls -la` gives the agent the pin's date, and a date plus a repo is most of a diff.
#
# BEST-EFFORT AND OBSERVABLE, NOT FAIL-CLOSED, and that is a deliberate retreat. An earlier version
# asserted the postcondition and exited 7; it fired on DIRECTORIES and the explanation for that is
# still unknown. An assert on an unexplained condition has now cost more than the leak it guards:
# reading an mtime yields the pin's DATE, and converting a date into the upstream fix needs network,
# which is closed. So normalise, PRINT what remains, and let the build live.
#
# `-depth` walks contents before their directory, which is the correct order for stamping a tree.
# The loop converges: a second pass has nothing left to modify.
RUN set -eu; \
    [ -d /opt/engine-src ] || { echo "FATAL: /opt/engine-src is missing; nothing to normalise" >&2; exit 7; }; \
    newer=""; \
    for pass in 1 2 3; do \
        find /opt/engine-src -depth -exec touch -h -d "2000-01-01T00:00:00Z" {} + || true; \
        newer="$(find /opt/engine-src -newermt "2000-01-02" -print -quit)"; \
        [ -n "$newer" ] || break; \
    done; \
    total="$(find /opt/engine-src -type f | wc -l)"; \
    residual="$(find /opt/engine-src -newermt "2000-01-02" | wc -l)"; \
    echo "[mtimes] /opt/engine-src normalised to 2000-01-01 ($total files, $residual still newer)"; \
    if [ "$residual" != "0" ]; then \
        echo "[mtimes] RESIDUAL DRIFT -- not fatal, but read this: the first 20 paths, with type and stamp" >&2; \
        find /opt/engine-src -newermt "2000-01-02" -printf "[mtimes]   %y %TY-%Tm-%Td %p\n" 2>/dev/null | head -20 >&2; \
    fi
