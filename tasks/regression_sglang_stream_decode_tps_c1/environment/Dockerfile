#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

ARG BASE_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu24.04
FROM ${BASE_IMAGE}

ARG PIP_INDEX_URL=https://pypi.org/simple

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_RETRIES=10 \
    PIP_TIMEOUT=60

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl git patch rsync \
        "python3.12" "python3.12-venv" "python3.12-dev" \
        build-essential cmake ninja-build ccache \
        libdw1 libelf1 libnuma1 libnuma-dev xz-utils \
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

RUN "python3.12" -m venv /opt/venv

COPY requirements.lock /opt/infb/requirements.lock

RUN extra=""; for u in https://docs.sglang.ai/whl/cu129/ https://download.pytorch.org/whl/cu128; do extra="${extra} --extra-index-url ${u}"; done; \
    echo "[task-image] index=${PIP_INDEX_URL}${extra}"; \
    n=0; until /opt/venv/bin/python -m pip install --no-deps \
        --index-url "${PIP_INDEX_URL}" ${extra} \
        -r /opt/infb/requirements.lock; do \
      n=$((n + 1)); \
      [ "$n" -ge 3 ] && { echo "[task-image] lock install failed 3x -- giving up"; exit 1; }; \
      echo "[task-image] lock install attempt ${n} failed; retrying in 45s"; sleep 45; \
    done; \
    { /opt/venv/bin/python -m pip check || true; }


RUN set -eu; \
    python3.12 -m venv /opt/aiperf-venv; \
    /opt/aiperf-venv/bin/python -m pip install -q --no-cache-dir --upgrade pip; \
    /opt/aiperf-venv/bin/python -m pip install --no-cache-dir "aiperf==0.12.0"; \
    /opt/aiperf-venv/bin/aiperf --version; \
    /opt/aiperf-venv/bin/python -c "import importlib.metadata as m; v=m.version('aiperf'); assert v=='0.12.0', f'aiperf is {v}, not the version this task was calibrated against'"; \
    echo "[task-image] aiperf 0.12.0 ready in its own venv (engine closure untouched)"


ENV INFB_VENV=/opt/venv

ENV VIRTUAL_ENV=/opt/venv \
    INFB_VLLM_PYTHON=/opt/venv/bin/python3 \
    CUDA_HOME=/usr/local/cuda-12.8 \
    TORCH_CUDA_ARCH_LIST=10.0+PTX \
    PATH=/opt/venv/bin:/usr/local/cuda-12.8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/usr/local/nvidia/lib64

ARG ENGINE_REF=163b97012707d712e141e7ee486b72e2b0d8e60d
RUN set -eu; mkdir -p /opt/infb-src/engine-src; cd /opt/infb-src/engine-src; \
    git init -q .; \
    git remote add origin https://github.com/sgl-project/sglang.git; \
    git fetch -q --depth 1 origin "${ENGINE_REF}"; \
    git checkout -q FETCH_HEAD; \
    git submodule update -q --init --recursive --depth 1; \
    head="$(git rev-parse HEAD)"; \
    if [ "$head" != "${ENGINE_REF}" ]; then \
        echo "FATAL: the tree is at $head, not the reviewed ref -- the ref has moved" >&2; exit 7; \
    fi; \
    if [ ! -d python/sglang ] || [ ! -f python/pyproject.toml ]; then \
        echo "FATAL: not an SGLang monorepo root (need python/sglang + python/pyproject.toml)" >&2; \
        exit 96; \
    fi; \
    rm -rf /opt/infb-src/engine-src/.git; \
    find /opt/infb-src/engine-src -maxdepth 4 -name .git -exec rm -rf {} + 2>/dev/null || true

RUN set -eu; \
    mkdir -p /opt/inference-bench; \
    archive=/opt/inference-bench/engine-src.tar.gz; \
    [ "$(basename /opt/infb-src/engine-src)" = "engine-src" ] || { \
        echo "FATAL: the source dir must be named engine-src; its name is hashed" >&2; exit 1; }; \
    tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner \
        -cf - -C /opt/infb-src engine-src | gzip -n > "${archive}"; \
    digest="$(gzip -dc "${archive}" | sha256sum | awk '{print $1}')"; \
    printf '%s  engine-src.tar (uncompressed stream of engine-src.tar.gz)\n' "${digest}" \
        > "${archive}.sha256"; \
    rm -rf /opt/infb-src; \
    echo "[task-image] pinned engine source archived: $(du -h "${archive}" | cut -f1)"; \
    echo "[task-image] engine-src.tar sha256 = ${digest}"; \
    echo "[task-image]   ^ commit this into tests/engine_src_archive.sha256, or every scored run fails closed."

ENV INFB_ENGINE_SRC_ARCHIVE=/opt/inference-bench/engine-src.tar.gz

RUN printf '#!/bin/sh\nldconfig 2>/dev/null || true\nexec "$@"\n' > /usr/local/bin/infb-entrypoint \
    && chmod 0755 /usr/local/bin/infb-entrypoint

RUN /opt/venv/bin/python - <<'PY'
from importlib import metadata
import re
pins = {}
for raw in open('/opt/infb/requirements.lock'):
    raw = raw.split(' #')[0].strip()
    m = re.match(r'^([A-Za-z0-9._-]+)\s*={2,3}\s*(\S+)$', raw)
    if m:
        pins[m.group(1).lower().replace('_', '-')] = m.group(2)
installed = {d.metadata['Name'].lower().replace('_', '-'): d.version
             for d in metadata.distributions() if d.metadata['Name']}
missing = sorted(n for n in pins if n not in installed)
wrong = sorted(f'{n}: want {v}, have {installed[n]}' for n, v in pins.items()
               if n in installed and installed[n] != v)
print(f'lock pins {len(pins)}, installed {len(installed)}, missing {len(missing)}, wrong {len(wrong)}')
assert not missing, f'lock pins not installed: {missing[:5]}'
assert not wrong, f'installed versions differ from the lock: {wrong[:5]}'
assert 'sglang' not in installed, (
    'an engine distribution is installed in this venv. A release or an editable install is on the '
    'FIXED side of the commit this task pins, so `diff` against the staged tree prints the answer.'
)
PY

WORKDIR /app

COPY public /app/public
COPY scripts /app/scripts

RUN set -eu; \
    found="$(find / -xdev -path '*/sglang/srt/managers/scheduler.py' -not -path '/proc/*' 2>/dev/null || true)"; \
    if [ -n "${found}" ]; then \
        echo "FATAL: a second copy of the engine survives in this image:" >&2; echo "${found}" >&2; exit 9; \
    fi; \
    if /opt/venv/bin/python -c 'import sglang' 2>/dev/null; then \
        echo "FATAL: this venv can satisfy 'import sglang' -- the tree on PYTHONPATH would not be the" >&2; \
        echo "       only engine, and diffing the two answers the question the task asks." >&2; \
        /opt/venv/bin/python -c 'import sglang; print(sglang.__file__)' >&2 || true; exit 9; \
    fi; \
    echo "[task-image] whole-image check: no engine Python outside the agent's tree, and this venv" \
         "cannot produce one"

ENTRYPOINT ["/usr/local/bin/infb-entrypoint"]

CMD ["/bin/bash", "-l"]

COPY deidentify_engine_version.py /tmp/deidentify_engine_version.py
RUN /opt/venv/bin/python /tmp/deidentify_engine_version.py --package sglang --scan \
    && rm -f /tmp/deidentify_engine_version.py

RUN rm -f /opt/infb/requirements.lock \
    && echo "[lock] /opt/infb/requirements.lock removed: build-time only, and it carried the engine pin"
