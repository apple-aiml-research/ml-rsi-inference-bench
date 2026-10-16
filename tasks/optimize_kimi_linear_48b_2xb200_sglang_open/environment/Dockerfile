#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
ARG ENGINE=engine

FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS engine

ARG PIP_INDEX_URL=https://pypi.org/simple

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_RETRIES=10 \
    PIP_TIMEOUT=60

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl git \
        "python3.12" "python3.12-venv" "python3.12-dev" \
        build-essential cmake ninja-build ccache \
        libdw1 libelf1 libnuma1 libnuma-dev xz-utils  \
        protobuf-compiler \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf "/usr/bin/python3.12" /usr/local/bin/python3 \
    && ln -sf "/usr/bin/python3.12" /usr/local/bin/python

RUN set -eu; \
    nsys_real="$(ls -1 /opt/nvidia/nsight-systems/*/target-linux-*/nsys \
                       /opt/nvidia/nsight-compute/*/host/target-linux-x64/nsys 2>/dev/null | head -1 || true)"; \
    if [ -n "${nsys_real}" ]; then \
        printf '%s\n' \
            '#!/bin/sh' \
            '# On-PATH `nsys`. A WRAPPER, not a symlink: a bind mount follows a symlink and would overwrite' \
            '# the Nsight binary itself. Re-globs at run time so a mounted host tree resolves too.' \
            'set -eu' \
            'for _c in /opt/nvidia/nsight-systems/*/target-linux-*/nsys \' \
            '          /opt/nvidia/nsight-compute/*/host/target-linux-x64/nsys; do' \
            '    [ -x "${_c}" ] || continue' \
            '    case "${_c}" in /usr/local/bin/nsys) continue ;; esac' \
            '    exec "${_c}" "$@"' \
            'done' \
            'echo "nsys: no Nsight binary found under /opt/nvidia" >&2' \
            'exit 127' \
            > /usr/local/bin/nsys; \
        chmod 0755 /usr/local/bin/nsys; \
        echo "[from-scratch] nsys on PATH: /usr/local/bin/nsys wrapper -> ${nsys_real}"; \
    else \
        echo "[from-scratch] nsys on PATH: SKIPPED — no Nsight tree in this base image"; \
    fi

RUN "python3.12" -m venv /opt/venv

RUN mkdir -p /opt/infb && cat > /opt/infb/requirements.lock <<'LOCK_EOF'
# Lockfile for this task's SGLang 0.5.10 runtime environment: a --no-deps pin set for the
# whole venv.
#
# pinned here : 174
#
# NOT pinned, and why:
#   sglang==0.5.10 — installed EDITABLE from source/python, because the installed
#     distribution's PEP 660 finder and .pth are stripped on purpose so `import sglang`
#     resolves to the agent's repo/. The build therefore installs it with `pip install -e`
#     from the public source at v0.5.10, with SETUPTOOLS_SCM_PRETEND_VERSION so the built
#     version is exactly this string rather than whatever git-describe reports.

aiohappyeyeballs===2.6.2
aiohttp===3.14.1
aiosignal===1.4.0
airportsdata===20260315
annotated-doc===0.0.4
annotated-types===0.7.0
anthropic===0.113.0
anyio===4.14.1
apache-tvm-ffi===0.1.12
asttokens===3.0.1
attrs===26.1.0
blobfile===3.0.0
build===1.5.0
certifi===2026.6.17
cffi===2.0.0
charset-normalizer===3.4.7
click===8.4.2
cloudpickle===3.1.2
compressed-tensors===0.15.0.1
cuda-bindings===12.9.7
cuda-pathfinder===1.5.5
cuda-python===12.9.0
cuda-tile===1.4.0
datasets===5.0.0
decorator===5.3.1
dill===0.4.1
diskcache===5.6.3
distro===1.9.0
docstring-parser===0.18.0
einops===0.8.2
executing===2.2.1
fastapi===0.138.2
filelock===3.29.4
flash-attn-4===4.0.0b19
flashinfer-cubin===0.6.7.post2
flashinfer-python===0.6.7.post2
frozenlist===1.8.0
fsspec===2026.4.0
gguf===0.19.0
grpcio===1.81.1
grpcio-health-checking===1.81.1
grpcio-reflection===1.81.1
h11===0.16.0
hf-xet===1.5.1
httpcore===1.0.9
httpx===0.28.1
huggingface-hub===1.21.0
idna===3.18
interegular===0.3.3
ipython===9.15.0
ipython-pygments-lexers===1.1.1
jedi===0.20.0
jinja2===3.1.6
jiter===0.16.0
jsonschema===4.26.0
jsonschema-specifications===2025.9.1
lark===1.3.1
llguidance===0.7.30
loguru===0.7.3
lxml===6.1.1
markdown-it-py===4.2.0
markupsafe===3.0.3
matplotlib-inline===0.2.2
mdurl===0.1.2
mistral-common===1.11.5
modelscope===1.37.1
mpmath===1.3.0
msgspec===0.21.1
multidict===6.7.1
multiprocess===0.70.19
nest-asyncio===1.6.0
networkx===3.6.1
ninja===1.13.0
numpy===2.3.5
nvidia-cublas-cu12===12.8.4.1
nvidia-cuda-cupti-cu12===12.8.90
nvidia-cuda-nvrtc-cu12===12.8.93
nvidia-cuda-runtime-cu12===12.8.90
nvidia-cudnn-cu12===9.10.2.21
nvidia-cudnn-frontend===1.25.0
nvidia-cufft-cu12===11.3.3.83
nvidia-cufile-cu12===1.13.1.3
nvidia-curand-cu12===10.3.9.90
nvidia-cusolver-cu12===11.7.3.90
nvidia-cusparse-cu12===12.5.8.93
nvidia-cusparselt-cu12===0.7.1
nvidia-cutlass-dsl===4.6.0.dev0
nvidia-cutlass-dsl-libs-base===4.6.0.dev0
nvidia-ml-py===13.610.43
nvidia-nccl-cu12===2.27.5
nvidia-nvjitlink-cu12===12.8.93
nvidia-nvshmem-cu12===3.3.20
nvidia-nvtx-cu12===12.8.90
openai===2.6.1
openai-harmony===0.0.4
orjson===3.11.9
outlines===0.1.11
outlines-core===0.1.26
packaging===26.2
pandas===3.0.3
parso===0.8.7
partial-json-parser===0.2.1.1.post7
pexpect===4.9.0
pillow===12.2.0
pip===26.1.2
prometheus-client===0.25.0
prompt-toolkit===3.0.52
propcache===0.5.2
protobuf===6.33.6
psutil===7.2.2
ptyprocess===0.7.0
pure-eval===0.2.3
py-spy===0.4.2
pyarrow===24.0.0
pybase64===1.4.3
pycountry===26.2.16
pycparser===3.0
pycryptodomex===3.23.0
pydantic===2.13.4
pydantic-core===2.46.4
pydantic-extra-types===2.11.1
pygments===2.20.0
pyproject-hooks===1.2.0
python-dateutil===2.9.0.post0
python-multipart===0.0.32
pyyaml===6.0.3
pyzmq===27.1.0
quack-kernels===0.5.3
referencing===0.37.0
regex===2026.6.28
requests===2.34.2
rich===15.0.0
rpds-py===2026.5.1
safetensors===0.8.0
scipy===1.18.0
sentencepiece===0.2.1
setproctitle===1.3.7
setuptools===82.0.1
sglang-kernel===0.4.1
shellingham===1.5.4
six===1.17.0
smg-grpc-proto===0.4.12
smg-grpc-servicer===0.6.0
sniffio===1.3.1
soundfile===0.13.1
stack-data===0.6.3
starlette===1.3.1
sympy===1.14.0
tabulate===0.10.0
tiktoken===0.13.0
timm===1.0.16
tokenizers===0.22.2
torch===2.9.1
torch-c-dlpack-ext===0.1.5
torch-memory-saver===0.0.9
torchao===0.9.0
torchaudio===2.9.1
torchcodec===0.9.1
torchvision===0.24.1
tqdm===4.68.3
traitlets===5.15.1
transformers===5.3.0
triton===3.5.1
typer===0.25.1
typing-extensions===4.15.0
typing-inspection===0.4.2
urllib3===2.7.0
uvicorn===0.49.0
uvloop===0.22.1
watchfiles===1.2.0
wcwidth===0.8.2
xgrammar===0.1.32
xxhash===3.8.0
yarl===1.24.2
LOCK_EOF

RUN extra=""; for u in ; do extra="${extra} --extra-index-url ${u}"; done; \
    echo "[from-scratch] index=${PIP_INDEX_URL}${extra}"; \
    n=0; until /opt/venv/bin/python -m pip install --no-deps \
        --index-url "${PIP_INDEX_URL}" ${extra} \
        -r /opt/infb/requirements.lock; do \
      n=$((n + 1)); \
      [ "$n" -ge 3 ] && { echo "[from-scratch] lock install failed 3x -- giving up"; exit 1; }; \
      echo "[from-scratch] lock install attempt ${n} failed; retrying in 45s"; sleep 45; \
    done; \
    { /opt/venv/bin/python -m pip check || true; }

RUN set -eu; mkdir -p /opt/engine-src; cd /opt/engine-src; \
    git clone --depth 1 --branch v0.5.10 --recurse-submodules --shallow-submodules \
        https://github.com/sgl-project/sglang.git .; \
    head="$(git rev-parse HEAD)"; \
    if [ "$head" != "1519acf37c23f2189adb93f57ca9cd2db1bebf18" ]; then \
        echo "FATAL: engine tree is at $head, not the reviewed 1519acf37c23f2189adb93f57ca9cd2db1bebf18 -- v0.5.10 has moved" >&2; exit 7; \
    fi; \
    rm -rf /opt/engine-src/.git; \
    find /opt/engine-src -maxdepth 4 -name .git -exec rm -rf {} + 2>/dev/null || true

RUN python3 - /opt/engine-src <<'INFB_PTX_PATCH'
#!/usr/bin/env python3
"""Rewrite the four CUDA-12.9-only `cuda::ptx` wrappers in sglang's dsv4 top-k JIT header as raw inline PTX.

WHY THIS RUNS HERE. On a CUDA-12.8 toolkit the serve-time JIT of this header fails: nvcc reports

    error: no instance of overloaded function "cuda::ptx::mbarrier_try_wait_parity" matches ...
    4 errors detected ... ninja: build stopped ...

and nothing that needs the kernel can run. Doing it in the Dockerfile keeps ONE copy that every
build route calls rather than a second copy that can drift away from it.

WHAT IT DOES. `sgl_kernel/deepseek_v4/topk/ptx.cuh` calls four `cuda::ptx` C++ wrappers
(mbarrier_try_wait_parity / mbarrier_arrive_expect_tx / mbarrier_arrive / cp_async_bulk) that exist only in CUDA
12.9's CCCL headers, so they are missing on a torch 2.11.0+cu128 / CUDA 12.8 toolkit (deliberately,
not by accident). The underlying PTX is ISA<=8.0 and IS accepted by 12.8 ptxas -- only the C++ wrappers
are missing -- so each call is rewritten to the raw inline PTX the wrapper emits, using ISA-8.0 .acquire/.release
ordering, which is STRONGER than the original .relaxed and therefore correctness-safe for a reference run.

Idempotent (marker-guarded by the caller and by the `#pragma once` tag) and fail-fast: it exits non-zero rather
than write a tree it did not fully rewrite.

    exit 0  patched
    exit 3  the expected shapes were not found (upstream drift) -- nothing written
    exit 4  a 12.9-only wrapper survived the rewrite -- nothing written

usage: patch_dsv4_topk_ptx_cuda128.py <path to ptx.cuh, or a source tree to search>

A DIRECTORY is accepted so the image build can run this with one argument and no shell plumbing:
the header only exists in the 0.5.12 trees (0.5.10 has no dsv4 top-k kernel at all), and "not present" is a
normal outcome that exits 0. The bash staging path keeps passing an exact file path; both reach the same code.
"""

import os
import sys, re

target = sys.argv[1]
if os.path.isdir(target):
    found = ""
    for root, _dirs, names in os.walk(target):
        if "ptx.cuh" in names and root.endswith(os.path.join("sgl_kernel", "deepseek_v4", "topk")):
            found = os.path.join(root, "ptx.cuh")
            break
    if not found:
        print(f"dsv4 topk ptx.cuh: not present under {target} -- nothing to patch (0.5.10 has no dsv4 top-k)")
        raise SystemExit(0)
    target = found
p = target
if "INFB-CUDA128-JIT-COMPAT" in open(p).read():
    print(f"dsv4 topk ptx.cuh: CUDA 12.8 JIT-compat patch already present in {p} (idempotent skip)")
    raise SystemExit(0)
s = open(p).read()
orig = s
# 1) mbarrier_try_wait_parity(sem_relaxed, scope_cta, addr, phase) -> raw PTX (acquire.cta)
s = s.replace(
    "  while (!cuda::ptx::mbarrier_try_wait_parity(cuda::ptx::sem_relaxed, cuda::ptx::scope_cta, addr, phase))\n    ;",
    "  uint32_t _smem = static_cast<uint32_t>(__cvta_generic_to_shared(addr));\n"
    "  uint32_t _done = 0;\n"
    "  while (!_done) {\n"
    "    asm volatile(\n"
    "        \"{\\n\\t\"\n"
    "        \".reg .pred P;\\n\\t\"\n"
    "        \"mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%1], %2;\\n\\t\"\n"
    "        \"selp.b32 %0, 1, 0, P;\\n\\t\"\n"
    "        \"}\"\n"
    "        : \"=r\"(_done)\n"
    "        : \"r\"(_smem), \"r\"(phase)\n"
    "        : \"memory\");\n"
    "  }")
# 2) mbarrier_arrive_expect_tx(sem_relaxed, scope_cta, space_shared, addr, tx) -> raw PTX (release.cta)
s = s.replace(
    "  cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_relaxed, cuda::ptx::scope_cta, cuda::ptx::space_shared, addr, tx);",
    "  uint32_t _smem = static_cast<uint32_t>(__cvta_generic_to_shared(addr));\n"
    "  uint64_t _state;\n"
    "  asm volatile(\"mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 %0, [%1], %2;\"\n"
    "               : \"=l\"(_state) : \"r\"(_smem), \"r\"(tx) : \"memory\");")
# 3) mbarrier_arrive(sem_relaxed, scope_cta, space_shared, addr) -> raw PTX (release.cta)
s = s.replace(
    "  cuda::ptx::mbarrier_arrive(cuda::ptx::sem_relaxed, cuda::ptx::scope_cta, cuda::ptx::space_shared, addr);",
    "  uint32_t _smem = static_cast<uint32_t>(__cvta_generic_to_shared(addr));\n"
    "  uint64_t _state;\n"
    "  asm volatile(\"mbarrier.arrive.release.cta.shared::cta.b64 %0, [%1];\"\n"
    "               : \"=l\"(_state) : \"r\"(_smem) : \"memory\");")
# 4) cp_async_bulk(space_shared, space_global, dst, src, num_bytes, mbar) -> raw PTX TMA load
s = s.replace(
    "  cuda::ptx::cp_async_bulk(cuda::ptx::space_shared, cuda::ptx::space_global, dst, src, num_bytes, mbar);",
    "  uint32_t _dsmem = static_cast<uint32_t>(__cvta_generic_to_shared(dst));\n"
    "  uint64_t _gsrc = static_cast<uint64_t>(__cvta_generic_to_global(src));\n"
    "  uint32_t _bsmem = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));\n"
    "  asm volatile(\"cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\"\n"
    "               : : \"r\"(_dsmem), \"l\"(_gsrc), \"r\"(num_bytes), \"r\"(_bsmem) : \"memory\");")
if s == orig:
    sys.stderr.write("FATAL: dsv4 topk ptx.cuh CUDA 12.8 JIT-compat patch replaced NOTHING (shape drift)\n")
    sys.exit(3)
# tag with the idempotency marker
s = s.replace("#pragma once\n", "#pragma once\n// INFB-CUDA128-JIT-COMPAT: raw-PTX rewrite of 4 cuda::ptx wrappers (12.9-only) for CUDA 12.8 nvcc\n", 1)
# all 4 wrapper calls must be gone
for bad in ("cuda::ptx::mbarrier_try_wait_parity", "cuda::ptx::mbarrier_arrive_expect_tx(",
            "cuda::ptx::mbarrier_arrive(cuda::ptx::sem_relaxed", "cuda::ptx::cp_async_bulk("):
    if bad in s:
        sys.stderr.write(f"FATAL: residual 12.9-only wrapper after patch: {bad}\n")
        sys.exit(4)
open(p, "w").write(s)
print("dsv4 topk ptx.cuh: rewrote 4 cuda::ptx wrappers to raw inline PTX (CUDA 12.8 JIT-compat)")
INFB_PTX_PATCH

RUN if [ -d /opt/engine-src/python/sglang ] && [ -f /opt/engine-src/python/pyproject.toml ]; then \
        echo "[from-scratch] sglang source baked at /opt/engine-src"; \
    elif [ -n "$(ls -A /opt/engine-src 2>/dev/null | grep -v '^\.keep$' || true)" ]; then \
        echo "FATAL: ENGINE_SRC_DIR was staged but is not an SGLang tree (need python/sglang + python/pyproject.toml); prepare_target_env.sh would exit 2 on it" >&2; \
        exit 96; \
    else \
        rm -rf /opt/engine-src; \
        echo "[from-scratch] no sglang source staged; the task will need INFB_SGLANG_SOURCE_DIR or a prestaged source tree"; \
    fi

RUN if [ -d /opt/engine-src/python ]; then \
        n=0; \
        until SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG="0.5.10" \
              /opt/venv/bin/python -m pip install --no-deps \
                  --index-url "${PIP_INDEX_URL}" \
                  -e /opt/engine-src/python; do \
          n=$((n + 1)); \
          [ "$n" -ge 3 ] && { echo "[from-scratch] editable sglang install failed 3x -- giving up"; exit 1; }; \
          echo "[from-scratch] editable sglang install attempt ${n} failed; retrying in 45s"; sleep 45; \
        done; \
        got="$(/opt/venv/bin/python -c 'from importlib import metadata; print(metadata.version("sglang"))')" \
        && if [ "${got}" != "0.5.10" ]; then \
               echo "FATAL: sglang installed as ${got}, expected 0.5.10 — the version drift this check exists for" >&2; \
               exit 97; \
           fi \
        && echo "[from-scratch] sglang ${got} installed editable from /opt/engine-src/python" \
        && rm -fv /opt/venv/lib/python*/site-packages/__editable__*sglang*.pth \
                  /opt/venv/lib/python*/site-packages/__editable___sglang_*_finder.py \
        && echo "[from-scratch] editable overlay stripped (MANDATORY — see prepare_target_env.sh)"; \
    fi

RUN n=0; until /opt/venv/bin/python -m pip install --no-cache-dir --prefix=/opt/infb/csrc-tools \
        --index-url "${PIP_INDEX_URL}" "cmake==3.31.10"; do \
      n=$((n + 1)); \
      [ "$n" -ge 3 ] && { echo "[from-scratch] csrc-tools install failed 3x -- giving up"; exit 1; }; \
      echo "[from-scratch] csrc-tools install attempt ${n} failed; retrying in 45s"; sleep 45; \
    done; \
    real="$(ls -d /opt/infb/csrc-tools/lib/python3.*/site-packages/cmake/data/bin | head -1)" \
    && for b in cmake ctest cpack; do \
           [ -x "${real}/${b}" ] && ln -sf "${real}/${b}" "/opt/infb/csrc-tools/bin/${b}"; \
       done \
    && /opt/infb/csrc-tools/bin/cmake --version | head -1 \
    && echo "[from-scratch] cmake 3.31.10 staged at /opt/infb/csrc-tools (apt's $(cmake --version | head -1 | awk '{print $3}') is too old for sgl-kernel)"

RUN n=0; until /opt/venv/bin/python -m pip install --no-cache-dir \
        --index-url "${PIP_INDEX_URL}" --prefix=/opt/infb/buildbackend scikit-build-core; do \
      n=$((n + 1)); \
      [ "$n" -ge 3 ] && { echo "[from-scratch] buildbackend install failed 3x -- giving up"; exit 1; }; \
      echo "[from-scratch] buildbackend install attempt ${n} failed; retrying in 45s"; sleep 45; \
    done; \
    site="$(ls -d /opt/infb/buildbackend/lib/python*/site-packages | head -1)" \
    && test -n "${site}" \
    && PYTHONPATH="${site}" /opt/venv/bin/python -c "import scikit_build_core.build" \
    && echo "[from-scratch] sgl-kernel build backend staged at ${site} (no network needed at rebuild time)"


RUN set -eu; mkdir -p /opt/csrc-deps/DeepGEMM; cd /opt/csrc-deps/DeepGEMM; \
    git init -q .; git remote add origin https://github.com/sgl-project/DeepGEMM; \
    if git fetch -q --depth 1 origin ffe2b6b97420a9f8c58268ca55755168e6e2f360 2>/dev/null; then \
        git checkout -q FETCH_HEAD; \
    else \
        echo "[from-scratch] DeepGEMM: fetch-by-id refused for ffe2b6b97420a9f8c58268ca55755168e6e2f360; fetching branches and tags"; \
        git fetch -q --tags origin; git checkout -q ffe2b6b97420a9f8c58268ca55755168e6e2f360; \
    fi; \
    git submodule update -q --init --recursive --depth 1 2>/dev/null || true; \
    rm -rf /opt/csrc-deps/DeepGEMM/.git; : > /opt/csrc-deps/DeepGEMM/.staged

RUN set -eu; mkdir -p /opt/csrc-deps/FlashMLA; cd /opt/csrc-deps/FlashMLA; \
    git init -q .; git remote add origin https://github.com/sgl-project/FlashMLA; \
    if git fetch -q --depth 1 origin 9804b12079e4c873514d3457aa588d3ccf40da28 2>/dev/null; then \
        git checkout -q FETCH_HEAD; \
    else \
        echo "[from-scratch] FlashMLA: fetch-by-id refused for 9804b12079e4c873514d3457aa588d3ccf40da28; fetching branches and tags"; \
        git fetch -q --tags origin; git checkout -q 9804b12079e4c873514d3457aa588d3ccf40da28; \
    fi; \
    git submodule update -q --init --recursive --depth 1 2>/dev/null || true; \
    rm -rf /opt/csrc-deps/FlashMLA/.git; : > /opt/csrc-deps/FlashMLA/.staged

RUN set -eu; mkdir -p /opt/csrc-deps/cutlass; cd /opt/csrc-deps/cutlass; \
    git init -q .; git remote add origin https://github.com/NVIDIA/cutlass; \
    if git fetch -q --depth 1 origin 57e3cfb47a2d9e0d46eb6335c3dc411498efa198 2>/dev/null; then \
        git checkout -q FETCH_HEAD; \
    else \
        echo "[from-scratch] cutlass: fetch-by-id refused for 57e3cfb47a2d9e0d46eb6335c3dc411498efa198; fetching branches and tags"; \
        git fetch -q --tags origin; git checkout -q 57e3cfb47a2d9e0d46eb6335c3dc411498efa198; \
    fi; \
    git submodule update -q --init --recursive --depth 1 2>/dev/null || true; \
    rm -rf /opt/csrc-deps/cutlass/.git; : > /opt/csrc-deps/cutlass/.staged

RUN set -eu; mkdir -p /opt/csrc-deps/flashinfer; cd /opt/csrc-deps/flashinfer; \
    git init -q .; git remote add origin https://github.com/flashinfer-ai/flashinfer.git; \
    if git fetch -q --depth 1 origin bc29697ba20b7e6bdb728ded98f04788e16ee021 2>/dev/null; then \
        git checkout -q FETCH_HEAD; \
    else \
        echo "[from-scratch] flashinfer: fetch-by-id refused for bc29697ba20b7e6bdb728ded98f04788e16ee021; fetching branches and tags"; \
        git fetch -q --tags origin; git checkout -q bc29697ba20b7e6bdb728ded98f04788e16ee021; \
    fi; \
    git submodule update -q --init --recursive --depth 1 2>/dev/null || true; \
    rm -rf /opt/csrc-deps/flashinfer/.git; : > /opt/csrc-deps/flashinfer/.staged

RUN set -eu; mkdir -p /opt/csrc-deps/fmt; cd /opt/csrc-deps/fmt; \
    git init -q .; git remote add origin https://github.com/fmtlib/fmt; \
    if git fetch -q --depth 1 origin 553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28 2>/dev/null; then \
        git checkout -q FETCH_HEAD; \
    else \
        echo "[from-scratch] fmt: fetch-by-id refused for 553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28; fetching branches and tags"; \
        git fetch -q --tags origin; git checkout -q 553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28; \
    fi; \
    git submodule update -q --init --recursive --depth 1 2>/dev/null || true; \
    rm -rf /opt/csrc-deps/fmt/.git; : > /opt/csrc-deps/fmt/.staged

RUN set -eu; mkdir -p /opt/csrc-deps/mscclpp; cd /opt/csrc-deps/mscclpp; \
    git init -q .; git remote add origin https://github.com/microsoft/mscclpp.git; \
    if git fetch -q --depth 1 origin 51eca89d20f0cfb3764ccd764338d7b22cd486a6 2>/dev/null; then \
        git checkout -q FETCH_HEAD; \
    else \
        echo "[from-scratch] mscclpp: fetch-by-id refused for 51eca89d20f0cfb3764ccd764338d7b22cd486a6; fetching branches and tags"; \
        git fetch -q --tags origin; git checkout -q 51eca89d20f0cfb3764ccd764338d7b22cd486a6; \
    fi; \
    git submodule update -q --init --recursive --depth 1 2>/dev/null || true; \
    rm -rf /opt/csrc-deps/mscclpp/.git; : > /opt/csrc-deps/mscclpp/.staged

RUN set -eu; mkdir -p /opt/csrc-deps/sgl-attn; cd /opt/csrc-deps/sgl-attn; \
    git init -q .; git remote add origin https://github.com/sgl-project/sgl-attn; \
    if git fetch -q --depth 1 origin bcf72ccc6816b36a5fae2c5a3c027604629785e0 2>/dev/null; then \
        git checkout -q FETCH_HEAD; \
    else \
        echo "[from-scratch] sgl-attn: fetch-by-id refused for bcf72ccc6816b36a5fae2c5a3c027604629785e0; fetching branches and tags"; \
        git fetch -q --tags origin; git checkout -q bcf72ccc6816b36a5fae2c5a3c027604629785e0; \
    fi; \
    git submodule update -q --init --recursive --depth 1 2>/dev/null || true; \
    rm -rf /opt/csrc-deps/sgl-attn/.git; : > /opt/csrc-deps/sgl-attn/.staged

RUN set -eu; mkdir -p /opt/csrc-deps/triton; cd /opt/csrc-deps/triton; \
    git init -q .; git remote add origin https://github.com/triton-lang/triton; \
    if git fetch -q --depth 1 origin v3.5.1 2>/dev/null; then \
        git checkout -q FETCH_HEAD; \
    else \
        echo "[from-scratch] triton: fetch-by-id refused for v3.5.1; fetching branches and tags"; \
        git fetch -q --tags origin; git checkout -q v3.5.1; \
    fi; \
    git submodule update -q --init --recursive --depth 1 2>/dev/null || true; \
    rm -rf /opt/csrc-deps/triton/.git; : > /opt/csrc-deps/triton/.staged

RUN { \
      printf '%s\t%s\t%s\t%s\n' 'DeepGEMM' 'FETCHCONTENT_SOURCE_DIR_REPO-DEEPGEMM' '-' 'repo-deepgemm'; \
      printf '%s\t%s\t%s\t%s\n' 'FlashMLA' 'FETCHCONTENT_SOURCE_DIR_REPO-FLASHMLA' '-' 'repo-flashmla'; \
      printf '%s\t%s\t%s\t%s\n' 'cutlass' 'FETCHCONTENT_SOURCE_DIR_REPO-CUTLASS' '-' 'repo-cutlass'; \
      printf '%s\t%s\t%s\t%s\n' 'flashinfer' 'FETCHCONTENT_SOURCE_DIR_REPO-FLASHINFER' '-' 'repo-flashinfer'; \
      printf '%s\t%s\t%s\t%s\n' 'fmt' 'FETCHCONTENT_SOURCE_DIR_REPO-FMT' '-' 'repo-fmt'; \
      printf '%s\t%s\t%s\t%s\n' 'mscclpp' 'FETCHCONTENT_SOURCE_DIR_REPO-MSCCLPP' '-' 'repo-mscclpp'; \
      printf '%s\t%s\t%s\t%s\n' 'sgl-attn' 'FETCHCONTENT_SOURCE_DIR_REPO-FLASH-ATTENTION' '-' 'repo-flash-attention'; \
      printf '%s\t%s\t%s\t%s\n' 'triton' 'FETCHCONTENT_SOURCE_DIR_REPO-TRITON' '-' 'repo-triton'; \
    } > /opt/csrc-deps/knobs.tsv

RUN if [ -s /opt/csrc-deps/knobs.tsv ]; then \
        echo "[from-scratch] sgl-kernel csrc deps staged:"; \
        while IFS="$(printf '\t')" read -r n k s d; do \
            if [ -d "/opt/csrc-deps/${n}" ]; then \
                printf '  %-14s %-46s %-24s %s\n' "$n" "$k" "$d" "$(du -sh "/opt/csrc-deps/${n}" | cut -f1)"; \
            else \
                printf '  %-14s %-46s %-24s MISSING (redirect will not be set)\n' "$n" "$k" "$d"; \
            fi; \
        done < /opt/csrc-deps/knobs.tsv; \
        true; \
    else \
        rm -rf /opt/csrc-deps; \
        echo "[from-scratch] no sgl-kernel csrc deps staged — a kernel rebuild will need network at configure time"; \
    fi

ENV INFB_SGLANG_SOURCE_DIR=/opt/engine-src

ENV VIRTUAL_ENV=/opt/venv \
    INFB_VLLM_PYTHON=/opt/venv/bin/python3 \
    CUDA_HOME=/usr/local/cuda-12.8 \
    TORCH_CUDA_ARCH_LIST=10.0+PTX \
    PATH=/opt/venv/bin:/usr/local/cuda-12.8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/usr/local/nvidia/lib64

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
_v = metadata.version('sglang')
print('sglang', _v)
assert _v.split('+')[0] == '0.5.10', _v
assert not missing, f'lock pins not installed: {missing[:5]}'
assert not wrong, f'installed versions differ from the lock: {wrong[:5]}'
PY

RUN /opt/venv/bin/python - <<'PY'
import base64, hashlib, pathlib
from importlib import metadata
REL = 'nvidia/nccl/lib/libnccl.so.2'
WANT = 'nvidia-nccl-cu12'  # this image's CUDA major
claim = {}
for dist in metadata.distributions():
    name = (dist.metadata['Name'] or '').lower().replace('_', '-')
    if not name.startswith('nvidia-nccl-cu'):
        continue
    for line in (dist.read_text('RECORD') or '').splitlines():
        path, _, rest = line.partition(',')
        if not path.replace('\\', '/').endswith(REL):
            continue
        digest = rest.split(',')[0]
        assert digest.startswith('sha256='), f'{name}: RECORD hash is {digest!r}, not sha256'
        claim[name] = (digest[len('sha256='):], pathlib.Path(str(dist.locate_file(path))))
print(f'nccl: want {WANT}; wheels claiming {REL}: {sorted(claim)}')
assert WANT in claim, f'{WANT} does not ship {REL}; claimants are {sorted(claim)}'
expected, target = claim[WANT]
assert target.exists(), f'{target} is missing'
h = hashlib.sha256()
with target.open('rb') as fh:
    for chunk in iter(lambda: fh.read(1 << 22), b''):
        h.update(chunk)
actual = base64.urlsafe_b64encode(h.digest()).decode().rstrip('=')
print(f'nccl: {target} is {target.stat().st_size} B, sha256(b64) {actual}')
assert actual == expected, (
    f'{REL} is NOT the one {WANT} shipped (want {expected}, have {actual}). Another wheel claiming\n'
    f'the same path overwrote it -- claimants: {sorted(claim)}. Every multi-GPU trial preloads this\n'
    f'file, so a CUDA-major mismatch here fails at ncclCommInitRank inside the engine instead.'
)
PY

RUN /opt/venv/bin/python - <<'PY'
import pathlib, re, struct, sysconfig
WANT = '12'  # this image's CUDA major
ROOTS = ['sgl_kernel', 'sglang', 'torch']
def dt_needed(path):
    # SEEKS rather than read_bytes(): sgl_kernel ships a 1.06 GB flash_ops.abi3.so and 12 more .so
    # files, and this needs a few KB of each. An ELF's section header table is at the END of the file,
    # so a truncated read cannot work either -- it has to be seeks.
    with path.open('rb') as fh:
        head = fh.read(0x40)
        if len(head) < 0x40 or head[:4] != b'\x7fELF' or head[4] != 2 or head[5] != 1:
            return []  # not a 64-bit little-endian ELF; manylinux x86_64 always is
        shoff, = struct.unpack_from('<Q', head, 0x28)
        shentsize, shnum = struct.unpack_from('<HH', head, 0x3a)
        fh.seek(shoff)
        tab = fh.read(shentsize * shnum)
        if len(tab) < shentsize * shnum:
            return []
        secs = []
        for i in range(shnum):
            o = i * shentsize
            sh_type, = struct.unpack_from('<I', tab, o + 4)
            sh_off, sh_size = struct.unpack_from('<QQ', tab, o + 0x18)
            sh_link, = struct.unpack_from('<I', tab, o + 0x28)
            secs.append((sh_type, sh_off, sh_size, sh_link))
        dyn = next(((o, n, l) for t, o, n, l in secs if t == 6), None)  # SHT_DYNAMIC
        if dyn is None:
            return []
        off, size, link = dyn
        fh.seek(off)
        dynblob = fh.read(size)
        fh.seek(secs[link][1])
        strblob = fh.read(secs[link][2])
    out = []
    for k in range(0, len(dynblob) - 15, 16):
        tag, val = struct.unpack_from('<qQ', dynblob, k)
        if tag == 0:
            break
        if tag == 1:  # DT_NEEDED
            end = strblob.find(b'\x00', val)
            out.append(strblob[val:end if end != -1 else None].decode())
    return out
site = pathlib.Path(sysconfig.get_paths()['purelib'])
found, scanned = {}, 0
for root in ROOTS:
    for so in sorted((site / root).rglob('*.so')):
        scanned += 1
        for lib in dt_needed(so):
            m = re.fullmatch(r'libcudart\.so\.(\d+)', lib)
            if m:
                found.setdefault(m.group(1), []).append(str(so.relative_to(site)))
print(f'cudart: want {WANT}; scanned {scanned} .so under {ROOTS}; majors '
      + repr({k: len(v) for k, v in sorted(found.items())}))
assert found, (
    f'no .so under {ROOTS} names a libcudart at all after scanning {scanned}. Either the engine\n'
    f'stopped shipping compiled extensions or this venv is not the one the lock built -- both are\n'
    f'worth failing the build for.'
)
wrong = {k: v[:3] for k, v in found.items() if k != WANT}
assert not wrong, (
    f'this image builds on CUDA {WANT} but the engine links libcudart.so.{sorted(wrong)}: {wrong}.\n'
    f'pip cannot see this (the cu12 and cu13 runtimes are different distributions and PEP 440\n'
    f'ignores a +cuXXX local label), and neither can the lock. Fix the image definition: the base\n'
    f'image, the CUDA directory and the torch index channel must all match the generation the\n'
    f'engine wheel was BUILT with. Unfixed, the first import in the trial raises\n'
    f'ImportError: libcudart.so.<N> and the run scores 0.0 with reason=live_reference_failed.'
)
PY

ENTRYPOINT ["/usr/local/bin/infb-entrypoint"]

CMD ["/bin/bash", "-l"]

FROM ${ENGINE}
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY cas/ /tmp/infb-cas/
RUN if ls /tmp/infb-cas/*.crt >/dev/null 2>&1; then \
        cp /tmp/infb-cas/*.crt /usr/local/share/ca-certificates/ && \
        update-ca-certificates; \
    else \
        echo "[task-image] no CA bundle shipped — skipping update-ca-certificates"; \
    fi; \
    rm -rf /tmp/infb-cas

# SGLang JIT-compiles RoPE/overlap kernels at serve time on sm100 (tvm-ffi -> nvcc, no prebuilt
# kernel). CUDA 12.8 nvcc rejects glibc 2.36+'s C23 `cospi`/`sinpi`/`tanpi` `__MATHCALL_VEC`
# declarations (host `noexcept(true)` vs CUDA `__host__ __device__`) -> "exception specification is
# incompatible" -> ninja status 2 -> server SIGQUIT. The patch is marker-guarded (idempotent) and
# fail-fast at BUILD time if the header differs.
RUN H=/usr/include/x86_64-linux-gnu/bits/mathcalls.h; \
    if [ -f "$H" ] && ! grep -q "patched: avoid CUDA __host__ __device__ redecl conflict" "$H"; then \
        sed -i \
          -e 's|^__MATHCALL_VEC (cospi,, (_Mdouble_ __x));|/* & -- patched: avoid CUDA __host__ __device__ redecl conflict */|' \
          -e 's|^__MATHCALL_VEC (sinpi,, (_Mdouble_ __x));|/* & -- patched */|' \
          -e 's|^__MATHCALL_VEC (tanpi,, (_Mdouble_ __x));|/* & -- patched */|' \
          "$H"; \
    fi; \
    if [ -f "$H" ] && grep -qE '^__MATHCALL_VEC \((cospi|sinpi|tanpi),' "$H"; then \
        echo "FATAL: glibc cospi/sinpi/tanpi still active in $H after patch (header format differs)" >&2; exit 1; \
    fi; \
    echo "[dockerfile] glibc mathcalls.h C23 cospi/sinpi/tanpi patched for B200 sm100 nvcc JIT"

WORKDIR /app

RUN rm -rf /app/public \
    && rm -rf /app/sealed
COPY public  /app/public
COPY scripts /app/scripts
RUN find /app/scripts -mindepth 1 -maxdepth 1 \
        ! -name _gpu_cleanup.py \
        ! -name _paired_helper.py \
        ! -name _run_benchmark_pd.py \
        ! -name _run_benchmark_serve.py \
        ! -name _target_env.py \
        ! -name build_mtp_prompt_bank.py \
        ! -name build_sgl_kernel.sh \
        ! -name build_vllm_csrc.sh \
        ! -name correctness_prompts.json \
        ! -name mtp_longctx_8k_prompts.spec.json \
        ! -name multi_turn \
        ! -name pd_bench_spec.py \
        ! -name pd_client_patch.py \
        ! -name pd_router.py \
        ! -name pd_topology.py \
        ! -name pd_turn_metrics.py \
        ! -name pier_healthcheck.sh \
        ! -name prepare_target_env.sh \
        ! -name run_benchmark.py \
        ! -name run_correctness.py \
        ! -name stage_editable_pkgs.sh \
        ! -name start_agent_timer.sh \
        -exec rm -rf {} +


CMD ["/bin/bash", "-l"]
