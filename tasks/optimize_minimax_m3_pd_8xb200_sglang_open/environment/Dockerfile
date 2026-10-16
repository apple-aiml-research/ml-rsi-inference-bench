#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
ARG ENGINE=engine

FROM nvidia/cuda:13.0.2-devel-ubuntu24.04 AS engine

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
        libdw1 libelf1 libnuma1 libnuma-dev xz-utils cuda-compat-13-0 libibverbs1 ibverbs-providers librdmacm1 libcurl4 clang libclang-rt-dev \
        protobuf-compiler \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf "/usr/bin/python3.12" /usr/local/bin/python3 \
    && ln -sf "/usr/bin/python3.12" /usr/local/bin/python

RUN set -e; \
    mkdir -p /opt/infb/cuda-compat/compat; \
    mv /usr/local/cuda-13.0/compat /opt/infb/cuda-compat/staged; \
    ln -s /opt/infb/cuda-compat/staged /usr/local/cuda-13.0/compat; \
    printf '%s\n' \
        '#!/bin/sh' \
        '# Decide ONCE per container whether the staged forward-compat driver is needed, and express' \
        '# the decision as the CONTENT of the directory that is on LD_LIBRARY_PATH. Idempotent.' \
        'D=/opt/infb/cuda-compat' \
        '[ -d "$D/staged" ] || exit 0' \
        '[ -e "$D/.decided" ] && exit 0' \
        'mkdir -p "$D/compat"' \
        'have=""' \
        'if [ -r /proc/driver/nvidia/version ]; then' \
        '  have=$(sed -n "s/.*[^0-9]\([0-9][0-9]*\)\.[0-9][0-9]*\..*/\1/p" /proc/driver/nvidia/version | head -1)' \
        'fi' \
        'case "$have" in "" | *[!0-9]*) have=0 ;; esac' \
        'min=@STAGED_MAJOR@' \
        'if [ "$have" -lt "$min" ]; then' \
        '  for f in "$D"/staged/*; do ln -sfn "$f" "$D/compat/${f##*/}"; done' \
        '  printf "forward-compat ACTIVE: host driver major %s < staged %s\n" "$have" "$min" > "$D/.decided"' \
        'else' \
        '  printf "forward-compat NOT needed: host driver major %s >= staged %s\n" "$have" "$min" > "$D/.decided"' \
        'fi' \
        'cat "$D/.decided"' \
        > /usr/local/bin/infb-forward-compat-activate; \
    printf '%s\n' \
        '# Sourced by bash through BASH_ENV -- the only hook that reaches a container `exec bash -c`' \
        '# command, because exec does not re-run the ENTRYPOINT and the baked ENV gets replaced.' \
        '# Keep it cheap: this runs on EVERY non-interactive bash. After the first invocation the' \
        '# activator is skipped by a builtin test, so the steady state costs no fork.' \
        'if [ -d /opt/infb/cuda-compat/compat ]; then' \
        '  [ -e /opt/infb/cuda-compat/.decided ] || /usr/local/bin/infb-forward-compat-activate >/dev/null 2>&1' \
        '  case ":${LD_LIBRARY_PATH-}:" in' \
        '    *:/opt/infb/cuda-compat/compat:*) ;;' \
        '    *) LD_LIBRARY_PATH="/opt/infb/cuda-compat/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; export LD_LIBRARY_PATH ;;' \
        '  esac' \
        'fi' \
        > /etc/infb-forward-compat.sh; \
    staged=$(ls /opt/infb/cuda-compat/staged/libcuda.so.*.* 2>/dev/null | head -1); \
    staged=${staged##*/libcuda.so.}; \
    staged=${staged%%.*}; \
    case "$staged" in "" | *[!0-9]*) echo "FATAL: cannot derive the staged libcuda major" >&2; exit 1 ;; esac; \
    sed -i "s/@STAGED_MAJOR@/$staged/" /usr/local/bin/infb-forward-compat-activate; \
    chmod 0755 /usr/local/bin/infb-forward-compat-activate; \
    chmod 0644 /etc/infb-forward-compat.sh; \
    echo "[from-scratch] staged forward-compat driver major $staged; used only on an older host:"; \
    ls /opt/infb/cuda-compat/staged

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
# Indexes these pins need beyond https://pypi.org/simple: PEP 440 local labels (+cu128, +cu129) that PyPI
# cannot serve by definition, so the project's own public index serves them. Folded in here rather
# than kept in a sibling manifest -- pip honours these lines directly.
--extra-index-url https://docs.sglang.ai/whl/cu130/
--extra-index-url https://download.pytorch.org/whl/cu130

# Full pin closure for this task's SGLang 0.5.18 + CUDA 13.0 runtime. Captured from a
# known-good built environment rather than re-resolved at build time, so this image
# reproduces that runtime exactly instead of drifting with today's index resolution.
#
# NOT pinned, and why:
#   sglang==0.5.18 — installed EDITABLE from source/python, because the editable
#     PEP 660 finder and .pth are stripped on purpose so `import sglang` resolves to the
#     agent's repo/. It is installed with `pip install -e` from the public source at
#     v0.5.18, with SETUPTOOLS_SCM_PRETEND_VERSION so the built version is exactly this
#     string rather than whatever git-describe reports.

#
# ADDED DELIBERATELY on top of the captured closure -- three pins:
#
#   nixl===1.4.0  nixl-cu12===1.4.0  nixl-cu13===1.4.0
#
# WHY. Where no RDMA HCA is present, SGLang's PD arm falls back to shipping KV over TCP
# (`No RDMA devices found` -> `Found 0 HCAs` -> `Using TCP transport (no HCA detected)`) while vLLM's
# NixlConnector uses `Backend UCX` (cuda_ipc/shm). That is not an engine difference, it is a wire
# difference, and it would make an SGLang-vs-vLLM comparison meaningless. SGLang ships its own NIXL
# backend (srt/disaggregation/nixl/conn.py, driving the same nixl_agent vLLM uses), so pinning these
# puts both engines on the same transport. mooncake's own NVLink path is not the alternative there,
# and SGLang's source calls that transport not-bug-free.
#
# SOUND TO ADD BY HAND: a lock is a --no-deps freeze, so a line is only correct if the distribution needs
# nothing that is not already pinned. Read from the wheels: nixl-cu12 and nixl-cu13 declare `torch, numpy`
# and nothing else; the `nixl` meta-package requires exactly those two, both pinned above. All three names
# are listed because the install is --no-deps: the meta-package alone would bring nothing.
aiohappyeyeballs===2.7.1
aiohttp===3.14.3
aiosignal===1.4.0
airportsdata===20260803
annotated-doc===0.0.5
annotated-types===0.8.0
anthropic===1.2.0
anyio===4.14.2
apache-tvm-ffi===0.1.11
asttokens===3.0.2
attrs===26.1.0
blobfile===3.0.0
build===1.6.0
certifi===2026.7.22
cffi===2.1.1
charset-normalizer===3.5.1
click===8.5.0
cloudpickle===3.1.2
compressed-tensors===0.18.0
cuda-bindings===13.3.1
cuda-core===1.0.1
cuda-pathfinder===1.8.0
cuda-python===13.3.1
cuda-tile===1.6.0rc5
cuda-toolkit===13.0.3.0
datasets===5.0.1
dill===0.4.1
diskcache===5.6.3
distro===1.9.0
docstring-parser===0.18.0
easydict===1.13
einops===0.8.2
executing===2.2.1
fastapi===0.141.1
filelock===3.32.4
flash-attn-4===4.0.0b19
flashinfer-python===0.6.17
frozenlist===1.8.0
fsspec===2026.6.0
gguf===0.19.0
grpcio===1.83.1
grpcio-health-checking===1.81.1
grpcio-reflection===1.81.1
h11===0.16.0
hf-xet===1.6.0
httpcore===1.0.9
httpcore2===2.12.0
httpx===0.28.1
httpx2===2.12.0
huggingface-hub===1.29.0
humming-kernels===0.1.10
idna===3.19
interegular===0.3.3
ipython===9.17.0
ipython-pygments-lexers===1.1.1
jedi===0.20.0
jinja2===3.1.6
jiter===0.16.0
jsonschema===4.26.0
jsonschema-specifications===2025.9.1
kernels===0.14.1
kernels-data===0.16.1
lark===1.3.1
llguidance===1.8.0
llvmlite===0.47.0
loguru===0.7.3
lxml===6.1.2
markdown-it-py===4.2.0
markupsafe===3.0.3
matplotlib-inline===0.2.2
mdurl===0.1.2
mistral-common===1.11.7
ml-dtypes===0.6.0
modelscope===1.39.1
modelscope-hub===0.3.0
mooncake-transfer-engine===0.3.13
mpmath===1.3.0
msgpack===1.2.2
msgspec===0.21.1
multidict===6.7.1
multiprocess===0.70.19
nccl4py===0.4.1
nest-asyncio===1.6.0
networkx===3.6.1
ninja===1.13.0
nixl===1.4.0
nixl-cu12===1.4.0
nixl-cu13===1.4.0
numba===0.65.1
numpy===2.3.5
nvidia-cublas===13.1.1.3
nvidia-cuda-cccl===13.3.3.4.1
nvidia-cuda-crt===13.3.73
nvidia-cuda-cupti===13.0.85
nvidia-cuda-nvcc===13.3.73
nvidia-cuda-nvdisasm===13.3.73
nvidia-cuda-nvrtc===13.0.88
nvidia-cuda-runtime===13.0.96
nvidia-cudnn-cu13===9.20.0.48
nvidia-cudnn-frontend===1.27.0
nvidia-cufft===12.0.0.61
nvidia-cufile===1.15.1.6
nvidia-curand===10.4.0.35
nvidia-cusolver===12.0.4.66
nvidia-cusparse===12.6.3.3
nvidia-cusparselt-cu13===0.8.1
nvidia-cutlass-dsl===4.6.2
nvidia-cutlass-dsl-libs-base===4.6.2
nvidia-cutlass-dsl-libs-core===4.6.2
nvidia-cutlass-dsl-libs-cu12===4.6.2
nvidia-cutlass-dsl-libs-cu13===4.6.2
nvidia-mathdx===25.6.0
nvidia-ml-py===13.610.43
nvidia-nccl-cu13===2.29.7
nvidia-nvjitlink===13.3.33
nvidia-nvshmem-cu13===3.4.5
nvidia-nvtx===13.0.85
nvidia-nvvm===13.3.73
openai===2.6.1
openai-harmony===0.0.4
orjson===3.12.0
outlines===0.1.11
outlines-core===0.1.26
packaging===26.3
pandas===3.0.5
parso===0.8.7
partial-json-parser===0.2.1.1.post7
pexpect===4.9.0
pillow===12.3.0
pip===24.0
prometheus-client===0.26.0
prompt-toolkit===3.0.53
propcache===0.5.2
protobuf===6.33.6
psutil===7.2.2
ptyprocess===0.7.0
pure-eval===0.2.3
py-spy===0.4.2
pyarrow===25.0.1
pybase64===1.5.0
pycountry===26.2.16
pycparser===3.0
pycryptodomex===3.23.0
pydantic===2.13.5
pydantic-core===2.46.5
pydantic-extra-types===2.11.1
pyelftools===0.33
pygments===2.21.0
pyproject-hooks===1.2.0
python-dateutil===2.9.0.post0
python-multipart===0.0.32
pyyaml===6.0.3
pyzmq===27.2.0
quack-kernels===0.6.4
referencing===0.37.0
regex===2026.7.19
requests===2.34.2
rich===15.0.0
rpds-py===2026.6.3
safetensors===0.8.0
scipy===1.18.1
sentencepiece===0.2.2
setproctitle===1.3.7
setuptools===84.0.0
sgl-deep-ep==0.1.0+cu130
sgl-deep-gemm==0.1.5.post3+cu130
sglang-kernel==0.4.6.post1+cu130
sglang-router===0.3.2
shellingham===1.5.4
six===1.17.0
smg-grpc-proto===0.4.16
smg-grpc-servicer===0.9.1
sniffio===1.3.1
soundfile===0.13.1
stack-data===0.6.3
starlette===1.6.0
sympy===1.14.0
tabulate===0.10.0
tiktoken===0.14.0
tilelang===0.1.11
timm===1.0.16
tokenizers===0.22.2
tokenspeed-mla===0.1.8
tokenspeed-triton===3.8.10.post20260721
tomlkit===0.15.1
torch==2.13.0+cu130
torch-c-dlpack-ext===0.1.5
torch-memory-saver===0.0.9.post1
torchaudio==2.11.0+cu130
torchcodec==0.15.0+cu130
torchvision==0.28.0+cu130
tqdm===4.70.0
traitlets===5.16.1
transformers===5.12.1
triton===3.7.1
truststore===0.10.4
typer===0.27.2
typing-extensions===4.16.0
typing-inspection===0.4.4
urllib3===2.7.0
uvicorn===0.52.4
uvloop===0.22.1
watchfiles===1.2.0
wcwidth===0.8.3
xgrammar===0.2.1
xxhash===4.0.1
yarl===1.24.5
z3-solver===4.15.4.0
zstandard===0.25.0
LOCK_EOF

RUN extra=""; for u in https://docs.sglang.ai/whl/cu130/ https://download.pytorch.org/whl/cu130; do extra="${extra} --extra-index-url ${u}"; done; \
    echo "[from-scratch] index=${PIP_INDEX_URL}${extra}"; \
    n=0; until /opt/venv/bin/python -m pip install --no-deps \
        --index-url "${PIP_INDEX_URL}" ${extra} \
        -r /opt/infb/requirements.lock; do \
      n=$((n + 1)); \
      [ "$n" -ge 3 ] && { echo "[from-scratch] lock install failed 3x -- giving up"; exit 1; }; \
      echo "[from-scratch] lock install attempt ${n} failed; retrying in 45s"; sleep 45; \
    done; \
    { /opt/venv/bin/python -m pip check || true; }

RUN set -eu; rm -rf /tmp/_vendor; mkdir -p /tmp/_vendor; cd /tmp/_vendor; \
    git clone --depth 1 --filter=blob:none --sparse --branch v3.7.1 https://github.com/triton-lang/triton src; \
    cd src; git sparse-checkout set python/triton_kernels; \
    if [ ! -d "python/triton_kernels" ]; then \
        echo "FATAL: https://github.com/triton-lang/triton@v3.7.1 has no python/triton_kernels" >&2; exit 7; \
    fi; \
    if ! /opt/venv/bin/python -m pip install --no-deps --no-build-isolation "python/triton_kernels"; then \
        echo "[from-scratch] pip install python/triton_kernels failed; copying the package dir instead" >&2; \
        sp="$(/opt/venv/bin/python -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"; \
        cp -r "python/triton_kernels/triton_kernels" "$sp/"; \
    fi; \
    /opt/venv/bin/python -c "import triton_kernels, triton_kernels.tensor"; \
    cd /; rm -rf /tmp/_vendor

RUN set -e; \
    url="https://static.rust-lang.org/dist/rust-1.92.0-x86_64-unknown-linux-gnu.tar.xz"; \
    echo "[from-scratch] rust toolchain 1.92.0 from ${url}"; \
    curl -fsSL "$url" -o /tmp/rust.tar.xz \
    && mkdir -p /tmp/rust-src && tar xJf /tmp/rust.tar.xz -C /tmp/rust-src --strip-components=1 \
    && /tmp/rust-src/install.sh --prefix=/usr/local --disable-ldconfig \
         --components=rustc,cargo,rust-std-x86_64-unknown-linux-gnu >/dev/null \
    && rm -rf /tmp/rust.tar.xz /tmp/rust-src \
    && echo "[from-scratch] $(rustc --version) / $(cargo --version)"

RUN set -eu; mkdir -p /opt/engine-src; cd /opt/engine-src; \
    git clone --depth 1 --branch v0.5.18 --recurse-submodules --shallow-submodules \
        https://github.com/sgl-project/sglang.git .; \
    head="$(git rev-parse HEAD)"; \
    if [ "$head" != "71de97b264b04dcd514cf904003028aefe9775c8" ]; then \
        echo "FATAL: engine tree is at $head, not the reviewed 71de97b264b04dcd514cf904003028aefe9775c8 -- v0.5.18 has moved" >&2; exit 7; \
    fi; \
    rm -rf /opt/engine-src/.git; \
    find /opt/engine-src -maxdepth 4 -name .git -exec rm -rf {} + 2>/dev/null || true

RUN python3 - /opt/engine-src <<'INFB_PTX_PATCH'
#!/usr/bin/env python3
"""Rewrite the four CUDA-12.9-only `cuda::ptx` wrappers in sglang's dsv4 top-k JIT header as raw inline PTX.

WHY THIS PATCH EXISTS. Without it the correctness reference server dies at serve time compiling
`sglang/jit_kernel/include/sgl_kernel/deepseek_v4/topk/ptx.cuh` --

    error: no instance of overloaded function "cuda::ptx::mbarrier_try_wait_parity" matches ...
    4 errors detected ... ninja: build stopped ... Received sigquit from a child process

-- which gives correctness 0.0 and zero benches. Applying it here, in the image build, means every build
route gets one copy of the rewrite rather than a second copy that can drift.

WHAT IT DOES. `sgl_kernel/deepseek_v4/topk/ptx.cuh` calls four `cuda::ptx` C++ wrappers
(mbarrier_try_wait_parity / mbarrier_arrive_expect_tx / mbarrier_arrive / cp_async_bulk) that exist only in CUDA
12.9's CCCL headers, while the venvs these tasks build against are torch 2.11.0+cu128 on a CUDA 12.8 toolkit.
The underlying PTX is ISA<=8.0 and IS accepted by 12.8 ptxas -- only the C++ wrappers
are missing -- so each call is rewritten to the raw inline PTX the wrapper emits, using ISA-8.0 .acquire/.release
ordering, which is STRONGER than the original .relaxed and therefore correctness-safe for a reference run.

Idempotent (marker-guarded by the caller and by the `#pragma once` tag) and fail-fast: it exits non-zero rather
than write a tree it did not fully rewrite.

    exit 0  patched
    exit 3  the expected shapes were not found (upstream drift) -- nothing written
    exit 4  a 12.9-only wrapper survived the rewrite -- nothing written

usage: one argument -- a path to ptx.cuh, or a source tree to search.

A DIRECTORY is accepted so this can run in-image with one argument and no shell plumbing: the header only
exists in the 0.5.12 trees (0.5.10 has no dsv4 top-k kernel at all), and "not present" is a normal outcome
that exits 0. A caller with an exact file path reaches the same code.
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
        echo "[from-scratch] no sglang source staged; the task will need INFB_SGLANG_SOURCE_DIR or a pre-staged source tree"; \
    fi

RUN if [ -d /opt/engine-src/python ]; then \
        n=0; \
        until SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG="0.5.18" \
              /opt/venv/bin/python -m pip install --no-deps \
                  --index-url "${PIP_INDEX_URL}" \
                  -e /opt/engine-src/python; do \
          n=$((n + 1)); \
          [ "$n" -ge 3 ] && { echo "[from-scratch] editable sglang install failed 3x -- giving up"; exit 1; }; \
          echo "[from-scratch] editable sglang install attempt ${n} failed; retrying in 45s"; sleep 45; \
        done; \
        got="$(/opt/venv/bin/python -c 'from importlib import metadata; print(metadata.version("sglang"))')" \
        && if [ "${got}" != "0.5.18" ]; then \
               echo "FATAL: sglang installed as ${got}, expected 0.5.18 — a version drift this build must not ship" >&2; \
               exit 97; \
           fi \
        && echo "[from-scratch] sglang ${got} installed editable from /opt/engine-src/python" \
        && rm -fv /opt/venv/lib/python*/site-packages/__editable__*sglang*.pth \
                  /opt/venv/lib/python*/site-packages/__editable___sglang_*_finder.py \
        && echo "[from-scratch] editable overlay stripped (MANDATORY — see prepare_target_env.sh)"; \
    fi

RUN /opt/venv/bin/python - <<'INFB_FLASHINFER_REALIGN'
import glob, pathlib, sys
old, new = "alignas(64) CUtensorMap", "alignas(128) CUtensorMap"
done = 0
for path in glob.glob("/opt/venv/lib/python*/site-packages/flashinfer/data/csrc/**/*.cu", recursive=True):
    text = pathlib.Path(path).read_text(encoding="utf-8")
    if old in text:
        pathlib.Path(path).write_text(text.replace(old, new), encoding="utf-8")
        done += 1
if not done:
    sys.exit(f"FATAL: no {old!r} under flashinfer/data/csrc; the pinned flashinfer changed, revisit this step")
print(f"[from-scratch] flashinfer CUtensorMap re-aligned to 128 in {done} file(s)")
INFB_FLASHINFER_REALIGN

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


RUN set -eu; mkdir -p /opt/csrc-deps/FlashMLA; cd /opt/csrc-deps/FlashMLA; \
    git init -q .; git remote add origin https://github.com/sgl-project/FlashMLA.git; \
    if git fetch -q --depth 1 origin 05e26647fe840b8baedae486c2d86d5ce4efeb7c 2>/dev/null; then \
        git checkout -q FETCH_HEAD; \
    else \
        echo "[from-scratch] FlashMLA: fetch-by-id refused for 05e26647fe840b8baedae486c2d86d5ce4efeb7c; fetching branches and tags"; \
        git fetch -q --tags origin; git checkout -q 05e26647fe840b8baedae486c2d86d5ce4efeb7c; \
    fi; \
    git submodule update -q --init --recursive --depth 1 2>/dev/null || true; \
    rm -rf /opt/csrc-deps/FlashMLA/.git; : > /opt/csrc-deps/FlashMLA/.staged

RUN set -eu; mkdir -p /opt/csrc-deps/cutlass; cd /opt/csrc-deps/cutlass; \
    git init -q .; git remote add origin https://github.com/NVIDIA/cutlass.git; \
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

RUN set -eu; mkdir -p /opt/csrc-deps/flashmla-cutlass; cd /opt/csrc-deps/flashmla-cutlass; \
    git init -q .; git remote add origin https://github.com/NVIDIA/cutlass.git; \
    if git fetch -q --depth 1 origin 147f5673d0c1c3dcf66f78d677fd647e4a020219 2>/dev/null; then \
        git checkout -q FETCH_HEAD; \
    else \
        echo "[from-scratch] flashmla-cutlass: fetch-by-id refused for 147f5673d0c1c3dcf66f78d677fd647e4a020219; fetching branches and tags"; \
        git fetch -q --tags origin; git checkout -q 147f5673d0c1c3dcf66f78d677fd647e4a020219; \
    fi; \
    git submodule update -q --init --recursive --depth 1 2>/dev/null || true; \
    rm -rf /opt/csrc-deps/flashmla-cutlass/.git; : > /opt/csrc-deps/flashmla-cutlass/.staged

RUN set -eu; mkdir -p /opt/csrc-deps/fmt; cd /opt/csrc-deps/fmt; \
    git init -q .; git remote add origin https://github.com/fmtlib/fmt.git; \
    if git fetch -q --depth 1 origin 553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28 2>/dev/null; then \
        git checkout -q FETCH_HEAD; \
    else \
        echo "[from-scratch] fmt: fetch-by-id refused for 553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28; fetching branches and tags"; \
        git fetch -q --tags origin; git checkout -q 553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28; \
    fi; \
    git submodule update -q --init --recursive --depth 1 2>/dev/null || true; \
    rm -rf /opt/csrc-deps/fmt/.git; : > /opt/csrc-deps/fmt/.staged

RUN set -eu; mkdir -p /opt/csrc-deps/sgl-attn; cd /opt/csrc-deps/sgl-attn; \
    git init -q .; git remote add origin https://github.com/sgl-project/sgl-attn.git; \
    if git fetch -q --depth 1 origin f89bc2306632d1ec5f97b014dded4254f5b4a907 2>/dev/null; then \
        git checkout -q FETCH_HEAD; \
    else \
        echo "[from-scratch] sgl-attn: fetch-by-id refused for f89bc2306632d1ec5f97b014dded4254f5b4a907; fetching branches and tags"; \
        git fetch -q --tags origin; git checkout -q f89bc2306632d1ec5f97b014dded4254f5b4a907; \
    fi; \
    git submodule update -q --init --recursive --depth 1 2>/dev/null || true; \
    rm -rf /opt/csrc-deps/sgl-attn/.git; : > /opt/csrc-deps/sgl-attn/.staged

RUN set -eu; mkdir -p /opt/csrc-deps/triton; cd /opt/csrc-deps/triton; \
    git init -q .; git remote add origin https://github.com/triton-lang/triton.git; \
    if git fetch -q --depth 1 origin v3.7.1 2>/dev/null; then \
        git checkout -q FETCH_HEAD; \
    else \
        echo "[from-scratch] triton: fetch-by-id refused for v3.7.1; fetching branches and tags"; \
        git fetch -q --tags origin; git checkout -q v3.7.1; \
    fi; \
    git submodule update -q --init --recursive --depth 1 2>/dev/null || true; \
    rm -rf /opt/csrc-deps/triton/.git; : > /opt/csrc-deps/triton/.staged

RUN { \
      printf '%s\t%s\t%s\t%s\n' 'FlashMLA' 'FETCHCONTENT_SOURCE_DIR_REPO-FLASHMLA' '-' 'repo-flashmla'; \
      printf '%s\t%s\t%s\t%s\n' 'cutlass' 'FETCHCONTENT_SOURCE_DIR_REPO-CUTLASS' '-' 'repo-cutlass'; \
      printf '%s\t%s\t%s\t%s\n' 'flashinfer' 'FETCHCONTENT_SOURCE_DIR_REPO-FLASHINFER' '-' 'repo-flashinfer'; \
      printf '%s\t%s\t%s\t%s\n' 'flashmla-cutlass' 'FETCHCONTENT_SOURCE_DIR_REPO-FLASHMLA-CUTLASS' '-' 'repo-flashmla-cutlass'; \
      printf '%s\t%s\t%s\t%s\n' 'fmt' 'FETCHCONTENT_SOURCE_DIR_REPO-FMT' '-' 'repo-fmt'; \
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
    CUDA_HOME=/usr/local/cuda-13.0 \
    TORCH_CUDA_ARCH_LIST=10.0+PTX \
    PATH=/opt/venv/bin:/usr/local/cuda-13.0/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/opt/infb/cuda-compat/compat:/usr/local/cuda-13.0/lib64:/usr/local/nvidia/lib64 \
    BASH_ENV=/etc/infb-forward-compat.sh

RUN printf '#!/bin/sh\nldconfig 2>/dev/null || true\nexec "$@"\n' > /usr/local/bin/infb-entrypoint \
    && chmod 0755 /usr/local/bin/infb-entrypoint

RUN printf '%s\n' \
        '#!/bin/sh' \
        'ldconfig 2>/dev/null || true' \
        '# STDERR, and neither /dev/null nor stdout. Discarding it leaves no record of which branch' \
        '# the activator took. Printing it to STDOUT contaminates the first command that runs through' \
        '# the entrypoint: it was once parsed as a dpkg PACKAGE name by the first command that ran' \
        '# through it, landing in the middle of that inventory.' \
        '/usr/local/bin/infb-forward-compat-activate >&2 || true' \
        'exec "$@"' \
        > /usr/local/bin/infb-entrypoint; \
    chmod 0755 /usr/local/bin/infb-entrypoint; \
    echo "[from-scratch] entrypoint activates forward compat at container start"

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
assert _v.split('+')[0] == '0.5.18', _v
assert not missing, f'lock pins not installed: {missing[:5]}'
assert not wrong, f'installed versions differ from the lock: {wrong[:5]}'
PY

RUN /opt/venv/bin/python - <<'PY'
import base64, hashlib, pathlib
from importlib import metadata
REL = 'nvidia/nccl/lib/libnccl.so.2'
WANT = 'nvidia-nccl-cu13'  # the CUDA major this image is built for
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
WANT = '13'  # the CUDA major this image is built for
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
    f'ignores a +cuXXX local label), and neither can the lock. Fix this Dockerfile: the base image,\n'
    f'CUDA_HOME and the --extra-index-url torch channel in the lock above must all match the CUDA\n'
    f'generation the engine wheel was BUILT with. Unfixed, the first import at run time raises\n'
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
