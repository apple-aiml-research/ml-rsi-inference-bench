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
# Indexes these pins need beyond https://pypi.org/simple: PEP 440 local labels (+cu128, +cu129) that PyPI
# cannot serve by definition, so the project's own public index serves them. Folded in here rather
# than kept in a sibling manifest -- pip honours these lines directly.
--extra-index-url https://docs.sglang.ai/whl/cu129/
--extra-index-url https://download.pytorch.org/whl/cu128

# Lockfile for this task's SGLang runtime env, taken as a freeze of a known-good resolution rather
# than from a build script.
#
# distributions resolved : 217
# pinned here            : 215
#
# NOT pinned, and why:
#   sglang==0.5.15 — installed EDITABLE from source/python, because the artifact's
#     PEP 660 finder and .pth are stripped on purpose so `import sglang` resolves to the
#     agent's repo/. This image therefore installs it with `pip install -e` from the
#     pinned source tree, with SETUPTOOLS_SCM_PRETEND_VERSION so the built version is
#     exactly this string rather than whatever git-describe reports. The build fails
#     closed if the installed version is anything else.
#   flash-mla==1.0.0+9241ae3 — BUILT FROM SOURCE at env-build time; the `+9241ae3` label is the
#     source revision, so no index has this file and none can. The build clones the
#     upstream repo at 9241ae3 and pip-installs the tree with --no-build-isolation.

aiohappyeyeballs===2.7.1
aiohttp===3.14.2
aiosignal===1.4.0
airportsdata===20260315
annotated-doc===0.0.4
annotated-types===0.7.0
anthropic===0.117.0
anyio===4.14.2
apache-tvm-ffi===0.1.9
asttokens===3.0.2
attrs===26.1.0
blobfile===3.0.0
build===1.5.0
certifi===2026.6.17
cffi===2.1.0
charset-normalizer===3.4.9
click===8.4.2
cloudpickle===3.1.2
compressed-tensors===0.17.1
cryptography===49.0.0
cuda-bindings===12.9.7
cuda-core===1.0.1
cuda-pathfinder===1.5.6
cuda-python===13.3.1
cuda-tile===1.4.0
cuda-toolkit===12.8.1
datasets===5.0.0
decorator===5.3.1
dill===0.4.1
diskcache===5.6.3
distro===1.9.0
dnspython===2.8.0
docstring-parser===0.18.0
easydict===1.13
einops===0.8.2
email-validator===2.3.0
executing===2.2.1
fastapi===0.139.2
filelock===3.31.1
flash-attn-4===4.0.0b15
flashinfer-cubin===0.6.11.post1
flashinfer-python===0.6.11.post1
frozenlist===1.8.0
fsspec===2026.6.0
gguf===0.19.0
grpcio===1.82.1
grpcio-health-checking===1.82.1
grpcio-reflection===1.82.1
h11===0.16.0
hf-xet===1.5.2
httpcore===1.0.9
httpx===0.28.1
huggingface-hub===1.24.0
id===1.6.1
idna===3.18
interegular===0.3.3
ipython===9.15.0
ipython-pygments-lexers===1.1.1
jedi===0.20.0
jinja2===3.1.6
jiter===0.16.0
jsonschema===4.26.0
jsonschema-specifications===2025.9.1
kernels===0.12.3
kernels-data===0.16.0
lark===1.3.1
llguidance===0.7.30
loguru===0.7.3
lxml===6.1.1
markdown-it-py===4.2.0
markupsafe===3.0.3
matplotlib-inline===0.2.2
mdurl===0.1.2
mistral-common===1.11.6
ml-dtypes===0.5.4
modelscope===1.38.1
modelscope-hub===0.1.7
mpmath===1.3.0
msgspec===0.21.1
multidict===6.7.1
multiprocess===0.70.19
nest-asyncio===1.6.0
networkx===3.6.1
ninja===1.13.0
numpy===2.5.1
nvidia-cublas===13.1.0.3
nvidia-cublas-cu12===12.8.4.1
nvidia-cuda-cupti===13.0.85
nvidia-cuda-cupti-cu12===12.8.90
nvidia-cuda-nvrtc===13.0.88
nvidia-cuda-nvrtc-cu12===12.8.93
nvidia-cuda-runtime===13.0.96
nvidia-cuda-runtime-cu12===12.8.90
nvidia-cudnn-cu12===9.19.0.56
nvidia-cudnn-frontend===1.26.0
nvidia-cufft===12.0.0.61
nvidia-cufft-cu12===11.3.3.83
nvidia-cufile===1.15.1.6
nvidia-cufile-cu12===1.13.1.3
nvidia-curand===10.4.0.35
nvidia-curand-cu12===10.3.9.90
nvidia-cusolver===12.0.4.66
nvidia-cusolver-cu12===11.7.3.90
nvidia-cusparse===12.6.3.3
nvidia-cusparse-cu12===12.5.8.93
nvidia-cusparselt-cu12===0.7.1
nvidia-cutlass-dsl===4.5.1
nvidia-cutlass-dsl-libs-base===4.5.1
nvidia-cutlass-dsl-libs-cu13===4.5.1
nvidia-ml-py===13.610.43
nvidia-nccl-cu12===2.28.9
nvidia-nvjitlink===13.0.88
nvidia-nvjitlink-cu12===12.8.93
nvidia-nvshmem-cu12===3.4.5
nvidia-nvtx===13.0.85
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
pillow===12.3.0
pip===26.1.2
platformdirs===4.10.1
prometheus-client===0.25.0
prompt-toolkit===3.0.52
propcache===0.5.2
protobuf===7.35.1
psutil===7.2.2
ptyprocess===0.7.0
pure-eval===0.2.3
py-spy===0.4.2
pyarrow===25.0.0
pyasn1===0.6.4
pybase64===1.4.3
pycountry===26.2.16
pycparser===3.0
pycryptodomex===3.23.0
pydantic===2.13.4
pydantic-core===2.46.4
pydantic-extra-types===2.11.1
pygments===2.20.0
pyjwt===2.13.0
pyopenssl===26.3.0
pyproject-hooks===1.2.0
python-dateutil===2.9.0.post0
python-multipart===0.0.32
pyyaml===6.0.3
pyzmq===27.1.0
quack-kernels===0.4.1
referencing===0.37.0
regex===2026.7.19
requests===2.34.2
rfc3161-client===1.0.7
rfc8785===0.1.4
rich===15.0.0
rpds-py===2026.6.3
safetensors===0.8.0
scipy===1.18.0
securesystemslib===1.4.0
sentencepiece===0.2.2
setproctitle===1.3.7
setuptools===81.0.0
sgl-deep-gemm==0.1.4.post1+cu129
sglang-kernel==0.4.5+cu129
shellingham===1.5.4
sigstore===4.4.0
sigstore-models===0.0.6
sigstore-rekor-types===0.0.18
six===1.17.0
smg-grpc-proto===0.4.14
smg-grpc-servicer===0.7.0
sniffio===1.3.1
soundfile===0.13.1
stack-data===0.6.3
starlette===1.3.1
sympy===1.14.0
tabulate===0.10.0
tiktoken===0.13.0
tilelang===0.1.8
timm===1.0.16
tokenizers===0.22.2
tokenspeed-mla===0.1.1
tokenspeed-triton===3.8.10.post20260709
tomlkit===0.15.1
torch==2.11.0+cu128
torch-c-dlpack-ext===0.1.5
torch-memory-saver===0.0.9.post1
torchao===0.17.0
torchaudio==2.11.0+cu128
torchcodec===0.11.1
torchvision==0.26.0+cu128
tqdm===4.69.0
traitlets===5.15.1
transformers===5.6.0
triton===3.6.0
tuf===7.0.0
typer===0.27.0
typing-extensions===4.16.0
typing-inspection===0.4.2
urllib3===2.7.0
uvicorn===0.51.0
uvloop===0.22.1
watchfiles===1.2.0
wcwidth===0.8.2
wheel===0.47.0
xgrammar===0.2.0
xxhash===3.8.1
yarl===1.24.5
z3-solver===4.15.4.0
LOCK_EOF

RUN extra=""; for u in https://docs.sglang.ai/whl/cu129/ https://download.pytorch.org/whl/cu128; do extra="${extra} --extra-index-url ${u}"; done; \
    echo "[from-scratch] index=${PIP_INDEX_URL}${extra}"; \
    n=0; until /opt/venv/bin/python -m pip install --no-deps \
        --index-url "${PIP_INDEX_URL}" ${extra} \
        -r /opt/infb/requirements.lock; do \
      n=$((n + 1)); \
      [ "$n" -ge 3 ] && { echo "[from-scratch] lock install failed 3x -- giving up"; exit 1; }; \
      echo "[from-scratch] lock install attempt ${n} failed; retrying in 45s"; sleep 45; \
    done; \
    { /opt/venv/bin/python -m pip check || true; }

RUN set -e; \
    url="https://static.rust-lang.org/dist/rust-1.90.0-x86_64-unknown-linux-gnu.tar.xz"; \
    echo "[from-scratch] rust toolchain 1.90.0 from ${url}"; \
    curl -fsSL "$url" -o /tmp/rust.tar.xz \
    && mkdir -p /tmp/rust-src && tar xJf /tmp/rust.tar.xz -C /tmp/rust-src --strip-components=1 \
    && /tmp/rust-src/install.sh --prefix=/usr/local --disable-ldconfig \
         --components=rustc,cargo,rust-std-x86_64-unknown-linux-gnu >/dev/null \
    && rm -rf /tmp/rust.tar.xz /tmp/rust-src \
    && echo "[from-scratch] $(rustc --version) / $(cargo --version)"

RUN set -e; \
    mkdir -p /opt/flash-mla-src && cd /opt/flash-mla-src && git init -q . \
    && git remote add origin "https://github.com/deepseek-ai/FlashMLA.git" \
    && ( git fetch -q --depth 1 origin "9241ae3ef9bac614dd25e45e507e089f888280e0" 2>/dev/null && git checkout -q FETCH_HEAD \
         || ( echo "[from-scratch] flash_mla: fetch-by-id refused for 9241ae3ef9bac614dd25e45e507e089f888280e0; fetching branches and tags" \
              && git fetch -q --tags origin && git checkout -q "9241ae3ef9bac614dd25e45e507e089f888280e0" ) ) \
    && git submodule update -q --init --recursive \
    && echo "[from-scratch] flash_mla source at $(git rev-parse --short HEAD)" \
    && CUDA_HOME="/usr/local/cuda-12.8" PATH="/usr/local/cuda-12.8/bin:${PATH}" \
       TORCH_CUDA_ARCH_LIST="9.0+PTX" \
       FLASH_MLA_DISABLE_SM100="1" \
       /opt/venv/bin/python -m pip install --no-cache-dir --no-build-isolation --no-deps . \
    && got="$(/opt/venv/bin/python -c 'from importlib import metadata; print(metadata.version("flash-mla"))')" \
    && echo "[from-scratch] flash_mla installed as ${got}" \
    && label="$(printf '%.7s' "9241ae3ef9bac614dd25e45e507e089f888280e0")" \
    && case "${got}" in *"${label}"*) ;; \
         *) echo "FATAL: flash_mla built as ${got}, which does not carry ${label}, the pinned 9241ae3ef9bac614dd25e45e507e089f888280e0 — the venv would misreport its own provenance" >&2; exit 98 ;; \
       esac

RUN set -eu; mkdir -p /opt/engine-src; cd /opt/engine-src; \
    git init -q .; \
    git remote add origin https://github.com/sgl-project/sglang.git; \
    git fetch -q --depth 1 origin 01a96720c6f690ac00adeed94b762b193d8472cd; \
    git checkout -q FETCH_HEAD; \
    git submodule update -q --init --recursive --depth 1; \
    head="$(git rev-parse HEAD)"; \
    if [ "$head" != "01a96720c6f690ac00adeed94b762b193d8472cd" ]; then \
        echo "FATAL: engine tree is at $head, not the reviewed 01a96720c6f690ac00adeed94b762b193d8472cd -- 01a96720c6f690ac00adeed94b762b193d8472cd has moved" >&2; exit 7; \
    fi; \
    rm -rf /opt/engine-src/.git; \
    find /opt/engine-src -maxdepth 4 -name .git -exec rm -rf {} + 2>/dev/null || true

RUN python3 - /opt/engine-src <<'INFB_PTX_PATCH'
#!/usr/bin/env python3
"""Rewrite the four CUDA-12.9-only `cuda::ptx` wrappers in sglang's dsv4 top-k JIT header as raw inline PTX.

WHY THIS FILE EXISTS AS A FILE. Without it, a serve-time JIT build of this header fails outright:

    error: no instance of overloaded function "cuda::ptx::mbarrier_try_wait_parity" matches ...
    4 errors detected ... ninja: build stopped ... Received sigquit from a child process

Keeping the rewrite in one file that every build route calls is what stops a second copy drifting.

WHAT IT DOES. `sgl_kernel/deepseek_v4/topk/ptx.cuh` calls four `cuda::ptx` C++ wrappers
(mbarrier_try_wait_parity / mbarrier_arrive_expect_tx / mbarrier_arrive / cp_async_bulk) that exist only in CUDA
12.9's CCCL headers, while this environment is torch 2.11.0+cu128 on a CUDA 12.8 toolkit. The underlying PTX is
ISA<=8.0 and IS accepted by 12.8 ptxas -- only the C++ wrappers are missing -- so each call is rewritten to the
raw inline PTX the wrapper emits, using ISA-8.0 .acquire/.release ordering, which is STRONGER than the original
.relaxed and therefore correctness-safe for a reference run.

Idempotent (marker-guarded by the caller and by the `#pragma once` tag) and fail-fast: it exits non-zero rather
than write a tree it did not fully rewrite.

    exit 0  patched
    exit 3  the expected shapes were not found (upstream drift) -- nothing written
    exit 4  a 12.9-only wrapper survived the rewrite -- nothing written

usage: patch_dsv4_topk_ptx_cuda128.py <path to ptx.cuh, or a source tree to search>

A DIRECTORY is accepted so this can run in-image with one argument and no shell plumbing: the header only
exists in some engine versions, and "not present" is a normal outcome that exits 0. A caller that already knows
the exact file path may pass it instead; both reach the same code.
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
        echo "[from-scratch] no sglang source staged; the task will need INFB_SGLANG_SOURCE_DIR or a mounted source tree"; \
    fi

RUN if [ -d /opt/engine-src/python ]; then \
        n=0; \
        until SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG="0.5.15" \
              /opt/venv/bin/python -m pip install --no-deps \
                  --index-url "${PIP_INDEX_URL}" \
                  -e /opt/engine-src/python; do \
          n=$((n + 1)); \
          [ "$n" -ge 3 ] && { echo "[from-scratch] editable sglang install failed 3x -- giving up"; exit 1; }; \
          echo "[from-scratch] editable sglang install attempt ${n} failed; retrying in 45s"; sleep 45; \
        done; \
        got="$(/opt/venv/bin/python -c 'from importlib import metadata; print(metadata.version("sglang"))')" \
        && if [ "${got}" != "0.5.15" ]; then \
               echo "FATAL: sglang installed as ${got}, expected 0.5.15" >&2; \
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
    TORCH_CUDA_ARCH_LIST=9.0+PTX \
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
assert _v.split('+')[0] == '0.5.15', _v
assert not missing, f'lock pins not installed: {missing[:5]}'
assert not wrong, f'installed versions differ from the lock: {wrong[:5]}'
PY

RUN /opt/venv/bin/python - <<'PY'
import base64, hashlib, pathlib
from importlib import metadata
REL = 'nvidia/nccl/lib/libnccl.so.2'
WANT = 'nvidia-nccl-cu12'  # the CUDA major this image is built for
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
WANT = '12'  # the CUDA major this image is built for
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
    f'this recipe builds on CUDA {WANT} but the engine links libcudart.so.{sorted(wrong)}: {wrong}.\n'
    f'pip cannot see this (the cu12 and cu13 runtimes are different distributions and PEP 440\n'
    f'ignores a +cuXXX local label), and neither can the lock. Fix the BUILD: the base image, the\n'
    f'CUDA directory and the torch extra-index channel must all match the generation the engine\n'
    f'wheel was BUILT with. Unfixed, the first import in the trial raises\n'
    f'ImportError: libcudart.so.<N>, the pristine arm cannot be benched, and the run is\n'
    f'recorded with reward 0.0 and reason=pristine_failed.'
)
PY

ENTRYPOINT ["/usr/local/bin/infb-entrypoint"]

CMD ["/bin/bash", "-l"]

FROM ${ENGINE}

RUN set -eu; \
    [ -d /opt/engine-src/python/sglang ] || { \
        echo "FATAL: the engine stage did not leave an SGLang monorepo at /opt/engine-src" >&2; exit 96; }; \
    [ "$(basename /opt/engine-src)" = "engine-src" ] || { \
        echo "FATAL: the source dir must be named engine-src; its name is hashed" >&2; exit 1; }; \
    mkdir -p /opt/inference-bench; \
    archive=/opt/inference-bench/engine-src.tar.gz; \
    tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner \
        --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
        --exclude='*.egg-info' --exclude='build' --exclude='sglang/_version.py' \
        --exclude='_core*.so' \
        -cf - -C /opt engine-src | gzip -n > "${archive}"; \
    digest="$(gzip -dc "${archive}" | sha256sum | awk '{print $1}')"; \
    printf '%s  engine-src.tar (uncompressed stream of engine-src.tar.gz)\n' "${digest}" \
        > "${archive}.sha256"; \
    rm -rf /opt/engine-src; \
    echo "[task-image] pinned engine source archived: $(du -h "${archive}" | cut -f1)"; \
    echo "[task-image] engine-src.tar sha256 = ${digest}"; \
    echo "[task-image]   ^ commit this into tests/engine_src_archive.sha256, or every scored run fails closed."

ENV INFB_ENGINE_SRC_ARCHIVE=/opt/inference-bench/engine-src.tar.gz

RUN set -eu; \
    /opt/venv/bin/python -m pip uninstall --yes sglang >/dev/null 2>&1 || true; \
    rm -rf /opt/venv/lib/python*/site-packages/sglang \
           /opt/venv/lib/python*/site-packages/__editable__*sglang* ; \
    if /opt/venv/bin/python -c 'from importlib import metadata; metadata.version("sglang")' 2>/dev/null; then \
        echo "FATAL: an sglang distribution is still installed after the uninstall" >&2; exit 95; \
    fi; \
    echo "[task-image] engine distribution evicted; only the archive remains"

RUN /opt/venv/bin/python - <<'PY'
from importlib import metadata
import pathlib
import re
lock = pathlib.Path('/opt/infb/requirements.lock')
assert lock.is_file(), 'the engine stage did not leave its lock at /opt/infb/requirements.lock'
pins = {}
for raw in lock.read_text().splitlines():
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
    'an engine distribution is installed in this venv. Anything resolvable as `sglang` outside the '
    'staged tree is a second engine the harness might bench instead of the one under test.'
)
# The mechanism under test is a Triton constexpr specialisation and the reference repair is
# `@triton.jit(do_not_specialize=[...])`. On a Triton without that argument the reference arm would
# be a silent no-op and the task would be unsolvable for a reason nothing reports.
import inspect
import triton
assert 'do_not_specialize' in inspect.signature(triton.jit).parameters, (
    f'triton {metadata.version("triton")} has no do_not_specialize argument'
)
print(f'triton {metadata.version("triton")}: do_not_specialize present')
PY

WORKDIR /app

COPY public /app/public
COPY scripts /app/scripts

RUN set -eu; \
    found="$(find / -xdev -name 'l2norm.py' -not -path '/proc/*' 2>/dev/null || true)"; \
    if [ -n "${found}" ]; then \
        echo "FATAL: a second copy of the engine survives in this image:" >&2; echo "${found}" >&2; exit 9; \
    fi; \
    if PYTHONPATH= /opt/venv/bin/python -c 'import sglang, sys; sys.stderr.write(sglang.__file__)' 2>/dev/null; then \
        echo "" >&2; \
        echo "FATAL: the venv can import an engine on its own, so PYTHONPATH does not decide which" >&2; \
        echo "  tree the harness benches -- see the path above." >&2; exit 9; \
    fi; \
    echo "[task-image] whole-image check: no engine Python outside the agent's tree, and the venv cannot import one"

RUN set -eu; \
    if [ -e "${INFB_SGLANG_SOURCE_DIR}" ]; then \
        echo "FATAL: INFB_SGLANG_SOURCE_DIR=${INFB_SGLANG_SOURCE_DIR} still resolves -- the task" >&2; \
        echo "  layer's eviction did not take, so the advertised path is a real second tree" >&2; exit 9; \
    fi; \
    [ -f "${INFB_ENGINE_SRC_ARCHIVE}" ] || { \
        echo "FATAL: INFB_ENGINE_SRC_ARCHIVE=${INFB_ENGINE_SRC_ARCHIVE} is missing; test.sh needs it" >&2; exit 9; }; \
    echo "[task-image] env check: the advertised source dir resolves to nothing, the archive resolves"

CMD ["/bin/bash", "-l"]

COPY deidentify_engine_version.py /tmp/deidentify_engine_version.py
RUN /opt/venv/bin/python /tmp/deidentify_engine_version.py --package sglang --scan \
    && rm -f /tmp/deidentify_engine_version.py

RUN rm -f /opt/infb/requirements.lock \
    && echo "[lock] /opt/infb/requirements.lock removed: build-time only, and it carried the engine pin"
