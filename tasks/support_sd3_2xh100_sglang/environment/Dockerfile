#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#


# Model and CLIP scorer weights are staged separately under their upstream licenses.


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
















accelerate===1.14.0
addict===2.4.0
aiohappyeyeballs===2.7.1
aiohttp===3.14.3
aiosignal===1.4.0
airportsdata===20260803
annotated-doc===0.0.5
annotated-types===0.8.0
anthropic===1.0.0
anyio===4.14.2
apache-tvm-ffi===0.1.13.post3
asttokens===3.0.2
attrs===26.1.0
av===16.1.0
blobfile===3.0.0
build===1.5.0
cache-dit===1.3.0
certifi===2026.7.22
cffi===2.1.1
charset-normalizer===3.5.1
click===8.4.2
cloudpickle===3.1.2
compressed-tensors===0.15.0.1
cuda-bindings===13.3.1
cuda-core===1.0.1
cuda-pathfinder===1.6.1
cuda-python===13.3.1
cuda-tile===1.5.0
datasets===5.0.1
decorator===5.3.1
diffusers===0.37.0
dill===0.4.1
diskcache===5.6.3
distro===1.9.0
docstring-parser===0.18.0
einops===0.8.2
executing===2.2.1
fastapi===0.141.1
filelock===3.32.4
flash-attn-4===4.0.0b26
flashinfer-cubin===0.6.7.post3
flashinfer-python===0.6.7.post3
frozenlist===1.8.0
fsspec===2026.6.0
gguf===0.19.0
grpcio===1.83.0
grpcio-health-checking===1.81.1
grpcio-reflection===1.81.1
h11===0.16.0
hf-xet===1.6.0
httpcore===1.0.9
httpcore2===2.12.0
httpx===0.28.1
httpx2===2.12.0
huggingface-hub===1.28.0
humanize===4.16.0
idna===3.19
imageio===2.36.0
imageio-ffmpeg===0.5.1
importlib-metadata===9.0.0
interegular===0.3.3
ipython===9.16.1
ipython-pygments-lexers===1.1.1
jedi===0.20.0
jinja2===3.1.6
jiter===0.16.0
jsonschema===4.26.0
jsonschema-specifications===2025.9.1
kernels===0.14.1
kernels-data===0.16.0
lark===1.3.1
lazy-loader===0.5
llguidance===0.7.30
loguru===0.7.3
lxml===6.1.2
markdown-it-py===4.2.0
markupsafe===3.0.3
matplotlib-inline===0.2.2
mdurl===0.1.2
mistral-common===1.11.7
modelscope===1.39.1
modelscope-hub===0.2.0
moviepy===2.2.1
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
nvidia-cudnn-frontend===1.27.0
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
opencv-python-headless===4.10.0.84
orjson===3.12.0
outlines===0.1.11
outlines-core===0.1.26
packaging===26.3
pandas===3.0.5
parso===0.8.7
partial-json-parser===0.2.1.1.post7
pexpect===4.9.0
pillow===11.3.0
pip===24.0
proglog===0.1.12
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
pydantic===2.13.4
pydantic-core===2.46.4
pydantic-extra-types===2.11.1
pygments===2.21.0
pyproject-hooks===1.2.0
python-dateutil===2.9.0.post0
python-dotenv===1.2.3
python-multipart===0.0.32
pyyaml===6.0.1
pyzmq===27.2.0
quack-kernels===0.5.3
referencing===0.37.0
regex===2026.7.19
remote-pdb===2.1.0
requests===2.34.2
rich===15.0.0
rpds-py===2026.6.3
runai-model-streamer===0.16.1
safetensors===0.8.0
scikit-image===0.25.2
scipy===1.18.1
sentencepiece===0.2.2
setproctitle===1.3.7
setuptools===84.0.0
sglang-kernel===0.4.1
shellingham===1.5.4
six===1.17.0
smg-grpc-proto===0.4.14
smg-grpc-servicer===0.8.0
sniffio===1.3.1
soundfile===0.13.1
st-attn===0.0.7
stack-data===0.6.3
starlette===1.6.0
sympy===1.14.0
tabulate===0.10.0
tifffile===2026.8.23
tiktoken===0.14.0
timm===1.0.16
tokenizers===0.22.2
tomlkit===0.15.1
torch===2.9.1
torch-c-dlpack-ext===0.1.5
torch-memory-saver===0.0.9
torchao===0.9.0
torchaudio===2.9.1
torchcodec===0.9.1
torchvision===0.24.1
tqdm===4.70.0
traitlets===5.16.1
transformers===5.3.0
trimesh===5.0.0
triton===3.5.1
truststore===0.10.4
typer===0.27.1
typing-extensions===4.16.0
typing-inspection===0.4.4
urllib3===2.7.0
uvicorn===0.52.4
uvloop===0.22.1
vsa===0.0.4
watchfiles===1.2.0
wcwidth===0.8.2
xatlas===0.0.11
xgrammar===0.1.32
xxhash===4.0.1
yarl===1.24.5
zipp===4.1.0
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
    git init -q .; \
    git remote add origin https://github.com/sgl-project/sglang.git; \
    git fetch -q --depth 1 origin 9e6d1c066ec3445b534eef6ccdcd752f1bddae1f; \
    git checkout -q FETCH_HEAD; \
    git submodule update -q --init --recursive --depth 1; \
    head="$(git rev-parse HEAD)"; \
    if [ "$head" != "9e6d1c066ec3445b534eef6ccdcd752f1bddae1f" ]; then \
        echo "FATAL: engine tree is at $head, not the pinned 9e6d1c066ec3445b534eef6ccdcd752f1bddae1f -- 9e6d1c066ec3445b534eef6ccdcd752f1bddae1f has moved" >&2; exit 7; \
    fi; \
    rm -rf /opt/engine-src/.git; \
    find /opt/engine-src -maxdepth 4 -name .git -exec rm -rf {} + 2>/dev/null || true

RUN python3 - /opt/engine-src <<'INFB_PTX_PATCH'
#!/usr/bin/env python3
"""Replace CUDA 12.9-only C++ PTX wrappers with CUDA 12.8-compatible inline PTX.

The rewrite is idempotent. Unsupported source layouts fail without writing.
A source tree without the affected header requires no changes.
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

s = s.replace(
    "  cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_relaxed, cuda::ptx::scope_cta, cuda::ptx::space_shared, addr, tx);",
    "  uint32_t _smem = static_cast<uint32_t>(__cvta_generic_to_shared(addr));\n"
    "  uint64_t _state;\n"
    "  asm volatile(\"mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 %0, [%1], %2;\"\n"
    "               : \"=l\"(_state) : \"r\"(_smem), \"r\"(tx) : \"memory\");")

s = s.replace(
    "  cuda::ptx::mbarrier_arrive(cuda::ptx::sem_relaxed, cuda::ptx::scope_cta, cuda::ptx::space_shared, addr);",
    "  uint32_t _smem = static_cast<uint32_t>(__cvta_generic_to_shared(addr));\n"
    "  uint64_t _state;\n"
    "  asm volatile(\"mbarrier.arrive.release.cta.shared::cta.b64 %0, [%1];\"\n"
    "               : \"=l\"(_state) : \"r\"(_smem) : \"memory\");")

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

s = s.replace("#pragma once\n", "#pragma once\n// INFB-CUDA128-JIT-COMPAT: raw-PTX rewrite of 4 cuda::ptx wrappers (12.9-only) for CUDA 12.8 nvcc\n", 1)

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
        echo "[from-scratch] no sglang source staged; the task will need INFB_SGLANG_SOURCE_DIR or a prepared source artifact"; \
    fi

RUN if [ -d /opt/engine-src/python ]; then \
        n=0; \
        until SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG="0.5.11.dev0" \
              /opt/venv/bin/python -m pip install --no-deps \
                  --index-url "${PIP_INDEX_URL}" \
                  -e /opt/engine-src/python; do \
          n=$((n + 1)); \
          [ "$n" -ge 3 ] && { echo "[from-scratch] editable sglang install failed 3x -- giving up"; exit 1; }; \
          echo "[from-scratch] editable sglang install attempt ${n} failed; retrying in 45s"; sleep 45; \
        done; \
        got="$(/opt/venv/bin/python -c 'from importlib import metadata; print(metadata.version("sglang"))')" \
        && if [ "${got}" != "0.5.11.dev0" ]; then \
               echo "FATAL: sglang installed as ${got}, expected 0.5.11.dev0" >&2; \
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
    PATH=/usr/local/bin:/opt/venv/bin:/usr/local/cuda-12.8/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
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
assert _v.split('+')[0] == '0.5.11.dev0', _v
assert not missing, f'lock pins not installed: {missing[:5]}'
assert not wrong, f'installed versions differ from the lock: {wrong[:5]}'
PY

RUN /opt/venv/bin/python - <<'PY'
import base64, hashlib, pathlib
from importlib import metadata
REL = 'nvidia/nccl/lib/libnccl.so.2'
WANT = 'nvidia-nccl-cu12'  # Expected CUDA major for this image.
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
WANT = '12'  # Expected CUDA major for this image.
ROOTS = ['sgl_kernel', 'sglang', 'torch']
def dt_needed(path):



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
    f'ignores a +cuXXX local label). The Dockerfile base image, CUDA directory and torch wheel\n'
    f'index must use the CUDA generation required by the engine wheel. Otherwise importing\n'
    f'the compiled extensions raises\n'
    f'ImportError: libcudart.so.<N>.'
)
PY

ENTRYPOINT ["/usr/local/bin/infb-entrypoint"]

CMD ["/bin/bash", "-l"]

FROM ${ENGINE}
ENV PYTHONUNBUFFERED=1 \
    INFB_MODEL_PATH=/opt/infb/models/sd3 \
    SD3_CLIP_MODEL=/opt/infb/models/clip \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONPATH=/app/repo/python \
    INFB_PYTHON_BIN_DIR=/usr/local/bin \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends rsync \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

COPY scripts /app/scripts

# Persist both default command names for fresh agent shells.
RUN /opt/venv/bin/python3 /app/scripts/python_entrypoints.py install \
    --workspace /app --bin-dir /usr/local/bin
