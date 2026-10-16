#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
ARG ENGINE=engine

FROM nvidia/cuda:12.9.1-devel-ubuntu24.04 AS engine

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
        libdw1 libelf1 libnuma1  \
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
--extra-index-url https://download.pytorch.org/whl/cu128

# Exact package closure for this task's runtime, captured from an installed environment
# rather than re-resolved at build time, so a rebuild reproduces the same versions.
#
# CUDA_HOME    : /usr/local/cuda-12.9
# arch list    : 10.0+PTX
# interpreter  : /opt/venv/bin/python3
# packages     : 197

Jinja2===3.1.6
MarkupSafe===3.0.3
pyarrow===25.0.1
PyJWT===2.13.0
PyYAML===6.0.3
Pygments===2.20.0
aiohappyeyeballs===2.6.2
aiohttp===3.13.5
aiosignal===1.4.0
annotated-doc===0.0.4
annotated-types===0.7.0
anthropic===0.104.1
anyio===4.13.0
apache-tvm-ffi===0.1.9
astor===0.8.1
attrs===26.1.0
blake3===1.0.8
cachetools===7.1.4
cbor2===6.1.1
certifi===2026.5.20
cffi===2.0.0
charset-normalizer===3.4.7
click===8.4.1
cloudpickle===3.1.2
compressed-tensors===0.15.0.1
cryptography===48.0.0
cuda-bindings===13.3.0
cuda-core===1.0.1
cuda-pathfinder===1.5.5
cuda-python===13.3.0
cuda-tile===1.4.0
cuda-toolkit===13.0.2
depyf===0.20.0
detect-installer===0.1.0
dill===0.4.1
diskcache===5.6.3
distro===1.9.0
dnspython===2.8.0
docstring_parser===0.18.0
einops===0.8.2
email-validator===2.3.0
fastapi-cli===0.0.24
fastapi-cloud-cli===0.18.0
fastapi===0.136.3
fastar===0.11.0
fastsafetensors===0.3.2
filelock===3.29.0
flashinfer-cubin===0.6.8.post1
flashinfer-python===0.6.8.post1
frozenlist===1.8.0
fsspec===2026.4.0
gguf===0.19.0
googleapis-common-protos===1.75.0
grpcio===1.80.0
h11===0.16.0
hf-xet===1.5.0
httpcore===1.0.9
httptools===0.8.0
httpx-sse===0.4.3
httpx===0.28.1
huggingface_hub===1.16.4
idna===3.16
ijson===3.5.0
interegular===0.3.3
jiter===0.15.0
jmespath===1.1.0
jsonschema-specifications===2025.9.1
jsonschema===4.26.0
lark===1.2.2
llguidance===1.3.0
llvmlite===0.47.0
lm-format-enforcer===0.11.3
loguru===0.7.3
markdown-it-py===4.2.0
mcp===1.27.1
mdurl===0.1.2
mistral_common===1.11.2
ml_dtypes===0.5.4
model-hosting-container-standards===0.1.15
mpmath===1.3.0
msgspec===0.21.1
multidict===6.7.1
networkx===3.6.1
ninja===1.13.0
numba===0.65.0
numpy===2.3.5
nvidia-cublas-cu12===12.8.4.1
nvidia-cublas===13.1.0.3
nvidia-cuda-cupti-cu12===12.8.90
nvidia-cuda-cupti===13.0.85
nvidia-cuda-nvcc-cu12===12.8.93
nvidia-cuda-nvrtc-cu12===12.8.93
nvidia-cuda-nvrtc===13.0.88
nvidia-cuda-runtime-cu12===12.8.90
nvidia-cuda-runtime===13.0.96
nvidia-cudnn-cu12===9.19.0.56
nvidia-cudnn-cu13===9.19.0.56
nvidia-cudnn-frontend===1.18.0
nvidia-cufft-cu12===11.3.3.83
nvidia-cufft===12.0.0.61
nvidia-cufile-cu12===1.13.1.3
nvidia-cufile===1.15.1.6
nvidia-curand-cu12===10.3.9.90
nvidia-curand===10.4.0.35
nvidia-cusolver-cu12===11.7.3.90
nvidia-cusolver===12.0.4.66
nvidia-cusparse-cu12===12.5.8.93
nvidia-cusparse===12.6.3.3
nvidia-cusparselt-cu12===0.7.1
nvidia-cusparselt-cu13===0.8.0
nvidia-cutlass-dsl-libs-base===4.5.2
nvidia-cutlass-dsl===4.5.2
nvidia-ml-py===13.595.45
nvidia-nccl-cu12===2.28.9
nvidia-nccl-cu13===2.28.9
nvidia-nvjitlink-cu12===12.8.93
nvidia-nvjitlink===13.0.88
nvidia-nvshmem-cu12===3.4.5
nvidia-nvshmem-cu13===3.4.5
nvidia-nvtx-cu12===12.8.90
nvidia-nvtx===13.0.85
openai-harmony===0.0.8
openai===2.38.0
opencv-python-headless===4.13.0.92
opentelemetry-api===1.42.1
opentelemetry-exporter-otlp-proto-common===1.42.1
opentelemetry-exporter-otlp-proto-grpc===1.42.1
opentelemetry-exporter-otlp-proto-http===1.42.1
opentelemetry-exporter-otlp===1.42.1
opentelemetry-proto===1.42.1
opentelemetry-sdk===1.42.1
opentelemetry-semantic-conventions-ai===0.5.1
opentelemetry-semantic-conventions===0.63b1
outlines_core===0.2.14
packaging===26.2
pandas===3.0.3
partial-json-parser===0.2.1.1.post7
pillow===12.2.0
prometheus-fastapi-instrumentator===7.1.0
prometheus_client===0.25.0
propcache===0.5.2
protobuf===6.33.6
psutil===7.2.2
py-cpuinfo===9.0.0
pybase64===1.4.3
pycountry===26.2.16
pycparser===3.0
pydantic-extra-types===2.11.1
pydantic-settings===2.14.1
pydantic===2.13.4
pydantic_core===2.46.4
python-dateutil===2.9.0.post0
python-dotenv===1.2.2
python-json-logger===4.1.0
python-multipart===0.0.29
pyzmq===27.1.0
quack-kernels===0.4.1
referencing===0.37.0
regex===2026.5.9
requests===2.34.2
rich-toolkit===0.19.10
rich===15.0.0
rignore===0.7.6
rpds-py===0.30.0
safetensors===0.7.0
sentencepiece===0.2.1
sentry-sdk===2.60.0
setproctitle===1.3.7
setuptools===80.10.2
shellingham===1.5.4
six===1.17.0
sniffio===1.3.1
sse-starlette===3.4.4
starlette===0.52.1
supervisor===4.3.0
sympy===1.14.0
tabulate===0.10.0
tiktoken===0.13.0
tilelang===0.1.9
tokenizers===0.22.2
torch==2.11.0+cu128
torch_c_dlpack_ext===0.1.5
torchaudio===2.11.0
torchvision==0.26.0+cu128
tqdm===4.67.3
transformers===5.9.0
triton===3.6.0
typer===0.25.1
typing-inspection===0.4.2
typing_extensions===4.15.0
urllib3===2.7.0
uvicorn===0.48.0
uvloop===0.22.1
vllm==0.20.1+cu129
watchfiles===1.2.0
websockets===16.0
wheel===0.47.0
xgrammar===0.2.1
yarl===1.24.2
z3-solver===4.15.4.0

# Bootstrap packages the venv creates itself. `pip freeze` omits them, so they
# were missing from the closure and a `pip install --upgrade pip` step left the
# built venv one pip version ahead of the environment this closure was captured
# from. Pinned here so the closure is exact.
pip===26.1.1
LOCK_EOF

RUN mkdir -p /opt/infb/wheels \
    && curl -fsSL "https://github.com/vllm-project/vllm/releases/download/v0.20.1/vllm-0.20.1%2Bcu129-cp38-abi3-manylinux_2_31_x86_64.whl" -o "/opt/infb/wheels/vllm-0.20.1+cu129-cp38-abi3-manylinux_2_31_x86_64.whl" \
    && echo "9b4e40078eea9777ec0e1ffebbcc352b60903ad044bdc3eb235a8a30e8da259a  /opt/infb/wheels/vllm-0.20.1+cu129-cp38-abi3-manylinux_2_31_x86_64.whl" | sha256sum -c -

RUN extra=""; for u in https://download.pytorch.org/whl/cu128; do extra="${extra} --extra-index-url ${u}"; done; \
    echo "[from-scratch] index=${PIP_INDEX_URL}${extra}"; \
    n=0; until /opt/venv/bin/python -m pip install --no-deps \
        --index-url "${PIP_INDEX_URL}" ${extra} \
        --find-links /opt/infb/wheels \
        -r /opt/infb/requirements.lock; do \
      n=$((n + 1)); \
      [ "$n" -ge 3 ] && { echo "[from-scratch] lock install failed 3x -- giving up"; exit 1; }; \
      echo "[from-scratch] lock install attempt ${n} failed; retrying in 45s"; sleep 45; \
    done; \
    { /opt/venv/bin/python -m pip check || true; }

RUN set -eu; mkdir -p /opt/engine-src; cd /opt/engine-src; \
    git init -q .; \
    git remote add origin https://github.com/vllm-project/vllm.git; \
    git fetch -q --depth 1 origin v0.20.1; \
    git checkout -q FETCH_HEAD; \
    git submodule update -q --init --recursive --depth 1; \
    head="$(git rev-parse HEAD)"; \
    if [ "$head" != "132765e3560659ff63ebd236203672e991b70e08" ]; then \
        echo "FATAL: engine tree is at $head, not the reviewed 132765e3560659ff63ebd236203672e991b70e08 -- v0.20.1 has moved" >&2; exit 7; \
    fi; \
    rm -rf /opt/engine-src/.git; \
    find /opt/engine-src -maxdepth 4 -name .git -exec rm -rf {} + 2>/dev/null || true

RUN if [ -f /opt/engine-src/pyproject.toml ] && [ -d /opt/engine-src/vllm ]; then \
        echo "[from-scratch] engine source baked at /opt/engine-src"; \
    elif [ -n "$(ls -A /opt/engine-src 2>/dev/null | grep -v '^\.keep$' || true)" ]; then \
        echo "FATAL: ENGINE_SRC_DIR was staged but is not a vLLM tree (need pyproject.toml + vllm/); prepare_target_env.sh would reject it" >&2; \
        exit 96; \
    else \
        rm -rf /opt/engine-src; \
        echo "[from-scratch] no engine source staged; the task will need INFB_VLLM_SOURCE_DIR to point at a vLLM tree"; \
    fi


RUN set -eu; mkdir -p /opt/csrc-deps/DeepGEMM; cd /opt/csrc-deps/DeepGEMM; \
    git init -q .; git remote add origin https://github.com/deepseek-ai/DeepGEMM.git; \
    if git fetch -q --depth 1 origin 891d57b4db1071624b5c8fa0d1e51cb317fa709f 2>/dev/null; then \
        git checkout -q FETCH_HEAD; \
    else \
        echo "[from-scratch] DeepGEMM: fetch-by-id refused for 891d57b4db1071624b5c8fa0d1e51cb317fa709f; fetching branches and tags"; \
        git fetch -q --tags origin; git checkout -q 891d57b4db1071624b5c8fa0d1e51cb317fa709f; \
    fi; \
    git submodule update -q --init --recursive --depth 1 2>/dev/null || true; \
    rm -rf /opt/csrc-deps/DeepGEMM/.git; : > /opt/csrc-deps/DeepGEMM/.staged

RUN set -eu; mkdir -p /opt/csrc-deps/FlashMLA; cd /opt/csrc-deps/FlashMLA; \
    git init -q .; git remote add origin https://github.com/vllm-project/FlashMLA; \
    if git fetch -q --depth 1 origin a6ec2ba7bd0a7dff98b3f4d3e6b52b159c48d78b 2>/dev/null; then \
        git checkout -q FETCH_HEAD; \
    else \
        echo "[from-scratch] FlashMLA: fetch-by-id refused for a6ec2ba7bd0a7dff98b3f4d3e6b52b159c48d78b; fetching branches and tags"; \
        git fetch -q --tags origin; git checkout -q a6ec2ba7bd0a7dff98b3f4d3e6b52b159c48d78b; \
    fi; \
    git submodule update -q --init --recursive --depth 1 2>/dev/null || true; \
    rm -rf /opt/csrc-deps/FlashMLA/.git; : > /opt/csrc-deps/FlashMLA/.staged

RUN set -eu; mkdir -p /opt/csrc-deps/cutlass; cd /opt/csrc-deps/cutlass; \
    git init -q .; git remote add origin https://github.com/nvidia/cutlass.git; \
    if git fetch -q --depth 1 origin v4.4.2 2>/dev/null; then \
        git checkout -q FETCH_HEAD; \
    else \
        echo "[from-scratch] cutlass: fetch-by-id refused for v4.4.2; fetching branches and tags"; \
        git fetch -q --tags origin; git checkout -q v4.4.2; \
    fi; \
    git submodule update -q --init --recursive --depth 1 2>/dev/null || true; \
    rm -rf /opt/csrc-deps/cutlass/.git; : > /opt/csrc-deps/cutlass/.staged

RUN set -eu; mkdir -p /opt/csrc-deps/flash-attention; cd /opt/csrc-deps/flash-attention; \
    git init -q .; git remote add origin https://github.com/vllm-project/flash-attention.git; \
    if git fetch -q --depth 1 origin f5bc33cfc02c744d24a2e9d50e6db656de40611c 2>/dev/null; then \
        git checkout -q FETCH_HEAD; \
    else \
        echo "[from-scratch] flash-attention: fetch-by-id refused for f5bc33cfc02c744d24a2e9d50e6db656de40611c; fetching branches and tags"; \
        git fetch -q --tags origin; git checkout -q f5bc33cfc02c744d24a2e9d50e6db656de40611c; \
    fi; \
    git submodule update -q --init --recursive --depth 1 2>/dev/null || true; \
    rm -rf /opt/csrc-deps/flash-attention/.git; : > /opt/csrc-deps/flash-attention/.staged

RUN set -eu; mkdir -p /opt/csrc-deps/qutlass; cd /opt/csrc-deps/qutlass; \
    git init -q .; git remote add origin https://github.com/IST-DASLab/qutlass.git; \
    if git fetch -q --depth 1 origin 830d2c4537c7396e14a02a46fbddd18b5d107c65 2>/dev/null; then \
        git checkout -q FETCH_HEAD; \
    else \
        echo "[from-scratch] qutlass: fetch-by-id refused for 830d2c4537c7396e14a02a46fbddd18b5d107c65; fetching branches and tags"; \
        git fetch -q --tags origin; git checkout -q 830d2c4537c7396e14a02a46fbddd18b5d107c65; \
    fi; \
    git submodule update -q --init --recursive --depth 1 2>/dev/null || true; \
    rm -rf /opt/csrc-deps/qutlass/.git; : > /opt/csrc-deps/qutlass/.staged

RUN set -eu; mkdir -p /opt/csrc-deps/triton; cd /opt/csrc-deps/triton; \
    git init -q .; git remote add origin https://github.com/triton-lang/triton.git; \
    if git fetch -q --depth 1 origin v3.6.0 2>/dev/null; then \
        git checkout -q FETCH_HEAD; \
    else \
        echo "[from-scratch] triton: fetch-by-id refused for v3.6.0; fetching branches and tags"; \
        git fetch -q --tags origin; git checkout -q v3.6.0; \
    fi; \
    git submodule update -q --init --recursive --depth 1 2>/dev/null || true; \
    rm -rf /opt/csrc-deps/triton/.git; : > /opt/csrc-deps/triton/.staged

RUN { \
      printf '%s\t%s\t%s\t%s\n' 'DeepGEMM' 'DEEPGEMM_SRC_DIR' '-' 'deepgemm'; \
      printf '%s\t%s\t%s\t%s\n' 'FlashMLA' 'FLASH_MLA_SRC_DIR' '-' 'flashmla'; \
      printf '%s\t%s\t%s\t%s\n' 'cutlass' 'VLLM_CUTLASS_SRC_DIR' '-' 'cutlass'; \
      printf '%s\t%s\t%s\t%s\n' 'flash-attention' 'VLLM_FLASH_ATTN_SRC_DIR' '-' 'vllm-flash-attn'; \
      printf '%s\t%s\t%s\t%s\n' 'qutlass' 'QUTLASS_SRC_DIR' '-' 'qutlass'; \
      printf '%s\t%s\t%s\t%s\n' 'triton' 'TRITON_KERNELS_SRC_DIR' 'python/triton_kernels/triton_kernels' 'triton_kernels'; \
    } > /opt/csrc-deps/knobs.tsv

RUN if [ -s /opt/csrc-deps/knobs.tsv ]; then \
        echo "[from-scratch] csrc deps staged:"; \
        while IFS="$(printf '\t')" read -r n k; do \
            if [ -d "/opt/csrc-deps/${n}" ]; then \
                printf '  %-18s %-26s %s\n' "$n" "$k" "$(du -sh "/opt/csrc-deps/${n}" | cut -f1)"; \
            else \
                printf '  %-18s %-26s MISSING (knob will stay unset)\n' "$n" "$k"; \
            fi; \
        done < /opt/csrc-deps/knobs.tsv; \
        true; \
    else \
        rm -rf /opt/csrc-deps; \
        echo "[from-scratch] no csrc deps staged — an agent csrc rebuild will need network at configure time"; \
    fi

ENV INFB_VLLM_SOURCE_DIR=/opt/engine-src

ENV VIRTUAL_ENV=/opt/venv \
    INFB_VLLM_PYTHON=/opt/venv/bin/python3 \
    CUDA_HOME=/usr/local/cuda-12.9 \
    TORCH_CUDA_ARCH_LIST=10.0+PTX \
    PATH=/opt/venv/bin:/usr/local/cuda-12.9/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:/usr/local/nvidia/lib64

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
print('vllm', metadata.version('vllm'))
assert not missing, f'lock pins not installed: {missing[:5]}'
assert not wrong, f'installed versions differ from the lock: {wrong[:5]}'
PY

RUN /opt/venv/bin/python - <<'PY'
import base64, hashlib, pathlib
from importlib import metadata
REL = 'nvidia/nccl/lib/libnccl.so.2'
WANT = 'nvidia-nccl-cu12'  # the CUDA major this image builds on
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
WANT = '12'  # the CUDA major this image builds on
ROOTS = ['vllm', 'torch']
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
    f'ignores a +cuXXX local label), and neither can the lock. Fix this Dockerfile: the base\n'
    f'image, CUDA_HOME and the torch index URL must match the generation the engine wheel was\n'
    f'BUILT with. Unfixed, the first import in the run raises\n'
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
