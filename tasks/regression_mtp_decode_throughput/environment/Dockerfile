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
--extra-index-url https://wheels.vllm.ai/c30b405b8f022d41f57042cad7032256d4831355/

# Lockfile for this task's vLLM H100 runtime env — a python3.12 venv resolved by pip from this
# image's OWN two indexes (download.pytorch.org/whl/cu128 and wheels.vllm.ai/c30b405b8f.../).
#
# The venv it was taken from installs vllm 0.11.0rc2.dev55+gc30b405b8 -- the pin itself -- plus
# torch 2.8.0+cu128, which is exactly what requirements/cuda.txt asks for at that commit. A LOCAL
# VENV IS NOT THE WHOLE CHECK: an image build is the only thing that checks the apt-side and
# CUDA-generation halves a venv-side resolve cannot see, which is what the asserts below are for.
#
# distributions in that environment : 149
# pinned here                       : 149

agent-detector===2.0.0
aiohappyeyeballs===2.7.1
aiohttp===3.14.3
aiosignal===1.4.0
annotated-doc===0.0.5
annotated-types===0.8.0
anyio===4.15.1
astor===0.8.1
attrs===26.1.0
blake3===1.0.9
cachetools===7.1.8
cbor2===6.1.4
certifi===2026.7.22
cffi===2.1.1
charset-normalizer===3.5.1
click===8.5.0
cloudpickle===3.1.2
compressed-tensors===0.11.0
cuda-pathfinder===1.8.1
cupy-cuda12x===14.2.0
depyf===0.19.0
detect-installer===0.2.1
dill===0.4.1
diskcache===5.6.3
dnspython===2.8.0
einops===0.8.2
email-validator===2.3.0
fastapi===0.141.1
fastapi-cli===0.0.32
fastapi-cloud-cli===0.26.0
fastar===0.12.0
filelock===3.32.3
frozendict===2.4.7
frozenlist===1.8.0
fsspec===2026.7.0
gguf===0.19.0
h11===0.16.0
hf-xet===1.6.0
httpcore===1.0.9
httpcore2===2.12.0
httptools===0.8.0
httpx===0.28.1
httpx2===2.12.0
huggingface-hub===0.36.2
idna===3.19
interegular===0.3.3
jinja2===3.1.6
jiter===0.17.0
jsonschema===4.26.0
jsonschema-specifications===2025.9.1
lark===1.2.2
llguidance===0.7.30
llvmlite===0.44.0
lm-format-enforcer===0.11.3
markdown-it-py===4.2.0
markupsafe===3.0.3
mdurl===0.1.2
mistral-common===1.11.7
mpmath===1.3.0
msgpack===1.2.2
msgspec===0.21.1
multidict===6.8.0
networkx===3.6.1
ninja===1.13.2
numba===0.61.2
numpy===2.2.6
nvidia-cublas-cu12===12.8.4.1
nvidia-cuda-cupti-cu12===12.8.90
nvidia-cuda-nvrtc-cu12===12.8.93
nvidia-cuda-runtime-cu12===12.8.90
nvidia-cudnn-cu12===9.10.2.21
nvidia-cufft-cu12===11.3.3.83
nvidia-cufile-cu12===1.13.1.3
nvidia-curand-cu12===10.3.9.90
nvidia-cusolver-cu12===11.7.3.90
nvidia-cusparse-cu12===12.5.8.93
nvidia-cusparselt-cu12===0.7.1
nvidia-nccl-cu12===2.27.3
nvidia-nvjitlink-cu12===12.8.93
nvidia-nvtx-cu12===12.8.90
openai===3.13.0
openai-harmony===0.0.8
opencv-python-headless===5.0.0.93
outlines-core===0.2.11
packaging===26.3
partial-json-parser===0.2.1.1.post7
pillow===12.3.0
pip===26.2.1
prometheus-client===0.26.0
prometheus-fastapi-instrumentator===8.1.0
propcache===0.5.2
protobuf===7.36.1
psutil===7.2.2
py-cpuinfo===9.0.0
pybase64===1.5.0
pycountry===26.2.16
pycparser===3.0
pydantic===2.13.5
pydantic-core===2.46.5
pydantic-extra-types===2.11.1
pydantic-settings===2.15.0
pygments===2.21.0
python-dotenv===1.2.3
python-json-logger===4.2.0
python-multipart===0.0.32
pyyaml===6.0.3
pyzmq===27.2.0
ray===2.58.0
referencing===0.37.0
regex===2026.9.10
requests===2.34.2
rich===15.0.0
rich-toolkit===0.20.5
rignore===0.8.1
rpds-py===2026.6.3
safetensors===0.8.0
scipy===1.18.1
sentencepiece===0.2.2
sentry-sdk===2.69.1
setproctitle===1.3.7
setuptools===78.1.0
shellingham===1.5.4
six===1.17.0
sniffio===1.3.1
soundfile===0.14.0
soxr===1.1.0
starlette===1.6.0
sympy===1.14.0
tiktoken===0.14.0
tokenizers===0.22.2
torch==2.8.0+cu128
torchaudio==2.8.0+cu128
torchvision==0.23.0+cu128
tqdm===4.70.1
transformers===4.57.6
triton===3.4.0
truststore===0.10.4
typer===0.27.2
typing-extensions===4.16.0
typing-inspection===0.4.4
urllib3===2.7.0
uvicorn===0.52.4
uvloop===0.22.1
vllm==0.11.0rc2.dev55+gc30b405b8
watchfiles===1.2.0
websockets===17.1
xformers===0.0.32.post1
xgrammar===0.1.25
yarl===1.24.5
LOCK_EOF

RUN mkdir -p /opt/infb/wheels

RUN extra=""; for u in https://download.pytorch.org/whl/cu128 https://wheels.vllm.ai/c30b405b8f022d41f57042cad7032256d4831355/; do extra="${extra} --extra-index-url ${u}"; done; \
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
    git fetch -q --depth 1 origin c30b405b8f022d41f57042cad7032256d4831355; \
    git checkout -q FETCH_HEAD; \
    git submodule update -q --init --recursive --depth 1; \
    head="$(git rev-parse HEAD)"; \
    if [ "$head" != "c30b405b8f022d41f57042cad7032256d4831355" ]; then \
        echo "FATAL: engine tree is at $head, not the reviewed c30b405b8f022d41f57042cad7032256d4831355 -- c30b405b8f022d41f57042cad7032256d4831355 has moved" >&2; exit 7; \
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
        echo "[from-scratch] no engine source staged; the task will need INFB_VLLM_SOURCE_DIR or a source tree staged at run time"; \
    fi

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
print('vllm', metadata.version('vllm'))
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
ROOTS = ['vllm', 'torch']
def dt_needed(path):
    # SEEKS rather than read_bytes(): a compiled extension here can be a gigabyte or more and this
    # needs a few KB of each. An ELF's section header table is at the END of the file,
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
    f'ignores a +cuXXX local label), and neither can the lock. Fix THIS IMAGE: the base image, the\n'
    f'CUDA_HOME directory and the torch index above must match the generation the engine wheel was\n'
    f'BUILT with. Unfixed, the first import in the trial raises\n'
    f'ImportError: libcudart.so.<N>, the pristine arm cannot be benched, and the run is\n'
    f'recorded with reward 0.0 and reason=pristine_failed.'
)
PY

ENTRYPOINT ["/usr/local/bin/infb-entrypoint"]

CMD ["/bin/bash", "-l"]

FROM ${ENGINE}
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        patch \
        procps \
        rsync \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY public /app/public

COPY scripts /app/scripts

RUN set -eu; \
    roots="/opt /usr/lib /usr/local /root /home"; \
    dest=/opt/infb/engine-ext/vllm; \
    n=0; \
    for pkg in $(find $roots -maxdepth 8 -type d -path '*/site-packages/vllm' 2>/dev/null); do \
      n=$((n + 1)); \
      mkdir -p "$dest"; \
      find "$pkg" -maxdepth 1 -name '*.so' -exec cp -a {} "$dest/" \; ; \
      if [ -f "$pkg/_version.py" ]; then cp -a "$pkg/_version.py" "$dest/"; fi; \
      if [ -d "$pkg/vllm_flash_attn" ]; then cp -a "$pkg/vllm_flash_attn" "$dest/"; fi; \
      echo "[evict] harvested $(find "$dest" -type f | wc -l) file(s) out of $pkg, removing the package"; \
      rm -rf "$pkg"; \
    done; \
    echo "[evict] evicted $n installed engine tree(s)"; \
    if [ "$n" -gt 0 ]; then \
      [ -n "$(find "$dest" -maxdepth 1 -name '*.so' -print -quit)" ] \
        || { echo "[evict] FATAL: harvest produced no compiled extension -- the overlay would have nothing to link"; exit 1; }; \
      stray=$(find "$dest" -name '*.py' ! -name '_version.py' ! -path "$dest/vllm_flash_attn/*" | wc -l); \
      [ "$stray" -eq 0 ] \
        || { echo "[evict] FATAL: $stray engine .py reached the harvest -- widen the allowlist deliberately or drop them:"; \
             find "$dest" -name '*.py' ! -name '_version.py' ! -path "$dest/vllm_flash_attn/*"; exit 1; }; \
      [ -n "$(find $roots -maxdepth 9 -type d -name 'vllm-*.dist-info' -print -quit)" ] \
        || { echo "[evict] FATAL: distribution metadata gone -- engines >= 0.11 read it to detect CUDA"; exit 1; }; \
    fi; \
    left=$(find $roots -maxdepth 8 -type d -path '*/site-packages/vllm' 2>/dev/null | wc -l); \
    [ "$left" -eq 0 ] \
      || { echo "[evict] FATAL: $left installed engine tree(s) survived:"; \
           find $roots -maxdepth 8 -type d -path '*/site-packages/vllm' 2>/dev/null; exit 1; }

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

COPY deidentify_engine_version.py /tmp/deidentify_engine_version.py
RUN /opt/venv/bin/python /tmp/deidentify_engine_version.py \
        --package vllm \
        --version-file /opt/infb/engine-ext/vllm/_version.py \
    && rm -f /tmp/deidentify_engine_version.py

RUN rm -f /opt/infb/requirements.lock \
    && echo "[lock] /opt/infb/requirements.lock removed: build-time only, and it carried the engine pin"
