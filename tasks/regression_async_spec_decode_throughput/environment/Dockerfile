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

# Lockfile for this task's vLLM runtime env.
#
# B200 build of the 342c4f14 pin: CUDA 12.8.1 base, arch 10.0+PTX. Measured 2026-09-14 on a B200
# host: the PyPI vllm==0.12.0 wheel loads and runs on sm100, so the wheel already carries B200
# kernels and only the SOURCE-built extension's arch changes.
#
# PROVENANCE. This resolution was produced by `uv pip install vllm==0.12.0 numba==0.61.2` into a
# venv against the same index and the cu128 torch channel, then `uv pip freeze`. That venv is not
# hypothetical: it is the one the task's own timing calibration ran on, so this is a resolution
# known to RUN, which is more than a fresh resolve guarantees.
#
# WHY 0.12.0 IS THE RIGHT WHEEL FOR THIS PIN. The build pairs a released wheel (compiled artefacts)
# with a raw source tree at the commit the agent edits. The pin
# 342c4f147266e7d01a8a6a64754df7e0f7f8991a is 2025-12-01; v0.11.2 is 2025-11-20 and does NOT contain
# it, v0.12.0 is 2025-12-03 and does (`compare` reports v0.12.0 ahead by 34, behind by 0). So 0.12.0
# is the FIRST release containing the commit and its .so are contemporaneous with the tree.
#
# CUDA GENERATION, measured rather than inferred: every one of the 12 compiled extensions in this
# venv links libcudart.so.12 (`vllm/_C.abi3.so`, `vllm/_moe_C.abi3.so`, ...) and torch reports
# 2.9.0+cu128. So this is CUDA 12.8 with the cu128 channel. pip and this lock cannot see that on
# their own; only the .so can.
#
# distributions resolved : 163
# pinned here            : 163
#
# NOT pinned, and why:
#   (nothing withheld -- the freeze is taken whole)

agent-detector===2.0.0
aiohappyeyeballs===2.7.1
aiohttp===3.14.3
aiosignal===1.4.0
annotated-doc===0.0.5
annotated-types===0.8.0
anthropic===0.71.0
anyio===4.15.1
apache-tvm-ffi===0.1.13.post3
astor===0.8.1
attrs===26.1.0
blake3===1.0.9
cachetools===7.1.8
cbor2===6.1.4
certifi===2026.7.22
charset-normalizer===3.5.1
click===8.5.0
cloudpickle===3.1.2
compressed-tensors===0.12.2
cuda-bindings===13.4.1
cuda-core===1.2.0
cuda-pathfinder===1.8.1
cuda-python===13.4.1
cupy-cuda12x===14.2.0
depyf===0.20.0
detect-installer===0.2.1
dill===0.4.1
diskcache===5.6.3
distro===1.9.0
dnspython===2.8.0
docstring-parser===0.18.0
einops===0.8.2
email-validator===2.3.0
fastapi-cli===0.0.32
fastapi-cloud-cli===0.26.0
fastapi===0.141.1
fastar===0.12.0
filelock===3.32.6
flashinfer-python===0.5.3
frozenlist===1.8.0
fsspec===2026.7.0
gguf===0.19.0
h11===0.16.0
hf-xet===1.6.0
httpcore2===2.12.0
httpcore===1.0.9
httptools===0.8.0
httpx2===2.12.0
httpx===0.28.1
huggingface-hub===0.36.2
idna===3.19
interegular===0.3.3
jinja2===3.1.6
jiter===0.17.0
jmespath===1.1.0
jsonschema-specifications===2025.9.1
jsonschema===4.26.0
lark===1.2.2
llguidance===1.3.0
llvmlite===0.44.0
lm-format-enforcer===0.11.3
loguru===0.7.3
markdown-it-py===4.2.0
markupsafe===3.0.3
mdurl===0.1.2
mistral-common===1.11.7
model-hosting-container-standards===0.1.16
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
nvidia-cuda-nvdisasm===13.4.49
nvidia-cuda-nvrtc-cu12===12.8.93
nvidia-cuda-runtime-cu12===12.8.90
nvidia-cudnn-cu12===9.10.2.21
nvidia-cudnn-frontend===1.28.0
nvidia-cufft-cu12===11.3.3.83
nvidia-cufile-cu12===1.13.1.3
nvidia-curand-cu12===10.3.9.90
nvidia-cusolver-cu12===11.7.3.90
nvidia-cusparse-cu12===12.5.8.93
nvidia-cusparselt-cu12===0.7.1
nvidia-cutlass-dsl-libs-base===4.7.1
nvidia-cutlass-dsl-libs-core===4.7.1
nvidia-cutlass-dsl-libs-cu12===4.7.1
nvidia-cutlass-dsl===4.7.1
nvidia-ml-py===13.610.43
nvidia-nccl-cu12===2.27.5
nvidia-nvjitlink-cu12===12.8.93
nvidia-nvshmem-cu12===3.3.20
nvidia-nvtx-cu12===12.8.90
openai-harmony===0.0.8
openai===3.13.0
opencv-python-headless===5.0.0.93
outlines-core===0.2.11
packaging===26.3
partial-json-parser===0.2.1.1.post7
pillow===12.3.0
prometheus-client===0.26.0
prometheus-fastapi-instrumentator===8.1.0
propcache===0.5.2
protobuf===7.36.1
psutil===7.2.2
py-cpuinfo===9.0.0
pybase64===1.5.0
pycountry===26.2.16
pydantic-core===2.46.5
pydantic-extra-types===2.11.1
pydantic-settings===2.15.0
pydantic===2.13.5
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
rich-toolkit===0.20.5
rich===15.0.0
rignore===0.8.1
rpds-py===2026.6.3
safetensors===0.8.0
scipy===1.18.1
sentencepiece===0.2.2
sentry-sdk===2.69.1
setproctitle===1.3.7
setuptools===80.10.2
shellingham===1.5.4
six===1.17.0
sniffio===1.3.1
starlette===1.6.0
supervisor===4.3.0
sympy===1.14.0
tabulate===0.10.0
tiktoken===0.14.0
tokenizers===0.22.2
torch==2.9.0+cu128
torchaudio==2.9.0+cu128
torchvision==0.24.0+cu128
tqdm===4.70.1
transformers===4.57.6
triton===3.5.0
truststore===0.10.4
typer===0.27.2
typing-extensions===4.16.0
typing-inspection===0.4.4
urllib3===2.7.0
uvicorn===0.52.4
uvloop===0.22.1
vllm===0.12.0
watchfiles===1.2.0
websockets===17.1
xgrammar===0.1.27
yarl===1.24.5
LOCK_EOF

RUN mkdir -p /opt/infb/wheels

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
    git fetch -q --depth 1 origin 342c4f147266e7d01a8a6a64754df7e0f7f8991a; \
    git checkout -q FETCH_HEAD; \
    git submodule update -q --init --recursive --depth 1; \
    head="$(git rev-parse HEAD)"; \
    if [ "$head" != "342c4f147266e7d01a8a6a64754df7e0f7f8991a" ]; then \
        echo "FATAL: engine tree is at $head, not the reviewed 342c4f147266e7d01a8a6a64754df7e0f7f8991a -- 342c4f147266e7d01a8a6a64754df7e0f7f8991a has moved" >&2; exit 7; \
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
        echo "[from-scratch] no engine source staged; the task will need INFB_VLLM_SOURCE_DIR or a mounted source tree"; \
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
print('vllm', metadata.version('vllm'))
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
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        patch \
        procps \
        rsync \
    && rm -rf /var/lib/apt/lists/*

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
    n=0; \
    for rec in $(find /opt /usr/lib /usr/local /root /home -maxdepth 10 -path '*/vllm-*.dist-info/RECORD' 2>/dev/null); do \
      echo "[evict] removing wheel manifest $rec ($(wc -l < "$rec") entries)"; rm -f "$rec"; n=$((n + 1)); \
    done; \
    echo "[evict] removed $n vllm RECORD manifest(s)"; \
    [ "$n" -gt 0 ] || { echo "[evict] FATAL: no vllm RECORD found -- the digest answer key may have moved"; exit 1; }

RUN /opt/venv/bin/python - <<'PY'
from importlib import metadata
v = metadata.version("vllm")
print(f"[evict] importlib.metadata.version('vllm') = {v} (still resolvable without RECORD)")
d = metadata.distribution("vllm")
assert (d.metadata["Name"] or "").lower().replace("_", "-") == "vllm", d.metadata["Name"]
# This is the call the platform plugin makes; without RECORD it must still work.
print(f"[evict] entry points: {len(list(metadata.entry_points()))} group(s) enumerable")
PY

WORKDIR /app

COPY public /app/public

COPY scripts /app/scripts

RUN /opt/venv/bin/python - <<'PY'
import sys
import time
from huggingface_hub import snapshot_download

REPO = "nvidia/gpt-oss-120b-Eagle3"
DEST = "/opt/infb/aux-models/nvidia/gpt-oss-120b-Eagle3"
# Mirror layout (<root>/<org>/<name>/<file>), not the hub cache layout: the bench hands this
# DIRECTORY straight to the engine, with no snapshots/ level to guess at. `local_dir` gives that.
last_exc = None
for attempt in range(3):
    try:
        snapshot_download(
            repo_id=REPO,
            allow_patterns=["config.json", "hf_quant_config.json", "model.safetensors"],
            local_dir=DEST,
            # `token=False`, explicitly: the repo is public, but huggingface_hub picks up an
            # ambient token from the environment or ~/.cache/huggingface/token, and a STALE one
            # turns a public repo into 401 RepositoryNotFoundError. A build that can be broken by
            # a credential it does not need should not read one.
            token=False,
        )
        last_exc = None
        break
    except Exception as exc:
        last_exc = exc
        if attempt < 2:
            print(f"[drafter] attempt {attempt + 1}/3 failed "
                  f"({type(exc).__name__}: {exc}); retrying", flush=True)
            time.sleep(10 * (attempt + 1))
if last_exc is not None:
    sys.exit(
        f"FATAL: could not fetch the speculative drafter {REPO!r} after 3 attempts: "
        f"{type(last_exc).__name__}: {last_exc}\n"
        f"Without it the engine cannot start with the speculative config this task scores, both "
        f"arms report no throughput, and the trial spends its whole budget producing nothing "
        f"attributable. The build needs network access to huggingface.co; the TRIAL does not."
    )
print(f"[drafter] staged {REPO} -> {DEST}")
PY

ENV INFB_SPEC_DRAFTER_PATH=/opt/infb/aux-models/nvidia/gpt-oss-120b-Eagle3

RUN test -f /opt/infb/aux-models/nvidia/gpt-oss-120b-Eagle3/config.json \
    && test -s /opt/infb/aux-models/nvidia/gpt-oss-120b-Eagle3/model.safetensors \
    || (echo "FATAL: the drafter directory is missing config.json or model.safetensors" && exit 1)

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
