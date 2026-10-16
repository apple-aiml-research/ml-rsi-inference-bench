#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# The recipe of the task image for support_deepseek_v41_flash_4xb200_vllm (DeepSWE idiom).
#
# Every input is public: the nvidia/cuda base by digest, hash-locked wheels from PyPI, the PyTorch
# cu130 index and flashinfer.ai (lock/, resolved by lock/resolve.sh), the vLLM tree at the base
# commit from GitHub, the kernel preparation commit this directory carries as a patch, and the
# checkpoint from the Hugging Face Hub at its pinned revision (model-manifest.json). The build
# needs network; the trial does not.
#
# The engine is a STAGE in this file and the final stage is `FROM ${ENGINE}`, so
#   docker build tasks/support_deepseek_v41_flash_4xb200_vllm/environment
# needs no registry and no prebuilt tag: the ARG default below is the whole public contract.
# infb-engine: task-authored
ARG ENGINE=engine

FROM docker.io/nvidia/cuda:13.0.3-devel-ubuntu24.04@sha256:b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266 AS engine

ARG BASE_SHA=e83d57d8ae14e240e2779c37db58b6dacd21bab2
ARG PREPARED_BASE_SHA=76578587913b5e1cec479d28fbb8e0912cb2834f
ARG RUST_TOOLCHAIN=1.95
# B200 only. Upstream's wheel builds many architectures; this image serves one machine, and
# restricting the list is the difference between a short and a multi-hour compile. The verifier
# rebuilds the candidate with the same value.
ARG TORCH_CUDA_ARCH_LIST=10.0a

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN for attempt in 1 2 3; do \
      if apt-get update && apt-get install -y --no-install-recommends \
           ca-certificates curl wget git git-lfs lsof netcat-openbsd kmod unzip locales procps \
           python3.12-full python3.12-dev python3.12-venv \
           build-essential cmake ninja-build perl patchelf ccache pkg-config \
           libnuma1 libnuma-dev numactl libssl-dev libcurl4-openssl-dev libdw-dev; then break; fi; \
      if [ "$attempt" = 3 ]; then echo "FATAL: apt failed on three attempts" >&2; exit 100; fi; \
      echo "apt attempt $attempt failed; retrying in 30 s" >&2; sleep 30; \
    done \
    && rm -rf /var/lib/apt/lists/*

COPY lock/requirements-tools.lock /opt/infb/lock/
RUN python3.12 -m venv /opt/infb/tools \
    && /opt/infb/tools/bin/pip install --no-cache-dir --retries 10 --timeout 120 --require-hashes \
         -r /opt/infb/lock/requirements-tools.lock

RUN python3.12 -m venv /opt/vllm
ENV PATH=/opt/vllm/bin:/root/.cargo/bin:${PATH}

RUN curl --proto '=https' --tlsv1.2 -sSf --retry 3 https://sh.rustup.rs \
      | sh -s -- -y --no-modify-path --profile minimal --default-toolchain ${RUST_TOOLCHAIN} \
    && rustc --version && cargo --version

COPY lock/requirements.lock /opt/infb/lock/
RUN pip install --no-cache-dir --retries 10 --timeout 120 --require-hashes --no-deps \
      --index-url https://pypi.org/simple \
      --extra-index-url https://download.pytorch.org/whl/cu130 \
      --extra-index-url https://flashinfer.ai/whl/cu130 \
      --extra-index-url https://flashinfer.ai/whl \
      -r /opt/infb/lock/requirements.lock

# 6. The vLLM tree at the base commit, in DeepSWE's git time-travel idiom: the default branch points
#    at the base with no future history, later branches and tags gone, the reflog expired and the
#    store garbage-collected so the reference solution cannot be recovered from history; hooks
#    disabled. Then the kernel-only preparation commit, reproduced here from the patch and the
#    message this directory carries, with pinned identity and dates so the commit is an object
#    rather than a local artefact -- and asserted, so the agent's starting point is the one the
#    oracle's patch applies to. The agent timer's files and the timer script live under /app and
#    are excluded from git so they never enter a submission.
WORKDIR /app
COPY prepared-base/ /opt/infb/prepared-base/
RUN git clone https://github.com/vllm-project/vllm.git . \
    && DEFAULT="$(git remote show origin | sed -n 's/.*HEAD branch: //p')" \
    && git checkout -B "$DEFAULT" "$BASE_SHA" \
    && git remote remove origin \
    && for b in $(git for-each-ref --format='%(refname:short)' refs/heads | grep -vx "$DEFAULT"); do git branch -D "$b" || true; done \
    && for t in $(git tag); do git merge-base --is-ancestor "$t" HEAD 2>/dev/null || git tag -d "$t"; done \
    && test "$(git rev-parse HEAD)" = "$BASE_SHA" \
    && git apply --binary --whitespace=nowarn /opt/infb/prepared-base/000-supplied-kernels.patch \
    && git add -A \
    && GIT_AUTHOR_NAME='Inference-Bench' GIT_AUTHOR_EMAIL='inference-bench@local' \
       GIT_COMMITTER_NAME='Inference-Bench' GIT_COMMITTER_EMAIL='inference-bench@local' \
       GIT_AUTHOR_DATE='1789040611 +0000' GIT_COMMITTER_DATE='1789040611 +0000' \
       git commit --quiet --no-verify -F /opt/infb/prepared-base/message.txt \
    && git reflog expire --expire=now --all \
    && git gc --prune=now \
    && git config core.hooksPath /dev/null \
    && printf '.timer/\nscripts/start_agent_timer.sh\n' >> .git/info/exclude \
    && head="$(git rev-parse HEAD)" \
    && if [ "$head" != "$PREPARED_BASE_SHA" ]; then \
         echo "FATAL: prepared base is $head, expected $PREPARED_BASE_SHA" >&2; exit 7; fi \
    && test -z "$(git status --porcelain)"

ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    VLLM_TARGET_DEVICE=cuda \
    CARGO_NET_OFFLINE=false \
    MAX_JOBS=64 \
    NVCC_THREADS=4 \
    CMAKE_BUILD_PARALLEL_LEVEL=64
RUN cd /app \
    && pip install --no-cache-dir --no-deps --no-build-isolation -e . \
    && python -c "import vllm; print('vllm', vllm.__version__, 'from', vllm.__file__)" \
    && python -c "import glob; so=sorted(glob.glob('/app/vllm/**/*.so', recursive=True)); print('compiled extensions:', len(so)); [print('   ', s) for s in so]" \
    && git status --porcelain | head -50 \
    && rm -rf /app/.deps/deepgemm-src/.git \
    && find /app/.deps -type d -name objects -prune -print | while read -r o; do \
         d="${o%/objects}"; [ -f "$d/HEAD" ] && [ -d "$d/refs" ] || continue; \
         h="$(git --git-dir="$d" rev-parse --verify -q HEAD)" || continue; \
         git --git-dir="$d" update-ref --no-deref HEAD "$h" \
         && git --git-dir="$d" for-each-ref --format='%(refname)' | while read -r r; do git --git-dir="$d" update-ref -d "$r" || exit 1; done \
         && echo "$h" > "$d/shallow" \
         && git --git-dir="$d" reflog expire --expire=now --all \
         && git --git-dir="$d" gc --quiet --prune=now \
         && echo "dependency store $d reduced to $h ($(git --git-dir="$d" rev-list --all | wc -l) commit)" || exit 1; \
       done \
    && find /root/.cargo/git/checkouts -mindepth 3 -maxdepth 3 -type d -name .git | while read -r d; do \
         h="$(git --git-dir="$d" rev-parse --verify -q HEAD)" || { echo "FATAL: cargo checkout $d has no HEAD" >&2; exit 1; }; \
         git --git-dir="$d" update-ref --no-deref HEAD "$h" \
         && git --git-dir="$d" for-each-ref --format='%(refname)' | while read -r r; do git --git-dir="$d" update-ref -d "$r" || exit 1; done \
         && echo "$h" > "$d/shallow" \
         && git --git-dir="$d" reflog expire --expire=now --all \
         && git --git-dir="$d" gc --quiet --prune=now \
         && echo "cargo checkout $d reduced to $h ($(git --git-dir="$d" rev-list --all | wc -l) commit)" || exit 1; \
       done \
    && for d in /root/.cargo/git/db/*; do \
         [ -d "$d/objects" ] && [ -d "$d/refs" ] || continue; \
         keep="$(git --git-dir="$d" for-each-ref --format='%(objectname)' refs/commit)"; \
         [ -n "$keep" ] || { echo "FATAL: cargo store $d holds no checked-out commit" >&2; exit 1; }; \
         git --git-dir="$d" for-each-ref --format='%(refname)' | while read -r r; do \
           case "$r" in refs/commit/*) ;; *) git --git-dir="$d" update-ref -d "$r" || exit 1 ;; esac; done \
         && printf '%s\n' $keep > "$d/shallow" \
         && git --git-dir="$d" reflog expire --expire=now --all \
         && git --git-dir="$d" gc --quiet --prune=now \
         && echo "cargo store $d reduced to $(git --git-dir="$d" rev-list --all | wc -l) commit(s)" || exit 1; \
       done \
    && CARGO_NET_OFFLINE=true cargo fetch --offline --locked --manifest-path /app/rust/Cargo.toml \
    && echo "cargo resolves the frontend's dependencies offline from the reduced stores"

RUN cd /app \
    && python -c "import pathlib,sys; \
want={'_C_stable_libtorch','_moe_C_stable_libtorch','cumem_allocator'}; \
got={p.name.split('.')[0] for p in pathlib.Path('/app/vllm').glob('*.so')}; \
fa=list(pathlib.Path('/app/vllm/vllm_flash_attn').glob('_vllm_fa*_C*.so')); \
missing=sorted(want-got); \
print('cuda extensions present:', sorted(got)); print('flash-attention:', [p.name for p in fa]); \
sys.exit(0 if not missing and fa else print('FATAL: missing', missing, 'flash-attn', fa) or 7)" \
    && test -z "$(git status --porcelain)"

# 7c. FlashInfer's precompiled JIT cache at the engine's own FlashInfer version (0.6.18.post1, cu130),
#     exactly as upstream's docker/Dockerfile at the base installs it into its release image. Without it
#     this model's first MoE forward JIT-compiles FlashInfer's TRT-LLM SM100 fused-MoE module in every
#     fresh container, and every verifier launch starts from a fresh container. Its own layer,
#     AFTER the compile, so adding it recompiled nothing; its own lock, from FlashInfer's own index.
COPY lock/requirements-flashinfer-jit.lock /opt/infb/lock/
RUN pip install --no-cache-dir --require-hashes --no-deps --retries 10 --timeout 120 \
      --index-url https://flashinfer.ai/whl/cu130 -r /opt/infb/lock/requirements-flashinfer-jit.lock \
    && python -c "import flashinfer_jit_cache, pathlib; p=pathlib.Path(flashinfer_jit_cache.__file__).parent; print('flashinfer-jit-cache', getattr(flashinfer_jit_cache, '__version__', '?'), len(list(p.rglob('*.so'))), 'modules')"

ENV CARGO_NET_OFFLINE=true \
    LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:/usr/local/cuda-13.0/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# 9. NO IMPORTABLE DeepSeek-V4.1 IMPLEMENTATION ON ANY INTERPRETER'S PATH. The model-author
#    implementation ships inside the checkpoint (inference/, under /model and the Hub cache), which is
#    on no interpreter's path; no Transformers release carries the model; the engine tree at the
#    prepared base carries no file naming it. The check asserts all three, so a later lock bump
#    cannot reopen the hole silently.
COPY checks/ /opt/infb/checks/
RUN python /opt/infb/checks/no_reference_impl.py

ENV INFB_VENV=/opt/vllm
WORKDIR /app
CMD ["/bin/bash"]

FROM ${ENGINE}

ENV HF_HOME=/hf HF_HUB_CACHE=/hf/hub
COPY model-manifest.json fetch_model.py /opt/infb/
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 0 --of 48
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 1 --of 48
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 2 --of 48
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 3 --of 48
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 4 --of 48
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 5 --of 48
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 6 --of 48
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 7 --of 48
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 8 --of 48
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 9 --of 48
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 10 --of 48
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 11 --of 48
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 12 --of 48
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 13 --of 48
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 14 --of 48
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 15 --of 48
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 16 --of 48
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 17 --of 48
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 18 --of 48
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 19 --of 48
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 20 --of 48
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 21 --of 48
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 22 --of 48
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 23 --of 48
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 24 --of 48
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 25 --of 48
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 26 --of 48
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 27 --of 48
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 28 --of 48
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 29 --of 48
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 30 --of 48
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 31 --of 48
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 32 --of 48
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 33 --of 48
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 34 --of 48
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 35 --of 48
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 36 --of 48
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 37 --of 48
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 38 --of 48
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 39 --of 48
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 40 --of 48
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 41 --of 48
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 42 --of 48
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 43 --of 48
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 44 --of 48
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 45 --of 48
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 46 --of 48
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 47 --of 48 --finalize

COPY author/requirements.lock /opt/infb/author/requirements.lock
RUN python3.12 -m venv /opt/model-author/venv \
    && /opt/model-author/venv/bin/pip install --no-cache-dir --retries 10 --timeout 120 --require-hashes --no-deps \
         --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu130 \
         -r /opt/infb/author/requirements.lock \
    && /opt/model-author/venv/bin/python -c "import torch, tilelang, tvm_ffi, safetensors, transformers, tokenizers, numpy; \
print('model-author runtime: torch', torch.__version__, 'tilelang', tilelang.__version__, 'numpy', numpy.__version__)"

RUN S=/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277 \
    && mkdir -p /model-author/deepseek-v41-mp4 && cd "$S/inference" \
    && /opt/model-author/venv/bin/python convert.py --hf-ckpt-path "$S" --save-path /tmp/infb-mp4 \
         --model-parallel 4 --expert-dtype fp4 --tokenizer-path "$S" \
    && mv /tmp/infb-mp4/model0-mp4.safetensors /tmp/infb-mp4/tokenizer.json /tmp/infb-mp4/tokenizer_config.json \
         /model-author/deepseek-v41-mp4/ \
    && rm -rf /tmp/infb-mp4 \
    && chmod a-w /model-author/deepseek-v41-mp4/model0-mp4.safetensors /model-author/deepseek-v41-mp4/tokenizer.json \
         /model-author/deepseek-v41-mp4/tokenizer_config.json
RUN S=/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277 \
    && cd "$S/inference" \
    && /opt/model-author/venv/bin/python convert.py --hf-ckpt-path "$S" --save-path /tmp/infb-mp4 \
         --model-parallel 4 --expert-dtype fp4 --tokenizer-path "$S" \
    && mv /tmp/infb-mp4/model1-mp4.safetensors /model-author/deepseek-v41-mp4/ && rm -rf /tmp/infb-mp4 \
    && chmod a-w /model-author/deepseek-v41-mp4/model1-mp4.safetensors
RUN S=/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277 \
    && cd "$S/inference" \
    && /opt/model-author/venv/bin/python convert.py --hf-ckpt-path "$S" --save-path /tmp/infb-mp4 \
         --model-parallel 4 --expert-dtype fp4 --tokenizer-path "$S" \
    && mv /tmp/infb-mp4/model2-mp4.safetensors /model-author/deepseek-v41-mp4/ && rm -rf /tmp/infb-mp4 \
    && chmod a-w /model-author/deepseek-v41-mp4/model2-mp4.safetensors
RUN S=/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277 \
    && cd "$S/inference" \
    && /opt/model-author/venv/bin/python convert.py --hf-ckpt-path "$S" --save-path /tmp/infb-mp4 \
         --model-parallel 4 --expert-dtype fp4 --tokenizer-path "$S" \
    && mv /tmp/infb-mp4/model3-mp4.safetensors /model-author/deepseek-v41-mp4/ && rm -rf /tmp/infb-mp4 \
    && chmod a-w /model-author/deepseek-v41-mp4/model3-mp4.safetensors /model-author/deepseek-v41-mp4 /model-author

COPY author/logprobs.py author/write_manifest.py author/expected-shards.json /opt/infb/author/
RUN S=/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277 \
    && cp -rL "$S/inference" "$S/encoding" /opt/model-author/ \
    && python3 -c "import hashlib, json, pathlib, sys; \
rec=json.load(open('/hf/hub/.infb/manifest.json'))['files']; root=pathlib.Path('/opt/model-author'); \
bad=[p for p in rec if p.startswith(('inference/', 'encoding/')) and hashlib.sha256((root/p).read_bytes()).hexdigest()!=rec[p]['sha256']]; \
n=sum(1 for p in rec if p.startswith(('inference/', 'encoding/'))); \
sys.exit(print('FATAL: the author tree differs from the checkpoint at', bad) or 7) if bad else print('author tree:', n, 'files as fetched')" \
    && install -m 755 /opt/infb/author/logprobs.py /opt/model-author/logprobs.py \
    && python3 /opt/infb/author/write_manifest.py --shards /model-author/deepseek-v41-mp4 --snapshot "$S" \
         --fetch-record /hf/hub/.infb/manifest.json --expected /opt/infb/author/expected-shards.json > /dev/null \
    && chmod a-w /model-author/deepseek-v41-mp4/manifest.json \
    && chmod -R a-w /opt/model-author/inference /opt/model-author/encoding

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1 \
    VLLM_NO_USAGE_STATS=1

COPY scripts/start_agent_timer.sh materials/message.txt eval/ /opt/infb/stage/
COPY materials/v51/ /opt/infb/stage-v51/
RUN --mount=type=bind,source=buildcheck,target=/infb-buildcheck \
    mkdir -p /app/scripts /app/eval /opt/infb/materials /model/deepseek-ai /logs/artifacts /logs/verifier \
    && mv /opt/infb/stage/start_agent_timer.sh /app/scripts/start_agent_timer.sh \
    && chmod 755 /app/scripts/start_agent_timer.sh \
    && mv /opt/infb/stage/message.txt /opt/infb/materials/message.txt \
    && mv /opt/infb/stage/deepseek_v41_flash /app/eval/deepseek_v41_flash \
    && rmdir /opt/infb/stage \
    && mkdir -p /opt/infb/v53-materials/tests \
    && mv /app/eval/deepseek_v41_flash/sparse_indexer /opt/infb/v53-materials/sparse_indexer \
    && for f in README.md recipe.py build.sh tests/native.py; do \
         mv "/app/eval/deepseek_v41_flash/$f" "/opt/infb/v53-materials/$f" \
         && mv "/opt/infb/stage-v51/$f" "/app/eval/deepseek_v41_flash/$f" || exit 7; done \
    && rm -r /opt/infb/stage-v51 \
    && ln -s /hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277 \
         /model/deepseek-ai/DeepSeek-V4.1-Flash \
    && test -f /model/deepseek-ai/DeepSeek-V4.1-Flash/config.json \
    && git config --system user.name agent && git config --system user.email agent@localhost \
    && cd /app \
    && mkdir -p eval/deepseek_v41_flash/datasets/gsm8k \
    && curl -fsSL --retry 10 -o eval/deepseek_v41_flash/datasets/gsm8k/train.jsonl \
         https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl \
    && curl -fsSL --retry 10 -o eval/deepseek_v41_flash/datasets/gsm8k/test.jsonl \
         https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl \
    && echo "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465  eval/deepseek_v41_flash/datasets/gsm8k/train.jsonl" | sha256sum -c - \
    && echo "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14  eval/deepseek_v41_flash/datasets/gsm8k/test.jsonl" | sha256sum -c - \
    && chmod -R u=rwX,go=rX eval \
    && chmod 755 eval/deepseek_v41_flash/run.py eval/deepseek_v41_flash/recipe.py eval/deepseek_v41_flash/build.sh \
    && git add -f eval/deepseek_v41_flash \
    && GIT_AUTHOR_NAME='Inference-Bench' GIT_AUTHOR_EMAIL='inference-bench@local' \
       GIT_COMMITTER_NAME='Inference-Bench' GIT_COMMITTER_EMAIL='inference-bench@local' \
       GIT_AUTHOR_DATE='1789040671 +0000' GIT_COMMITTER_DATE='1789040671 +0000' \
       git commit --quiet --no-verify -F /opt/infb/materials/message.txt \
    && git reflog expire --expire=now --all \
    && head="$(git rev-parse HEAD)" \
    && echo "materials head: $head" \
    && if [ "$head" != "992b504750b47a6a5bcfd9617b5b49eff8065fa6" ]; then \
         echo "FATAL: materials head is $head, not the expected 992b504750b47a6a5bcfd9617b5b49eff8065fa6 -- a materials file differs" >&2; exit 7; fi \
    && python3 /infb-buildcheck/no_answer_address.py \
    && test -d /usr/local/cuda-13.0/compat \
    && test -z "$(git status --porcelain)"

RUN --mount=type=bind,source=sparse-indexer,target=/infb-sparse-indexer \
    --mount=type=bind,source=buildcheck,target=/infb-buildcheck \
    cd /app \
    && cp -R /opt/infb/v53-materials/. /app/eval/deepseek_v41_flash/ && rm -r /opt/infb/v53-materials \
    && chmod -R u=rwX,go=rX eval \
    && chmod 755 eval/deepseek_v41_flash/run.py eval/deepseek_v41_flash/recipe.py eval/deepseek_v41_flash/build.sh \
    && python3 -B /infb-sparse-indexer/prepare.py --app /app --work /tmp/infb-sparse-prep \
    && git add -A && git add -f eval/deepseek_v41_flash \
    && GIT_AUTHOR_NAME='Inference-Bench' GIT_AUTHOR_EMAIL='inference-bench@local' \
       GIT_COMMITTER_NAME='Inference-Bench' GIT_COMMITTER_EMAIL='inference-bench@local' \
       GIT_AUTHOR_DATE='1789040731 +0000' GIT_COMMITTER_DATE='1789040731 +0000' \
       git commit --quiet --no-verify -F /infb-sparse-indexer/message.txt \
    && git reflog expire --expire=now --all \
    && head="$(git rev-parse HEAD)" \
    && echo "sparse-indexer preparation head: $head" \
    && if [ "$head" != "6ead86daeaed3fb406414cc61db23cb214bd7668" ]; then \
         echo "FATAL: the sparse-indexer preparation head is $head, not 6ead86daeaed3fb406414cc61db23cb214bd7668 -- a preparation input differs" >&2; exit 7; fi \
    && pip install --no-cache-dir --no-deps --no-build-isolation -e . \
    && python tools/build_deepgemm_v41_sparse.py third_party/deepgemm_v41_sparse vllm/third_party/deep_gemm_v41_sparse /opt/vllm/bin/python \
    && test -f vllm/_deepselect_C.abi3.so && test -f vllm/third_party/deep_gemm_v41_sparse/build-manifest.json \
    && ls vllm/third_party/deep_gemm_v41_sparse/_C.*.so && test -d vllm/third_party/deep_gemm_v41_sparse/include/deep_gemm \
    && find /app/.deps -type d -name objects -prune -print | while read -r o; do \
         d="${o%/objects}"; [ -f "$d/HEAD" ] && [ -d "$d/refs" ] || continue; \
         h="$(git --git-dir="$d" rev-parse --verify -q HEAD)" || continue; \
         keep=" $(git --git-dir="$d" for-each-ref --points-at="$h" --format='%(refname)' refs/tags | tr '\n' ' ')"; \
         git --git-dir="$d" update-ref --no-deref HEAD "$h" \
         && git --git-dir="$d" for-each-ref --format='%(refname)' | while read -r r; do \
              case "$keep" in *" $r "*) ;; *) git --git-dir="$d" update-ref -d "$r" || exit 1 ;; esac; done \
         && rm -f "$d/FETCH_HEAD" "$d/ORIG_HEAD" \
         && echo "$h" > "$d/shallow" \
         && git --git-dir="$d" reflog expire --expire=now --all \
         && git --git-dir="$d" gc --quiet --prune=now \
         && echo "dependency store $d reduced to $h ($(git --git-dir="$d" rev-list --all | wc -l) commit; tags at it kept:${keep% })" || exit 1; \
       done \
    && find /root/.cargo/git \( -name FETCH_HEAD -o -name ORIG_HEAD \) -type f -delete \
    && test -z "$(find /app/.deps /root/.cargo/git \( -name FETCH_HEAD -o -name ORIG_HEAD \) -print)" \
    && for s in cutlass:v4.7.1 triton_kernels:v3.5.1; do \
         g="/app/.deps/${s%%:*}-src/.git"; t="${s#*:}"; \
         test "$(git --git-dir="$g" rev-parse --verify -q "$t^{commit}")" = "$(git --git-dir="$g" rev-parse --verify -q HEAD)" \
           || { echo "FATAL: $g does not resolve FetchContent's GIT_TAG $t to its checked-out commit: an offline rebuild would fetch" >&2; exit 1; }; \
         echo "offline rebuild: $g resolves FetchContent's GIT_TAG $t to its checked-out commit"; \
       done \
    && python3 -B /infb-sparse-indexer/audit_image.py \
         --git-root /app --git-root /opt --git-root /usr/local --git-root /root \
         --root /app --root /opt --root /usr/local --root /root \
    && python3 /infb-buildcheck/no_answer_address.py \
    && test -z "$(git status --porcelain)"

# 15. THE RUNTIME LIBRARY PATH WITHOUT THE FORWARD-COMPATIBILITY DIRECTORY. Section 8's path puts
#     the image's compat libcuda ahead of the host's, and on a runtime that keeps the image's
#     environment over a newer host driver CUDA then fails to initialise (cuInit 803). This last
#     line keeps every other entry, in order. Metadata only: no layer of the image changes.
ENV LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
