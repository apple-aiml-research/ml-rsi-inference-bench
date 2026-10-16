#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# infb-engine: task-authored
ARG ENGINE=engine

FROM docker.io/nvidia/cuda:13.0.3-devel-ubuntu24.04@sha256:b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266 AS engine-build

# The Rust toolchain vLLM's frontend extension is built with (rust-toolchain.toml at the base
# commit pins 1.95).
ARG RUST_TOOLCHAIN=1.95
# B200 only. Upstream's wheel builds eight architectures; this image serves one machine, and
# restricting the list is the difference between a 20-minute and a multi-hour compile. The verifier
# rebuilds the candidate with the same value.
ARG TORCH_CUDA_ARCH_LIST=10.0a

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# System packages: the runtime subset of upstream's docker/Dockerfile at the base commit (Python
# 3.12 -- Ubuntu 24.04's own -- and the toolchain the CUDA and Triton kernels compile with). The
# Rust frontend's build.rs uses protox, a pure-Rust protobuf compiler, so no protoc is needed here.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl wget git git-lfs lsof netcat-openbsd kmod unzip locales procps \
      python3.12-full python3.12-dev python3.12-venv \
      build-essential cmake ninja-build perl patchelf ccache pkg-config \
      libnuma1 libnuma-dev numactl libssl-dev libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Three virtual environments, hash-locked, and deliberately separate:
#    /opt/infb/tools  only what this recipe itself runs (huggingface_hub for the checkpoint);
#    /opt/vllm        the engine's, filled from the lock below and nothing else;
#    /opt/evalscope   EvalScope 1.10.0, the evaluator the upstream acceptance test calls. It lives
#                     in its own interpreter so the engine's resolved dependency set is never moved
#                     by EvalScope's, while the public copy of that test under /public stays
#                     runnable inside the episode.
# Only the .lock files enter the image, never lock/resolve.sh or lock/RESOLVED.md: those are
# provenance for whoever re-resolves. Only the tools lock is copied here, because the checkpoint
# fetch below needs it.
COPY lock/requirements-tools.lock /opt/infb/lock/
RUN python3.12 -m venv /opt/infb/tools \
    && /opt/infb/tools/bin/pip install --no-cache-dir --require-hashes -r /opt/infb/lock/requirements-tools.lock \
    && python3.12 -m venv /opt/vllm \
    && python3.12 -m venv /opt/evalscope
ENV PATH=/opt/vllm/bin:/root/.cargo/bin:${PATH}

RUN curl --proto '=https' --tlsv1.2 -sSf --retry 3 https://sh.rustup.rs \
      | sh -s -- -y --no-modify-path --profile minimal --default-toolchain ${RUST_TOOLCHAIN} \
    && rustc --version && cargo --version

# The checkpoint from the Hub at the pinned revision, into the image's Hugging Face cache. The
# upstream acceptance test launches `vllm serve Qwen/Qwen3.8-Flash-Next-FP8` -- the repository id,
# not a path -- so the cache layout plus HF_HUB_OFFLINE=1 is what lets that test run VERBATIM
# offline; /model below is a symlink onto the same snapshot for the instruction's sake. One layer
# per group of shards (64 groups, largest about 3.4 GB): registries cap a single blob upload, and
# Docker's overlay2 allows 128 layers. Every file is verified against model-manifest.json (size, and
# the Hub's sha256 for the 133 LFS files).
ENV HF_HOME=/hf HF_HUB_CACHE=/hf/hub
COPY model-manifest.json fetch_model.py /opt/infb/
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 0 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 1 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 2 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 3 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 4 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 5 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 6 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 7 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 8 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 9 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 10 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 11 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 12 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 13 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 14 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 15 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 16 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 17 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 18 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 19 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 20 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 21 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 22 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 23 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 24 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 25 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 26 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 27 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 28 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 29 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 30 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 31 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 32 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 33 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 34 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 35 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 36 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 37 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 38 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 39 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 40 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 41 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 42 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 43 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 44 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 45 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 46 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 47 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 48 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 49 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 50 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 51 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 52 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 53 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 54 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 55 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 56 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 57 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 58 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 59 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 60 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 61 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 62 --of 64
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 63 --of 64 --finalize

# vLLM's dependencies at the base commit, hash-locked from the public indexes (lock/RESOLVED.md):
# requirements/cuda.txt (which pulls in common.txt) and requirements/build/cuda.txt, so the editable
# install below needs no build isolation and no network, plus pytest and tblib for the vendored
# upstream test's conftest chain, and transformers pinned at 5.12.1. The engine's two locks are
# copied HERE, after the 64 checkpoint layers, so a lock change no longer re-downloads 185 GB; the
# big FlashInfer artifacts get their own layer so a retry of that fetch does not redo the dependency
# set.
COPY lock/requirements.lock lock/requirements-evalscope.lock /opt/infb/lock/
RUN pip install --no-cache-dir --retries 10 --timeout 120 --require-hashes --no-deps \
      --index-url https://pypi.org/simple \
      --extra-index-url https://download.pytorch.org/whl/cu130 \
      --extra-index-url https://flashinfer.ai/whl/cu130 \
      --extra-index-url https://flashinfer.ai/whl \
      -r /opt/infb/lock/requirements.lock

RUN /opt/evalscope/bin/pip install --no-cache-dir --retries 10 --timeout 120 --require-hashes --no-deps \
      -r /opt/infb/lock/requirements-evalscope.lock \
    && rm -rf /opt/evalscope/lib/python3.12/site-packages/transformers \
              /opt/evalscope/lib/python3.12/site-packages/transformers-*.dist-info \
    && /opt/evalscope/bin/python -c "import evalscope; print('evalscope', evalscope.__version__)"

# The vLLM tree at the base commit, in DeepSWE's git time-travel idiom: the default branch points
# at the base with no future history, later branches and tags gone, the reflog expired and the store
# garbage-collected so the reference solution cannot be recovered from history; hooks disabled. Then
# the kernel-only preparation commit, reproduced here from the patch and the message this directory
# carries, with pinned identity and dates so the commit is an object rather than a local artefact --
# and asserted, so the agent's starting point is the one the oracle's patch applies to. The agent
# timer's files and the timer script live under /app and are excluded from git so they never enter a
# submission.
ARG BASE_SHA=687db597440bf2b0cc65404e5f8cc238632127e0
ARG KERNEL_PREPARATION_SHA=6a88e6efdce8c965a8424cd9066179c88aa70554
WORKDIR /app
COPY prepared-base/ /opt/infb/prepared-base/
RUN git clone https://github.com/vllm-project/vllm.git . \
    && DEFAULT="$(git remote show origin | sed -n 's/.*HEAD branch: //p')" \
    && git checkout -B "$DEFAULT" "$BASE_SHA" \
    && git remote remove origin \
    && for b in $(git for-each-ref --format='%(refname:short)' refs/heads | grep -vx "$DEFAULT"); do git branch -D "$b" || true; done \
    && for t in $(git tag); do git merge-base --is-ancestor "$t" HEAD 2>/dev/null || git tag -d "$t"; done \
    && test "$(git rev-parse HEAD)" = "$BASE_SHA" \
    && git apply --index --binary --whitespace=nowarn /opt/infb/prepared-base/000-supplied-kernels.patch \
    && git diff --cached --name-only -z > /tmp/preparation-paths \
    && GIT_AUTHOR_NAME='Inference-Bench' GIT_AUTHOR_EMAIL='inference-bench@local' \
       GIT_COMMITTER_NAME='Inference-Bench' GIT_COMMITTER_EMAIL='inference-bench@local' \
       GIT_AUTHOR_DATE='1788153123 +0000' GIT_COMMITTER_DATE='1788153123 +0000' \
       xargs -0 git commit -o --quiet --no-verify -F /opt/infb/prepared-base/kernels-message.txt -- < /tmp/preparation-paths \
    && rm /tmp/preparation-paths \
    && git reflog expire --expire=now --all \
    && git gc --prune=now \
    && git config core.hooksPath /dev/null \
    && printf '.timer/\nscripts/start_agent_timer.sh\n' >> .git/info/exclude \
    && head="$(git rev-parse HEAD)" \
    && if [ "$head" != "$KERNEL_PREPARATION_SHA" ]; then \
         echo "FATAL: prepared base is $head, expected $KERNEL_PREPARATION_SHA" >&2; exit 7; fi \
    && test -z "$(git status --porcelain)"

# Editable install of the tree without build isolation (the build backend is in the lock) and
# without touching the index, so exactly this step is what the verifier re-runs on the candidate's
# tree. The CUDA extension and the Rust frontend are compiled here; .gitignore covers every artefact
# they leave, so the tree must still be clean afterwards.
#
# The same RUN then reduces every Git store the install fetched -- the FetchContent clones under
# /app/.deps with their submodule stores, and Cargo's git database and checkouts -- to its
# checked-out commit: no remote-tracking ref, no ref off that commit, no history beyond it, no
# reflog, no alternates. It has to be this RUN: a later step deleting history would leave it in this
# layer, which the image still carries. A tag AT the checked-out commit stays, because FetchContent's
# update step resolves a GIT_TAG such as cutlass's v4.4.2 against it, which is what keeps the
# verifier's offline rebuild of the candidate from needing a fetch.
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    VLLM_TARGET_DEVICE=cuda \
    CARGO_NET_OFFLINE=false \
    MAX_JOBS=64 \
    NVCC_THREADS=4 \
    CMAKE_BUILD_PARALLEL_LEVEL=64
RUN --mount=type=bind,source=buildcheck,target=/infb-buildcheck \
    cd /app \
    && pip install --no-cache-dir --no-deps --no-build-isolation -e . \
    && python -c "import vllm; print('vllm', vllm.__version__, 'from', vllm.__file__)" \
    && python -c "import glob; so=sorted(glob.glob('/app/vllm/**/*.so', recursive=True)); print('compiled extensions:', len(so)); [print('   ', s) for s in so]" \
    && git status --porcelain | head -50 \
    && echo "c80168e45cb9b62a203d7c2a784153e5391e3221ca4b594ddd7e095634d3cb41  /infb-buildcheck/reduce_git_stores.py" | sha256sum -c - \
    && python3 /infb-buildcheck/reduce_git_stores.py /app/.deps /root/.cargo/git

# The assertions on what the install produced, in their own layer so tightening them costs seconds
# rather than the whole compile.
#
# Two things this layer deliberately does NOT do. It does not look for `vllm._C`: at this commit
# setup.py adds that name only on the ROCm and CPU paths, and the CUDA custom ops are
# `_C_stable_libtorch` and `_moe_C_stable_libtorch` (`if _build_custom_ops()`) -- the kind of thing a
# recipe written from an older vLLM layout gets wrong once. And it does not IMPORT any of them: a
# container build has no GPU and therefore no `libcuda.so.1`, so importing a driver-linked extension
# fails here for a reason that says nothing about the image. The existence and the names are what the
# build can honestly assert; the imports are asserted by the verifier's own L0 node at grading time.
RUN cd /app \
    && python -c "import pathlib,sys; \
want={'_C_stable_libtorch','_moe_C_stable_libtorch','cumem_allocator','spinloop','fs_io_C'}; \
got={p.name.split('.')[0] for p in pathlib.Path('/app/vllm').glob('*.so')}; \
fa=list(pathlib.Path('/app/vllm/vllm_flash_attn').glob('_vllm_fa*_C*.so')); \
missing=sorted(want-got); \
print('cuda extensions present:', sorted(got)); print('flash-attention:', [p.name for p in fa]); \
sys.exit(0 if not missing and fa else print('FATAL: missing', missing, 'flash-attn', fa) or 7)" \
    && test -z "$(git status --porcelain)"

FROM engine-build AS engine-runtime

# FlashInfer's pre-compiled JIT modules (flashinfer-jit-cache, the same release as
# flashinfer-python), in their own layer because the wheel is a 1 GB fetch and placed after the
# engine build so it invalidates none of those layers. Without it the TRT-LLM fused-MoE module
# (`fused_moe_trtllm_sm100`, which vLLM selects for block-FP8 MoE whenever the per-rank intermediate
# is 128-aligned) is JIT-compiled on first use, and that compile is not reliable. The reference's own
# path is unaffected -- it refines the FP8 block scales to 32 and selects Triton -- but an agent's EP
# or replicated-expert layout is not.
COPY lock/requirements-flashinfer-jit-cache.lock /opt/infb/lock/
RUN pip install --no-cache-dir --retries 10 --timeout 120 --require-hashes --no-deps \
      --index-url https://pypi.org/simple \
      --extra-index-url https://flashinfer.ai/whl/cu130 \
      -r /opt/infb/lock/requirements-flashinfer-jit-cache.lock \
    && python -c "import flashinfer, flashinfer_jit_cache, pathlib, sys; \
so=pathlib.Path(flashinfer_jit_cache.__file__).parent/'jit_cache'/'fused_moe_trtllm_sm100'/'fused_moe_trtllm_sm100.so'; \
print('flashinfer', flashinfer.__version__, 'jit cache', so.is_file()); sys.exit(0 if so.is_file() else 7)"

FROM engine-runtime AS engine
# A neutral git identity, system-wide and AFTER the preparation commit, so that commit's hash does
# not move. The submission is the committed tree, and without it `git commit` works only where the
# hostname carries a domain; a Docker-default hostname does not, and the same commit fails with
# "Author identity unknown".
RUN git config --system user.name agent \
    && git config --system user.email agent@localhost \
    && test -z "$(git -C /app status --porcelain)"

# Runtime environment. The library path below names the image's CUDA 13 forward-compatibility driver
# directory first, which is what lets the engine run on a host whose driver is older than CUDA 13's
# (570.x). The task layer at the end of this file takes that directory off the runtime path: on a
# host whose driver is NEWER than the compat libcuda (580.126.20) it shadows the real driver and CUDA
# does not initialize (cuInit 803). The Hub, ModelScope and Cargo are never contacted at trial time.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1 \
    VLLM_NO_USAGE_STATS=1 \
    MODELSCOPE_CACHE=/hf/modelscope \
    CARGO_NET_OFFLINE=true \
    LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:/usr/local/cuda-13.0/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# The public material the instruction names outside /app/eval (/public): model card and configs, the
# public copy of the acceptance test, the offline dataset copies EvalScope reads and the
# whole-serving sample texts. /model is a symlink onto the checkpoint snapshot in the Hub cache, so
# the instruction can name one path while the upstream test resolves the repository id. GSM8K is
# fetched from its public source and pinned by sha256 rather than committed twice into the task
# directory; AIME25 (16 KB) is committed, because its source needs a client rather than a URL.
COPY public/ /public/
RUN mkdir -p /public/datasets/gsm8k \
    && curl -fsSL --retry 10 -o /tmp/gsm8k_train.jsonl \
         https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl \
    && curl -fsSL --retry 10 -o /tmp/gsm8k_test.jsonl \
         https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl \
    && echo "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465  /tmp/gsm8k_train.jsonl" | sha256sum -c - \
    && echo "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14  /tmp/gsm8k_test.jsonl" | sha256sum -c - \
    && mv /tmp/gsm8k_train.jsonl /public/datasets/gsm8k/main_train.jsonl \
    && mv /tmp/gsm8k_test.jsonl /public/datasets/gsm8k/main_test.jsonl \
    && ln -s /hf/hub/models--Qwen--Qwen3.8-Flash-Next-FP8/snapshots/236dfdf285828023ca3bcd3f37366c58a3469b13 /model \
    && ln -s /model /public/model-card \
    && test -f /model/config.json

ARG FP8_PREPARATION_SHA=ada6bd3c2d576cc95a9c7a834690c5fd2ff34b13
COPY preparation-fp8/ /opt/infb/preparation-fp8/
RUN cd /app \
    && git apply --index --binary --whitespace=nowarn /opt/infb/preparation-fp8/001-sm100-fp8-default.patch \
    && GIT_AUTHOR_NAME='Inference-Bench' GIT_AUTHOR_EMAIL='inference-bench@local' \
       GIT_COMMITTER_NAME='Inference-Bench' GIT_COMMITTER_EMAIL='inference-bench@local' \
       GIT_AUTHOR_DATE='1788153124 +0000' GIT_COMMITTER_DATE='1788153124 +0000' \
       git commit -o --quiet --no-verify -F /opt/infb/preparation-fp8/message.txt -- \
       vllm/model_executor/layers/fused_moe/fused_moe.py \
    && test "$(git rev-parse HEAD)" = "$FP8_PREPARATION_SHA" \
    && test -z "$(git status --porcelain)"

ARG PREPARED_BASE_SHA=a5e63c9ee859ceb2f299ff55f063c7c32f94f08b
COPY eval/ /app/eval/
COPY prepared-base/evaluation-message.txt /opt/infb/prepared-base/evaluation-message.txt
RUN cd /app \
    && git ls-files --others --exclude-standard -z -- eval/qwen38 > /tmp/evaluation-paths \
    && xargs -0 git add -- < /tmp/evaluation-paths \
    && GIT_AUTHOR_NAME='Inference-Bench' GIT_AUTHOR_EMAIL='inference-bench@local' \
       GIT_COMMITTER_NAME='Inference-Bench' GIT_COMMITTER_EMAIL='inference-bench@local' \
       GIT_AUTHOR_DATE='1788153125 +0000' GIT_COMMITTER_DATE='1788153125 +0000' \
       xargs -0 git commit -o --quiet --no-verify -F /opt/infb/prepared-base/evaluation-message.txt -- < /tmp/evaluation-paths \
    && rm /tmp/evaluation-paths \
    && test "$(git rev-parse HEAD)" = "$PREPARED_BASE_SHA" \
    && test -z "$(git status --porcelain)"

RUN cd /app \
    && printf 'eval/qwen38/reference/model_author/\n' >> .git/info/exclude \
    && M=/app/eval/qwen38/reference/model_author && mkdir -p "$M" \
    && curl -fsSL --retry 10 -o "$M/modeling_qwen4_exp.py" \
         https://raw.githubusercontent.com/huggingface/transformers/v5.17.0/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py \
    && curl -fsSL --retry 10 -o "$M/configuration_qwen4_exp.py" \
         https://raw.githubusercontent.com/huggingface/transformers/v5.17.0/src/transformers/models/qwen4_exp/configuration_qwen4_exp.py \
    && echo "ca9f00bbd73cfcfbad7ba6073b5ecc23ca27bdace58b746fe6ead6ab175efc0c  $M/modeling_qwen4_exp.py" | sha256sum -c - \
    && echo "b78132d8cd935437208ee281fa4569b771a63fcb58ebffe84f3e62f5b86235ca  $M/configuration_qwen4_exp.py" | sha256sum -c - \
    && printf '%s\n' \
         'Model-author implementation of qwen4_exp, taken verbatim from Hugging Face Transformers' \
         'at tag v5.17.0 (the first release carrying the FP8Embedding class that loads this' \
         'checkpoint n-gram table with its per-tensor scale, transformers PR #48368).' \
         '' \
         'src/transformers/models/qwen4_exp/modeling_qwen4_exp.py' \
         '  sha256 ca9f00bbd73cfcfbad7ba6073b5ecc23ca27bdace58b746fe6ead6ab175efc0c' \
         'src/transformers/models/qwen4_exp/configuration_qwen4_exp.py' \
         '  sha256 b78132d8cd935437208ee281fa4569b771a63fcb58ebffe84f3e62f5b86235ca' \
         '' \
         'Transformers is pinned at 5.12.1 in this image and does NOT know this model type, so' \
         'these files are here to read, not to import.' \
         > "$M/PROVENANCE.md" \
    && test -z "$(git status --porcelain)"

COPY scripts/start_agent_timer.sh /app/scripts/start_agent_timer.sh

# NO IMPORTABLE qwen4_exp ANYWHERE ON AN INTERPRETER'S PATH.
#
#     The EvalScope set pins transformers 5.17.0, and 5.17.0 is the release that carries
#     `transformers/models/qwen4_exp/` -- a complete, working implementation of the model this task
#     asks for, and the very implementation the fidelity gate measures agreement against. EvalScope
#     is installed in its own venv and that venv is bridged onto the engine interpreter, which
#     leaves the ENGINE's transformers 5.12.1 winning every `import transformers` (site.addsitedir
#     APPENDS) -- but precedence does not protect the grade: a candidate does not have to shadow the
#     package, it can name the path, and the verifier runs the same recipe. The fidelity node would
#     then score a delegation to the reference rather than an implementation of it.
#
#     Supplying `modeling_qwen4_exp.py` under /public to READ AND PORT is the task; shipping an
#     installed runtime to DELEGATE to is not.
#
#     The whole transformers tree goes, not only the model package, and nothing is lost by it: the
#     engine's 5.12.1 already wins at runtime through the bridge, so EvalScope has never once
#     resolved this copy.
COPY checks/ /opt/infb/checks/
RUN rm -rf /opt/evalscope/lib/python3.12/site-packages/transformers \
           /opt/evalscope/lib/python3.12/site-packages/transformers-*.dist-info \
    && python /opt/infb/checks/no_reference_impl.py \
    && /opt/evalscope/bin/python -c "import evalscope; print('evalscope still imports:', evalscope.__version__)"

# The two build checks. They are BIND-MOUNTED, never copied, because they have to carry the strings
# they guard against and a copy in the image would be the very leak they guard. The dependency store
# audit is pinned by digest, so a change to it changes this step; it reads every Git object store on
# the image and fails on a withheld file's blob id, a distinctive symbol of the reference solution, a
# remote-tracking ref, alternates, an unreachable object, or a dependency store holding more than its
# checked-out commit. It prints counts only.
RUN --mount=type=bind,source=buildcheck,target=/infb-buildcheck \
    python3 /infb-buildcheck/no_answer_address.py \
    && python3 /infb-buildcheck/no_withheld_code.py \
    && echo "ef86558c16a92957bf4fbe1a5fece00c8179b2b17b7d6a03129b5ab9ec1c06c0  /infb-buildcheck/dependency_stores.py" | sha256sum -c - \
    && python3 /infb-buildcheck/dependency_stores.py

# The EvalScope bridge and the final assertions. site.addsitedir APPENDS the path, so the engine's
# own packages keep precedence over EvalScope's copies of anything they share.
RUN echo /opt/evalscope/lib/python3.12/site-packages \
      > /opt/vllm/lib/python3.12/site-packages/infb_evalscope.pth \
    && python -c "import evalscope, transformers; print('bridge ok: evalscope', evalscope.__version__, '| transformers', transformers.__version__)" \
    && chmod 755 /app/scripts/start_agent_timer.sh \
    && test -x /app/eval/qwen38/build.sh \
    && mkdir -p /logs/artifacts /logs/verifier \
    && test -d /usr/local/cuda-13.0/compat \
    && test -z "$(git -C /app status --porcelain)"

# The engine interpreter, declared: it is /opt/vllm.
ENV INFB_VENV=/opt/vllm
WORKDIR /app
CMD ["/bin/bash"]

# The task layer: the engine stage above is the task image, and the engine is a STAGE in this file
# with the final stage `FROM ${ENGINE}`, so `docker build tasks/<slug>/environment` needs no registry
# and no prebuilt tag -- Pier's docker backend passes no --build-arg, so the ARG default at the top is
# the whole public contract. This seam exists so tests/Dockerfile can be this whole file plus the
# tests. Its only change is the runtime library path without the forward-compat directory.
FROM ${ENGINE}
WORKDIR /app
ENV LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
