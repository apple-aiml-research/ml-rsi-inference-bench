#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# The recipe of the task image for support_minimax_m3_4xb200_vllm (DeepSWE idiom).
#
# Pier BUILDS this file: task.toml pins no docker_image, so the task image is whatever this recipe
# produces.
# Every input is public: the nvidia/cuda base by digest, hash-locked wheels from PyPI, the PyTorch
# cu130 index and flashinfer.ai (lock/, resolved by lock/resolve.sh), the vLLM tree at the base
# commit from GitHub, the kernel preparation commit this directory carries as a patch, the
# checkpoint from the Hugging Face Hub at its pinned revision (model-manifest.json), and the public
# material under /public fetched by URL and sha256. The build needs network; the trial does not.
#
# SHAPE: the engine is a STAGE in this file and the final stage is `FROM ${ENGINE}`, so
#   docker build tasks/support_minimax_m3_4xb200_vllm/environment
# needs no registry and no prebuilt tag. Pier's docker backend passes no --build-arg, so the ARG
# default below is the whole public contract.
#
# The engine stage is this task's OWN: a Day-0 task's premise is one upstream commit and a checkpoint
# at one revision.
# infb-engine: task-authored
ARG ENGINE=engine

FROM docker.io/nvidia/cuda:13.0.3-devel-ubuntu24.04@sha256:b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266 AS engine-build

ARG RUST_TOOLCHAIN=1.95
# B200 only: this image serves one machine. The verifier rebuilds the candidate with the same value.
ARG TORCH_CUDA_ARCH_LIST=10.0a

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# 1. System packages: Python 3.12 (Ubuntu 24.04's own) and the toolchain the CUDA and Triton
#    kernels compile with.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl wget git git-lfs lsof netcat-openbsd kmod unzip locales procps \
      python3.12-full python3.12-dev python3.12-venv \
      build-essential cmake ninja-build perl patchelf ccache pkg-config \
      libnuma1 libnuma-dev numactl libssl-dev libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Two virtual environments, hash-locked and separate: /opt/infb/tools holds only what this
#    recipe itself runs (huggingface_hub for the checkpoint); /opt/vllm is the engine's, filled from
#    the engine lock below and nothing else. Only the .lock files enter the image, never
#    lock/resolve.sh or lock/RESOLVED.md. The engine lock is copied where it is installed, AFTER the
#    checkpoint layers, so a lock change does not re-fetch the checkpoint.
COPY lock/requirements-tools.lock /opt/infb/lock/
RUN python3.12 -m venv /opt/infb/tools \
    && /opt/infb/tools/bin/pip install --no-cache-dir --require-hashes -r /opt/infb/lock/requirements-tools.lock \
    && python3.12 -m venv /opt/vllm
ENV PATH=/opt/vllm/bin:/root/.cargo/bin:${PATH}

# 3. The Rust toolchain vLLM's frontend extension is built with (rust/rust-toolchain.toml at the
#    base commit pins 1.95).
RUN curl --proto '=https' --tlsv1.2 -sSf --retry 3 https://sh.rustup.rs \
      | sh -s -- -y --no-modify-path --profile minimal --default-toolchain ${RUST_TOOLCHAIN} \
    && rustc --version && cargo --version

# 4. The checkpoint at the pinned revision, into the image's Hugging Face cache; /model below is a
#    symlink onto the snapshot. One layer per safetensors shard (31, the largest 16.1 GB) and one for
#    the 21 small files: registries cap a single blob upload, and Docker's overlay2 allows 128
#    layers. Every file is verified against model-manifest.json (size; the Hub's sha256 for the 32
#    LFS files; the git object id of every other file) and the computed digests are recorded at
#    $HF_HUB_CACHE/.infb/manifest.json for preflight. fetch_model.py downloads from the Hub; a
#    builder holding a copy of the files may bind it read-only at /infb-model-staging/<repo_id>/,
#    and a staged file is accepted only if it passes the same verification.
ENV HF_HOME=/hf HF_HUB_CACHE=/hf/hub
COPY model-manifest.json fetch_model.py /opt/infb/
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 0 --of 32
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 1 --of 32
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 2 --of 32
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 3 --of 32
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 4 --of 32
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 5 --of 32
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 6 --of 32
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 7 --of 32
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 8 --of 32
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 9 --of 32
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 10 --of 32
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 11 --of 32
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 12 --of 32
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 13 --of 32
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 14 --of 32
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 15 --of 32
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 16 --of 32
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 17 --of 32
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 18 --of 32
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 19 --of 32
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 20 --of 32
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 21 --of 32
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 22 --of 32
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 23 --of 32
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 24 --of 32
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 25 --of 32
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 26 --of 32
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 27 --of 32
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 28 --of 32
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 29 --of 32
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 30 --of 32
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 31 --of 32 --finalize

# 5. vLLM's dependencies at the base commit, hash-locked from the public indexes (lock/RESOLVED.md):
#    every requirements/test/cuda.txt pin, the rest of requirements/cuda.txt and
#    requirements/build/cuda.txt resolved as of the base commit, and transformers 5.5.3, so the
#    editable install below needs no build isolation and no network.
COPY lock/requirements.lock /opt/infb/lock/
RUN pip install --no-cache-dir --retries 10 --timeout 120 --require-hashes --no-deps \
      --index-url https://pypi.org/simple \
      --extra-index-url https://download.pytorch.org/whl/cu130 \
      --extra-index-url https://flashinfer.ai/whl/cu130 \
      -r /opt/infb/lock/requirements.lock

# 6. The vLLM tree at the base commit, in DeepSWE's git time-travel idiom: the default branch points
#    at the base with no future history, later branches and tags gone, the reflog expired and the
#    store garbage-collected so no later object can be recovered from history; hooks disabled. Then
#    the kernel-only preparation commit, reproduced from the patch and message this directory
#    carries, with pinned identity and date, and asserted. The agent timer's files and the timer
#    script live under /app and are excluded from git so they never enter a submission.
ARG BASE_SHA=a3195fab7b1227e75403fef83891e961e69228a6
ARG KERNEL_PREPARATION_SHA=5c1e3b901b554c2b4d44414cf51a7b2c84c1c54c
WORKDIR /app
COPY prepared-base/000-supplied-kernels.patch prepared-base/kernels-message.txt /opt/infb/prepared-base/
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
       GIT_AUTHOR_DATE='1781541473 +0000' GIT_COMMITTER_DATE='1781541473 +0000' \
       xargs -0 git commit -o --quiet --no-verify -F /opt/infb/prepared-base/kernels-message.txt -- < /tmp/preparation-paths \
    && rm /tmp/preparation-paths \
    && git reflog expire --expire=now --all \
    && git gc --prune=now \
    && git config core.hooksPath /dev/null \
    && printf '.timer/\nscripts/start_agent_timer.sh\n' >> .git/info/exclude \
    && head="$(git rev-parse HEAD)" \
    && if [ "$head" != "$KERNEL_PREPARATION_SHA" ]; then \
         echo "FATAL: kernel preparation is $head, expected $KERNEL_PREPARATION_SHA" >&2; exit 7; fi \
    && test -z "$(git status --porcelain)"

# 7. Editable install of the tree without build isolation (the build backend is in the lock) and
#    without touching the index, so exactly this step is what /public/build.sh and the verifier
#    re-run on a candidate's tree. The CUDA extensions and the Rust frontend are compiled here, and
#    the Rust workspace's crates are fetched so `cargo test --offline` works in the episode and in
#    the verifier; .gitignore covers every artefact they leave, so the tree must still be clean.
#    Every Git store the build's FetchContent clones leave under /app/.deps (their submodule stores
#    included) is then reduced to its checked-out commit in this same layer: every ref not pointing
#    at HEAD deleted, the store shallow at HEAD, reflogs expired, unreachable objects pruned. A full
#    clone carries its upstream's other branches and their history; FetchContent's update step on a
#    rebuild still finds the pinned commit, or the pinned tag at HEAD, locally, so the rebuild needs no
#    fetch. Step 11b's audit reads every one of these stores.
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
    && (cd /app/rust && cargo fetch --locked) \
    && find /app/.deps -type d -name objects -prune -printf '%h\n' | while read -r g; do \
         test -f "$g/HEAD" || continue; \
         head="$(git --git-dir="$g" rev-parse HEAD)" || exit 1; \
         git --git-dir="$g" for-each-ref --format='%(refname) %(objectname) %(*objectname)' \
           | while read -r ref obj peeled; do \
               [ "$obj" = "$head" ] || [ "$peeled" = "$head" ] || git --git-dir="$g" update-ref --no-deref -d "$ref" || exit 1; \
             done || exit 1; \
         echo "$head" > "$g/shallow" \
           && git --git-dir="$g" reflog expire --expire=now --all \
           && git --git-dir="$g" gc --prune=now --quiet \
           && echo "store $g at $head: $(git --git-dir="$g" count-objects -v | awk '/^(count|in-pack):/ {n += $2} END {print n}') objects" \
           || exit 1; \
       done \
    && git status --porcelain | head -50

# 7b. The assertions on what step 7 produced, in their own layer so tightening them costs seconds
#     rather than the whole compile. Existence and names only: a build container has no GPU, so the
#     driver-linked imports are asserted by the image smoke and by the verifier's L0 node.
RUN cd /app \
    && python -c "import pathlib,sys; \
want={'_C','_C_stable_libtorch','_moe_C_stable_libtorch','cumem_allocator','spinloop'}; \
got={p.name.split('.')[0] for p in pathlib.Path('/app/vllm').glob('*.so')}; \
fa=list(pathlib.Path('/app/vllm/vllm_flash_attn').glob('_vllm_fa*_C*.so')); \
rust=sorted(p.name for p in pathlib.Path('/app/vllm').glob('_rust_*.so')); \
missing=sorted(want-got); \
print('cuda extensions present:', sorted(got)); print('flash-attention:', [p.name for p in fa]); print('rust:', rust); \
sys.exit(0 if not missing and fa and rust else print('FATAL: missing', missing, 'flash-attn', fa, 'rust', rust) or 7)" \
    && test -z "$(git status --porcelain)"

FROM engine-build AS engine-runtime

# 7c. FlashInfer's pre-compiled JIT modules (flashinfer-jit-cache, the engine's FlashInfer release),
#     in their own layer because the wheel is a large fetch; upstream's docker/Dockerfile at the base
#     installs the same wheel the same way.
COPY lock/requirements-flashinfer-jit-cache.lock /opt/infb/lock/
RUN pip install --no-cache-dir --retries 10 --timeout 120 --require-hashes --no-deps \
      --index-url https://pypi.org/simple \
      --extra-index-url https://flashinfer.ai/whl/cu130 \
      -r /opt/infb/lock/requirements-flashinfer-jit-cache.lock \
    && python -c "import flashinfer, flashinfer_jit_cache; print('flashinfer', flashinfer.__version__, 'jit cache', flashinfer_jit_cache.__file__)"

FROM engine-runtime AS engine
#     A neutral git identity, system-wide and AFTER the preparation commit, so that commit's hash does
#     not move and `git commit` works under any hostname.
RUN git config --system user.name agent \
    && git config --system user.email agent@localhost \
    && test -z "$(git -C /app status --porcelain)"

# 8. Runtime environment. The image's CUDA forward-compatibility driver library is first on the
#    library path while the image builds. The task layer at the end of this file takes it off the
#    runtime path: on a host whose driver is newer than that library it would shadow the host's
#    own, and CUDA would not initialize. This declaration stays, because a runtime may put the
#    directory back on a host whose driver predates CUDA 13. The
#    Hub, ModelScope and Cargo are never contacted at trial time.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1 \
    VLLM_NO_USAGE_STATS=1 \
    MODELSCOPE_CACHE=/hf/modelscope \
    CARGO_NET_OFFLINE=true \
    LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:/usr/local/cuda-13.0/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# 9. The public material the instruction names (/public): the MSA paper in full, the model-author
#    implementation from Transformers v5.13.0 (NOT installed here: the engine's Transformers is
#    5.5.3), the GSM8K test split, the efficiency client and its public sample, the public fidelity
#    examples, and the rebuild script. Each fetched file is pinned by sha256. /model is a symlink
#    onto the checkpoint snapshot in the Hub cache.
COPY public/ /public/
RUN mkdir -p /public/datasets/gsm8k /public/reference-impl /public/msa \
    && curl -fsSL --retry 10 -o /public/datasets/gsm8k/main_test.jsonl \
         https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl \
    && echo "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14  /public/datasets/gsm8k/main_test.jsonl" | sha256sum -c - \
    && curl -fsSL --retry 10 -o /public/reference-impl/modeling_minimax_m3_vl.py \
         https://raw.githubusercontent.com/huggingface/transformers/v5.13.0/src/transformers/models/minimax_m3_vl/modeling_minimax_m3_vl.py \
    && curl -fsSL --retry 10 -o /public/reference-impl/configuration_minimax_m3_vl.py \
         https://raw.githubusercontent.com/huggingface/transformers/v5.13.0/src/transformers/models/minimax_m3_vl/configuration_minimax_m3_vl.py \
    && echo "fd17ebfc48e8597607c017674053338029871fe9c924042d64cc8f92fa3b2584  /public/reference-impl/modeling_minimax_m3_vl.py" | sha256sum -c - \
    && echo "7ee031c3220fed242265388bdd8d7d31780ce268ed5244f5923002ebee695017  /public/reference-impl/configuration_minimax_m3_vl.py" | sha256sum -c - \
    && curl -fsSL --retry 10 -o /public/msa/MiniMaxSparseAttention.pdf \
         https://raw.githubusercontent.com/MiniMax-AI/MSA/f63ed8aacacbca83f4d0f6f87a849cf384e440a7/docs/MiniMaxSparseAttention.pdf \
    && echo "c68e3e1c178a0f2d0e3f4a4aa6a3d6af97499869090ce77f6364870fedf233c9  /public/msa/MiniMaxSparseAttention.pdf" | sha256sum -c - \
    && ln -s /hf/hub/models--MiniMaxAI--MiniMax-M3-MXFP8/snapshots/1c4e6a69f3278df8dd6c9693fcca241efe439b7d /model \
    && test -f /model/config.json

# 9b. THE FILES THE ENGINE'S OWN RUST TESTS READ AT RUN TIME, SO THAT NO TEST NEEDS THE NETWORK: the Hugging Face
#     files of the eight models the `vllm-chat` round-trip tests load, in the image's Hub cache where the Rust client
#     resolves them (it does not honor HF_HUB_OFFLINE), and the gpt-oss harmony encoding's vocabulary, which
#     openai-harmony loads from TIKTOKEN_ENCODINGS_BASE and checks against the sha256 pinned in the crate. Each file
#     is pinned by repository revision (or URL), size and sha256 in offline-test-files.json.
COPY offline-test-files.json fetch_offline_test_files.py /opt/infb/
RUN /opt/infb/tools/bin/python /opt/infb/fetch_offline_test_files.py /opt/infb/offline-test-files.json
ENV TIKTOKEN_ENCODINGS_BASE=/opt/infb/tiktoken

# 10. The evaluation resources, a separate deterministic source commit: their edits do not
#     invalidate the compilation of the unchanged engine source.
ARG PREPARED_BASE_SHA=138747893727914cd4f499376e82c3b4bf5f0bba
COPY eval/ /app/eval/
COPY prepared-base/evaluation-message.txt /opt/infb/prepared-base/evaluation-message.txt
RUN cd /app \
    && git ls-files --others --exclude-standard -z -- eval/minimax_m3 > /tmp/evaluation-paths \
    && xargs -0 git add -- < /tmp/evaluation-paths \
    && GIT_AUTHOR_NAME='Inference-Bench' GIT_AUTHOR_EMAIL='inference-bench@local' \
       GIT_COMMITTER_NAME='Inference-Bench' GIT_COMMITTER_EMAIL='inference-bench@local' \
       GIT_AUTHOR_DATE='1781541474 +0000' GIT_COMMITTER_DATE='1781541474 +0000' \
       xargs -0 git commit -o --quiet --no-verify -F /opt/infb/prepared-base/evaluation-message.txt -- < /tmp/evaluation-paths \
    && rm /tmp/evaluation-paths \
    && git rev-parse HEAD \
    && if [ "$PREPARED_BASE_SHA" != "0000000000000000000000000000000000000000" ]; then test "$(git rev-parse HEAD)" = "$PREPARED_BASE_SHA"; fi \
    && test -z "$(git status --porcelain)"

COPY scripts/start_agent_timer.sh /app/scripts/start_agent_timer.sh

# 11. NO IMPORTABLE MODEL-AUTHOR OR MSA IMPLEMENTATION ON ANY INTERPRETER'S PATH: Transformers 5.12.0
#     and later carry minimax_m3_vl and an MSA integration, and the MSA package is withheld. The
#     model-author source under /public is there to read, not to import.
COPY checks/ /opt/infb/checks/
RUN python /opt/infb/checks/no_reference_impl.py

# 11b. NOTHING ON THE AGENT'S SURFACE SAYS WHICH PULL REQUESTS THIS TASK RECONSTRUCTS, AND NO WITHHELD
#      IMPLEMENTATION IS PRESENT: the task-authored files, the checkpoint's text files, the prepared
#      commits and every Git object, every other Git store in the image (the dependency clones and
#      their submodule stores, cargo's git database; each ref at its store's checked-out commit),
#      site-packages, /public, /opt, /root/.cache and /tmp. The checks are BIND-MOUNTED, never copied:
#      they carry the strings they look for.
RUN --mount=type=bind,source=buildcheck,target=/infb-buildcheck \
    python3 /infb-buildcheck/no_answer_address.py \
    && python3 /infb-buildcheck/no_withheld_code.py

# 12. Final assertions.
RUN chmod 755 /app/scripts/start_agent_timer.sh /public/build.sh \
    && mkdir -p /logs/artifacts /logs/verifier \
    && test -d /usr/local/cuda-13.0/compat \
    && test -z "$(git -C /app status --porcelain)"

# 13. The engine venv, declared: the runtime refuses an image it cannot prove carries an engine, and
#     the separate-verifier guard requires the declaration of every Day-0 task.
ENV INFB_VENV=/opt/vllm
WORKDIR /app
CMD ["/bin/bash"]

# ── the task layer ───────────────────────────────────────────────────────────────────
# It adds one line, the runtime library path without the compat directory; everything else is the
# engine stage above. This seam exists so the engine stage can be substituted for a prebuilt image,
# and so tests/Dockerfile can be this whole file plus the tests.
FROM ${ENGINE}
WORKDIR /app
ENV LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
