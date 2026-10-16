#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# The recipe of the task image for support_diffusiongemma_26b_a4b_2xh100_vllm (DeepSWE idiom).
#
# Every input is public: the nvidia/cuda base by digest, hash-locked wheels from PyPI, the PyTorch
# cu130 index and flashinfer.ai (lock/, resolved by lock/resolve.sh), the vLLM tree at the base
# commit from GitHub, the engine's FlashAttention fork and CUTLASS at their pinned commits from
# GitHub (prepare_fa_vendor.py, checked file by file against fa-materials.json), the two
# checkpoints from the Hugging Face Hub at their pinned revisions (model-manifest.json), the GSM8K
# splits and the model-author files by sha256. The build needs network; the trial does not.
#
# The engine is a STAGE in this file and the final stage is `FROM ${ENGINE}`, so
#   docker build tasks/support_diffusiongemma_26b_a4b_2xh100_vllm/environment
# needs no registry and no prebuilt tag: the ARG default below is the whole public contract.
# infb-engine: task-authored
ARG ENGINE=engine

FROM docker.io/nvidia/cuda:13.0.3-devel-ubuntu24.04@sha256:b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266 AS engine-build

# The upstream base commit and the one preparation commit built on it before the build, which
# vendors the FlashAttention fork at the engine's own pin (no kernel is prepared). Both are asserted
# below, so a drift in either input fails the build instead of producing a different task. The
# commit MESSAGE says what it supplies and not where it came from: it is the top of `git log` in
# /app, the first thing an agent reads.
ARG RUST_TOOLCHAIN=1.95
# H100 only. Restricting the list is the difference between a one-hour and a multi-hour compile;
# the verifier rebuilds the candidate with the same value.
ARG TORCH_CUDA_ARCH_LIST=9.0a

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# 1. System packages: the runtime subset of upstream's docker/Dockerfile at the base commit (Python
#    3.12, Ubuntu 24.04's own, and the toolchain the CUDA and Triton kernels compile with).
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl wget git git-lfs lsof netcat-openbsd kmod unzip locales procps \
      python3.12-full python3.12-dev python3.12-venv \
      build-essential cmake ninja-build perl patchelf ccache pkg-config \
      libnuma1 libnuma-dev numactl libssl-dev libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Two virtual environments, hash-locked and separate: /opt/infb/tools holds only what this recipe
#    runs (huggingface_hub for the checkpoints); /opt/vllm is the engine's, filled from the lock and
#    nothing else. Only the .lock files enter the image, never lock/resolve.sh or lock/RESOLVED.md.
#    The engine lock is copied where it is installed, AFTER the checkpoint layers, so a lock change
#    does not re-download 52 GB.
COPY lock/requirements-tools.lock /opt/infb/lock/
RUN python3.12 -m venv /opt/infb/tools \
    && /opt/infb/tools/bin/pip install --no-cache-dir --require-hashes -r /opt/infb/lock/requirements-tools.lock \
    && python3.12 -m venv /opt/vllm
ENV PATH=/opt/vllm/bin:/root/.cargo/bin:${PATH}

# 3. The Rust toolchain vLLM's frontend extension is built with (rust-toolchain.toml at the base
#    commit pins 1.95).
RUN curl --proto '=https' --tlsv1.2 -sSf --retry 3 https://sh.rustup.rs \
      | sh -s -- -y --no-modify-path --profile minimal --default-toolchain ${RUST_TOOLCHAIN} \
    && rustc --version && cargo --version

# 4. The checkpoints from the Hub at their pinned revisions, into the image's Hugging Face cache: one
#    layer per safetensors shard of google/diffusiongemma-26B-A4B-it and one for its nine small files
#    (twelve groups), then one for Qwen/Qwen2.5-0.5B-Instruct. Registries cap a single blob upload, so
#    no layer here exceeds 5 GB. Every file is verified against model-manifest.json (size
#    and sha256) and the digests are recorded at $HF_HUB_CACHE/.infb/manifest.json for preflight.
ENV HF_HOME=/hf HF_HUB_CACHE=/hf/hub
COPY model-manifest.json fetch_model.py /opt/infb/
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 0 --of 12
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 1 --of 12
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 2 --of 12
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 3 --of 12
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 4 --of 12
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 5 --of 12
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 6 --of 12
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 7 --of 12
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 8 --of 12
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 9 --of 12
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 10 --of 12
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 11 --of 12 --finalize
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --repo Qwen/Qwen2.5-0.5B-Instruct --group 0 --of 1 --finalize

# 5. vLLM's dependencies at the base commit, hash-locked from the public indexes (lock/RESOLVED.md):
#    every package the base's requirements/test/cuda.txt pins, at its pin, and the rest of
#    requirements/cuda.txt and requirements/build/cuda.txt resolved at the base commit's date, so the
#    editable install below needs no build isolation and no network. transformers is 5.5.3.
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
#    disabled. Then the one preparation commit: the engine's FlashAttention fork at its own pin,
#    vendored under third_party/vllm-flash-attn so its sources are editable in the tree and build
#    offline, and a three-line CMake default that builds from it (prepare_fa_vendor.py, every file
#    checked against fa-materials.json). The script is BIND-MOUNTED, never copied: it also carries,
#    for its authoring mode, the identities the build checks below look for, and a copy in the image
#    would be the very leak those checks guard. The commit is made with a pinned identity and dates,
#    so it is an object rather than a local artefact, and asserted. The agent timer's files and the
#    timer script live under /app and are excluded from git so they never enter a submission.
ARG BASE_SHA=1ce3cdc5c14f656f81f91e264d557e0b6e6fea54
ARG VENDOR_SHA=e21e4b8f5010a1dd3c4b95c34c348f88f906f686
WORKDIR /app
COPY prepared-base/ /opt/infb/prepared-base/
RUN --mount=type=bind,source=prepare_fa_vendor.py,target=/infb-vendor/prepare_fa_vendor.py \
    --mount=type=bind,source=fa-materials.json,target=/infb-vendor/fa-materials.json \
    git clone https://github.com/vllm-project/vllm.git . \
    && DEFAULT="$(git remote show origin | sed -n 's/.*HEAD branch: //p')" \
    && git checkout -B "$DEFAULT" "$BASE_SHA" \
    && git remote remove origin \
    && for b in $(git for-each-ref --format='%(refname:short)' refs/heads | grep -vx "$DEFAULT"); do git branch -D "$b" || true; done \
    && for t in $(git tag); do git merge-base --is-ancestor "$t" HEAD 2>/dev/null || git tag -d "$t"; done \
    && test "$(git rev-parse HEAD)" = "$BASE_SHA" \
    && python3.12 /infb-vendor/prepare_fa_vendor.py --materials /infb-vendor/fa-materials.json \
         --export /app/third_party/vllm-flash-attn --cmake /app/cmake/external_projects/vllm_flash_attn.cmake \
    && git add -f -- third_party/vllm-flash-attn cmake/external_projects/vllm_flash_attn.cmake \
    && n="$(git diff --cached --name-only -- third_party/vllm-flash-attn | wc -l)" \
    && want="$(python3.12 -c "import json; m=json.load(open('/infb-vendor/fa-materials.json')); print(m['fork']['files'] + m['cutlass']['files'])")" \
    && if [ "$n" != "$want" ]; then echo "FATAL: vendored $n files, expected $want" >&2; exit 7; fi \
    && test "$(git diff --cached --name-only | wc -l)" = "$((want + 1))" \
    && GIT_AUTHOR_NAME='Inference-Bench' GIT_AUTHOR_EMAIL='inference-bench@local' \
       GIT_COMMITTER_NAME='Inference-Bench' GIT_COMMITTER_EMAIL='inference-bench@local' \
       GIT_AUTHOR_DATE='1781241375 +0000' GIT_COMMITTER_DATE='1781241375 +0000' \
       git commit --quiet --no-verify -F /opt/infb/prepared-base/vendor-message.txt \
    && git reflog expire --expire=now --all \
    && git gc --prune=now \
    && git config core.hooksPath /dev/null \
    && printf '.timer/\nscripts/start_agent_timer.sh\n' >> .git/info/exclude \
    && head="$(git rev-parse HEAD)" \
    && echo "vendor commit: $head" \
    && if [ "$head" != "$VENDOR_SHA" ]; then echo "FATAL: vendor commit is $head, expected $VENDOR_SHA" >&2; exit 7; fi \
    && test -z "$(git status --porcelain)"

# 7. Editable install of the tree without build isolation (the build backend is in the lock) and
#    without touching the index; the FlashAttention extensions compile from the vendored source, so
#    no fetch happens. These commands are /app/eval/diffusion_gemma/build.sh, which the verifier
#    re-runs on the candidate's tree (the image build alone may reach the network, for the Rust
#    frontend's crates). setup.py copies the fork's CuTe-DSL sources through the link CMake makes in
#    its build directory, and the copies would not import; so, as build.sh does, the copy is removed
#    before the install and replaced by that link after it. The
#    editable install builds in a temporary directory, so the build is repeated once through
#    `setup.py build_ext --inplace --build-temp /app/build/temp`, which keeps CMake's build directory
#    in the image and makes build.sh's rebuilds incremental.
#    .gitignore covers every artefact the build leaves, so the tree must still be clean afterwards.
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    VLLM_TARGET_DEVICE=cuda \
    CARGO_NET_OFFLINE=false \
    MAX_JOBS=48 \
    NVCC_THREADS=4 \
    CMAKE_BUILD_PARALLEL_LEVEL=48
RUN cd /app \
    && rm -rf vllm/vllm_flash_attn/cute \
    && pip install --no-cache-dir --no-deps --no-build-isolation -e . \
    && rm -rf vllm/vllm_flash_attn/cute \
    && python setup.py build_ext --inplace --build-temp /app/build/temp \
    && rm -rf vllm/vllm_flash_attn/cute \
    && ln -s ../../third_party/vllm-flash-attn/flash_attn/cute vllm/vllm_flash_attn/cute \
    && python -c "import vllm; print('vllm', vllm.__version__, 'from', vllm.__file__)" \
    && python -c "import glob; so=sorted(glob.glob('/app/vllm/**/*.so', recursive=True)); print('compiled extensions:', len(so)); [print('   ', s) for s in so]" \
    && git status --porcelain | head -50

# 7b. The assertions on what step 7 produced, in their own layer so tightening them costs seconds
#     rather than the compile. The extension names are those setup.py builds for CUDA at this commit;
#     nothing is imported, because a build container has no GPU and no libcuda.so.1. The FA4 CuTe-DSL
#     sources must be served from the vendored tree through a symlink, so a candidate's edits there
#     take effect without a rebuild.
RUN cd /app \
    && python -c "import pathlib,sys; \
want={'_C_stable_libtorch','_moe_C_stable_libtorch','cumem_allocator','spinloop'}; \
got={p.name.split('.')[0] for p in pathlib.Path('/app/vllm').glob('*.so')}; \
fa={p.name.split('.')[0] for p in pathlib.Path('/app/vllm/vllm_flash_attn').glob('_vllm_fa*_C*.so')}; \
missing=sorted(want-got) + sorted({'_vllm_fa2_C','_vllm_fa3_C'}-fa); \
print('cuda extensions present:', sorted(got)); print('flash-attention:', sorted(fa)); \
sys.exit(0 if not missing else print('FATAL: missing', missing) or 7)" \
    && test -L /app/vllm/vllm_flash_attn/cute \
    && case "$(readlink -f /app/vllm/vllm_flash_attn/cute)" in /app/third_party/vllm-flash-attn/flash_attn/cute) ;; *) echo "FATAL: cute -> $(readlink -f /app/vllm/vllm_flash_attn/cute)" >&2; exit 7;; esac \
    && test -z "$(git status --porcelain)"

FROM engine-build AS engine-runtime

# 7c. FlashInfer's pre-compiled JIT modules (flashinfer-jit-cache, the same release as
#     flashinfer-python), in their own layer because the wheel is a large fetch, placed after the
#     engine build so it invalidates none of its layers.
COPY lock/requirements-flashinfer-jit-cache.lock /opt/infb/lock/
RUN pip install --no-cache-dir --retries 10 --timeout 120 --require-hashes --no-deps \
      --index-url https://pypi.org/simple \
      --extra-index-url https://flashinfer.ai/whl/cu130 \
      -r /opt/infb/lock/requirements-flashinfer-jit-cache.lock \
    && python -c "import flashinfer, flashinfer_jit_cache; print('flashinfer', flashinfer.__version__, 'jit cache', flashinfer_jit_cache.__file__)"

FROM engine-runtime AS engine
#     A neutral git identity, system-wide and AFTER the preparation commit, so that commit's hash does
#     not move. The submission is the committed tree, and without it `git commit` works only where the
#     hostname carries a domain.
RUN git config --system user.name agent \
    && git config --system user.email agent@localhost \
    && test -z "$(git -C /app status --porcelain)"

# 8. Runtime environment. This stage puts the image's CUDA forward-compatibility driver library
#    first on the library path; the task layer at the end of this file takes it off, so a runtime
#    that keeps the image's environment on a host whose driver supersedes the compat library loads
#    the host's libcuda instead of failing CUDA initialization. The Hub and Cargo are never
#    contacted at trial time.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1 \
    VLLM_NO_USAGE_STATS=1 \
    CARGO_NET_OFFLINE=true \
    LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:/usr/local/cuda-13.0/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# 9. The public material the instruction names outside /app/eval (/public): the whole-serving sample
#    texts and the GSM8K splits, fetched from their public source and pinned by sha256. /model and
#    /models/qwen2.5-0.5b-instruct are symlinks onto the checkpoint snapshots in the Hub cache.
COPY public/ /public/
RUN mkdir -p /public/datasets/gsm8k /models \
    && curl -fsSL --retry 10 -o /tmp/gsm8k_train.jsonl \
         https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl \
    && curl -fsSL --retry 10 -o /tmp/gsm8k_test.jsonl \
         https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl \
    && echo "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465  /tmp/gsm8k_train.jsonl" | sha256sum -c - \
    && echo "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14  /tmp/gsm8k_test.jsonl" | sha256sum -c - \
    && mv /tmp/gsm8k_train.jsonl /public/datasets/gsm8k/main_train.jsonl \
    && mv /tmp/gsm8k_test.jsonl /public/datasets/gsm8k/main_test.jsonl \
    && ln -s /hf/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/0f28bc42f588fbd8f71e08102b1c3960298a1358 /model \
    && ln -s /hf/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 /models/qwen2.5-0.5b-instruct \
    && test -f /model/config.json && test -f /models/qwen2.5-0.5b-instruct/config.json

ARG PREPARED_BASE_SHA=928e87391806edec70f4d81a0fadab20593dce48
COPY eval/ /app/eval/
RUN cd /app \
    && git ls-files --others --exclude-standard -z -- eval/diffusion_gemma > /tmp/evaluation-paths \
    && xargs -0 git add -- < /tmp/evaluation-paths \
    && GIT_AUTHOR_NAME='Inference-Bench' GIT_AUTHOR_EMAIL='inference-bench@local' \
       GIT_COMMITTER_NAME='Inference-Bench' GIT_COMMITTER_EMAIL='inference-bench@local' \
       GIT_AUTHOR_DATE='1781241376 +0000' GIT_COMMITTER_DATE='1781241376 +0000' \
       xargs -0 git commit -o --quiet --no-verify -F /opt/infb/prepared-base/evaluation-message.txt -- < /tmp/evaluation-paths \
    && rm /tmp/evaluation-paths \
    && head="$(git rev-parse HEAD)" \
    && echo "prepared base: $head" \
    && if [ "$head" != "$PREPARED_BASE_SHA" ]; then echo "FATAL: prepared base is $head, expected $PREPARED_BASE_SHA" >&2; exit 7; fi \
    && echo "$head" > /opt/infb/prepared-base/PREPARED_BASE_SHA \
    && test -z "$(git status --porcelain)"

# 11. The model-author implementation from Transformers v5.13.0 (which is NOT installed here), to read
#     beside the evaluation resources: fetched by sha256 after the evaluation commit and excluded from
#     git, so it never enters the prepared base, a submission or a candidate's patch. Its
#     PROVENANCE.md is part of the evaluation commit.
RUN cd /app \
    && M=eval/diffusion_gemma/reference/model_author && mkdir -p "$M" \
    && U=https://raw.githubusercontent.com/huggingface/transformers/v5.13.0/src/transformers \
    && for f in modeling_diffusion_gemma.py generation_diffusion_gemma.py configuration_diffusion_gemma.py; do \
         printf '%s\n' "$M/$f" >> .git/info/exclude; \
         curl -fsSL --retry 10 -o "$M/$f" "$U/models/diffusion_gemma/$f"; done \
    && for f in cache_utils.py masking_utils.py; do \
         printf '%s\n' "$M/$f" >> .git/info/exclude; \
         curl -fsSL --retry 10 -o "$M/$f" "$U/$f"; done \
    && printf '%s  %s\n' \
         3a114c48ccd270577c6caf4f04009442f706aea4d5cb0a48f08beaa0c2b602ae "$M/modeling_diffusion_gemma.py" \
         0a4bf0c9fbcaef2cf86d3bb2177eb985bb83568af815a6812a4c8f7a1eeb71de "$M/generation_diffusion_gemma.py" \
         69131b618dcdf8e95b4b17fadf40c658c53cd3a4a7f4ae410b26673b7d552ec0 "$M/configuration_diffusion_gemma.py" \
         16bbbba2394c0151af53009800e5a4052479417a8c7e5edfbd5ef128a46fcbc1 "$M/cache_utils.py" \
         ad7b312bc3830894a198760520cc1baf02e35d327d216d41f00b0c61e9604334 "$M/masking_utils.py" \
       | sha256sum -c - \
    && test -z "$(git status --porcelain)"

COPY scripts/start_agent_timer.sh /app/scripts/start_agent_timer.sh

# 12. NO IMPORTABLE DiffusionGemma IMPLEMENTATION ANYWHERE ON AN INTERPRETER'S PATH: Transformers
#     5.11.0 and later carry one, and the engine's Transformers is 5.5.3; the model-author files above
#     are there to read, not to import.
COPY checks/ /opt/infb/checks/
RUN python /opt/infb/checks/no_reference_impl.py

# 12b. NOTHING ON THE AGENT'S SURFACE SAYS WHICH PULL REQUESTS THIS IS, AND NO WITHHELD CODE IS
#      RECOVERABLE. Both checks are BIND-MOUNTED, never copied: they carry the strings they look for.
RUN --mount=type=bind,source=buildcheck,target=/infb-buildcheck \
    python3 /infb-buildcheck/no_answer_address.py \
    && python3 /infb-buildcheck/no_withheld_code.py

# 13. Final assertions, and the engine venv declared: on the default launch path the runtime refuses an
#     image it cannot prove carries an engine, and it accepts either an `# infb-recipe:` marker or this
#     declaration.
RUN chmod 755 /app/scripts/start_agent_timer.sh \
    && test -x /app/eval/diffusion_gemma/build.sh \
    && mkdir -p /logs/artifacts /logs/verifier \
    && test -d /usr/local/cuda-13.0/compat \
    && test -z "$(git -C /app status --porcelain)"
ENV INFB_VENV=/opt/vllm
WORKDIR /app
CMD ["/bin/bash"]

# ── the task layer ───────────────────────────────────────────────────────────────────
# It adds one line, on purpose: the library path without the compat directory, so that a runtime
# that keeps the image's environment on a host whose driver supersedes the compat library loads the
# host's libcuda instead of failing CUDA initialization. Otherwise the engine stage above IS the task
# image; the seam lets tests/Dockerfile be this whole file plus the tests, which makes the verifier
# image the task image by construction instead of by a registry reference
FROM ${ENGINE}
WORKDIR /app
ENV LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

COPY l3/fetch_l3_hf.py l3/check_l3_cache.py l3/agent-l3-hf-manifest.json /opt/infb/l3/
RUN /opt/vllm/bin/python /opt/infb/l3/fetch_l3_hf.py /opt/infb/l3/agent-l3-hf-manifest.json "$HF_HUB_CACHE" \
    && /opt/vllm/bin/python /opt/infb/l3/check_l3_cache.py /opt/infb/l3/agent-l3-hf-manifest.json "$HF_HUB_CACHE"

# infb-verifier-tail
ENV CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda

COPY config.json /tests/config.json
COPY eval/ /tests/eval/
COPY infb_efficiency/ /tests/infb_efficiency/
COPY reference/ /tests/reference/
COPY test.sh test.patch grader.py preflight.py /tests/
COPY infb_verify/ /tests/infb_verify/
COPY hidden/ /tests/hidden/
COPY public/ /tests/public/
COPY _upstream/ /tests/_upstream/
RUN chmod +x /tests/test.sh && test -s /tests/hidden/policy.json && test -f /tests/public/efficiency/client.py \
    && test -f /tests/infb_verify/test_tp1.py && test -f /tests/grader.py && test -f /tests/test.patch
COPY l3/ /opt/infb/l3/
RUN /opt/vllm/bin/python /opt/infb/l3/fetch_l3_hf.py /opt/infb/l3/l3-hf-manifest.json "$HF_HUB_CACHE"
RUN python3 /tests/infb_efficiency/snapshot.py --config /tests/config.json
RUN /opt/vllm/bin/python /tests/eval/diffusion_gemma/run.py gencfg --out /opt/infb/gencfg --output /tmp/gencfg.json \
    && test -f /opt/infb/gencfg/tp1_fixed/generation_config.json && test -f /opt/infb/gencfg/tp2_fixed/generation_config.json
