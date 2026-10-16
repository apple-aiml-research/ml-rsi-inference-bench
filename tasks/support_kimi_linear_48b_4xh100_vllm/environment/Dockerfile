#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# The recipe of the task image for support_kimi_linear_48b_4xh100_vllm (DeepSWE idiom).
#
# Pier BUILDS this file: task.toml pins no docker_image, so the task image is whatever this recipe
# produces.
# Every input is public: the nvidia/cuda base by digest, hash-locked wheels from PyPI, the PyTorch
# cu129 channel and flashinfer.ai (lock/, resolved by lock/resolve.sh), the vLLM tree at the base
# commit from GitHub, the four CMake externals at their pinned revisions, and the checkpoint from the
# Hugging Face Hub at its pinned revision (model-manifest.json). The build needs network; the trial
# does not.
#
# SHAPE: the engine is a STAGE in this file and the final stage is `FROM ${ENGINE}`, so
#   docker build tasks/support_kimi_linear_48b_4xh100_vllm/environment
# needs no registry and no prebuilt tag. Pier's docker backend passes no --build-arg, so the ARG
# default below is the whole public contract.
#
# The engine stage is this task's OWN: a Day-0 task's premise is one upstream commit and a checkpoint
# at one revision. The marker below says which kind of engine stage this is.
# infb-engine: task-authored
ARG ENGINE=engine

# Isolated build-only inputs: none of this stage's layers is an ancestor of the final image.
FROM docker.io/nvidia/cuda:12.9.1-devel-ubuntu24.04@sha256:020bc241a628776338f4d4053fed4c38f6f7f3d7eb5919fecb8de313bb8ba47c AS source-preparation
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git python3 \
    && rm -rf /var/lib/apt/lists/*
RUN --mount=type=bind,source=sourceprep,target=/sourceprep python3 /sourceprep/prepare.py

FROM docker.io/nvidia/cuda:12.9.1-devel-ubuntu24.04@sha256:020bc241a628776338f4d4053fed4c38f6f7f3d7eb5919fecb8de313bb8ba47c AS engine

# Constructed source tree and root commit; source preparation is a separate build stage.
# Only its clean export enters this image. Ordinary dependencies and compiled sources are unchanged.
ARG BASE_SHA=d89bba62d459db1df4eb397a938022d7454b1214
ARG BASE_TREE=f02109e92a704dc7d6abe7d49d87662ff4146a24
# H100 only. Upstream's wheel builds eight architectures; this image serves one machine, and
# restricting the list is the difference between a 20-minute and a multi-hour compile. /app/eval/kimi_linear/build.sh carries the same value, and the verifier rebuilds the candidate
# with it, so oracle and candidate take one build path.
ARG TORCH_CUDA_ARCH_LIST=9.0a
# The four CMake externals, at the revisions this commit's own cmake files pin
# (CMakeLists.txt:285 and cmake/external_projects/*.cmake, read at the base commit). They are cloned
# into the image so the VERIFIER's offline rebuild of a candidate tree needs no network.
ARG CUTLASS_REVISION=v4.2.1
ARG FLASH_ATTN_REVISION=a893712401d70362fbb299cd9c4b3476e8e9ed54
ARG FLASHMLA_REVISION=46d64a8ebef03fa50b4ae74937276a5c940e3f95
ARG QUTLASS_REVISION=830d2c4537c7396e14a02a46fbddd18b5d107c65

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# 1. System packages: the runtime subset of upstream's docker/Dockerfile at the base commit. Python
#    3.12 is Ubuntu 24.04's own, which is the version upstream's Dockerfile declares
#    (`ARG PYTHON_VERSION=3.12`) and reaches through deadsnakes on the older bases it uses; taking
#    the distro's removes both deadsnakes and a pinned python-build-standalone from this recipe.
#    ccache is here because the verifier rebuilds the engine on every submission.
#    No protoc and no Rust: this commit has no rust-toolchain.toml and no Rust extension.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl wget git git-lfs lsof netcat-openbsd kmod unzip locales procps \
      python3.12-full python3.12-dev python3.12-venv \
      build-essential cmake ninja-build perl patchelf ccache pkg-config \
      libnuma1 libnuma-dev numactl libssl-dev libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Two virtual environments, deliberately separate:
#    /opt/infb/tools  only what this recipe itself runs (huggingface_hub for the checkpoint);
#    /opt/vllm        the engine's, filled from the locks below and nothing else.
#    Only the .lock files enter the image, never lock/resolve.sh or lock/RESOLVED.md: those are
#    provenance for whoever re-resolves, and a resolver script is a natural place for a reference
#    commit to hide. Only the tools lock is copied here, because the checkpoint fetch below needs it;
#    the engine's locks are copied where they are installed, AFTER the checkpoint layers, so a lock
#    change no longer re-downloads 98 GB.
COPY lock/requirements-tools.lock /opt/infb/lock/
RUN python3.12 -m venv /opt/infb/tools \
    && /opt/infb/tools/bin/pip install --no-cache-dir --require-hashes -r /opt/infb/lock/requirements-tools.lock \
    && python3.12 -m venv /opt/vllm
ENV PATH=/opt/vllm/bin:${PATH}

# 3. The checkpoint from the Hub at the pinned revision, into the image's Hugging Face cache. The
#    mandatory upstream test this task freezes is `tests/models/test_initialization.py`, whose
#    registry entry names the repository id `moonshotai/Kimi-Linear-48B-A3B-Instruct` rather than a
#    path, and the recipe the instruction gives the agent names the same id -- so the cache layout
#    plus HF_HUB_OFFLINE=1 is what lets that test run VERBATIM offline and the recipe serve as
#    written. /model below is a symlink onto the same snapshot for the instruction's sake.
#    One layer per group of shards (20 groups, largest about 5.0 GB): registries cap a single blob
#    upload, and Docker's overlay2 allows 128 layers. Every file is verified against
#    model-manifest.json (size, and the Hub's sha256 for the 21 LFS files) and the digests computed
#    here are recorded at $HF_HUB_CACHE/.infb/manifest.json for preflight.
ENV HF_HOME=/hf HF_HUB_CACHE=/hf/hub
COPY model-manifest.json fetch_model.py /opt/infb/
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 0 --of 20
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 1 --of 20
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 2 --of 20
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 3 --of 20
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 4 --of 20
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 5 --of 20
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 6 --of 20
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 7 --of 20
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 8 --of 20
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 9 --of 20
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 10 --of 20
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 11 --of 20
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 12 --of 20
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 13 --of 20
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 14 --of 20
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 15 --of 20
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 16 --of 20
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 17 --of 20
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 18 --of 20
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 19 --of 20 --finalize

# 4. vLLM's dependencies at the base commit, hash-locked from the public indexes (lock/RESOLVED.md):
#    requirements/cuda.txt (which pulls in common.txt) and requirements/build.txt, so the
#    editable install below needs no build isolation and no network, plus pytest and tblib for the
#    vendored upstream test's conftest chain, and transformers pinned at 4.57.1 (D-LOCK-2).
#    flashinfer is a SECOND lock, installed in its own layer, because its declared
#    apache-tvm-ffi pre-release has been withdrawn from PyPI and it can only be pinned
#    dependency-free (D-LOCK-1). Packages the engine install alone needs stay LATE so they cannot
#    invalidate the checkpoint layers above.
COPY lock/requirements.lock /opt/infb/lock/
RUN pip install --no-cache-dir --retries 10 --timeout 120 --require-hashes --no-deps \
      --index-url https://pypi.org/simple \
      --extra-index-url https://download.pytorch.org/whl/cu129 \
      -r /opt/infb/lock/requirements.lock \
    && python -c "import torch; print('torch', torch.__version__)"

# 4b. flashinfer, in its own layer, with its lock copied in its OWN COPY -- measured: sharing one
#     COPY with the engine lock meant a change to either invalidated the 5-10 minute engine
#     install as well.
COPY lock/requirements-flashinfer.lock /opt/infb/lock/
# 4b. flashinfer. Its own declared dependency no longer exists, so it is installed
#     dependency-free against the surviving apache-tvm-ffi; whether it IMPORTS is asserted here, and
#     whether the recipe still serves with it present is measured when the oracle serves.
RUN pip install --no-cache-dir --retries 10 --timeout 120 --require-hashes --no-deps \
      --index-url https://pypi.org/simple \
      --extra-index-url https://flashinfer.ai/whl \
      -r /opt/infb/lock/requirements-flashinfer.lock \
    && python -c "import flashinfer; print('flashinfer', flashinfer.__version__)"

# 5. The four CMake externals at the revisions this commit pins, cloned into the image so the
#    VERIFIER's offline rebuild of a candidate tree needs no network. Each is handed to CMake by the
#    environment variable its own cmake file reads, so the build never consults a remote:
#    VLLM_CUTLASS_SRC_DIR (CMakeLists.txt:288), VLLM_FLASH_ATTN_SRC_DIR, FLASH_MLA_SRC_DIR and
#    QUTLASS_SRC_DIR (cmake/external_projects/*.cmake). qutlass builds only for 12.0a and 10.0a, so it
#    is SKIPPED at 9.0a and is cloned only because its cmake file is still evaluated.
#    In the SAME RUN, every Git store under /opt/infb/externals -- the four clones and
#    their submodules' stores -- is reduced to its checked-out commit (buildcheck/reduce_git_stores.py,
#    bind-mounted and pinned by sha256), so no layer holds an external's other history; the CMake builds
#    read the checked-out sources through the variables below and never consult a store. The task layer's
#    dependency store audit (buildcheck/dependency_stores.py) reads every object store on the image.
RUN --mount=type=bind,source=buildcheck,target=/infb-buildcheck \
    mkdir -p /opt/infb/externals && cd /opt/infb/externals \
    && git clone --depth 1 --branch ${CUTLASS_REVISION} https://github.com/nvidia/cutlass.git cutlass \
    && git clone https://github.com/vllm-project/flash-attention.git vllm-flash-attn \
    && git -C vllm-flash-attn checkout -q ${FLASH_ATTN_REVISION} \
    && git clone https://github.com/vllm-project/FlashMLA flashmla \
    && git -C flashmla checkout -q ${FLASHMLA_REVISION} \
    && git clone https://github.com/IST-DASLab/qutlass.git qutlass \
    && git -C qutlass checkout -q ${QUTLASS_REVISION} \
    && git -C vllm-flash-attn submodule update --init --recursive \
    && git -C flashmla submodule update --init --recursive \
    && echo "3d51b52a1618a9e6665633e3a5ffd804e7c017f6ac413123f8e4cbd68e82b3a2  /infb-buildcheck/reduce_git_stores.py" \
       | sha256sum -c - \
    && python3 /infb-buildcheck/reduce_git_stores.py /opt/infb/externals \
    && for d in cutlass vllm-flash-attn flashmla qutlass; do \
         printf '%-18s %s\n' "$d" "$(git -C $d rev-parse HEAD)"; done
ENV VLLM_CUTLASS_SRC_DIR=/opt/infb/externals/cutlass \
    VLLM_FLASH_ATTN_SRC_DIR=/opt/infb/externals/vllm-flash-attn \
    FLASH_MLA_SRC_DIR=/opt/infb/externals/flashmla \
    QUTLASS_SRC_DIR=/opt/infb/externals/qutlass \
    FETCHCONTENT_BASE_DIR=/opt/infb/fc \
    CCACHE_DIR=/opt/infb/ccache \
    CCACHE_MAXSIZE=8G

# 6. Copy only the clean prepared tree, never the reconstruction inputs or upstream history.
WORKDIR /app
COPY --from=source-preparation /prepared/ /app/
# Preserve the integration-only image's version despite replacing its Git history.
# setuptools-scm still writes _version.py; setup.py adds the same CUDA suffix as before.
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.11.1rc5.dev52+g1994de99e
RUN test "$(git rev-parse HEAD)" = "${BASE_SHA}" \
    && test "$(git rev-parse 'HEAD^{tree}')" = "${BASE_TREE}" \
    && test "$(git rev-list --all --count)" = 1 \
    && test -z "$(git status --porcelain)"

# 7. Editable install of the tree without build isolation (the build backend is in the lock) and
#    without touching the index, so exactly this step is what the verifier re-runs on the candidate's
#    tree. The CUDA extensions are compiled here; .gitignore covers every artefact they leave
#    (/.deps/, *.so, vllm/vllm_flash_attn/*, /vllm/_version.py), so the tree must still be clean
#    afterwards. The prints are separated from the assertions by design: a wrong assertion then costs
#    seconds against the cached compile instead of the whole build.
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    VLLM_TARGET_DEVICE=cuda \
    MAX_JOBS=64 \
    NVCC_THREADS=4 \
    CMAKE_BUILD_PARALLEL_LEVEL=64
RUN cd /app \
    && pip install --no-cache-dir --no-deps --no-build-isolation -e . \
    && python -c "import vllm; print('vllm', vllm.__version__, 'from', vllm.__file__)" \
    && python -c "import glob; so=sorted(glob.glob('/app/vllm/**/*.so', recursive=True)); print('compiled extensions:', len(so)); [print('   ', s) for s in so]" \
    && ccache --show-stats | head -8 \
    && git status --porcelain | head -50

# 7b. The assertions on what step 7 produced, in their own layer.
#
#     The names are read from `setup.py` AT THIS COMMIT and not carried from another task: here
#     `_build_custom_ops()` is true on CUDA, so `vllm._C` IS built, and the Hopper set is
#     _C, _moe_C, _flashmla_C, _flashmla_extension_C, cumem_allocator and the two flash-attention
#     extensions -- seven, which is what this tree builds. (A later vLLM renamed the first
#     two to `_C_stable_libtorch` and `_moe_C_stable_libtorch`; that is not this tree, and a recipe
#     written from the other one gets it wrong exactly once.) `_qutlass_C` is deliberately absent:
#     qutlass builds only for 12.0a and 10.0a.
#
#     This layer asserts EXISTENCE AND NAMES and does not IMPORT them: a `podman build` container has
#     no GPU and therefore no libcuda.so.1, so importing a driver-linked extension fails here for a
#     reason that says nothing about the image. The imports are asserted by the image smoke on four
#     devices and again by the verifier's own L0 node at grading time.
RUN cd /app \
    && python -c "import pathlib,sys; \
want={'_C','_moe_C','cumem_allocator','_flashmla_C','_flashmla_extension_C'}; \
got={p.name.split('.')[0] for p in pathlib.Path('/app/vllm').glob('*.so')}; \
fa=sorted(p.name for p in pathlib.Path('/app/vllm/vllm_flash_attn').glob('_vllm_fa*_C*.so')); \
ver=pathlib.Path('/app/vllm/_version.py'); \
missing=sorted(want-got); \
print('cuda extensions present:', sorted(got)); print('flash-attention:', fa); print('_version.py:', ver.is_file()); \
sys.exit(0 if not missing and fa and ver.is_file() else print('FATAL: missing', missing, 'flash-attn', fa, 'version', ver.is_file()) or 7)" \
    && test -z "$(git status --porcelain)"

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1 \
    VLLM_NO_USAGE_STATS=1 \
    MODELSCOPE_CACHE=/hf/modelscope

# 9. /model is a symlink onto the checkpoint snapshot in the Hub cache, so the instruction can name
#    one short path while the upstream test resolves the repository id. Every agent-facing material lives in the tracked evaluation tree /app/eval/kimi_linear/, which the
#    task layer below adds with the base commit; nothing is shipped under /public any more.
RUN ln -s /hf/hub/models--moonshotai--Kimi-Linear-48B-A3B-Instruct/snapshots/e1df551a447157d4658b573f9a695d57658590e9 /model \
    && test -f /model/config.json \
    && test -f /model/modeling_kimi.py \
    && test -f /model/configuration_kimi.py

# 9b. The evaluator's datasets, from their public source and pinned by sha256 rather than committed
#     into the task directory twice. They back the tracked symlink eval/kimi_linear/datasets/gsm8k.
#     The names are the ones the evaluator itself caches (`/tmp/<basename of the URL>`), so copying
#     these two files into /tmp is the whole of running it offline -- which is why the evaluator
#     needed no edit (see /app/eval/kimi_linear/tests/upstream/PROVENANCE.md).
RUN mkdir -p /opt/infb/datasets/gsm8k \
    && curl -fsSL --retry 10 -o /tmp/train.jsonl \
         https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl \
    && curl -fsSL --retry 10 -o /tmp/test.jsonl \
         https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl \
    && echo "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465  /tmp/train.jsonl" | sha256sum -c - \
    && echo "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14  /tmp/test.jsonl" | sha256sum -c - \
    && mv /tmp/train.jsonl /tmp/test.jsonl /opt/infb/datasets/gsm8k/ \
    && test "$(wc -l < /opt/infb/datasets/gsm8k/test.jsonl)" = "1319" \
    && test "$(wc -l < /opt/infb/datasets/gsm8k/train.jsonl)" = "7473"

COPY scripts/start_agent_timer.sh /app/scripts/start_agent_timer.sh

# 11. The engine stage's final assertions, and a neutral git identity so a plain `git commit` in /app
#     cannot fail with "Author identity unknown" on a Docker-default hostname; it writes
#     /etc/gitconfig, not the tree. The engine stage keeps the adapted tree's own clean root commit;
#     the task layer re-roots it with the evaluation materials.
RUN git config --system user.name agent \
    && git config --system user.email agent@localhost \
    && mkdir -p /logs/artifacts /logs/verifier \
    && test -z "$(git -C /app status --porcelain)" \
    && test "$(git -C /app rev-parse HEAD)" = "${BASE_SHA}" \
    && /opt/vllm/bin/python -c "import transformers; assert transformers.__version__ == '4.57.1', transformers.__version__; print('transformers', transformers.__version__)" \
    && grep -q '^flashinfer-python==0.4.1$' /app/requirements/cuda.txt \
    && echo "requirements/cuda.txt is upstream's own: the flashinfer pin is intact"

# 12. The engine venv, declared. On the default launch path the runtime refuses an image it cannot
#     prove carries an engine, and it accepts either an `# infb-recipe:` marker or this declaration;
#     the separate-verifier guard requires it of every Day-0 task. It is the truth about this image:
#     the engine interpreter is /opt/vllm.
ENV INFB_VENV=/opt/vllm
WORKDIR /app
CMD ["/bin/bash"]

# ── the task layer ───────────────────────────────────────────────────────────────────
# The agent-facing evaluation materials and the task's base commit. The
# engine stage above is the compiled adapted tree; this layer adds the tracked evaluation tree
# /app/eval/kimi_linear/ and the three-line upstream test-registry entry, and re-roots /app as ONE
# parentless commit over exactly those files (baseprep/prepare_base.py; it asserts the tree and
# commit it produces). Its inputs live outside sourceprep/ so the engine's source stage, and with it
# the compile, keeps its cache when they change. It sits after the engine on purpose: an edit to the evaluation materials never
# invalidates the engine compile. tests/Dockerfile is this whole file plus the
# tests, so the verifier's /app is the same tree by construction.
#
# One step, BIND-MOUNTED inputs, nothing copied from the build context: every Dockerfile step on an
# image this size costs a fixed commit overhead, and the answer-address scan has to carry the very
# strings it forbids, so a copy of it in the image would be
# the leak it guards. The scan covers the task-authored files the image ships (/opt/infb,
# /app/eval, the checkpoint's text files), HEAD's commit and the whole object store. The
# no-reference-implementation check runs again over the final image's interpreters.
FROM ${ENGINE}
RUN --mount=type=bind,source=baseprep,target=/infb-baseprep \
    --mount=type=bind,source=eval,target=/infb-eval \
    --mount=type=bind,source=checks,target=/infb-checks \
    --mount=type=bind,source=buildcheck,target=/infb-buildcheck \
    python3 /infb-baseprep/prepare_base.py \
    && mkdir -p /opt/infb/checks \
    && cp /infb-checks/no_reference_impl.py /opt/infb/checks/ \
    && /opt/vllm/bin/python /opt/infb/checks/no_reference_impl.py \
    && python3 /infb-buildcheck/no_answer_address.py
WORKDIR /app
