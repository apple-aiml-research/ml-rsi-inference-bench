#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# infb-engine: task-authored
#
# The task image for support_deepseek_v4_flash_4xb200_sglang, in DeepSWE's git time-travel idiom.
# Every input is public: the nvidia/cuda base by digest, hash-locked wheels from PyPI, the PyTorch
# cu130 index and flashinfer.ai (lock/, resolved once by lock/resolve.sh), FlashMLA and the CUTLASS
# it builds against at pinned commits from GitHub, the SGLang tree at the base commit from GitHub,
# and the checkpoint from the Hugging Face Hub at its pinned revision (model-manifest.json). The
# build needs network; the trial does not.
#
# With NO build argument, ENGINE defaults to the engine stage below, so docker build on this
# directory alone produces the runnable image from public sources.
ARG ENGINE=engine

FROM docker.io/nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04@sha256:8b2705ea7a8653ad3451b46ab835eced92d77b44e671b9cf3ad4f95fbb2efe5e AS vendor
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git python3 \
    && rm -rf /var/lib/apt/lists/*
COPY kernel-materials.json prepare_kernel_materials.py /opt/infb/
ARG FLASH_MLA_COMMIT=9241ae3ef9bac614dd25e45e507e089f888280e0
ARG CUTLASS_COMMIT=147f5673d0c1c3dcf66f78d677fd647e4a020219
RUN git clone https://github.com/deepseek-ai/FlashMLA.git /tmp/flash-mla \
    && git -C /tmp/flash-mla checkout -q ${FLASH_MLA_COMMIT} \
    && git init -q /tmp/cutlass \
    && git -C /tmp/cutlass remote add origin https://github.com/NVIDIA/cutlass.git \
    && git -C /tmp/cutlass fetch -q --depth 1 origin ${CUTLASS_COMMIT} \
    && git -C /tmp/cutlass checkout -q FETCH_HEAD \
    && python3 /opt/infb/prepare_kernel_materials.py --manifest /opt/infb/kernel-materials.json \
         --flashmla /tmp/flash-mla --cutlass /tmp/cutlass --variant supplied \
         --out-flashmla /opt/infb-vendor/flash_mla --out-cutlass /opt/infb-deps/flashmla-cutlass \
         --record /opt/infb-deps/kernel-materials-record.json \
    && rm -rf /tmp/flash-mla /tmp/cutlass

FROM docker.io/nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04@sha256:8b2705ea7a8653ad3451b46ab835eced92d77b44e671b9cf3ad4f95fbb2efe5e AS engine

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl wget git git-lfs lsof netcat-openbsd kmod unzip locales procps \
      python3.12-full python3.12-dev python3.12-venv \
      build-essential cmake ninja-build perl patchelf ccache pkg-config \
      libopenmpi-dev libnuma1 libnuma-dev numactl \
      libibverbs-dev libibverbs1 libibumad3 librdmacm1 libnl-3-200 libnl-route-3-200 \
      libnl-3-dev libnl-route-3-dev ibverbs-providers rdma-core infiniband-diags \
      libssl-dev libcurl4-openssl-dev \
    && ln -sf /usr/lib/x86_64-linux-gnu/libmlx5.so.1 /usr/lib/x86_64-linux-gnu/libmlx5.so \
    && rm -rf /var/lib/apt/lists/*

COPY lock/requirements-tools.lock /opt/infb/lock/requirements-tools.lock
RUN python3.12 -m venv /opt/infb/tools \
    && /opt/infb/tools/bin/pip install --no-cache-dir --require-hashes -r /opt/infb/lock/requirements-tools.lock \
    && python3.12 -m venv /opt/sglang
ENV PATH=/opt/sglang/bin:/root/.cargo/bin:${PATH}

RUN curl --proto '=https' --tlsv1.2 -sSf --retry 3 https://sh.rustup.rs \
      | sh -s -- -y --no-modify-path --profile minimal --default-toolchain 1.90 \
    && rustc --version && cargo --version

# 4. The checkpoint from the Hub at the pinned revision, in the layout upstream's own test helper
#    resolves offline (DEFAULT_MODEL_CACHE_DIR/<repo id>, read by sglang.test.test_utils's
#    try_cached_model, which the registered acceptance test calls). One layer per group of 46: the
#    largest safetensors shard is 3.60 GB, and 46 layers leave the
#    whole image far below Docker overlay2's 128-layer limit that the public path must respect.
#    Every file is verified against model-manifest.json (size, and the Hub's sha256 for the LFS
#    files); the digests computed here are recorded at /model/.infb/manifest.json for preflight.
COPY model-manifest.json fetch_model.py /opt/infb/
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 0 --of 46
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 1 --of 46
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 2 --of 46
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 3 --of 46
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 4 --of 46
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 5 --of 46
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 6 --of 46
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 7 --of 46
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 8 --of 46
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 9 --of 46
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 10 --of 46
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 11 --of 46
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 12 --of 46
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 13 --of 46
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 14 --of 46
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 15 --of 46
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 16 --of 46
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 17 --of 46
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 18 --of 46
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 19 --of 46
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 20 --of 46
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 21 --of 46
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 22 --of 46
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 23 --of 46
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 24 --of 46
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 25 --of 46
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 26 --of 46
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 27 --of 46
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 28 --of 46
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 29 --of 46
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 30 --of 46
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 31 --of 46
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 32 --of 46
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 33 --of 46
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 34 --of 46
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 35 --of 46
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 36 --of 46
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 37 --of 46
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 38 --of 46
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 39 --of 46
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 40 --of 46
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 41 --of 46
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 42 --of 46
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 43 --of 46
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 44 --of 46
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 45 --of 46 --finalize

COPY lock/requirements.lock lock/requirements-flashinfer.lock /opt/infb/lock/
RUN pip install --no-cache-dir --require-hashes --no-deps --retries 10 --timeout 120 \
      --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu130 \
      -r /opt/infb/lock/requirements.lock \
    && find /opt/sglang/lib/python3.12/site-packages/sgl_kernel -name 'flashmla_ops*.so' -print -delete \
    && test -z "$(find / -xdev -name '*flashmla_ops*' -not -path '/proc/*' 2>/dev/null)"
RUN pip install --no-cache-dir --require-hashes --no-deps --retries 10 --timeout 120 \
      --index-url https://flashinfer.ai/whl/cu130 --extra-index-url https://flashinfer.ai/whl \
      -r /opt/infb/lock/requirements-flashinfer.lock

COPY --from=vendor /opt/infb-deps/ /opt/infb-deps/
RUN test -e /usr/local/cuda/include/cuda || ln -s /usr/local/cuda/include/cccl/cuda /usr/local/cuda/include/cuda \
    && test -f /opt/infb-deps/flashmla-cutlass/include/cutlass/cutlass.h

WORKDIR /app
ARG BASE_SHA=55224fff08518b2402680a5b1b9189dcc5b27dda
RUN git clone https://github.com/sgl-project/sglang.git . \
    && DEFAULT="$(git remote show origin | sed -n 's/.*HEAD branch: //p')" \
    && git checkout -B "$DEFAULT" "$BASE_SHA" \
    && head="$(git rev-parse HEAD)" \
    && if [ "$head" != "55224fff08518b2402680a5b1b9189dcc5b27dda" ]; then \
         echo "FATAL: engine tree is at $head, not the reviewed 55224fff08518b2402680a5b1b9189dcc5b27dda -- github.com/sgl-project/sglang has moved" >&2; \
         exit 7; \
       fi \
    && git config core.hooksPath /dev/null \
    && git remote remove origin \
    && for b in $(git for-each-ref --format='%(refname:short)' refs/heads | grep -vx "$DEFAULT"); do git branch -D "$b" || true; done \
    && for t in $(git tag); do git merge-base --is-ancestor "$t" HEAD 2>/dev/null || git tag -d "$t"; done \
    && git reflog expire --expire=now --all \
    && git gc --prune=now \
    && printf '.timer/\nscripts/start_agent_timer.sh\n' >> .git/info/exclude \
    && test "$(git rev-parse HEAD)" = "$BASE_SHA" \
    && for blob in 145ab1fb081e879959bde07bbc838bcc1744a98c 3a89e8114ce58b35776bfde401a0596fe83508dc \
                   498f2601eeaa1756cde5911141121a15aa1feeb9 b497470606cfe86ef17056606d5a346295d60c58 \
                   d3953578b925b8c1eac2e82f84d77077601479ce 49f6f5596377f87863c4216109757a666c54182c; do \
         if git cat-file -e "$blob" 2>/dev/null; then echo "FATAL: withheld source blob $blob is recoverable" >&2; exit 8; fi; \
       done
COPY --from=vendor /opt/infb-vendor/flash_mla /app/third_party/flash_mla
COPY kernel-prereq.patch kernel-prereq.msg /opt/infb/
ARG BASE_PRIME_SHA=0a4f56af55881720f8ef6f6589ed55926a647d6d
RUN git apply --binary --whitespace=nowarn /opt/infb/kernel-prereq.patch \
    && git add -A python/sglang/jit_kernel third_party/flash_mla \
    && test "$(git ls-files third_party/flash_mla | wc -l)" = "$(find third_party/flash_mla -type f -o -type l | wc -l)" \
    && GIT_AUTHOR_NAME="Inference-Bench" GIT_AUTHOR_EMAIL="inference-bench@local" \
       GIT_COMMITTER_NAME="Inference-Bench" GIT_COMMITTER_EMAIL="inference-bench@local" \
       GIT_AUTHOR_DATE="2026-05-08T01:28:23+0000" GIT_COMMITTER_DATE="2026-05-08T01:28:23+0000" \
       git commit -q --no-verify -F /opt/infb/kernel-prereq.msg \
    && test "$(git rev-parse HEAD)" = "$BASE_PRIME_SHA" \
    && test -z "$(git diff --name-only $BASE_SHA HEAD -- . ':!python/sglang/jit_kernel' ':!third_party/flash_mla')" \
    && git reflog expire --expire=now --all \
    && test -z "$(git status --porcelain)" \
    && test "$(git rev-parse HEAD)" = "$BASE_PRIME_SHA"

RUN apt-get update && apt-get install -y --no-install-recommends \
      libprotobuf-dev protobuf-compiler protobuf-compiler-grpc \
    && rm -rf /var/lib/apt/lists/* \
    && protoc --version

RUN cd /app/python \
    && pip install --no-cache-dir --no-deps --no-build-isolation -e . \
    && cd /app \
    && kernels lock python \
    && (kernels download python || echo "WARNING: hub kernel download failed; kernels will be JIT-compiled at runtime") \
    && mkdir -p /root/.cache/sglang \
    && (test -f python/kernels.lock && mv python/kernels.lock /root/.cache/sglang/ || true) \
    && python -c "import sglang, sglang.srt.models.registry as r; print('sglang from', sglang.__file__)" \
    && test -z "$(git status --porcelain)" \
    && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PIP_NO_INDEX=1 CARGO_NET_OFFLINE=true \
       pip install --no-cache-dir --no-deps --no-build-isolation --no-index -e /app/python \
    && test -z "$(git status --porcelain)"

COPY eval/deepseek_v4_flash/limits.json /opt/infb/build-helper/deepseek_v4_flash/limits.json
COPY eval/deepseek_v4_flash/kernels/build.py /opt/infb/build-helper/deepseek_v4_flash/kernels/build.py
RUN python /opt/infb/build-helper/deepseek_v4_flash/kernels/build.py --tree /app --vendor-only \
    && echo /app/third_party/flash_mla > /opt/sglang/lib/python3.12/site-packages/infb-flash-mla.pth \
    && cd / && python -c "import flash_mla, flash_mla.cuda; assert flash_mla.__file__.startswith('/app/third_party/flash_mla/'), flash_mla.__file__; print('flash_mla from', flash_mla.__file__)" \
    && cd /app && test -z "$(git status --porcelain)"

ENV DEFAULT_MODEL_CACHE_DIR=/model \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    CARGO_NET_OFFLINE=true \
    LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:/usr/local/cuda-13.0/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# A NEUTRAL GIT IDENTITY, for the committed-tree contract. With a Docker-default hostname
# `git commit` fails, "Author identity unknown" (rc 128), so on `--env docker` an agent's first
# commit would fail. System-wide, after the preparation commits so their hashes do not move (both
# are made under a pinned identity regardless).
ARG BASE_EVAL_SHA=eac98398e7adc70dd10d8db3ff77226b026000ce
COPY eval/deepseek_v4_flash/ /app/eval/deepseek_v4_flash/
COPY eval-materials.msg /opt/infb/
COPY scripts/start_agent_timer.sh /app/scripts/start_agent_timer.sh
RUN chmod 755 /app/scripts/start_agent_timer.sh \
    && mkdir -p /logs/artifacts /logs/verifier \
    && test -d /usr/local/cuda-13.0/compat \
    && curl -fsSL --retry 3 -o /tmp/gsm8k_test.jsonl \
      https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl \
    && echo "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14  /tmp/gsm8k_test.jsonl" | sha256sum -c - \
    && mkdir -p /app/eval/deepseek_v4_flash/tests/fixtures \
    && mv /tmp/gsm8k_test.jsonl /app/eval/deepseek_v4_flash/tests/fixtures/gsm8k_test.jsonl \
    && chmod 644 /app/eval/deepseek_v4_flash/tests/fixtures/gsm8k_test.jsonl \
    && cd /app \
    && test "$(git rev-parse HEAD)" = "$BASE_PRIME_SHA" \
    && git add -f eval/deepseek_v4_flash \
    && GIT_AUTHOR_NAME="Inference-Bench" GIT_AUTHOR_EMAIL="inference-bench@local" \
       GIT_COMMITTER_NAME="Inference-Bench" GIT_COMMITTER_EMAIL="inference-bench@local" \
       GIT_AUTHOR_DATE="2026-05-08T01:28:24+0000" GIT_COMMITTER_DATE="2026-05-08T01:28:24+0000" \
       git commit -q --no-verify -F /opt/infb/eval-materials.msg \
    && echo "eval materials commit: $(git rev-parse HEAD)" \
    && test "$(git rev-parse HEAD)" = "$BASE_EVAL_SHA" \
    && test "$(git ls-files eval/deepseek_v4_flash | wc -l)" = "$(find eval/deepseek_v4_flash -type f | wc -l)" \
    && test -z "$(git diff --name-only $BASE_PRIME_SHA HEAD -- . ':!eval/deepseek_v4_flash')" \
    && test -z "$(git status --porcelain)" \
    && git config --system user.name agent && git config --system user.email agent@localhost
ENV INFB_VENV=/opt/sglang

WORKDIR /app
CMD ["/bin/bash"]

# The task image. With no build argument this is the stage above, built from public sources; with
# --build-arg ENGINE=<ref> it starts from that prebuilt image and the stage above is pruned (see the
# header). Its one line takes the forward-compatibility directory off the runtime library path.
FROM ${ENGINE}
ENV LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# infb-verifier-tail
ENV CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda

# The human reference's tree, built here from the still-pristine image and never at grading time:
# a shared clone of /app at the base, tests/reference/solution.patch (solution/solution.patch byte
# for byte) applied, its tree checked against the closure tree oracle.lock records, and the engine's
# git-ignored build products under python/sglang copied with their digests. test.sh's reference step
# checks all of it again before it serves the reference. /app is not written.
COPY . /tests/
RUN chmod +x /tests/test.sh && python3 /tests/infb_efficiency/snapshot.py --config /tests/config.json
