#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# infb-engine: task-authored
ARG ENGINE=engine

FROM docker.io/nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04@sha256:a85c9f5af049f0ab679c1669ae6fa8393022886739af7361e85bb96878e8cdd4 AS engine

ARG BASE_SHA=a9944aec01e281f5c12c9651f981a53bdece6989
ARG GDRCOPY_VERSION=2.5.1
ARG RUST_TOOLCHAIN=1.92

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

COPY lock/ /opt/infb/lock/
RUN python3.12 -m venv /opt/infb/tools \
    && /opt/infb/tools/bin/pip install --no-cache-dir --require-hashes -r /opt/infb/lock/requirements-tools.lock \
    && python3.12 -m venv /opt/sglang
ENV PATH=/opt/sglang/bin:/root/.cargo/bin:${PATH}

RUN curl -fsSL --retry 3 -o /tmp/gdrcopy.tar.gz https://github.com/NVIDIA/gdrcopy/archive/refs/tags/v${GDRCOPY_VERSION}.tar.gz \
    && tar -xzf /tmp/gdrcopy.tar.gz -C /tmp \
    && make -C /tmp/gdrcopy-${GDRCOPY_VERSION} prefix=/usr/local CUDA=/usr/local/cuda lib lib_install \
    && ldconfig \
    && rm -rf /tmp/gdrcopy.tar.gz /tmp/gdrcopy-${GDRCOPY_VERSION}

RUN curl --proto '=https' --tlsv1.2 -sSf --retry 3 https://sh.rustup.rs \
      | sh -s -- -y --no-modify-path --profile minimal --default-toolchain ${RUST_TOOLCHAIN} \
    && rustc --version && cargo --version

COPY model-manifest.json fetch_model.py /opt/infb/
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 0 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 1 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 2 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 3 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 4 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 5 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 6 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 7 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 8 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 9 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 10 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 11 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 12 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 13 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 14 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 15 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 16 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 17 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 18 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 19 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 20 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 21 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 22 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 23 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 24 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 25 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 26 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 27 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 28 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 29 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 30 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 31 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 32 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 33 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 34 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 35 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 36 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 37 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 38 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 39 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 40 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 41 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 42 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 43 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 44 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 45 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 46 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 47 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 48 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 49 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 50 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 51 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 52 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 53 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 54 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 55 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 56 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 57 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 58 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 59 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 60 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 61 --of 63
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 62 --of 63 --finalize

RUN pip install --no-cache-dir --require-hashes --no-deps \
      --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu130 \
      -r /opt/infb/lock/requirements.lock \
    && pip install --no-cache-dir --require-hashes --no-deps \
      --index-url https://flashinfer.ai/whl/cu130 --extra-index-url https://flashinfer.ai/whl \
      -r /opt/infb/lock/requirements-flashinfer.lock

WORKDIR /app
RUN git clone https://github.com/sgl-project/sglang.git . \
    && DEFAULT="$(git remote show origin | sed -n 's/.*HEAD branch: //p')" \
    && git checkout -B "$DEFAULT" "$BASE_SHA" \
    && git remote remove origin \
    && for b in $(git for-each-ref --format='%(refname:short)' refs/heads | grep -vx "$DEFAULT"); do git branch -D "$b" || true; done \
    && for t in $(git tag); do git merge-base --is-ancestor "$t" HEAD 2>/dev/null || git tag -d "$t"; done \
    && git reflog expire --expire=now --all \
    && git gc --prune=now \
    && git config core.hooksPath /dev/null \
    && printf '.timer/\nscripts/start_agent_timer.sh\n' >> .git/info/exclude \
    && test "$(git rev-parse HEAD)" = "$BASE_SHA"

RUN cd /app/python \
    && pip install --no-cache-dir --no-deps --no-build-isolation -e . \
    && cd /app \
    && kernels lock python \
    && (kernels download python || echo "WARNING: hub kernel download failed; kernels will be JIT-compiled at runtime") \
    && mkdir -p /root/.cache/sglang \
    && (test -f python/kernels.lock && mv python/kernels.lock /root/.cache/sglang/ || true) \
    && python -c "import sglang, sglang.srt.models.registry as r; print('sglang from', sglang.__file__)" \
    && test -z "$(git status --porcelain)"

ENV DEFAULT_MODEL_CACHE_DIR=/model \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    SGLANG_RUST_BUILD_MODE=never \
    CARGO_NET_OFFLINE=true \
    LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:/usr/local/cuda-13.0/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

ARG PREPARED_BASE_SHA=9f6b624b4b4460f4b62aed3297cd2574dbf02b92

COPY public/ /app/eval/glm53_flash/
RUN curl -fsSL --retry 3 -o /tmp/gsm8k_test.jsonl \
      https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl \
    && echo "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14  /tmp/gsm8k_test.jsonl" | sha256sum -c - \
    && mkdir -p /app/eval/glm53_flash/tests/data/gsm8k \
    && mv /tmp/gsm8k_test.jsonl /app/eval/glm53_flash/tests/data/gsm8k/test.jsonl \
    && cp /app/eval/glm53_flash/tests/data/gsm8k/test.jsonl /tmp/test.jsonl \
    && find /app/eval \( -name __pycache__ -o -name '*.pyc' \) -prune -exec rm -rf {} + \
    && rm -f /app/eval/glm53_flash/instruction.md \
    && cd /app \
    && git add --chmod=-x eval \
    && GIT_AUTHOR_NAME='Inference-Bench' GIT_AUTHOR_EMAIL='inference-bench@local' \
       GIT_COMMITTER_NAME='Inference-Bench' GIT_COMMITTER_EMAIL='inference-bench@local' \
       GIT_AUTHOR_DATE='1790156485 +0000' GIT_COMMITTER_DATE='1790156485 +0000' \
       git commit --quiet --no-verify -m "Add eval/glm53_flash: checks and measurement tools for GLM-5.3-Flash" \
    && git reflog expire --expire=now --all \
    && head="$(git rev-parse HEAD)" \
    && if [ "$head" != "$PREPARED_BASE_SHA" ]; then \
         echo "FATAL: prepared base is $head, expected $PREPARED_BASE_SHA" >&2; exit 7; fi \
    && test "$(git rev-parse HEAD^)" = "$BASE_SHA" \
    && test -z "$(git diff --name-only "$BASE_SHA" HEAD -- . ':(exclude)eval')" \
    && test -z "$(git status --porcelain)"
COPY scripts/start_agent_timer.sh /app/scripts/start_agent_timer.sh
RUN chmod 755 /app/scripts/start_agent_timer.sh \
    && mkdir -p /logs/artifacts /logs/verifier \
    && test -d /usr/local/cuda-13.0/compat \
    && test -z "$(git -C /app status --porcelain)"
RUN git config --system user.name agent \
    && git config --system user.email agent@localhost
ENV INFB_VENV=/opt/sglang

WORKDIR /app
CMD ["/bin/bash"]

FROM ${ENGINE}
WORKDIR /app
ENV LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# infb-verifier-tail
ENV CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda

COPY . /tests/
COPY public/glm53_protocols/ /tests/glm53_protocols/
RUN chmod +x /tests/test.sh && test -s /tests/reference/solution.patch

RUN python3 /tests/infb_efficiency/snapshot.py --config /tests/config.json
