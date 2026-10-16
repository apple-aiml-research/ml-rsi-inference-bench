#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Public, pinned recipe. Only sanitized dependencies and prepared source enter
# the engine stage; original wheels, source archives and Git history stay in
# intermediate stages. The default build requires no private registry.
# infb-engine: task-authored
ARG ENGINE=engine
FROM docker.io/nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04@sha256:0f8250615943f311785f9ce6379a49520a4b53c124d22b42ba859edf93af3991 AS common
ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl wget git git-lfs lsof netcat-openbsd unzip locales procps \
      python3.12-full python3.12-dev python3.12-venv \
      build-essential cmake ninja-build pkg-config \
      libnuma1 libnuma-dev numactl libibverbs1 libnl-3-200 libnl-route-3-200 \
    && rm -rf /var/lib/apt/lists/*
COPY lock/requirements-tools.lock /opt/infb/lock/requirements-tools.lock
RUN python3.12 -m venv /opt/infb/tools \
    && /opt/infb/tools/bin/pip install --no-cache-dir --require-hashes -r /opt/infb/lock/requirements-tools.lock
ENV HF_HUB_CACHE=/model-cache
COPY model-manifest.json fetch_model.py /opt/infb/
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 0 --of 16
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 1 --of 16
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 2 --of 16
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 3 --of 16
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 4 --of 16
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 5 --of 16
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 6 --of 16
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 7 --of 16
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 8 --of 16
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 9 --of 16
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 10 --of 16
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 11 --of 16
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 12 --of 16
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 13 --of 16
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 14 --of 16
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 15 --of 16 --finalize

FROM common AS dependencies
COPY lock/requirements.lock /opt/infb/lock/requirements.lock
RUN python3.12 -m venv /opt/sglang \
    && /opt/sglang/bin/pip install --no-cache-dir --require-hashes --retries 10 --timeout 120 \
      --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu126 \
      -r /opt/infb/lock/requirements.lock
COPY sanitize_dependencies.py /tmp/sanitize_dependencies.py
RUN /opt/sglang/bin/python /tmp/sanitize_dependencies.py --record /tmp/dependency-sanitization.json

FROM common AS source
ARG SOURCE_COMMIT=556e4143f0ba8e480a73fc0e08f05d7ea83880cc
RUN git init /source-fetch \
    && git -C /source-fetch fetch --depth=1 https://github.com/sgl-project/sglang.git "$SOURCE_COMMIT" \
    && test "$(git -C /source-fetch rev-parse FETCH_HEAD^{tree})" = 85bd0aca6e92e6c36430daddeab653768a72b2b3 \
    && mkdir /app \
    && git -C /source-fetch archive FETCH_HEAD | tar -xf - -C /app
COPY prepare-memory-pool.patch /tmp/prepare-memory-pool.patch
RUN git -C /app apply --check /tmp/prepare-memory-pool.patch \
    && git -C /app apply /tmp/prepare-memory-pool.patch \
    && rm /tmp/prepare-memory-pool.patch
COPY preparation.json /opt/infb/preparation.json
COPY kernels/ /app/third_party/triton_kernels/
COPY eval/ /app/eval/
COPY prepare_tree.py /tmp/prepare_tree.py
RUN python3 /tmp/prepare_tree.py

FROM common AS engine
COPY --from=dependencies /opt/sglang/ /opt/sglang/
COPY --from=source /app/ /app/
COPY --from=source /opt/infb/prepared.json /opt/infb/prepared.json
COPY --from=source /opt/infb/preparation.json /opt/infb/preparation.json
COPY capture.py /opt/infb/capture.py
ENV PATH=/opt/sglang/bin:${PATH}
COPY verify_stack.py /tmp/verify_stack.py
RUN pip install --no-cache-dir --no-deps --no-build-isolation -e /app/third_party/triton_kernels -e /app/python \
    && python /tmp/verify_stack.py \
    && rm /tmp/verify_stack.py \
    && test -z "$(git -C /app status --porcelain)"
# Independent mathematics and format documentation are supplied as reading material.
COPY public/gpt_oss/ /public/gpt_oss/
COPY public/harmony/ /public/harmony/
COPY public/measure_serving.py /public/measure_serving.py
COPY scripts/start_agent_timer.sh /app/scripts/start_agent_timer.sh
RUN chmod 755 /app/scripts/start_agent_timer.sh \
    && git config --system user.name agent \
    && git config --system user.email agent@localhost \
    && mkdir -p /logs/artifacts /logs/verifier \
    && rm /opt/infb/fetch_model.py \
    && test -L /model && test -f /model/config.json \
    && test -z "$(git -C /app status --porcelain)"
ENV INFB_VENV=/opt/sglang \
    HF_HUB_OFFLINE=1 \
    PYTHONPATH=/app/eval \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_DATASETS_OFFLINE=1 \
    SGLANG_RUST_BUILD_MODE=never \
    CARGO_NET_OFFLINE=true \
    INFB_MODEL_RECORD=/opt/infb/checkpoint.json
WORKDIR /app
CMD ["/bin/bash"]

# The final stage permits an identical prebuilt engine to be supplied by a backend.
FROM ${ENGINE}
WORKDIR /app
