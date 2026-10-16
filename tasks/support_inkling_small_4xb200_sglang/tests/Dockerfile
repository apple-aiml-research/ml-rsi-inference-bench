#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# infb-engine: task-authored
#
# The task image for support_inkling_small_4xb200_sglang, in DeepSWE's git time-travel idiom.
# Every input is public: the nvidia/cuda base by digest, hash-locked wheels from PyPI,
# the PyTorch cu130 index and flashinfer.ai (lock/, resolved once by lock/resolve.sh as of the landing's
# merge), the SGLang tree at the base commit from GitHub, and the checkpoint from the Hugging Face Hub at
# its pinned revision (model-manifest.json). The build needs network; the trial does not.
#
# With NO build argument -- which is what Pier passes on the public path -- `ENGINE` defaults to the
# `engine` stage below, so `docker build` on this directory alone produces the runnable image from
# public sources. The task ships no docker_image.
#
# What the image supplies and withholds: the engine's ordinary
# dependencies at the base's own pins, plain FlashAttention-4 4.0.0b15 among them, plus the two
# generic libraries the model's landing added (Helion 0.2.6, Numba 0.65.1). There is no
# kernel-preparation commit. The model's relative attention and its model-specific kernels are the
# task; build checks (steps 11 and 12) assert that no implementation of them reached the image.
ARG ENGINE=engine

FROM docker.io/nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04@sha256:8b2705ea7a8653ad3451b46ab835eced92d77b44e671b9cf3ad4f95fbb2efe5e AS engine

ARG BASE_SHA=829e9ce9d596cd6dc871319086d0303b30346806

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# 1. System packages: the runtime subset of upstream's docker/Dockerfile (Python 3.12, the toolchain the
#    JIT kernels compile with, the RDMA/InfiniBand userspace the engine's transport probes load even on a
#    single node). DeepEP, NVSHMEM and GDRCopy are not installed: the recipe is TP4 on one node.
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

# 2. Two virtual environments: /opt/infb/tools carries only what this recipe itself runs
#    (huggingface_hub for the checkpoint download); /opt/sglang is the engine's, filled from the lock in
#    step 5 and nothing else. Both hash-locked. Only the tools lock is copied here, so re-resolving the
#    engine lock does not invalidate the checkpoint layers of step 4.
COPY lock/requirements-tools.lock /opt/infb/lock/requirements-tools.lock
RUN python3.12 -m venv /opt/infb/tools \
    && /opt/infb/tools/bin/pip install --no-cache-dir --require-hashes -r /opt/infb/lock/requirements-tools.lock \
    && python3.12 -m venv /opt/sglang
ENV PATH=/opt/sglang/bin:/root/.cargo/bin:${PATH}

# 3. The Rust toolchain the base tree's setuptools-rust extension is built with
#    (`sglang.srt.grpc._core`; rust/sglang-grpc/rust-toolchain.toml pins channel 1.90). It stays at
#    rustup's default /root/.cargo, the path task.toml puts on PATH, so the registry cache this build
#    warms is the one the verifier's offline re-install reads; step 8 proves that offline path.
RUN curl --proto '=https' --tlsv1.2 -sSf --retry 3 https://sh.rustup.rs \
      | sh -s -- -y --no-modify-path --profile minimal --default-toolchain 1.90 \
    && rustc --version && cargo --version

# 4. The checkpoint from the Hub at the pinned revision, in the layout upstream's test helpers resolve
#    offline (DEFAULT_MODEL_CACHE_DIR/<repo id>). One layer per safetensors file -- the nine model shards
#    and mtp.safetensors, the largest 19.97 GB -- and one for the eleven small files; fetch_model.py
#    asserts that shape. Every file is verified against model-manifest.json (size, and the Hub's sha256
#    for the LFS files); the digests computed here are recorded at /model/.infb/manifest.json for
#    preflight.
COPY model-manifest.json fetch_model.py /opt/infb/
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 0 --of 11
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 1 --of 11
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 2 --of 11
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 3 --of 11
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 4 --of 11
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 5 --of 11
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 6 --of 11
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 7 --of 11
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 8 --of 11
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 9 --of 11
RUN /opt/infb/tools/bin/python /opt/infb/fetch_model.py --group 10 --of 11 --finalize

# 5. SGLang's dependency set, hash-locked from the public indexes (lock/RESOLVED.md), resolved from the
#    pyproject of the model's landing: the base's pins plus helion==0.2.6 and numba==0.65.1, the
#    landing's only dependency change, which the image supplies and the instruction states. Then
#    FlashInfer's prebuilt cubins and JIT cache from flashinfer.ai, so no kernel is fetched at runtime.
#    FlashAttention-4 is installed from its wheel only (--only-binary): the lock's first hash is the
#    wheel's, and step 5b proves the installed files are that wheel's.
COPY lock/requirements.lock lock/requirements-flashinfer.lock /opt/infb/lock/
RUN pip install --no-cache-dir --require-hashes --no-deps --retries 10 --timeout 120 \
      --only-binary flash-attn-4 \
      --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu130 \
      -r /opt/infb/lock/requirements.lock
RUN pip install --no-cache-dir --require-hashes --no-deps --retries 10 --timeout 120 \
      --index-url https://flashinfer.ai/whl/cu130 --extra-index-url https://flashinfer.ai/whl \
      -r /opt/infb/lock/requirements-flashinfer.lock

# 5b. The lock facts, asserted: the pins the task's material boundary depends on, and FlashAttention-4
#     4.0.0b15 as exactly the PyPI wheel (sha256 30ad1f66...), every installed file hashed against the
#     wheel's own RECORD.
RUN python -c "import importlib.metadata as m; want = {'nvidia-cutlass-dsl': '4.5.2', 'flash-attn-4': '4.0.0b15', 'transformers': '5.12.1', 'quack-kernels': '0.5.0', 'helion': '0.2.6', 'numba': '0.65.1', 'apache-tvm-ffi': '0.1.11', 'flashinfer-python': '0.6.14', 'sglang-kernel': '0.4.5', 'kernels': '0.14.1'}; got = {k: m.version(k) for k in want}; assert got == want, got; print('pins', got)" \
    && pip download --no-cache-dir --no-deps --only-binary :all: --index-url https://pypi.org/simple \
         --dest /tmp/fa4 flash-attn-4==4.0.0b15 \
    && echo "30ad1f66d251938808d2fe68cca664a3e5c0143359c27079d848617a29b00076  /tmp/fa4/flash_attn_4-4.0.0b15-py3-none-any.whl" | sha256sum -c - \
    && python -c "import base64, csv, hashlib, io, zipfile, importlib.metadata as m; w = zipfile.ZipFile('/tmp/fa4/flash_attn_4-4.0.0b15-py3-none-any.whl'); rows = [r for r in csv.reader(io.StringIO(w.read('flash_attn_4-4.0.0b15.dist-info/RECORD').decode())) if r and r[1]]; d = m.distribution('flash-attn-4'); bad = [p for p, h, _ in rows if 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(d.locate_file(p).read_bytes()).digest()).rstrip(b'=').decode() != h]; assert len(rows) > 10 and not bad, (len(rows), bad[:5]); print('flash-attn-4 4.0.0b15: %d installed files match the pinned wheel' % len(rows))" \
    && rm -rf /tmp/fa4

# 6. The SGLang tree at the base, in DeepSWE's git time-travel idiom: the parent of the model's landing.
#    Afterwards the default branch points at the base with no future history, later branches and tags
#    are gone, the reflog is expired and the store garbage-collected so no later commit can be recovered
#    from history; hooks are disabled. The agent timer's files and the timer script (repository contract,
#    task.toml healthcheck) live under /app and are excluded from git so they never enter a submission.
WORKDIR /app
RUN git clone https://github.com/sgl-project/sglang.git . \
    && DEFAULT="$(git remote show origin | sed -n 's/.*HEAD branch: //p')" \
    && git checkout -B "$DEFAULT" "$BASE_SHA" \
    && head="$(git rev-parse HEAD)" \
    && if [ "$head" != "829e9ce9d596cd6dc871319086d0303b30346806" ]; then \
         echo "FATAL: engine tree is at $head, not the reviewed 829e9ce9d596cd6dc871319086d0303b30346806" >&2; \
         exit 7; \
       fi \
    && git config core.hooksPath /dev/null \
    && git remote remove origin \
    && for b in $(git for-each-ref --format='%(refname:short)' refs/heads | grep -vx "$DEFAULT"); do git branch -D "$b" || true; done \
    && for t in $(git tag); do git merge-base --is-ancestor "$t" HEAD 2>/dev/null || git tag -d "$t"; done \
    && git reflog expire --expire=now --all \
    && git gc --prune=now \
    && printf '.timer/\nscripts/start_agent_timer.sh\n' >> .git/info/exclude \
    && test "$(git rev-parse HEAD)" = "$BASE_SHA"

# 7. The protobuf compiler the Rust extension's build.rs needs (`rust/sglang-grpc/build.rs` runs
#    tonic_build over `proto/`, which shells out to `protoc`). Upstream's own docker/Dockerfile
#    installs the same three packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libprotobuf-dev protobuf-compiler protobuf-compiler-grpc \
    && rm -rf /var/lib/apt/lists/* \
    && protoc --version

# 8. Editable install of the tree without build isolation (the build backend is in the lock), the hub
#    kernels the pyproject declares fetched once so nothing is downloaded at trial time, and two
#    assertions the verifier depends on: the tree is clean afterwards, and the SAME editable install
#    succeeds a second time with the network refused (--no-index, CARGO_NET_OFFLINE, HF_HUB_OFFLINE),
#    which is what infb_verify/test_l0.py runs inside the offline verifier container.
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

# 9. Runtime environment. The library path below names the image's CUDA 13 forward-compatibility
#    driver directory first. The task image's one line after FROM ${ENGINE} takes that directory
#    off the image's own runtime path: on a host whose driver is newer than the compat libcuda, the
#    compat directory shadows the driver and CUDA does not initialize (cuInit 803). The checkpoint resolves offline through DEFAULT_MODEL_CACHE_DIR; the Hub and Cargo
#    are never contacted at trial time.
ENV DEFAULT_MODEL_CACHE_DIR=/model \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    CARGO_NET_OFFLINE=true \
    LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:/usr/local/cuda-13.0/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# 10. THE EVALUATION MATERIALS, THE AUTHOR IMPLEMENTATION FOR READING AND THE AGENT TIMER.
#     /app/eval/inkling_small/ holds the public protocols the grader shares with the agent (the
#     registered test and its helper, the relative-attention operator contract with its plain-PyTorch
#     reference and checking, measuring and attribution code, the serving protocols and the efficiency
#     client), small examples, and the public GSM8K test file (fetched from its public source and pinned
#     by sha256). They are committed on top of the base as one preparation commit, reproduced bit for bit
#     (pinned author, committer, dates and message) with its sha asserted, so the prepared base the agent
#     starts from is the same object wherever this recipe runs. /public/reference-impl/ holds the
#     model-author implementation for reading (Transformers v5.16.1, not importable), with its provenance.
#     The last steps of the stage, and as few as the work allows: every step on an image this size mounts
#     and commits all of its layers.
#
# A NEUTRAL GIT IDENTITY, for the committed-tree contract: with a Docker-default hostname an unconfigured
# `git commit` fails, "Author identity unknown". System-wide, after the preparation commit so its hash
# does not move.
ARG BASE_EVAL_SHA=1183049084a198c8d133edcb843a4e55076ed1ab
COPY eval/inkling_small/ /app/eval/inkling_small/
COPY eval-materials.msg /opt/infb/
COPY public/reference-impl/ /public/reference-impl/
COPY scripts/start_agent_timer.sh /app/scripts/start_agent_timer.sh
RUN chmod 755 /app/scripts/start_agent_timer.sh \
    && mkdir -p /logs/artifacts /logs/verifier \
    && test -d /usr/local/cuda-13.0/compat \
    && echo "4afd169e8a0ae652f0da0f696e8b06aae4c287da7d7ca33cf68dee90764026a8  /public/reference-impl/modeling_inkling.py" | sha256sum -c - \
    && echo "20f2c4923d7cbea5293b08aa2f2b0c41601319b528390ed0abaa4eb2fdca6187  /public/reference-impl/configuration_inkling.py" | sha256sum -c - \
    && curl -fsSL --retry 3 -o /tmp/gsm8k_test.jsonl \
      https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl \
    && echo "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14  /tmp/gsm8k_test.jsonl" | sha256sum -c - \
    && mkdir -p /app/eval/inkling_small/fixtures \
    && mv /tmp/gsm8k_test.jsonl /app/eval/inkling_small/fixtures/gsm8k_test.jsonl \
    && chmod 644 /app/eval/inkling_small/fixtures/gsm8k_test.jsonl \
    && cd /app \
    && test "$(git rev-parse HEAD)" = "$BASE_SHA" \
    && git add -f eval/inkling_small \
    && GIT_AUTHOR_NAME="Inference-Bench" GIT_AUTHOR_EMAIL="inference-bench@local" \
       GIT_COMMITTER_NAME="Inference-Bench" GIT_COMMITTER_EMAIL="inference-bench@local" \
       GIT_AUTHOR_DATE="2026-07-20T05:40:49+0000" GIT_COMMITTER_DATE="2026-07-20T05:40:49+0000" \
       git commit -q --no-verify -F /opt/infb/eval-materials.msg \
    && echo "eval materials commit: $(git rev-parse HEAD)" \
    && test "$(git rev-parse HEAD)" = "$BASE_EVAL_SHA" \
    && test "$(git ls-files eval/inkling_small | wc -l)" = "$(find eval/inkling_small -type f | wc -l)" \
    && test -z "$(git diff --name-only $BASE_SHA HEAD -- . ':!eval/inkling_small')" \
    && test -z "$(git status --porcelain)" \
    && git config --system user.name agent && git config --system user.email agent@localhost

# 11. NO IMPLEMENTATION OF THE WITHHELD WORK IS IMPORTABLE, in the engine interpreter: no Transformers
#     `models/inkling` (the engine's 5.12.1 has none; >= 5.14.0 would), no `sglang.jit_kernel.flash_attn`
#     (the landing's in-tree FlashAttention-4 fork), and no FlashAttention-4 `flash_attn_varlen_func`
#     accepting `rel_bias`.
COPY checks/ /opt/infb/checks/
RUN python /opt/infb/checks/no_reference_impl.py

# 12. NOTHING ON THE AGENT'S SURFACE CARRIES THE WITHHELD CODE OR SAYS WHERE THE ANSWER IS. The two
#     checks scan the working tree, every Git object, the installed packages, the caches, the
#     task-authored files and the checkpoint's text files, and prove the landing's objects are not in
#     /app's store. They are BIND-MOUNTED, never copied: they carry the identifiers and the addresses
#     they look for, and a copy of them in the image would be the very leak they guard.
RUN --mount=type=bind,source=buildcheck,target=/infb-buildcheck \
    python3 /infb-buildcheck/no_answer_address.py \
    && python3 /infb-buildcheck/no_withheld_code.py

# THE ENGINE VENV THIS IMAGE CARRIES, stated rather than inferred.
ENV INFB_VENV=/opt/sglang

WORKDIR /app
CMD ["/bin/bash"]

# The task image. With no build argument this is the stage above, built from public sources; with
# --build-arg ENGINE=<ref> it is that prebuilt image and the stage above is pruned (see the header).
# Either way its one line re-declares the runtime library path without the forward-compat directory.
FROM ${ENGINE}
ENV LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# infb-verifier-tail
#
# Everything above this marker is environment/Dockerfile verbatim -- byte-identical instructions
# over byte-identical context files, duplicated under tests/. Pier builds a separate verifier from
# tests/ alone and uploads no tests into it, so the verifier
# recipe has to be self-contained; making it the task recipe verbatim means a public `docker build`
# of tests/ is a cache hit after the task image, and the verifier can never run on a different
# engine than the agent did. Regenerate with:
#   cat environment/Dockerfile <the tail below> > tests/Dockerfile
#
# The verifier image provides its own CUDA toolkit, inherited from the task image whose recipe
# declares CUDA_HOME. Restated here explicitly, because a bare FROM can read as "no toolkit".
# Inert on Docker (the base image carries the same values).
ENV CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda

# THE TESTS, in as few steps as the work allows: every step on this image mounts and commits all of
# its layers whatever it copies. One COPY places tests/ at /tests, as Pier's own verifier upload would: the grader's
# files together with the recipe inputs duplicated beside them, which no step reads (the preflight's recorded digest of
# /tests covers them with the rest).
#
# The human reference's tree, built here from the still-pristine image and never at grading time:
# a shared clone of /app at the base, tests/reference/solution.patch
# (solution/solution.patch byte for byte) applied, its tree checked against the closure tree oracle.lock
# records, and the engine's git-ignored build products under python/sglang copied with their digests.
# test.sh's reference step checks all of it again before it serves the reference. /app is not written.
COPY . /tests/
RUN chmod +x /tests/test.sh && python3 /tests/infb_efficiency/snapshot.py --config /tests/config.json
