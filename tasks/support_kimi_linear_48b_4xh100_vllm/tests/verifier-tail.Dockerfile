#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# infb-verifier-tail
# Everything above this marker is environment/Dockerfile, byte for byte, and that is the whole point:
# the verifier image is the task image plus the tests, by construction rather than by a registry
# reference. The recipe's COPY and bind-mount sources are therefore committed under BOTH directories,
# byte-identically. A public `docker build` of tests/ after the task image is layer cache hits down
# to this line.
#
# Assembled, never hand-edited:
#   cat environment/Dockerfile > tests/Dockerfile && printf '\n' >> tests/Dockerfile \
#     && cat tests/verifier-tail.Dockerfile >> tests/Dockerfile
# tests/ is the build context; the agent never sees this container.
#
# The verifier image provides its own CUDA toolkit (inherited from the task image, whose recipe
# declares CUDA_HOME). Restated here explicitly, because a bare FROM reads as "no toolkit". Inert on
# Docker (same values as the base).
ENV CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda

# ONE copy of the whole build context, one step of checks and the snapshot. Every
# Dockerfile step on an image this size costs a fixed commit overhead, whatever it copies. The
# contents are unchanged: the runner, grader, preflight and configuration; the private checks
# (infb_verify/); the hidden material, which test_hidden.py reads at IMPORT time, so a verifier image
# without it fails at COLLECTION and takes the whole suite with it; the grader's copy of the
# evaluation tree (eval/, whose supplied counterpart the base commit carries at /app/eval; test.sh
# compares the two by content after its restore, before preflight); the task-owned efficiency gate and
# the reference patch its snapshot step applies. The snapshot builds the reference tree at
# /opt/infb-reference from the pristine /app of this image plus that patch, checks it against the
# target tree in config.json, copies the base's git-ignored build products (vllm/'s compiled
# extensions) into it, and byte-compiles it. It is the one step the separate-verifier guard allows
# beyond COPYs and chmod, as its own RUN, and it runs before any submission is applied.
COPY . /tests/
RUN chmod +x /tests/test.sh \
    && test -s /tests/hidden/probes.json \
    && test -s /tests/hidden/fidelity-sequences.json \
    && test -s /tests/hidden/fidelity-reference.json \
    && test -s /tests/hidden/decode-prompts.json \
    && test -s /tests/hidden/long-context.json \
    && test -s /tests/hidden/concurrency-prompts.json \
    && test -s /tests/hidden/router-probe.json \
    && test -s /tests/eval/kimi_linear/tests/upstream/gsm8k_eval.py
RUN python3 /tests/infb_efficiency/snapshot.py --config /tests/config.json
