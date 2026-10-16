#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# infb-verifier-tail
# Everything above this marker is environment/Dockerfile, byte for byte: the verifier image is the task
# image plus the tests, by construction. The build uses tests/ as the only build context, so every
# context read of the recipe is committed under both directories, byte-identical. The agent never sees
# this container.
#
# The CUDA toolkit is restated for runtimes that choose a host-toolkit bind mount per Dockerfile.
# Inert on Docker (same values as the base).
ENV CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda

COPY config.json /tests/config.json
COPY eval/ /tests/eval/
COPY infb_efficiency/ /tests/infb_efficiency/
COPY reference/ /tests/reference/
# The verifier's own tree: the separate verifier runs /tests/test.sh from this image, with nothing
# mounted; every file test.sh and its modules read is here, checked below.
COPY test.sh test.patch grader.py preflight.py /tests/
COPY infb_verify/ /tests/infb_verify/
COPY hidden/ /tests/hidden/
COPY public/ /tests/public/
COPY _upstream/ /tests/_upstream/
RUN chmod +x /tests/test.sh && test -s /tests/hidden/policy.json && test -f /tests/public/efficiency/client.py \
    && test -f /tests/infb_verify/test_tp1.py && test -f /tests/grader.py && test -f /tests/test.patch
# The L3 instances' Hugging Face files: the non-weight files of every non-gated repository an L3
# instance reads, at pinned revisions, each checked by sha256, into the offline cache; the gated
# repositories are excluded (l3/l3-hf-manifest.json names them and why).
COPY l3/ /opt/infb/l3/
RUN /opt/vllm/bin/python /opt/infb/l3/fetch_l3_hf.py /opt/infb/l3/l3-hf-manifest.json "$HF_HUB_CACHE"
# The reference tree, built here and nowhere else: an independent copy of the prepared base with
# tests/reference/solution.patch applied, its tree identity checked, the native-source rule asserted,
# the prepared base's compiled products copied, the CuTe-DSL link re-created into the copy, and the
# copy byte-compiled.
RUN python3 /tests/infb_efficiency/snapshot.py --config /tests/config.json
# The fixed-work generation configurations the verifier's sessions serve (recipes.py's GENCFG over the
# checkpoint's generation_config.json), written once here.
RUN /opt/vllm/bin/python /tests/eval/diffusion_gemma/run.py gencfg --out /opt/infb/gencfg --output /tmp/gencfg.json \
    && test -f /opt/infb/gencfg/tp1_fixed/generation_config.json && test -f /opt/infb/gencfg/tp2_fixed/generation_config.json
