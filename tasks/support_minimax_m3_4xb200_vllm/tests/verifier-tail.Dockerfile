#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Everything above this marker is environment/Dockerfile, byte for byte: the verifier image is the task image plus
# the tests, by construction. This file is built with tests/ as the only build context and no build arguments,
# so the recipe's COPY and bind sources are committed under BOTH directories, byte-identical. A public
# `docker build` of tests/ after the task image is layer-cache hits down to this line. tests/ is the build
# context; the agent never sees this container.
#
# The verifier image provides its own CUDA toolkit (inherited from the task image), restated here explicitly
# because a bare FROM reads as "no toolkit".
ENV CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda

COPY test.sh /tests/test.sh
COPY test.patch /tests/test.patch
COPY grader.py /tests/grader.py
COPY config.json /tests/config.json
COPY preflight.py /tests/preflight.py
COPY infb_verify/ /tests/infb_verify/
COPY eval/ /tests/eval/
COPY _upstream/ /tests/_upstream/
# The hidden material (private MSA case identities, fidelity and engine-agreement series, replay and chat inputs,
# the witness prompts, the efficiency seed, the frozen policy). Every node module reads its own at import or call
# time, so none of it is optional.
COPY hidden/ /tests/hidden/
# The task-owned efficiency implementation and the reference patch the snapshot step applies. Both only under
# tests/, so the hidden text pool and the seed never reach the agent's image.
COPY infb_efficiency/ /tests/infb_efficiency/
COPY reference/ /tests/reference/
# THE REFERENCE TREE, BUILT HERE AND NOWHERE ELSE. `grader.py prepare` applies the candidate's
# model.patch first and L0 rebuilds the candidate's products in place, so this RUN -- before a byte of candidate
# content exists -- is the one moment the reference can be built from the pristine tree: an independent Git copy,
# the patch, the tree-identity guard, /app/.deps copied in, the documented editable build run offline in the copy
# (the venv's own install of /app restored after), byte-compiled, every product hashed into snapshot.json. Its
# measured wall sets the verifier-image build allowance. One step with the chmod and the hidden-material
# assertions, so the tail pays one step's mount and commit on this large image; the snapshot
# needs only the standard library and runs the engine's interpreter itself.
# A build step has no driver. The image ENV no longer lists the compat directory, and the reference build's
# import check must load libcuda.so.1, so this one step puts the compat directory back on its own path. The
# shipped ENV is unchanged.
RUN chmod +x /tests/test.sh \
    && test -s /tests/hidden/msa_cases.json \
    && test -s /tests/hidden/fidelity.json.xz \
    && test -s /tests/hidden/fidelity-sources.json \
    && test -s /tests/hidden/engine_agreement.json.xz \
    && test -s /tests/hidden/replay.json \
    && test -s /tests/hidden/chat_cases.json \
    && test -s /tests/hidden/witness.json \
    && test -s /tests/hidden/efficiency_seed.json \
    && test -s /tests/hidden/policy.json \
    && LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH python3 /tests/infb_efficiency/snapshot.py --config /tests/config.json
