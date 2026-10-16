#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# The verifier tail: the task image above, plus the tests.
# The toolkit is restated from the task image so a build that inspects only this file still declares
# it. Inert on Docker (same values as the base).
# infb-verifier-tail
ENV CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda

COPY test.sh /tests/test.sh
COPY test.patch /tests/test.patch
COPY grader.py /tests/grader.py
COPY config.json /tests/config.json
COPY preflight.py /tests/preflight.py
COPY infb_verify/ /tests/infb_verify/
COPY _upstream/ /tests/_upstream/
# The hidden material -- the prefix-reuse cases, the fidelity sequences with their reference
# log-probabilities, and the efficiency draw's seed. It is NOT optional: both hidden modules read
# theirs at import time, so a verifier image without it fails at COLLECTION and takes the whole
# suite with it.
COPY hidden/ /tests/hidden/
COPY infb_efficiency/ /tests/infb_efficiency/
COPY reference/ /tests/reference/
RUN chmod +x /tests/test.sh && test -s /tests/hidden/prefix_reuse.json
RUN chmod +x /tests/test.sh && test -s /tests/hidden/prefix_witness.json
RUN chmod +x /tests/test.sh && test -s /tests/hidden/fidelity.json
RUN chmod +x /tests/test.sh && test -s /tests/hidden/efficiency_seed.json
# THE REFERENCE'S TREE, BUILT HERE AND NOWHERE ELSE.
#
# The paired gate serves the closure oracle beside the candidate in one run on the same GPUs, and the
# reference tree cannot be built at grading time: `grader.py prepare` applies the candidate's
# model.patch first, and L0 runs the candidate's own setup.py and CMake as root and replaces the
# engine's compiled extensions IN PLACE under /app/vllm. By then there is no pristine tree left to
# copy the build products from. This RUN is the one moment it can be taken, before a byte of
# candidate content exists in the image.
RUN python3 /tests/infb_efficiency/snapshot.py --config /tests/config.json
