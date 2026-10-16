#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""One launch per deployment module: `test_bf16kv.py` and `test_fp8kv.py` each run in their own pytest process (their
own group in test.sh), and every node of the module is served by that module's one launch, in file order."""

from __future__ import annotations

import pytest

from infb_verify import serve


@pytest.fixture(scope="module")
def session(request):
    yield from serve.launch(request.module.DEPLOYMENT)
