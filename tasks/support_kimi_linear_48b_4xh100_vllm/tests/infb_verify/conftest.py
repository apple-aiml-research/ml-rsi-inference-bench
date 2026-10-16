#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""One server for every served-API module in this package.

The accuracy node, the behaviour probes and the fidelity gate all want exactly the same
server at exactly the same recipe, and a model load is minutes. The fixture is therefore
session-scoped and lives here, so one `pytest /tests/infb_verify/test_hidden.py ...` launches once.
"""

from __future__ import annotations

import pytest

from infb_verify import serve


@pytest.fixture(scope="session")
def server():
    yield from serve.launch()
