#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""One server for every served-API module in this package.

An earlier form launched its own server per module; a second module (the fidelity gate) and a second
launch would cost another model load per trial for nothing -- the two modules want exactly the same
server at exactly the same recipe. The fixture is therefore session-scoped and lives here, so one
`pytest /tests/infb_verify/test_fidelity.py /tests/infb_verify/test_hidden.py` launches once.
"""

from __future__ import annotations

import pytest

from infb_verify import serve


@pytest.fixture(scope="session")
def server():
    yield from serve.launch()
