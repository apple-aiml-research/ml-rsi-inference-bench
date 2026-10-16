#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""One server for every served-API module in this package.

Every served-API node of this task -- the efficiency pair, the hidden GSM8K slice and the fidelity
gate -- wants exactly the same server at exactly the same recipe, and each launch of this 510 GB
checkpoint costs minutes. The fixture is therefore session-scoped and lives here, so one
`pytest /tests/infb_verify/test_hidden.py /tests/infb_verify/test_fidelity.py` launches once.
"""

from __future__ import annotations

import pytest

from infb_verify import serve


@pytest.fixture(scope="session")
def server():
    yield from serve.launch()


@pytest.fixture(scope="session")
def instrumented_server():
    """The deployment node's own launch: the recipe plus the engine's profiler configuration (runtime_evidence.py), in
    its own group after the timed session, so profiling never touches a timed measurement."""
    from infb_verify import runtime_evidence as rt  # noqa: PLC0415

    trace_dir = rt.OUT / "traces"
    with rt.instrumented_server(trace_dir) as server:
        yield rt.Instrumented(server, trace_dir)
