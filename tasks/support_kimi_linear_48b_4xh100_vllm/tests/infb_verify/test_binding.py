#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Actual-serving evidence on a separate instrumented launch; never a timing server."""

import json

import pytest

from infb_verify import kda, records, serve


@pytest.fixture
def binding_server():
    yield from serve.launch(binding=True)


def test_kda_serving_binding(binding_server):
    cfg = json.loads((kda.TESTS / "config.json").read_text())

    def body(record):
        record["problems"] = problems = kda.check_binding(cfg, binding_server.url_root, kda.RECORDS / "traces")
        return not problems, problems

    problems = records.run("test_kda_serving_binding", {"evidence": str(kda.RECORDS / "candidate-binding.json")}, body)
    assert not problems, problems
