#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Existing GDN output/state behavior, independently of native Kimi model support."""

import json

from infb_verify import kda, records


def test_gdn_output_and_state():
    cfg = json.loads((kda.TESTS / "config.json").read_text())

    def body(record):
        record["problems"] = problems = kda.check_gdn(cfg, "candidate")
        return not problems, problems

    problems = records.run("test_gdn_output_and_state", {"evidence": str(kda.RECORDS / "candidate-gdn.json")}, body)
    assert not problems, problems
