#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Independent model-author fidelity at the ordinary recipe (private inputs and calibration; the
executable protocol is the public `qwen38.serving.fidelity`)."""
from __future__ import annotations

import json

from infb_verify import served


def test_fidelity_kl(server):
    result = served.fidelity(server, "ordinary")
    report = {key: value for key, value in result.items() if key not in ("reads", "sequences")}
    assert result["passed"], json.dumps(report, allow_nan=False)
