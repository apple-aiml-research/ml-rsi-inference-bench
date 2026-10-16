#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""test_execution_witness: the served model runs the `attend` implementation.

The public protocol (eval/diffusion_gemma/serving/binding.py) with the hidden window prompts and the
identity sets the attend group's timing collection recorded. The trace wait is
max(300, 300 * ceil(2E / 300)) for the worst measured trace export E -> 300 s.
"""
from __future__ import annotations

import json
import os

from eval.diffusion_gemma.serving import binding as B
from eval.diffusion_gemma.serving import reads as R

from infb_verify import common
from infb_verify.common import SESSIONS
from infb_verify.serving_nodes import prose

BINDING = common.RECORDS / "attend_binding.json"
TRACE_WAIT_S = int(os.environ.get("INFB_TRACE_WAIT_S", "300"))


def run(deployment: str) -> dict:
    recipe = "tp1" if deployment == "tp1" else "tp2_fixed"
    srv = SESSIONS.get(recipe)
    if not BINDING.exists():
        return {"binding": False, "pass": False, "error": f"blocked_by: no identity record at {BINDING}"}
    identity = json.loads(BINDING.read_text())
    data = common.hidden("reads.json.xz")
    rd = next(r for r in data["reads"] if r["set"] == "reads" and r["band"] == 4096)
    body = R.text_read(data["prompts"][rd["prompt"]]["ids"], rd["seed"], rd["targets"], 1)
    w1 = (prose(6144, 900_000), [prose(2000, 920_000 + 3000 * k) for k in range(3)])
    rec = B.witness(srv, SESSIONS.profile_root(recipe), identity, w1, body, 1 if deployment == "tp1" else 2, TRACE_WAIT_S)
    rec["binding"] = True
    return rec
