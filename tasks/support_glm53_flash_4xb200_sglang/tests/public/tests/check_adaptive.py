#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""LowLatency's adaptive speculation schedule, read from the server's own log.

    python3 /app/eval/glm53_flash/tests/check_adaptive.py --log /tmp/low_latency.log

Give it the log of one LowLatency server, from its launch through the workload you ran on it (`serve.py --log`
keeps one). It reads the schedule's start-up line and its switch lines and describes them: the tier ladders and whether
they hold the recipe's `--speculative-num-steps`, and the switches logged under that workload. An active controller may
hold one tier under a given workload, so the switch count is an observation, not a verdict. Exit 0 when the schedule
was read, 2 when the log has no start-up line.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from glm53_protocols.adaptive import read_schedule, verdict  # noqa: E402
from glm53_protocols.recipe import recipe_flags  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--log", required=True, help="the LowLatency server's own log, launch through workload")
    ap.add_argument("--steps", type=int, default=None,
                    help="the recipe's --speculative-num-steps (default: read from the LowLatency class)")
    a = ap.parse_args()
    steps = a.steps
    if steps is None:
        from acceptance import server_args  # noqa: PLC0415 - needs the engine to import
        steps = int(recipe_flags(server_args("low_latency"))["speculative_num_steps"])
    schedule = read_schedule(Path(a.log).read_text(errors="replace").splitlines())
    status, reason = verdict(schedule, steps)
    print(json.dumps({"status": status, "reason": reason, "recipe_steps": steps, **schedule}, indent=1))
    return 0 if status == "observed" else 2


if __name__ == "__main__":
    sys.exit(main())
