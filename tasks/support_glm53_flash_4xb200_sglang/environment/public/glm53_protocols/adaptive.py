#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The adaptive speculation schedule, read from a server's own log lines."""
from __future__ import annotations

import re
from collections import Counter

LADDER = re.compile(r"AdaptiveSpeculativeParams initialized: steps=(\d+), candidate_steps=\[([0-9,\s]*)\]")
SWITCH = re.compile(r"Adaptive spec params updated: steps (\d+) -> (\d+)")


def read_schedule(lines: list[str]) -> dict:
    """Every start-up ladder and every switch in `lines`."""
    ladders = []
    switches: Counter = Counter()
    for line in lines:
        m = LADDER.search(line)
        if m:
            ladders.append({"steps": int(m.group(1)),
                            "candidate_steps": [int(x) for x in m.group(2).replace(" ", "").split(",") if x]})
            continue
        m = SWITCH.search(line)
        if m:
            switches[f"{m.group(1)}->{m.group(2)}"] += 1
    return {"ladders": ladders, "switches": dict(sorted(switches.items())), "n_switches": sum(switches.values())}


def verdict(schedule: dict, recipe_steps: int) -> tuple[str, str]:
    """("observed" | "not_measured", a description)."""
    if not schedule["ladders"]:
        return ("not_measured", "the server's log carries no adaptive-schedule start-up line "
                                "(`AdaptiveSpeculativeParams initialized: … candidate_steps=[…]`), so the schedule "
                                "cannot be read")
    ladders = [sorted(set(ladder["candidate_steps"])) for ladder in schedule["ladders"]]
    holds = all(recipe_steps in tiers for tiers in ladders)
    return ("observed", f"ladders {ladders} ({'holding' if holds else 'not all holding'} the recipe's "
                        f"--speculative-num-steps {recipe_steps}); {schedule['n_switches']} switches logged under "
                        f"this workload")
