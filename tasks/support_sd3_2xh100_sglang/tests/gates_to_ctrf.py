#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Convert gate JSON into CTRF for the task-owned grader.

The filename and gate identity must agree, and passed must be a boolean.
The grader scores only the required IDs in config.json; missing or malformed
results never receive credit. Unscored observations do not affect reward.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

VERIFIER_DIR = Path(os.environ.get("INFB_VERIFIER_DIR", "/logs/verifier"))
GATES_DIR = VERIFIER_DIR / "gates"


def main() -> int:
    tests = []
    for p in sorted(GATES_DIR.glob("*.gate.json")):
        try:
            g = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        name = p.name[: -len(".gate.json")]


        valid = isinstance(g, dict) and g.get("gate") == name and type(g.get("passed")) is bool
        status = "passed" if valid and g["passed"] is True else "failed"
        node = {"name": name, "status": status}
        if status != "passed":
            node["message"] = (str(g.get("error") or "gate failed")[:2000] if valid
                               else "malformed gate: require matching identity and boolean passed")
        tests.append(node)

    passed = sum(1 for t in tests if t["status"] == "passed")
    ctrf = {
        "reportFormat": "CTRF",
        "specVersion": "1.0.0",
        "results": {
            "tool": {"name": "inference-bench-gates"},
            "summary": {
                "tests": len(tests),
                "passed": passed,
                "failed": len(tests) - passed,
                "skipped": 0,
                "pending": 0,
                "other": 0,
            },
            "tests": tests,
        },
    }
    out = VERIFIER_DIR / "gates-ctrf.json"
    out.write_text(json.dumps(ctrf, indent=2))
    print(f"[gates_to_ctrf] {len(tests)} gates ({passed} passed) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
