#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The configuration a freshly launched server reports back equals the one it was launched with.

    python3 /app/eval/glm53_flash/tests/check_recipe.py --class low_latency --url http://127.0.0.1:30000

Run it once after the server is healthy and before it serves any request. It reads `GET /server_info` and compares
every flag the class's launch argv names (from `test_glm53_flash_b200.py`), plus the attention backends in
`recipe_backends.json`, at the top level and in every data-parallel scheduler's `internal_states` entry (a scheduler's
report, not every GPU rank's). It prints every mismatch; exit 0 when there is none.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from acceptance import CLASSES, server_args  # noqa: E402
from glm53_protocols.recipe import compare, expected_fields, server_info  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--class", dest="cls", choices=sorted(CLASSES), required=True)
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    a = ap.parse_args()
    resolved = json.loads((HERE / "recipe_backends.json").read_text())[a.cls]
    result = compare(expected_fields(server_args(a.cls), resolved), server_info(a.url.rstrip("/")))
    print(f"{len(result['expected'])} fields at the top level and on {result['n_dp_schedulers']} data-parallel scheduler(s)")
    for m in result["mismatches"]:
        print(f"MISMATCH {m['view']}: {m['flag']} expected {m['expected']!r}, served {m['served']!r}")
    return 1 if result["mismatches"] else 0


if __name__ == "__main__":
    sys.exit(main())
