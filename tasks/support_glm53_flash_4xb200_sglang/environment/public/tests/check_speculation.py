#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""LowLatency's speculative decoding with the checkpoint's NextN draft: acceptance and single-request speed.

    python3 /app/eval/glm53_flash/tests/check_speculation.py --url http://127.0.0.1:30000

Against a running LowLatency server it runs, in turn:
1. the one-prompt acceptance probe (`glm53_protocols/speculation.py`: upstream's `send_one_prompt`, 2,048 new tokens,
   up to three attempts with a prefix-cache flush before each), printing the average acceptance length and the speed
   of each attempt against the acceptance file's own `accept_length_thres`;
2. the acceptance file's own `test_bs_1_speed`, unchanged, with its own thresholds and attempts.
Exit 0 when the probe's last attempt is above the threshold and `test_bs_1_speed` passes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from acceptance import MODEL, acceptance_class, bound, run_methods  # noqa: E402
from glm53_protocols.speculation import acceptance_probe  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()
    url = a.url.rstrip("/")
    cls = acceptance_class("low_latency")
    bar = float(cls.accept_length_thres)
    acc, speed, readings = acceptance_probe(url, bar)
    for i, (x, s) in enumerate(readings, 1):
        print(f"probe attempt {i}: acceptance length {x:.3f}, {s:.2f} tok/s")
    print(f"acceptance length {acc:.3f} {'above' if acc > bar else 'NOT above'} {bar} (accept_length_thres)")
    result = run_methods(bound(cls, url, a.model), ["test_bs_1_speed"])
    return 0 if acc > bar and result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
