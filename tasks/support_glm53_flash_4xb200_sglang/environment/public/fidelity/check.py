#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Compare recorded public examples with a running server's log-probabilities.

    python3 /app/eval/glm53_flash/fidelity/check.py --url http://127.0.0.1:30000

The examples contain prose, code and math prompts and one longer
code prompt (4,608 tokens), each with a 512-token continuation and original
model-author reference values. The command reports divergence, not a task pass
verdict. Use --examples for compatible recorded data.
The shared request and comparison functions live in glm53_protocols/fidelity.py.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from glm53_protocols.fidelity import k3_per_sequence, served_continuation_logprobs  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--examples", default=str(HERE / "examples.json"))
    a = ap.parse_args()
    refs = json.loads(Path(a.examples).read_text())["sequences"]
    try:
        served = served_continuation_logprobs(a.url.rstrip("/"), refs)
    except AssertionError as exc:
        print(f"not measured: {exc}")
        return 1
    k3 = k3_per_sequence(refs, served)
    for r, v in zip(refs, k3):
        print(f"{r['id']:9s} {r['prompt_len'] + r['cont_len']:5d} tokens  k3 {v:.6f}")
    print(f"median k3 over {len(refs)} sequences: {statistics.median(k3):.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
