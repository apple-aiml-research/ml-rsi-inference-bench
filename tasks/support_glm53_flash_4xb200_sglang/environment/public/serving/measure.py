#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Your server's per-user decode speed and output throughput, one axis at a time, with a prefix-cache flush before each.

    python3 /app/eval/glm53_flash/serving/measure.py --class low_latency --url http://127.0.0.1:30000

For each axis it flushes the server's prefix cache (`client.flush_prefix_cache`: `POST /flush_cache`), then runs
`client.py measure` for that axis alone, in its own process from outside the engine tree, on the natural-text sample
beside this file, with the checkpoint's tokenizer and the configuration's load:
- `decode_c1`: per-user decode speed at concurrency 1 (S1);
- `throughput`: output throughput at concurrency 64 in HighThroughput, 16 in LowLatency (S_C).
Both axes of one run use one draw of prompts; the seed is printed, and `--seed` repeats it. The client prints each
value in tokens per second and a final `INFB_EFFICIENCY {...}` line with the details (TTFT, and any request that did
not count, and why). A flush that fails is printed and the measurement still runs. Faster is better.
"""
from __future__ import annotations

import argparse
import secrets
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from client import AXES, flush_prefix_cache  # noqa: E402

LOAD = {"high_throughput": 64, "low_latency": 16}
TOKENIZER = "/model/zai-org/GLM-5.3-Flash/tokenizer.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--class", dest="cls", choices=sorted(LOAD), required=True)
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--axis", choices=(*AXES, "both"), default="both")
    ap.add_argument("--seed", default=None, help="the draw's seed (default: random, printed)")
    ap.add_argument("--texts", default=str(HERE / "sample.jsonl.xz"))
    ap.add_argument("--tokenizer", default=TOKENIZER)
    a = ap.parse_args()
    url = a.url.rstrip("/")
    seed = a.seed or secrets.token_hex(8)
    print(f"[measure] seed {seed}", flush=True)
    worst = 0
    for axis in (AXES if a.axis == "both" else (a.axis,)):
        print(f"[measure] {axis}: flush {flush_prefix_cache(url)}", flush=True)
        argv = [sys.executable, "-I", str(HERE / "client.py"), "measure", "--engine", "sglang", "--base-url", url,
                "--tokenizer", a.tokenizer, "--texts", a.texts, "--concurrency", str(LOAD[a.cls]),
                "--axis", axis, "--seed", seed]
        worst = max(worst, subprocess.run(argv, cwd="/tmp").returncode)
    return worst


if __name__ == "__main__":
    sys.exit(main())
