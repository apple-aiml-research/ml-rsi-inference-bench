#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Measure a running server as the grader measures a tree: per-user decode speed at concurrency 1 (S1) and output
throughput at concurrency 16 (S_C), with `client.py` beside this file, the same client the grader runs."""
from __future__ import annotations

import argparse
import json
import secrets
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOKENIZER = "/model/thinkingmachines/Inkling-Small-NVFP4/tokenizer.json"
CONCURRENCY = 16


def draw(base_url: str, axis: str, seed: str, deadline_s: float | None) -> dict:
    argv = [sys.executable, "-I", str(HERE / "client.py"), "measure", "--engine", "sglang", "--base-url", base_url,
            "--tokenizer", TOKENIZER, "--texts", str(HERE / "sample.jsonl.xz"), "--concurrency", str(CONCURRENCY),
            "--axis", axis, "--seed", seed, "--quiet"]
    if deadline_s:
        argv += ["--deadline-s", f"{deadline_s:g}"]
    p = subprocess.run(argv, capture_output=True, text=True)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("INFB_EFFICIENCY ")]
    report = json.loads(lines[-1][len("INFB_EFFICIENCY "):]) if lines else {}
    entry = (report.get("axes") or {}).get(axis) or {}
    return {"exit": p.returncode, "value_tok_s": entry.get("value_tok_s"), "status": entry.get("status"),
            "invalid": entry.get("invalid"), "ttft_median_s": entry.get("ttft_median_s"),
            "duration_s": entry.get("duration_s"), "error": None if lines else (p.stdout + p.stderr)[-2000:]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="S1 and S_C on a running server, three draws of one draw per axis")
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--axis", choices=("decode_c1", "throughput", "both"), default="both")
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--seed", help="the draw's seed (default: random, printed; the same seed repeats the draw)")
    ap.add_argument("--deadline-s", type=float, help="stop a draw, censored, after this many seconds")
    args = ap.parse_args(argv)
    seed = args.seed or secrets.token_hex(8)
    print(f"seed {seed}", flush=True)
    axes = ["decode_c1", "throughput"] if args.axis == "both" else [args.axis]
    worst = 0
    for axis in axes:
        values = []
        for i in range(args.draws):
            d = draw(args.base_url, axis, seed, args.deadline_s)
            worst = max(worst, d["exit"])
            label = "S1 (per-user decode, c=1)" if axis == "decode_c1" else f"S_C (output throughput, c={CONCURRENCY})"
            print(f"{label} draw {i + 1}: " + json.dumps(d, sort_keys=True), flush=True)
            if isinstance(d["value_tok_s"], (int, float)) and d["status"] == "complete":
                values.append(d["value_tok_s"])
        if values:
            print(f"{axis}: median of {len(values)} complete draw(s) {statistics.median(values):.2f} tok/s "
                  f"(draws {[round(v, 2) for v in values]})", flush=True)
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
