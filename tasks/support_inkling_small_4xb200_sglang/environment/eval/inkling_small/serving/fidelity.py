#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Prompt log-probabilities against the author's reference series (README, `run.py fidelity`).

    python fidelity.py --base-url http://127.0.0.1:30000 --inputs ../examples/fidelity.json [--output /tmp/f.json]

Each input sequence is sent once to `/generate` with `input_ids`, `max_new_tokens 0`, `return_logprob` and
`logprob_start_len 0`. Structural checks come first: exactly one log-probability for each of positions 1..n-1, every
one finite and <= 1e-6. Then, per position, with `ref` the author series (Transformers v5.16.1, the model author's
code, over the checkpoint dequantized to BF16, README) and `served` the server's value:
`k3 = expm1(ref - served) - (ref - served)`; the per-sequence statistic is the mean over positions, and bands are
summarized by their median. Grading applies the same statistic to hidden sequences against private bars; this prints
measurements, not a verdict.

Input format: {"sequences": [{"id": ..., "band": ..., "token_ids": [...], "author_logprobs": [n - 1 values]}]}.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from server import generate  # noqa: E402

LONG = 64000


def prompt_logprobs(base_url: str, ids: list[int]) -> tuple[list[float | None], dict]:
    out = generate(base_url, ids, max_new_tokens=0, timeout=1800.0 if len(ids) >= LONG else 900.0,
                   return_logprob=True, logprob_start_len=0)
    meta = out["meta_info"]
    return [x[0] for x in meta["input_token_logprobs"]], meta


def score(ids: list[int], ref: list[float], served: list) -> dict:
    n = len(ids)
    problems = []
    if len(served) != n:
        problems.append(f"{len(served)} input log-probabilities for {n} tokens")
    vals = served[1:n]
    if len(ref) != n - 1:
        problems.append(f"reference has {len(ref)} values for {n - 1} positions")
    if any(v is None or not isinstance(v, (int, float)) or not math.isfinite(v) or v > 1e-6 for v in vals):
        problems.append("a served log-probability at positions 1..n-1 is missing, non-finite or > 1e-6")
    if problems:
        return {"valid": False, "problems": problems}
    k3 = [math.expm1(r - s) - (r - s) for r, s in zip(ref, vals)]
    return {"valid": True, "mean_k3": sum(k3) / len(k3), "max_k3": max(k3),
            "max_abs_d": max(abs(r - s) for r, s in zip(ref, vals)), "positions": len(k3)}


def run(base_url: str, sequences: list[dict]) -> dict:
    results = []
    for s in sequences:
        served, meta = prompt_logprobs(base_url, s["token_ids"])
        rec = {"id": s.get("id"), "band": s.get("band"), "length": len(s["token_ids"]),
               "cached_tokens": meta.get("cached_tokens"), **score(s["token_ids"], s["author_logprobs"], served)}
        results.append(rec)
        print(json.dumps(rec), flush=True)
    bands = {}
    for r in results:
        if r.get("valid"):
            bands.setdefault(str(r["band"]), []).append(r["mean_k3"])
    summary = {b: {"sequences": len(v), "median_of_means": statistics.median(v), "max_mean": max(v)} for b, v in bands.items()}
    return {"sequences": results, "bands": summary}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    rec = run(args.base_url, json.loads(args.inputs.read_text())["sequences"])
    print("bands:", json.dumps(rec["bands"]))
    if args.output:
        args.output.write_text(json.dumps(rec, indent=1))
    return 0 if all(r.get("valid") for r in rec["sequences"]) else 1


if __name__ == "__main__":
    sys.exit(main())
