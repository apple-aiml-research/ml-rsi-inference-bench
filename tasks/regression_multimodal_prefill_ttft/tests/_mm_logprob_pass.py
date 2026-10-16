#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Output-agreement pass for regression_multimodal_prefill_ttft. Run by the verifier.

Sends sixteen of the ANCHOR'S OWN requests -- rebuilt from the same seed through the client
beside this file, so they are the bytes the scored point sent -- one at a time, greedy, and reads the
per-position prompt logprobs back off the live server. Both arms see identical inputs, so the
verifier compares the two lists position by position (check_outputs_agree) with no reference file
and no cross-process handoff.

WHY ONE AT A TIME. Batched matmul is not bitwise reproducible: gathered, sixteen different image
sizes read pass_rate 0.999 on a byte-identical tree. At concurrency 1 each request is its own batch
and the floor is exactly 1.0.

WHY THESE SIXTEEN. Picked by rank of predicted image-token total so the smallest and the largest
request are always in, with the single-image-area extremes reserved as well. A fixed-size version of
this check could be passed by a candidate that caps image resolution: such a cap would change scored
traffic while sparing every request a fixed set sends, so the set has to span the range the scored
shapes actually cover.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _mm_ttft_client import build_payloads, parse_hw  # noqa: E402


def pick_indices(predicted: list[int], sizes: list[list[list[int]]], k: int) -> list[int]:
    """Rank extremes first (token total and single-image area), then an even spread by token total."""
    order = sorted(range(len(predicted)), key=lambda i: predicted[i])
    area = [max((h * w for h, w in per), default=0) for per in sizes]
    k = max(1, min(k, len(order)))
    picks: list[int] = []
    for reserved in (order[0], order[-1],
                     min(range(len(area)), key=lambda i: area[i]),
                     max(range(len(area)), key=lambda i: area[i])):
        if reserved not in picks and len(picks) < k:
            picks.append(reserved)
    for j in range(k):
        if len(picks) >= k:
            break
        cand = order[round(j * (len(order) - 1) / max(1, k - 1))]
        if cand not in picks:
            picks.append(cand)
    return sorted(picks)


def prompt_logprobs(meta: dict) -> list[float | None]:
    """Slot 0 of each `input_token_logprobs` entry; a malformed entry is None (skipped, never a diff)."""
    out: list[float | None] = []
    for entry in (meta or {}).get("input_token_logprobs") or []:
        if isinstance(entry, (list, tuple)) and entry and entry[0] is not None:
            out.append(float(entry[0]))
        else:
            out.append(None)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--input-len", type=int, required=True)
    ap.add_argument("--num-prompts", type=int, required=True)
    ap.add_argument("--image-count", type=int, required=True)
    ap.add_argument("--image-format", default="jpeg")
    ap.add_argument("--image-min-hw", required=True)
    ap.add_argument("--image-max-hw", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--request-timeout-sec", type=float, default=180.0)
    ap.add_argument("--budget-sec", type=float, default=180.0)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    started = time.time()
    out: dict = {"comparison": "prompt_token_logprobs", "n": args.n, "concurrency": 1,
                 "logprobs": None, "lengths": [], "error": None}
    try:
        payloads = build_payloads(args.seed, args.num_prompts, args.image_count,
                                  parse_hw(args.image_min_hw), parse_hw(args.image_max_hw),
                                  args.image_format, args.input_len)
        predicted = [p["predicted"] for p in payloads]
        picks = pick_indices(predicted, [p["sizes"] for p in payloads], args.n)
        out["picked_predicted_tokens"] = [predicted[i] for i in picks]
        out["picked_sizes"] = [payloads[i]["sizes"] for i in picks]
        sampling = {"temperature": 0.0, "max_new_tokens": 1,
                    "top_p": 1.0, "top_k": -1, "ignore_eos": True}
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        logprobs: list[list[float | None]] = []
        for i in picks:
            if time.time() - started > args.budget_sec:
                raise TimeoutError(f"budget {args.budget_sec:.0f}s exceeded after {len(logprobs)} requests")
            body = {"text": payloads[i]["prompt"], "image_data": payloads[i]["images"],
                    "sampling_params": sampling, "return_logprob": True, "logprob_start_len": 0,
                    "stream": False}
            req = urllib.request.Request(args.base_url.rstrip("/") + "/generate",
                                         data=json.dumps(body).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
            with opener.open(req, timeout=args.request_timeout_sec) as fh:  # noqa: S310
                payload = json.loads(fh.read().decode("utf-8"))
            item = payload[0] if isinstance(payload, list) else payload
            meta = item.get("meta_info") if isinstance(item, dict) else None
            if not isinstance(meta, dict):
                raise ValueError(f"response carries no meta_info: {str(payload)[:200]}")
            logprobs.append(prompt_logprobs(meta))
        out["logprobs"] = logprobs
        out["lengths"] = [len(x) for x in logprobs]
    except Exception as exc:  # noqa: BLE001 -- recorded; the verifier fails closed on None
        out["error"] = f"{type(exc).__name__}: {exc}"[:500]
    out["elapsed_sec"] = round(time.time() - started, 2)
    path = Path(args.out_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[logprob_pass] {len(out['lengths'])} request(s), {sum(out['lengths'])} positions, "
          f"error={out['error']}", flush=True)
    return 0 if out["error"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
