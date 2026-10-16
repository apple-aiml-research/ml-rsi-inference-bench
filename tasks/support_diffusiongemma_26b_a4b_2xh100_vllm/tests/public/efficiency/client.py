#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Whole-serving efficiency, public: the client both arms are measured with.

Workload: 4,096 input and 1,024 output tokens per request (ignore_eos), counted with /model's
tokenizer, on a deployment's fixed-work configuration (recipes.py: every canvas runs all 48
denoising steps, so the work is fixed). Every request carries a fresh cache salt.

    C1  output tokens per second of one request end to end (1,024 / wall), the best of up to three
        attempts
    C4  aggregate output tokens per second over a draw of 16 requests at concurrency 4; three draws,
        their median

    python /public/efficiency/client.py --base-url http://127.0.0.1:8000 --mode c1 --output /tmp/c1.json
    python /public/efficiency/client.py --base-url http://127.0.0.1:8000 --mode c4 --output /tmp/c4.json

The public prompts are natural-text documents from sample.jsonl.xz, tokenized and cut to 4,096
tokens; grading supplies its own prompts of the same length.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import lzma
import statistics
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

SERVED_NAME = "google/diffusiongemma-26B-A4B-it"
IN_TOKENS, OUT_TOKENS = 4096, 1024
C1_ATTEMPTS, C4_REQUESTS, C4_CONCURRENCY, C4_DRAWS = 3, 16, 4, 3
SAMPLE = Path(__file__).with_name("sample.jsonl.xz")
_NOPROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _post(base_url: str, body: dict, timeout: float = 1800.0) -> dict:
    req = urllib.request.Request(base_url + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    try:
        with _NOPROXY.open(req, timeout=timeout) as r:
            out = json.loads(r.read())
            out["_status"] = r.status
    except urllib.error.HTTPError as e:
        out = {"_status": e.code, "_error": e.read().decode(errors="replace")[:2000]}
    except Exception as e:  # noqa: BLE001
        out = {"_status": None, "_error": f"{type(e).__name__}: {e}"[:2000]}
    out["_wall_s"] = time.monotonic() - t0
    return out


def one(base_url: str, prompt: list[int]) -> float:
    """The wall of one request; raises unless it returns exactly OUT_TOKENS tokens."""
    r = _post(base_url, {"model": SERVED_NAME, "prompt": list(prompt), "max_tokens": OUT_TOKENS, "ignore_eos": True,
                         "temperature": 1.0, "cache_salt": uuid.uuid4().hex})
    n = (r.get("usage") or {}).get("completion_tokens")
    if r.get("_status") != 200 or n != OUT_TOKENS:
        raise RuntimeError(f"efficiency request failed: {r.get('_status')} {n} {(r.get('_error') or '')[:200]}")
    return r["_wall_s"]


def c1(base_url: str, prompts: list[list[int]]) -> dict:
    """prompts: at least C1_ATTEMPTS prompts of IN_TOKENS tokens, one per attempt."""
    walls = [one(base_url, prompts[k]) for k in range(C1_ATTEMPTS)]
    return {"walls": walls, "first": OUT_TOKENS / walls[0], "value": max(OUT_TOKENS / w for w in walls)}


def c4(base_url: str, draws: list[list[list[int]]]) -> dict:
    """draws: C4_DRAWS lists of C4_REQUESTS prompts of IN_TOKENS tokens."""
    rates = []
    for prompts in draws[:C4_DRAWS]:
        t0 = time.monotonic()
        with cf.ThreadPoolExecutor(C4_CONCURRENCY) as ex:
            list(ex.map(lambda p: one(base_url, p), prompts))
        rates.append(len(prompts) * OUT_TOKENS / (time.monotonic() - t0))
    return {"draws": rates, "value": statistics.median(rates)}


def public_prompts(n: int, model: str = "/model") -> list[list[int]]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    rows = [json.loads(line) for line in lzma.open(SAMPLE, "rt")]
    out = []
    for row in rows:
        ids = tok(row["text"], add_special_tokens=False)["input_ids"]
        if len(ids) >= IN_TOKENS:
            out.append(ids[:IN_TOKENS])
        if len(out) == n:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--mode", choices=("c1", "c4"), required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if args.mode == "c1":
        rec = c1(args.base_url, public_prompts(C1_ATTEMPTS))
    else:
        ps = public_prompts(C4_REQUESTS * C4_DRAWS)
        rec = c4(args.base_url, [ps[d * C4_REQUESTS:(d + 1) * C4_REQUESTS] for d in range(C4_DRAWS)])
    Path(args.output).write_text(json.dumps(rec, indent=1))
    print(json.dumps({"mode": args.mode, "value": rec["value"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
