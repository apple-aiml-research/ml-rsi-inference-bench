#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Structure probes of the served API (README, `run.py api`): what the API-probe node checks, run against your server.

    python api.py --base-url http://127.0.0.1:30000 [--output /tmp/api.json]

Structure only, no numerical bar:
  * `/generate` from text and from `input_ids`, with `return_logprob`, `logprob_start_len 0`, `top_logprobs_num 5`:
    one input log-probability per prompt token, one output log-probability per generated token, five top entries;
  * `/v1/completions` from a text prompt and from a token-ID prompt, with `logprobs 5`;
  * streaming for both endpoints;
  * eight concurrent `/generate` requests completing;
  * every log-probability finite and <= 0, and every length exact.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import sys

import requests

TEXT = ("The relative position bias of this model is learned per token and per head. Summarize, in plain words, why a "
        "sliding window of 512 tokens and a relative extent of 1,024 tokens differ.")
IDS = [1000 + (37 * i) % 5000 for i in range(96)]
NEW = 16
TIMEOUT = 900.0


def lp_ok(values) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) and v <= 0 for v in values)


def generate_probe(base: str, prompt: dict, stream: bool = False) -> dict:
    body = {**prompt, "sampling_params": {"max_new_tokens": NEW, "temperature": 0, "ignore_eos": True},
            "return_logprob": True, "logprob_start_len": 0, "top_logprobs_num": 5, "stream": stream}
    r = requests.post(f"{base}/generate", json=body, timeout=TIMEOUT, stream=stream)
    r.raise_for_status()
    if stream:
        last = None
        chunks = 0
        for line in r.iter_lines(decode_unicode=True):
            if line and line.startswith("data:") and line[5:].strip() != "[DONE]":
                last = json.loads(line[5:])
                chunks += 1
        out = last
    else:
        out, chunks = r.json(), None
    meta = out["meta_info"]
    inp = [x[0] for x in meta["input_token_logprobs"]]
    outp = [x[0] for x in meta["output_token_logprobs"]]
    n_prompt = meta["prompt_tokens"]
    problems = []
    if len(inp) != n_prompt:
        problems.append(f"{len(inp)} input log-probabilities for {n_prompt} prompt tokens")
    if "input_ids" in prompt and n_prompt != len(prompt["input_ids"]):
        problems.append(f"prompt_tokens {n_prompt} != {len(prompt['input_ids'])} ids sent")
    if len(outp) != NEW or meta.get("completion_tokens") != NEW:
        problems.append(f"{len(outp)} output log-probabilities, completion_tokens {meta.get('completion_tokens')}, want {NEW}")
    if not lp_ok(inp[1:]) or not lp_ok(outp):
        problems.append("a log-probability is missing, non-finite or positive")
    tops = [t for t in (meta.get("output_top_logprobs") or [])]
    if len(tops) != NEW or any(len(t) != 5 for t in tops):
        problems.append("output_top_logprobs is not five entries per generated token")
    return {"stream_chunks": chunks, "prompt_tokens": n_prompt, "problems": problems}


def completions_probe(base: str, prompt, stream: bool = False) -> dict:
    body = {"model": "default", "prompt": prompt, "max_tokens": NEW, "temperature": 0, "logprobs": 5,
            "ignore_eos": True, "stream": stream}
    r = requests.post(f"{base}/v1/completions", json=body, timeout=TIMEOUT, stream=stream)
    r.raise_for_status()
    token_lps, tops, chunks = [], [], 0
    if stream:
        for line in r.iter_lines(decode_unicode=True):
            if line and line.startswith("data:") and line[5:].strip() != "[DONE]":
                ch = json.loads(line[5:])["choices"][0]
                lp = ch.get("logprobs") or {}
                token_lps += lp.get("token_logprobs") or []
                tops += lp.get("top_logprobs") or []
                chunks += 1
    else:
        lp = r.json()["choices"][0]["logprobs"]
        token_lps, tops = lp["token_logprobs"], lp["top_logprobs"]
    problems = []
    if len(token_lps) != NEW:
        problems.append(f"{len(token_lps)} token log-probabilities, want {NEW}")
    if not lp_ok(token_lps):
        problems.append("a log-probability is missing, non-finite or positive")
    if len(tops) != NEW or any(len(t) != 5 for t in tops):
        problems.append("top_logprobs is not five entries per generated token")
    return {"stream_chunks": chunks if stream else None, "problems": problems}


def run(base: str) -> dict:
    rec = {}
    for name, fn in (
        ("generate_text", lambda: generate_probe(base, {"text": TEXT})),
        ("generate_ids", lambda: generate_probe(base, {"input_ids": IDS})),
        ("generate_text_stream", lambda: generate_probe(base, {"text": TEXT}, stream=True)),
        ("generate_ids_stream", lambda: generate_probe(base, {"input_ids": IDS}, stream=True)),
        ("completions_text", lambda: completions_probe(base, TEXT)),
        ("completions_ids", lambda: completions_probe(base, IDS)),
        ("completions_text_stream", lambda: completions_probe(base, TEXT, stream=True)),
        ("completions_ids_stream", lambda: completions_probe(base, IDS, stream=True)),
    ):
        try:
            rec[name] = fn()
        except Exception as exc:
            rec[name] = {"problems": [f"{type(exc).__name__}: {exc}"[:500]]}
    with concurrent.futures.ThreadPoolExecutor(8) as ex:
        futs = [ex.submit(generate_probe, base, {"input_ids": IDS[i:] + IDS[:i]}) for i in range(8)]
        res = []
        for f in futs:
            try:
                res.append(f.result())
            except Exception as exc:
                res.append({"problems": [f"{type(exc).__name__}: {exc}"[:500]]})
    rec["concurrent_8"] = {"problems": [p for r in res for p in r["problems"]]}
    rec["all_structural_checks_pass"] = not any(v["problems"] for v in rec.values())
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--output", type=argparse.FileType("w"))
    args = ap.parse_args()
    rec = run(args.base_url)
    for k, v in rec.items():
        if isinstance(v, dict):
            print(f"{'ok  ' if not v['problems'] else 'FAIL'} {k}: {'; '.join(v['problems'])}")
    if args.output:
        json.dump(rec, args.output, indent=1)
    return 0 if rec["all_structural_checks_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
