#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Compare generated-token log-probabilities with fresh-prefill log-probabilities.

Each draw flushes the prefix cache, submits one batch of greedy generations, and re-scores
each prompt plus continuation individually. The metric is the maximum across requests of
the median absolute decode/prefill log-probability difference across generated tokens.
Missing or non-finite values produce an infinite difference.

The command line builds public prompts from a text sample:

    python3 /app/eval/gpt_oss/decode_batch/probe.py --base-url http://127.0.0.1:8000 --log server.log \
        --tokenizer /model/tokenizer.json --texts /app/eval/gpt_oss/efficiency/sample.jsonl.xz

With `--log` (the server's stdout/stderr log) the batch size defaults to its largest reported
CUDA-graph capture; `--n` supplies a batch size explicitly. The client prints each request's
median, the maximum and a JSON summary, and exits 1 when the maximum reaches a supplied `--bar`.
The check uses the standard library; building prompts additionally needs the supplied
`tokenizers` package. These observations do not establish complete task acceptance.
"""
from __future__ import annotations

import argparse
import json
import lzma
import math
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gpt_oss.fidelity import aligned_values

PROMPT_TOKENS = 256
NEW_TOKENS = 192
RESCORE_CONCURRENCY = 1

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def captured_sizes(log_text: str) -> list[int]:
    """The decode batch sizes rank 0 captured as CUDA graphs, from its `Capture cuda graph bs [...]` line; empty if
    the log has none."""
    lines = re.findall(r"\bTP0\] Capture cuda graph bs \[([0-9, ]*)\]", log_text)
    return sorted({int(x) for x in lines[-1].split(",") if x.strip()}) if lines else []


def _post(base_url: str, path: str, body: dict, timeout: float):
    req = urllib.request.Request(f"{base_url}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with _OPENER.open(req, timeout=timeout) as r:
        return json.loads(r.read())


def flush_cache(base_url: str, attempts: int = 3, wait_s: float = 2.0) -> dict:
    """SGLang's `/flush_cache`. It empties the radix cache, so every prompt is prefilled anew. The engine does it only
    when no request is running or waiting, and answers 200 when it has."""
    last = None
    for i in range(1, attempts + 1):
        try:
            with _OPENER.open(urllib.request.Request(f"{base_url}/flush_cache", data=b"", method="POST"),
                              timeout=300) as r:
                if r.status == 200:
                    return {"flushed": True, "attempts": i}
                last = r.status
        except urllib.error.HTTPError as exc:
            last = exc.code
        except Exception as exc:  # noqa: BLE001 - reported with the draw
            last = f"{type(exc).__name__}: {exc}"[:200]
        time.sleep(wait_s)
    return {"flushed": False, "attempts": attempts, "last": last}


def decode_batch(base_url: str, prompts: list[list[int]], new_tokens: int = NEW_TOKENS,
                 timeout: float = 900.0, raw: list | None = None) -> list[tuple[list, list]]:
    """Greedy continuations of every prompt, in one batched `/generate` request (`input_ids` a list of prompts): per
    prompt, in order, each generated token's decode log-probability, and the tokens. With `raw`, each response's
    request ID and finish reason are appended to it."""
    outs = _post(base_url, "/generate", {"input_ids": prompts, "return_logprob": True,
                                         "sampling_params": {"max_new_tokens": new_tokens, "temperature": 0.0,
                                                             "ignore_eos": True}}, timeout)
    if not isinstance(outs, list) or len(outs) != len(prompts):
        raise AssertionError(f"a batched /generate of {len(prompts)} prompts returned "
                             f"{len(outs) if isinstance(outs, list) else type(outs).__name__} results")
    res = []
    for out in outs:
        dec = out["meta_info"]["output_token_logprobs"]
        if len(dec) != new_tokens:
            raise AssertionError(f"asked for {new_tokens} greedy tokens with ignore_eos, got {len(dec)}")
        tokens = [e[1] for e in dec]
        res.append((aligned_values(dec, tokens, first_null=False), tokens))
        if raw is not None:
            raw.append({"id": out["meta_info"].get("id"), "finish_reason": out["meta_info"].get("finish_reason"),
                        "completion_tokens": out["meta_info"].get("completion_tokens")})
    return res


def prefill(base_url: str, prompt: list[int], tokens: list[int], timeout: float = 900.0) -> list:
    """The same tokens' log-probabilities when prompt + continuation is prefilled from scratch (`logprob_start_len` 0
    keeps the radix cache out of it). `max_new_tokens` 0 where the engine accepts it; one extra token changes no input
    value."""
    last = None
    for max_new in (0, 1):
        try:
            out = _post(base_url, "/generate", {"input_ids": prompt + tokens, "return_logprob": True,
                                                "logprob_start_len": 0,
                                                "sampling_params": {"max_new_tokens": max_new, "temperature": 0.0}},
                        timeout)["meta_info"]
            break
        except urllib.error.HTTPError as exc:
            last = exc
    else:
        raise last
    vals = aligned_values(out["input_token_logprobs"], prompt + tokens)
    if len(vals) != len(prompt) + len(tokens):
        raise AssertionError(f"prefill returned {len(vals)} input logprobs for {len(prompt) + len(tokens)} tokens")
    return vals[len(prompt):]


def medians(decoded: list, prefilled: list) -> list[float]:
    """Per request, the median over generated tokens of |decode - prefill|; a missing or non-finite value on either
    side counts as infinitely far."""
    result = []
    for (dec, _), pre in zip(decoded, prefilled):
        d = [abs(a - b) if a is not None and b is not None and math.isfinite(a) and math.isfinite(b) else math.inf
             for a, b in zip(dec, pre)]
        result.append(statistics.median(d))
    return result


def probe(base_url: str, prompts: list[list[int]], new_tokens: int = NEW_TOKENS,
          rescore_concurrency: int = RESCORE_CONCURRENCY, raw: list | None = None) -> dict:
    """One draw: flush, every prompt in one batched request, then every sequence re-scored by prefill, one at a time.
    A teacher-forced request returns every position's float32 log-probabilities over the whole vocabulary, so the
    re-scoring is sequential."""
    n = len(prompts)
    flush = flush_cache(base_url)
    decoded = decode_batch(base_url, prompts, new_tokens, raw=raw)
    with ThreadPoolExecutor(rescore_concurrency) as ex:
        prefilled = list(ex.map(lambda i: prefill(base_url, prompts[i], decoded[i][1]), range(n)))
    values = medians(decoded, prefilled)
    return {"n": n, "medians": values, "worst": max(values), "median": statistics.median(values),
            "flush": flush}


def text_prompts(tokenizer: str, texts: str, n: int, length: int = PROMPT_TOKENS) -> list[list[int]]:
    """n consecutive windows of `length` tokens from a .jsonl or .jsonl.xz of {"text": ...} records."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tokenizer)
    ids: list[int] = []
    with (lzma.open(texts, "rt") if texts.endswith(".xz") else open(texts)) as f:
        for line in f:
            ids.extend(tok.encode(json.loads(line)["text"], add_special_tokens=False).ids)
            if len(ids) >= n * length:
                break
    if len(ids) < n * length:
        raise SystemExit(f"{texts} holds {len(ids)} tokens, fewer than {n} prompts of {length}")
    return [ids[i * length:(i + 1) * length] for i in range(n)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--log", help="the server's stdout/stderr file, to default --n to its largest captured graph")
    ap.add_argument("--n", type=int, help="the batch size")
    ap.add_argument("--tokenizer", default="/model/tokenizer.json")
    ap.add_argument("--texts", default="/app/eval/gpt_oss/efficiency/sample.jsonl.xz")
    ap.add_argument("--bar", type=float, default=None, help="exit 1 if the worst request's median reaches it")
    args = ap.parse_args()
    n = args.n
    if n is None:
        if not args.log:
            raise SystemExit("give --log (the server's launch log) or --n")
        sizes = captured_sizes(open(args.log, errors="replace").read())
        if not sizes:
            raise SystemExit(f"{args.log} has no 'Capture cuda graph bs [...]' line on rank 0")
        n = max(sizes)
    r = probe(args.base_url, text_prompts(args.tokenizer, args.texts, n))
    for i, m in enumerate(r["medians"]):
        print(f"request {i:3d}: median |decode - prefill| {m:.6g}")
    print(json.dumps({k: r[k] for k in ("n", "worst", "median", "flush")}))
    return 1 if args.bar is not None and r["worst"] >= args.bar else 0


if __name__ == "__main__":
    sys.exit(main())
