#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Request-state lifecycle examples and the continuation comparison of `policy.json`.

Start a deployment with `/app/eval/kimi_linear/launch.py --deployment <name>`, then:

    python client.py sample --deployment tp4|dp2|apc|apc-pressure --url http://127.0.0.1:8000 --out record.json

Stop that server, start a cold `tp4` server (`launch.py`, prefix caching off), then:

    python client.py compare --records record.json --cold-url http://127.0.0.1:8000 --out result.json

`sample` runs the deployment's representative example and records every generation: prompt ids, returned ids
and the log-probability returned for each, by the reading rules of `policy.json`.

* tp4: one ordinary fresh one-token request, and the mixed fresh/continued loaded-layer example
  (`../kda/client.py`'s sample RPC, case `mixed_fresh_continued`, assessed by `../kda/numerical-policy.json`);
* dp2: one ordinary concurrent HTTP workload;
* apc and apc-pressure: one warm-prefix/sibling pair, run alone: a 1,024-token prefix with one suffix token,
  then the same prefix with another, sharing one cache salt. The engine's `vllm:prefix_cache_hits` delta for
  the sibling is reported, or reported unavailable when the server exports no such counter.

A feature the deployment does not serve is reported as unsupported with its error; nothing is substituted.

`compare` teacher-forces every recorded prompt plus its exact continuation on the cold server
(`prompt_logprobs` 0) and applies `policy.json`'s rule: the k3 statistic of `../fidelity/kl.py` over the whole
continuation and over every contiguous window of 4 and 16 generated positions. Against your own cold server
this is self-consistency of one implementation, and it is labelled so: it cannot establish agreement with
an independent qualified reference, which evaluation also uses. The examples illustrate the protocol;
passing them does not establish acceptance.

Requests are bounded by the published operation limits (`../operation-limits.json`). The generation and
teacher-forced readers, the window enumeration and the rule are this file's functions, used unchanged by
evaluation.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import math
import struct
import sys
import time
import uuid
from hashlib import sha256
from pathlib import Path

HERE = Path(__file__).resolve().parent
POLICY = json.loads((HERE / "policy.json").read_text())
LIMITS = json.loads((HERE.parent / "operation-limits.json").read_text())
B = POLICY["block_tokens"]
WINDOWS = tuple(POLICY["estimator"]["windows"])
DEPLOYMENTS = ("tp4", "dp2", "apc", "apc-pressure")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


check = _load("infb_lifecycle_fidelity_check", HERE.parent / "fidelity" / "check.py")
kl = check.kl
serving = _load("infb_lifecycle_serving_client", HERE.parent / "serving" / "client.py")


def token_digest(ids) -> str:
    """SHA-256 of the ids as little-endian int32: the token-prefix digest records are joined by."""
    ids = list(ids)
    return sha256(struct.pack(f"<{len(ids)}i", *ids)).hexdigest()


def http_limit_s(prompts) -> float:
    """The published limit of one request or one submitted concurrent batch (operation-limits.json)."""
    longest = max((len(p) for p in prompts), default=0)
    limits = LIMITS["limits_s"]
    return float(limits["http_workload_long"] if longest >= LIMITS["long_input_tokens"] else limits["http_workload"])


class _Cancel(Exception):
    """Raised inside the stream to close the connection: the client disconnect/abort path."""


def generation_body(prompt, max_tokens: int, salt: str, request_id: str, model: str | None = None) -> dict:
    body = {"prompt": list(prompt), "max_tokens": int(max_tokens), "temperature": 0, "ignore_eos": True,
            "skip_special_tokens": False, "logprobs": 1, "return_token_ids": True, "stream": True,
            "cache_salt": salt, "request_id": request_id}
    if model:
        body["model"] = model
    return body


async def generate(base_url: str, prompt, max_tokens: int, *, salt: str, request_id: str, deadline: float,
                   model: str | None = None, on_first=None, cancel_after_first: bool = False) -> dict:
    """One streamed greedy generation, read by `policy.json`'s generation rules into a record.

    `deadline` is a `time.perf_counter()` value: the published limit of the request or of its batch.
    `on_first()` is called when the first returned token arrives; with `cancel_after_first`, the connection is
    closed right after it (the server's disconnect/abort path). The record keeps everything returned before
    an error; `reading_problems` judges it."""
    record = {"request_id": request_id, "salt": salt, "prompt": list(prompt), "prompt_digest": token_digest(prompt),
              "max_tokens": int(max_tokens), "ids": [], "logprobs": [], "finish_reason": None, "status": None,
              "error": None, "cancelled": False, "chunks": 0, "malformed": [], "t_send_unix": time.time(),
              "t_first_unix": None, "t_end_unix": None}

    def on_event(event, arrival):
        if not isinstance(event, dict):
            record["malformed"].append(f"event is {type(event).__name__}")
            return
        if event.get("error") or event.get("object") == "error":
            record["error"] = "server error event: " + json.dumps(event.get("error") or event)[:300]
            return
        for choice in event.get("choices") or []:
            ids = choice.get("token_ids")
            values = (choice.get("logprobs") or {}).get("token_logprobs") if isinstance(
                choice.get("logprobs"), dict) else None
            if not ids and not values:
                if choice.get("finish_reason"):
                    record["finish_reason"] = str(choice["finish_reason"])
                continue
            record["chunks"] += 1
            if not isinstance(ids, list) or not isinstance(values, list) or len(ids) != len(values):
                record["malformed"].append(
                    f"chunk {record['chunks']}: {len(ids) if isinstance(ids, list) else ids!r} token ids and "
                    f"{len(values) if isinstance(values, list) else values!r} log-probabilities")
                continue
            record["ids"].extend(ids)
            record["logprobs"].extend(values)
            if choice.get("finish_reason"):
                record["finish_reason"] = str(choice["finish_reason"])
            if ids and record["t_first_unix"] is None:
                record["t_first_unix"] = time.time()
                if on_first is not None:
                    on_first()
                if cancel_after_first:
                    raise _Cancel()

    host, port = serving.split_url(base_url)
    sse = serving.SSE(on_event)
    body = json.dumps(generation_body(prompt, max_tokens, salt, request_id, model)).encode()
    try:
        status, _, error = await serving.http_stream(host, port, "/v1/completions", body, sse.feed, deadline=deadline)
        record["status"] = status
        if status != 200:
            record["error"] = f"HTTP {status}: {error[:300].decode('utf-8', 'replace')}"
    except _Cancel:
        record["cancelled"], record["status"] = True, 200
    except asyncio.TimeoutError:
        record["error"] = "the request did not complete within its published operation limit"
        record["operation_limit"] = True
    except (OSError, serving.HttpError, ValueError, asyncio.IncompleteReadError) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"[:300]
    record["t_end_unix"] = time.time()
    return record


def reading_problems(record: dict, *, partial: bool = False) -> list[str]:
    """The strict reading rules: the requested length (finish reason `length`), integer ids and one finite
    log-probability per returned token. `partial` is a deliberately cancelled request: at least one returned
    token, every returned token read."""
    problems = list(record.get("malformed") or [])
    if record.get("error"):
        problems.append(record["error"])
    ids, values = record.get("ids") or [], record.get("logprobs") or []
    if not all(isinstance(t, int) and not isinstance(t, bool) for t in ids):
        problems.append("the returned token ids are not all integers")
    bad = [i for i, v in enumerate(values) if not check.finite_number(v)]
    if bad:
        problems.append(f"log-probabilities at {bad[:4]} are not finite numbers: {[values[i] for i in bad[:4]]!r}")
    if len(ids) != len(values):
        problems.append(f"{len(ids)} token ids but {len(values)} log-probabilities")
    if partial:
        if not ids:
            problems.append("no output was returned before cancellation")
    else:
        if record.get("cancelled"):
            problems.append("cancelled")
        if len(ids) != record["max_tokens"]:
            problems.append(f"{len(ids)} output tokens returned for {record['max_tokens']} requested with ignore_eos")
        if record.get("finish_reason") != "length":
            problems.append(f"finish reason {record.get('finish_reason')!r}, not 'length'")
    return problems


async def gather(coroutines):
    return await asyncio.gather(*coroutines)


def run(coroutine):
    return asyncio.run(coroutine)


def model_name(base_url: str) -> str:
    import requests  # noqa: PLC0415 - the evaluation tree's readers use it

    response = requests.get(base_url.rstrip("/") + "/v1/models", timeout=LIMITS["limits_s"]["action"])
    response.raise_for_status()
    return response.json()["data"][0]["id"]


def counters(base_url: str) -> dict:
    """The engine's prefix-cache counters, summed over label sets; `available` False when none are exported."""
    names = ("vllm:prefix_cache_hits", "vllm:prefix_cache_queries")
    got = serving.scrape_counters(base_url, names, time.perf_counter() + LIMITS["limits_s"]["action"])
    if got["available"] and all(v is None for v in got["counters"].values()):
        got["available"] = False
    return got


def counter_delta(before: dict, after: dict, name: str = "vllm:prefix_cache_hits"):
    """The delta of one counter, or None when either reading is unavailable: never a fabricated zero."""
    if not before.get("available") or not after.get("available"):
        return None
    a, b = before["counters"].get(name), after["counters"].get(name)
    return None if a is None or b is None else b - a


def teacher_force(base_url: str, prompt, ids, salt: str) -> list[float]:
    """The cold server's log-probability of each continuation token, teacher-forced (`prompt_logprobs` 0), read by
    `../fidelity/check.py`'s strict reader. Raises its MalformedObservation or OperationLimitExceeded."""
    values = check.candidate_logprobs(base_url.rstrip("/") + "/v1/completions", list(prompt) + list(ids), salt)
    return values[len(prompt) - 1:]


def window_readings(reference, candidate, windows=WINDOWS) -> dict:
    """k3 over the whole continuation and over every contiguous window of each size: `../fidelity/kl.py`'s
    `k3_per_sequence(reference, candidate)` unchanged on each slice. A continuation shorter than a window has
    one window, its complete available length."""
    n = len(candidate)
    if n == 0 or len(reference) != n:
        raise ValueError(f"{len(reference)} reference and {n} candidate log-probabilities")
    out = {"n": n, "full": kl.k3_per_sequence(reference, candidate), "windows": {}}
    for w in windows:
        size = min(int(w), n)
        values = [kl.k3_per_sequence(reference[s:s + size], candidate[s:s + size]) for s in range(n - size + 1)]
        worst = max(range(len(values)), key=lambda s: (not math.isfinite(values[s]), values[s]))
        out["windows"][str(w)] = {"size": size, "max": values[worst], "start": worst, "values": values}
    return out


def tolerance():
    """The frozen T_lifecycle, or None while the policy is pending qualification."""
    value = POLICY["tolerance"]["T_lifecycle"]
    return None if value is None else float(value)


def judge(readings: dict, T: float) -> dict:
    """`policy.json`'s rule: every full and window value finite and at most T."""
    violations = []
    if not (math.isfinite(readings["full"]) and readings["full"] <= T):
        violations.append({"window": "full", "start": 0, "value": readings["full"]})
    for w, r in readings["windows"].items():
        violations += [{"window": w, "start": s, "value": v} for s, v in enumerate(r["values"])
                       if not (math.isfinite(v) and v <= T)]
    return {"T_lifecycle": T, "passed": not violations, "violations": violations}


def _public_tokens(tokenizer_path: str, count: int, offset_documents: int = 0):
    """(`count` documents of the public sample, tokenized, in id order from `offset_documents`; the tokenizer)."""
    tokenizer = serving.load_tokenizer(tokenizer_path)
    docs = serving.load_texts(HERE.parent / "serving" / "sample.jsonl.xz")
    return [tokenizer.encode(d["text"]) for d in docs[offset_documents:offset_documents + count]], tokenizer


def _rpc(base_url: str, method: str, kwargs: dict, timeout: float):
    import requests  # noqa: PLC0415

    response = requests.post(base_url.rstrip("/") + "/collective_rpc",
                             json={"method": method, "kwargs": kwargs, "timeout": timeout}, timeout=timeout + 60)
    response.raise_for_status()
    return response.json()["results"]


def sample(deployment: str, base_url: str, tokenizer_path: str = "/model") -> dict:
    examples = POLICY["examples"]
    nonce = uuid.uuid4().hex[:12]
    model = model_name(base_url)
    record = {"schema": "infb-kimi-state-lifecycle-sample/1", "deployment": deployment, "url": base_url,
              "model": model, "nonce": nonce, "examples": {}, "unsupported": []}

    def one(prompt, n, salt, rid):
        limit = http_limit_s([prompt])
        return generate(base_url, prompt, n, salt=salt, request_id=rid, model=model,
                        deadline=time.perf_counter() + limit)

    if deployment == "tp4":
        documents, tokenizer = _public_tokens(tokenizer_path, 0)
        prompt = tokenizer.encode(examples["one_token"])
        generation = run(one(prompt, examples["one_token_max_tokens"], f"example-{nonce}-one",
                             f"example-{nonce}-one"))
        record["examples"]["fresh_one_token"] = {"generations": [generation],
                                                 "problems": reading_problems(generation)}
        kda = _load("infb_lifecycle_kda_client", HERE.parent / "kda" / "client.py")
        try:
            ranks = _rpc(base_url, "infb_kda_sample", {"seed": str(kda.PROTOCOL["sample"]["seed_default"]),
                                                       "case": examples["loaded_layer_case"]},
                         kda.INVOCATION_LIMIT_S)
            record["examples"]["mixed_loaded_layer"] = {
                "result": ranks, "assessment": kda._assessment().sample_report(ranks)}
        except Exception as exc:  # noqa: BLE001 - reported as unsupported, never substituted
            record["unsupported"].append({"example": "mixed_loaded_layer", "error": f"{type(exc).__name__}: {exc}"[:500]})
    elif deployment == "dp2":
        documents, _ = _public_tokens(tokenizer_path, examples["concurrent_requests"])
        prompts = [d[:examples["concurrent_prompt_tokens"]] for d in documents]
        limit = http_limit_s(prompts)
        generations = run(gather([
            generate(base_url, p, examples["concurrent_max_tokens"], salt=f"example-{nonce}-dp-{i}",
                     request_id=f"example-{nonce}-dp-{i}", model=model, deadline=time.perf_counter() + limit)
            for i, p in enumerate(prompts)]))
        record["examples"]["concurrent"] = {"generations": generations,
                                            "problems": [p for g in generations for p in reading_problems(g)]}
    elif deployment in ("apc", "apc-pressure"):
        documents, _ = _public_tokens(tokenizer_path, 1, offset_documents=1)
        tokens = documents[0]
        prefix = tokens[:examples["warm_prefix_tokens"]]
        suffixes = [tokens[examples["warm_prefix_tokens"]], tokens[examples["warm_prefix_tokens"] + 1]]
        if suffixes[0] == suffixes[1]:
            suffixes[1] = tokens[examples["warm_prefix_tokens"] + 2]
        salt = f"example-{nonce}-pair"
        warm = run(one(prefix + [suffixes[0]], examples["warm_prefix_max_tokens"], salt, f"example-{nonce}-warm"))
        before = counters(base_url)
        sibling = run(one(prefix + [suffixes[1]], examples["warm_prefix_max_tokens"], salt,
                          f"example-{nonce}-sibling"))
        after = counters(base_url)
        hits = counter_delta(before, after)
        record["examples"]["warm_prefix_pair"] = {
            "prefix_tokens": len(prefix), "generations": [warm, sibling],
            "sibling_hit_tokens": hits if hits is not None else "unavailable",
            "counters": {"before": before, "after": after},
            "problems": reading_problems(warm) + reading_problems(sibling)}
    else:
        raise ValueError(f"unknown deployment {deployment!r}; one of {DEPLOYMENTS}")
    return record


def compare(record: dict, cold_url: str) -> dict:
    nonce = uuid.uuid4().hex[:12]
    T = tolerance()
    out = {"schema": "infb-kimi-state-lifecycle-compare/1", "cold_url": cold_url, "deployment": record["deployment"],
           "authority": ("self-consistency: this implementation's cold tp4 server against its own "
                         f"{record['deployment']} deployment. It cannot establish agreement with the independent "
                         "qualified reference that evaluation also uses."),
           "T_lifecycle": T if T is not None else "pending qualification: no verdict",
           "cases": []}
    for name, example in record["examples"].items():
        for generation in example.get("generations", []):
            case = {"example": name, "request_id": generation["request_id"], "n": len(generation["ids"])}
            problems = reading_problems(generation, partial=generation.get("cancelled", False))
            if problems:
                case["reading_problems"] = problems
                out["cases"].append(case)
                continue
            try:
                reference = teacher_force(cold_url, generation["prompt"], generation["ids"], f"compare-{nonce}-"
                                          + generation["request_id"])
            except check.MalformedObservation as exc:
                case["teacher_forcing_error"] = f"{type(exc).__name__}: {exc}"
                out["cases"].append(case)
                continue
            case["readings"] = window_readings(reference, generation["logprobs"])
            if T is not None:
                case["verdict"] = judge(case["readings"], T)
            out["cases"].append(case)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sample", help="run the deployment's representative example and record it")
    s.add_argument("--deployment", choices=DEPLOYMENTS, required=True)
    s.add_argument("--url", default="http://127.0.0.1:8000")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--tokenizer", default="/model", help="the checkpoint directory (its tokenizer)")
    c = sub.add_parser("compare", help="teacher-force the recorded continuations on a cold tp4 server")
    c.add_argument("--records", type=Path, required=True)
    c.add_argument("--cold-url", required=True)
    c.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "sample":
        result = sample(args.deployment, args.url, args.tokenizer)
        problems = [p for e in result["examples"].values() for p in e.get("problems", [])]
        status = 1 if problems or result["unsupported"] else 0
    else:
        result = compare(json.loads(args.records.read_text()), args.cold_url)
        failed = [c for c in result["cases"] if "readings" not in c
                  or not c.get("verdict", {"passed": True})["passed"]]
        status = 1 if failed else 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    summary = {k: v for k, v in result.items() if k not in ("examples", "cases")}
    if args.command == "compare":
        summary["cases"] = [{k: v for k, v in c.items() if k != "readings"}
                            | ({"full": c["readings"]["full"],
                                **{f"max_{w}": r["max"] for w, r in c["readings"]["windows"].items()}}
                               if "readings" in c else {}) for c in result["cases"]]
    print(json.dumps(summary, indent=1, default=str))
    return status


if __name__ == "__main__":
    sys.exit(main())
