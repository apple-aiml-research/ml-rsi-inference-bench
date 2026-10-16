#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Serving recipe, reasoning-effort and decode/prefill measurement protocols.

These are the functions the private verifier runs, with its separate inputs.
"""
import math
import argparse
import json
from pathlib import Path
import re
import statistics
from concurrent.futures import ThreadPoolExecutor
import requests
from .fidelity import aligned_values

LEVELS = ("low", "medium", "high")
TP = 4
ATTENTION_BACKEND = "triton"
DECODE_NEW_TOKENS = 64

def completion_tokens(base_url: str, prompt: str, effort: str, timeout: float = 900.0, *, model="openai/gpt-oss-120b", max_tokens=4096) -> int:
    body = {"model": model,
            "messages": [{"role": "user", "content": prompt}],
            "reasoning_effort": effort, "temperature": 0.0, "max_tokens": max_tokens}
    r = requests.post(f"{base_url}/v1/chat/completions", json=body, timeout=timeout)
    assert r.status_code == 200, f"/v1/chat/completions at reasoning_effort={effort!r} returned {r.status_code}: {r.text[:300]}"
    return int(r.json()["usage"]["completion_tokens"])


def effort_lengths(base_url: str, prompts: list[str], *, model="openai/gpt-oss-120b", max_tokens=4096) -> dict:
    """Completion tokens over prompts at each level, their mean and median; the levels are asked one after another."""
    out = {}
    for effort in LEVELS:
        with ThreadPoolExecutor(len(prompts)) as ex:
            n = list(ex.map(lambda p: completion_tokens(base_url, p, effort, model=model, max_tokens=max_tokens), prompts))
        out[effort] = {"mean": statistics.fmean(n), "median": statistics.median(n), "per_prompt": n,
                       "at_max_tokens": sum(x >= max_tokens for x in n)}
    return out


def recipe_findings(info: dict, log_text: str, *, model="openai/gpt-oss-120b") -> tuple[dict, list[str]]:
    """What the recipe fixes, read from /get_server_info and the launch log; the list is every mismatch."""
    served = {k: info.get(k) for k in (
        "model_path", "tp_size", "dp_size", "ep_size", "enable_dp_attention", "enable_ep_moe", "enable_deepep_moe",
        "disable_cuda_graph", "cuda_graph_max_bs", "mem_fraction_static", "quantization", "speculative_algorithm",
        "attention_backend", "enable_triton_kernel_moe", "max_total_tokens", "reasoning_parser")}
    bad = []
    want = {"model_path": model, "tp_size": TP, "dp_size": 1,
            "ep_size": 1, "enable_dp_attention": False, "enable_ep_moe": False, "enable_deepep_moe": False,
            "disable_cuda_graph": False, "cuda_graph_max_bs": 200, "mem_fraction_static": 0.93,
            "speculative_algorithm": None}
    for k, v in want.items():
        if k not in info:
            if k in ("enable_ep_moe", "enable_deepep_moe"):
                continue
            bad.append(f"{k}: absent from the server's own configuration")
        elif info[k] != v:
            bad.append(f"{k}: served {info[k]!r}, the recipe holds {v!r}")
    if info.get("quantization") not in (None, "mxfp4"):
        bad.append(f"quantization: served {info.get('quantization')!r}; the checkpoint declares mxfp4 and the recipe "
                   "does not override it")
    # The attention backend graded here is Triton, the only backend at this base that carries the attention sink.
    # Graded where it is built: at this base the model runner records the prefill and decode backends it constructs
    # in its scheduler's global server arguments, after every model-specific adjustment, and /get_server_info returns
    # each scheduler's as `internal_states`. The tokenizer's own field above and the launch log's server_args line
    # are printed, not graded: both are written before the model runner runs.
    built = [[s.get("prefill_attention_backend"), s.get("decode_attention_backend")]
             for s in info.get("internal_states") or [] if isinstance(s, dict)]
    served["attention_backend_built"] = built
    if not built or any(b != [ATTENTION_BACKEND, ATTENTION_BACKEND] for b in built):
        bad.append(f"the attention backend each scheduler built (prefill, decode): {built or 'not reported'}; the "
                   f"instruction names {ATTENTION_BACKEND!r}")
    served["attention_backend_in_launch_log"] = re.findall(
        r"server_args=ServerArgs\(.*?\battention_backend=(None|'[^']*')", log_text)
    ranks = sorted({int(m) for m in re.findall(r"\bTP(\d+)\] KV Cache is allocated", log_text)})
    served["ranks_reporting_kv_cache"] = ranks
    if ranks != list(range(TP)):
        bad.append(f"ranks reporting a KV cache in the launch log: {ranks}, not all {TP} tensor-parallel ranks")
    graphs = bool(re.search(r"\bTP0\] Capture cuda graph end", log_text))
    served["rank0_captured_cuda_graphs"] = graphs
    if not graphs:
        bad.append("the launch log has no 'Capture cuda graph end' on rank 0: CUDA graphs were not captured")
    return served, bad


def _generate(base_url: str, body: dict, timeout: float = 900.0) -> dict:
    r = requests.post(f"{base_url}/generate", json=body, timeout=timeout)
    assert r.status_code == 200, f"/generate returned {r.status_code}: {r.text[:300]}"
    return r.json()["meta_info"]


def decode_prefill_deltas(base_url: str, prefix: list[int], n_new: int = DECODE_NEW_TOKENS) -> list[float]:
    """|decode - prefill| log-probability of each of n_new greedily decoded tokens after `prefix`.

    The decode request returns each generated token with the log-probability the decode path gave it; the
    scoring request prefills prefix + continuation from scratch (`logprob_start_len` 0 keeps the radix cache
    out of it) and returns every input token's log-probability, entry i being token i's given tokens < i.
    A missing or non-finite value on either side counts as infinitely far."""
    out = _generate(base_url, {"input_ids": prefix, "return_logprob": True,
                               "sampling_params": {"max_new_tokens": n_new, "temperature": 0.0, "ignore_eos": True}})
    dec = out["output_token_logprobs"]
    assert len(dec) == n_new, f"asked for {n_new} greedy tokens with ignore_eos, got {len(dec)}"
    toks = [e[1] for e in dec]
    last = None
    for max_new in (0, 1):
        try:
            pre = _generate(base_url, {"input_ids": prefix + toks, "return_logprob": True, "logprob_start_len": 0,
                                       "sampling_params": {"max_new_tokens": max_new, "temperature": 0.0}})
            break
        except AssertionError as exc:
            last = exc
    else:
        raise last
    vals = aligned_values(pre["input_token_logprobs"], prefix + toks)
    aligned_values(dec, toks, first_null=False)
    assert len(vals) == len(prefix) + n_new, f"prefill returned {len(vals)} input logprobs for {len(prefix) + n_new} tokens"
    deltas = []
    for j, e in enumerate(dec):
        a, b = e[0], vals[len(prefix) + j]
        ok = a is not None and b is not None and math.isfinite(a) and math.isfinite(b)
        deltas.append(abs(a - b) if ok else math.inf)
    return deltas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", choices=("recipe", "effort", "decode"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--inputs", type=Path, help="JSON list of questions (effort) or token IDs (decode)")
    parser.add_argument("--log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.protocol == "recipe":
        if args.log is None:
            parser.error("recipe requires --log with the server's launch output")
        response = requests.get(args.base_url.rstrip("/") + "/get_server_info", timeout=300)
        response.raise_for_status()
        observed, mismatches = recipe_findings(response.json(), args.log.read_text(), model=args.model)
        result = {"observed": observed, "mismatches": mismatches}
    elif args.protocol == "effort":
        questions = json.loads(args.inputs.read_text()) if args.inputs else [
            "How many positive divisors does 360 have?", "What is the sum of the first 20 odd numbers?"]
        result = effort_lengths(args.base_url, questions, model=args.model)
    else:
        if args.inputs is None:
            parser.error("decode requires --inputs with a JSON list of prefix token IDs")
        result = {"deltas": decode_prefill_deltas(args.base_url, json.loads(args.inputs.read_text()))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
