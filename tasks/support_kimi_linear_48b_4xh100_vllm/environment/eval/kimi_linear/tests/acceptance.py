#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Accuracy, API, concurrency and served-recipe checks against a server at the prescribed recipe.

Start the server with `/app/eval/kimi_linear/launch.py` (keep its output with `--log` for the
recipe check), then run one mode at a time:

    python acceptance.py --mode accuracy --url http://127.0.0.1:8000
    python acceptance.py --mode api --url http://127.0.0.1:8000
    python acceptance.py --mode concurrency --url http://127.0.0.1:8000
    python acceptance.py --mode recipe --log /tmp/kimi-server.log [--deployment dp2]

Each mode prints its observations and fails on the conditions it checks locally, including
malformed answers and departures from the supplied serving configuration. Accuracy and
near-tie/concurrency measurements do not by themselves determine task acceptance. A compared
generation must carry its token ids and, at every position, the token and a map of finite top
log-probabilities; one that does not is a failed observation, and nothing read after it replaces it.
Requests are bounded by the published operation limits (`../operation-limits.json`).

* accuracy: vLLM's own GSM8K evaluator (`upstream/gsm8k_eval.py`, verbatim), all 1,319 test
  questions, 5-shot, 256 tokens, temperature 0, seed 42, run offline against the server: it
  reports accuracy and the invalid-answer rate. The evaluator submits every question at once, so
  the run is one concurrently submitted batch under the 900 s request limit; its one request
  clock (upstream's 600 s) is set to that limit.
* api: the served model list; a greedy completion that is non-empty with finite
  log-probabilities; `max_tokens=1` returning exactly one token; two identical greedy requests
  agreeing; a chat completion through the checkpoint's chat template; and batch comparisons:
  greedy prompts served one at a time, then concurrently, compared token by token.
  Where a batched answer diverges, the reading at the first
  divergence is how far each run's token is from the other run's top-1, from the top-5
  log-probabilities (infinite when either token is outside the other's top-5).
* concurrency: prompts of 1,024 to 12,000 tokens decoded greedily for 256 tokens, first one at a
  time and then all at once; per prompt the same first-divergence reading from the top-20
  log-probabilities, and medians for the input groups in the supplied fixture.
* recipe: the launch's own output: the deployment's arguments as parsed, the engine's configuration,
  the model compiled, CUDA graphs in FULL_AND_PIECEWISE mode and FlashMLA on every worker, for the
  `launch.py` deployment named by `--deployment` (default `tp4`).
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import concurrent.futures as cf
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
MODEL = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
MAX_MODEL_LEN = 1048576
DATASET_DIR = HERE.parent / "datasets" / "gsm8k"
GSM8K = {"num_questions": 1319, "num_shots": 5, "max_tokens": 256, "seed": 42}
LIMITS = json.loads((HERE.parent / "operation-limits.json").read_text())


class MalformedResponse(AssertionError):
    """A served response that is not a structurally valid observation. Terminal for its check."""


def request_limit_s(prompt) -> float:
    """The operation limit of one request: token prompts by their length, text prompts by their character count,
    which is never below their token count."""
    limits = LIMITS["limits_s"]
    return float(limits["http_workload_long"] if len(prompt) >= LIMITS["long_input_tokens"] else limits["http_workload"])


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def generation(out: dict, *, expect_tokens: int | None = None) -> dict:
    """The token ids, tokens and top log-probabilities of one generation, validated before any comparison."""
    try:
        c = out["choices"][0]
    except (KeyError, IndexError, TypeError):
        raise MalformedResponse(f"the response is not a completion: {str(out)[:300]}") from None
    ids, lp = c.get("token_ids"), c.get("logprobs")
    if not isinstance(ids, list) or not all(isinstance(t, int) and not isinstance(t, bool) for t in ids):
        raise MalformedResponse(f"the completion carries no list of token ids: {str(ids)[:120]}")
    if expect_tokens is not None and len(ids) != expect_tokens:
        raise MalformedResponse(f"the completion returned {len(ids)} token ids, not {expect_tokens}")
    if not isinstance(lp, dict):
        raise MalformedResponse("the completion carries no log-probabilities")
    toks, top = lp.get("tokens"), lp.get("top_logprobs")
    if not isinstance(toks, list) or not isinstance(top, list) or not (len(toks) == len(top) == len(ids)):
        raise MalformedResponse(f"{len(ids)} token ids against {len(toks) if isinstance(toks, list) else toks!r} tokens "
                                f"and {len(top) if isinstance(top, list) else top!r} top log-probability maps")
    for i, entry in enumerate(top):
        if not isinstance(entry, dict) or not entry or not all(_finite(v) for v in entry.values()):
            raise MalformedResponse(f"position {i}: the top log-probabilities are {str(entry)[:120]}, not a map of "
                                    "finite numbers")
    return {"ids": list(ids), "toks": list(toks), "top": list(top)}


def completion(completions_url: str, prompt, *, model: str = MODEL, **kw) -> dict:
    """One /v1/completions request, with the fields every timed or compared request carries."""
    import requests

    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": kw.pop("max_tokens", 32),
        "temperature": 0,
        "logprobs": kw.pop("logprobs", 1),
        "cache_salt": kw.pop("cache_salt", "infb-probe"),
    }
    body.update(kw)
    limit = request_limit_s(prompt)
    try:
        r = requests.post(completions_url, json=body, timeout=limit)
    except requests.exceptions.Timeout:
        raise MalformedResponse(f"a request did not complete within its {limit:g} s limit") from None
    if r.status_code != 200:
        raise MalformedResponse(f"the server answered HTTP {r.status_code}: {r.text[:300]!r}")
    try:
        return r.json()
    except ValueError:
        raise MalformedResponse(f"the response is not JSON: {r.text[:300]!r}") from None


def near_tie_gap(a: dict, b: dict, p: int) -> float:
    """How far each run's token at position p is from the other run's top-1, the larger of the two; infinite when
    either token is outside the other's returned top-k."""
    try:
        ta, tb = a["top"][p] or {}, b["top"][p] or {}
        ga = max(ta.values()) - ta.get(b["toks"][p], float("-inf"))
        gb = max(tb.values()) - tb.get(a["toks"][p], float("-inf"))
    except (IndexError, ValueError, TypeError):
        return float("inf")
    return max(ga, gb)


def load_evaluator(path: Path = HERE / "upstream" / "gsm8k_eval.py"):
    spec = importlib.util.spec_from_file_location("infb_gsm8k_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seed_datasets(dataset_dir: Path = DATASET_DIR) -> None:
    for name in ("train.jsonl", "test.jsonl"):
        src, dst = Path(dataset_dir) / name, Path("/tmp") / name
        assert src.is_file(), f"the offline dataset is missing: {src}"
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            dst.write_bytes(src.read_bytes())


class _RequestClock:
    """The evaluator's `aiohttp`, with its one request clock at the batch's published limit. Every question is
    dispatched at once, so each request's remaining allowance is the whole batch limit; upstream's 600 s would end
    requests the limit still allows."""

    def __init__(self, module, limit_s: float):
        self._module, self._limit_s = module, limit_s

    def __getattr__(self, name):
        return getattr(self._module, name)

    def ClientTimeout(self, *args, **kwargs):  # noqa: N802 - aiohttp's own name
        return self._module.ClientTimeout(total=self._limit_s)


def accuracy(port: int, *, dataset_dir: Path = DATASET_DIR, evaluator: Path = HERE / "upstream" / "gsm8k_eval.py",
             num_questions: int = 1319, num_shots: int = 5, max_tokens: int = 256, seed: int = 42) -> dict:
    """Upstream's evaluator, upstream's protocol, this task's recipe: accuracy and the invalid-answer rate."""
    seed_datasets(dataset_dir)
    module = load_evaluator(evaluator)
    limit = LIMITS["limits_s"]["http_workload"]
    module.aiohttp = _RequestClock(module.aiohttp, limit)
    try:
        result = module.evaluate_gsm8k(
            num_questions=num_questions, num_shots=num_shots, max_tokens=max_tokens, port=port,
            temperature=0.0, seed=seed)
    except (TimeoutError, asyncio.TimeoutError):
        raise MalformedResponse(f"the GSM8K batch did not complete within its {limit:g} s limit") from None
    print("INFB_GSM8K " + json.dumps({k: v for k, v in result.items()}))
    return result


def api_probes(url_for, probes: dict, *, model: str = MODEL, top_k: int = 5, near_tie_nats: float | None = None) -> list:
    """Check ordinary APIs and compare serial with concurrent greedy generations.

    Batch comparisons use token IDs and report the first-divergence near-tie gap defined by
    `near_tie_gap`. `near_tie_nats`, when supplied, bounds that gap; otherwise the gaps are
    reported without a numerical verdict. Returns the batch report.
    """
    import requests

    def post(prompt, **kw):
        return completion(url_for("v1/completions"), prompt, model=model, **kw)

    served = [m["id"] for m in requests.get(url_for("v1/models"), timeout=LIMITS["limits_s"]["action"]).json().get("data", [])]
    print("INFB_SERVED " + json.dumps(served))
    assert served == [model], (
        f"the server serves {served}, not exactly [{model!r}]"
    )

    out = post(probes["single"]["prompt"], max_tokens=32)
    choice = out["choices"][0]
    assert choice["text"], f"empty completion: {out}"
    lp = [v for v in (choice.get("logprobs") or {}).get("token_logprobs", []) if v is not None]
    assert lp, f"no log-probabilities returned: {out}"
    assert all(v == v and v != float("-inf") for v in lp), f"non-finite log-probabilities: {lp[:8]}"

    one = post(probes["single"]["prompt"], max_tokens=1)
    assert one["usage"]["completion_tokens"] == 1, one["usage"]

    a = post(probes["single"]["prompt"], max_tokens=24)
    b = post(probes["single"]["prompt"], max_tokens=24)
    assert a["choices"][0]["text"] == b["choices"][0]["text"], (
        "two identical greedy requests disagreed:\n"
        f"  {a['choices'][0]['text']!r}\n  {b['choices'][0]['text']!r}"
    )

    chat = requests.post(
        url_for("v1/chat/completions"),
        json={
            "model": model,
            "messages": probes["chat"]["messages"],
            "max_tokens": 24,
            "temperature": 0,
        },
        timeout=LIMITS["limits_s"]["http_workload"],
    )
    chat.raise_for_status()
    assert chat.json()["choices"][0]["message"]["content"].strip(), chat.json()

    prompts = probes["batch_invariance"]["prompts"]
    n_tokens = probes["batch_invariance"]["max_tokens"]

    def run_for(prompt: str) -> dict:
        return generation(post(prompt, max_tokens=n_tokens, logprobs=top_k, return_token_ids=True))

    serial = [run_for(p) for p in prompts]
    assert all(s["ids"] for s in serial), "a serial probe returned no tokens"

    report = []
    for size in probes["batch_invariance"]["group_sizes"]:
        group = prompts[:size]
        with cf.ThreadPoolExecutor(max_workers=size) as pool:
            concurrent = list(pool.map(run_for, group))
        for i, (alone, batched) in enumerate(zip(serial[:size], concurrent, strict=True)):
            if alone["ids"] == batched["ids"]:
                report.append({"size": size, "prompt": i, "equal": True})
                continue
            p = next((k for k, (x, y) in enumerate(zip(alone["ids"], batched["ids"])) if x != y),
                     min(len(alone["ids"]), len(batched["ids"])))
            gap = near_tie_gap(alone, batched, p)
            report.append({"size": size, "prompt": i, "equal": False, "first_divergence": p, "gap": gap})
            if near_tie_nats is not None:
                assert gap <= near_tie_nats, (
                    f"batch invariance failed at group size {size}, prompt {i}: the batched run leaves the solo "
                    f"run at token {p} by {gap:.3f} nats (a correct server's divergences are near-ties, within "
                    f"{near_tie_nats} nats and inside each other's top-{top_k}).\n"
                    f"  alone:      {alone['ids'][max(0, p - 4):p + 8]}\n"
                    f"  in a batch: {batched['ids'][max(0, p - 4):p + 8]}"
                )
    print("INFB_BATCH_INVARIANCE " + json.dumps(report))
    return report


def concurrency(url_for, prompts: list, bands: list, *, model: str = MODEL, max_tokens: int = 256, top_k: int = 20,
                bars: dict | None = None) -> dict:
    """Each prompt decoded greedily, first alone and then all at once; per prompt the first-divergence gap.

    Every prompt carries its `band`. Each band's reading is the median gap over its prompts. With `bars`
    (a bar per band), a band whose median is not below its bar is re-drawn once, the whole batch sent at
    once again, and passes when either draw is below its bar; without `bars`, one at-once draw is
    reported. Returns every draw's rows and medians, and `passed` when bars were given.
    """
    by_band = {band: [p for p in prompts if p["band"] == band] for band in bands}

    def run_for(p: dict, tag: str) -> dict:
        return generation(completion(url_for("v1/completions"), p["ids"], model=model, max_tokens=max_tokens,
                                     logprobs=top_k, ignore_eos=True, skip_special_tokens=False, return_token_ids=True,
                                     cache_salt=f"infb-concurrency-{p['id']}-{tag}"), expect_tokens=max_tokens)

    alone = {p["id"]: run_for(p, "alone") for p in prompts}

    def at_once(draw: int) -> dict:
        with cf.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            answers = list(pool.map(lambda p: run_for(p, f"at-once-d{draw}"), prompts))
        rows = {}
        for p, batched in zip(prompts, answers, strict=True):
            solo = alone[p["id"]]
            if solo["ids"] == batched["ids"]:
                rows[p["id"]] = {"gap": 0.0, "equal": True}
                continue
            k = next((i for i, (x, y) in enumerate(zip(solo["ids"], batched["ids"])) if x != y),
                     min(len(solo["ids"]), len(batched["ids"])))
            rows[p["id"]] = {"gap": near_tie_gap(solo, batched, k), "equal": False, "first_divergence": k,
                             "alone": solo["ids"][max(0, k - 4):k + 8], "at_once": batched["ids"][max(0, k - 4):k + 8]}
        return {"rows": rows, "medians": {band: median(rows[p["id"]]["gap"] for p in ps) for band, ps in by_band.items()}}

    # Written `not (value < bar)` on purpose: a NaN compares False against everything.
    draws = [at_once(1)]
    out = {"top_k": top_k, "draws": draws, "band_medians": None}
    if bars is not None:
        if any(not (draws[0]["medians"][band] < bars[band]) for band in bands):
            draws.append(at_once(2))
        out["passed"] = {band: any(d["medians"][band] < bars[band] for d in draws) for band in bands}
    out["band_medians"] = [d["medians"] for d in draws]
    return out


RECIPE_ARGS = {"tensor_parallel_size": 4, "max_model_len": MAX_MODEL_LEN, "trust_remote_code": True,
               "long_prefill_token_threshold": 2048, "worker_extension_cls": "worker.KDAWorker",
               "enable_prefix_caching": False}
DEPLOYMENT_ARGS = {
    "tp4": {},
    "dp2": {"data_parallel_size": 2},
    "apc": {"enable_prefix_caching": True, "mamba_block_size": 512},
    "apc-pressure": {"enable_prefix_caching": True, "mamba_block_size": 512, "max_model_len": 16384,
                     "num_gpu_blocks_override": 64, "max_num_seqs": 16, "max_num_batched_tokens": 8192},
}


def recipe_problems(text: str, *, max_model_len: int | None = None, deployment: str = "tp4") -> tuple[list[str], dict]:
    """The launch serves `deployment` as `launch.py` states it, read from the server's own output and from every
    worker's own lines:

      * the arguments the deployment names, as the server parsed them -- for `tp4`, tensor-parallel size 4,
        max-model-len 1,048,576, trust-remote-code, long-prefill-token-threshold 2,048 and prefix caching off;
        each other deployment with its stated difference -- and nothing beyond `launch.py`'s model, port, seed
        and the declared verification worker extension;
      * the defaults that stay, as the engine configured itself: one pipeline stage, the deployment's
        data-parallel size, no quantization, not eager, BF16 weights and KV cache, prefix caching as the
        deployment sets it, chunked prefill on with max_num_batched_tokens 8,192, compilation mode VLLM_COMPILE
        with CUDA graphs in FULL_AND_PIECEWISE mode, and FlashMLA for the MLA layers on every worker (four, or
        eight for `dp2`); for `apc-pressure`, the 64-block cache override applied;
      * and the model compiled: the compilation monitor's `torch.compile takes ... s in total`, which the
        engine prints only when it compiled the model (on each data-parallel replica).
    Returns (problems, the parsed summary).
    """
    want = {**RECIPE_ARGS, **DEPLOYMENT_ARGS[deployment]}
    if max_model_len is not None:
        want["max_model_len"] = max_model_len
    replicas = want.get("data_parallel_size", 1)
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    problems = []

    parsed = re.findall(r"non-default args: (\{.*\})", text)
    args = ast.literal_eval(parsed[-1]) if parsed else {}
    if not parsed:
        problems.append("the server printed no `non-default args`")
    for key, value in want.items():
        if parsed and args.get(key) != value:
            problems.append(f"the argument {key} is {args.get(key)!r}; the {deployment} deployment's is {value!r}")
    extra = sorted(set(args) - set(want) - {"model", "model_tag", "port", "seed"})
    if extra:
        problems.append(f"arguments beyond the {deployment} deployment: {extra}")

    configs = re.findall(r"with config: (.*)", text)
    if not configs:
        problems.append("the server printed no engine configuration")
    else:
        config = configs[-1]
        for field, value in (("tensor_parallel_size", "4"), ("pipeline_parallel_size", "1"),
                             ("data_parallel_size", str(replicas)), ("quantization", "None"),
                             ("enforce_eager", "False"), ("kv_cache_dtype", "auto"), ("dtype", "torch.bfloat16"),
                             ("max_seq_len", str(want["max_model_len"])),
                             ("enable_prefix_caching", str(want["enable_prefix_caching"])),
                             ("chunked_prefill_enabled", "True"), ("speculative_config", "None")):
            got = re.search(rf"(?:^|, ){field}=([^,]*)", config)
            if not got or got.group(1) != value:
                problems.append(f"the engine's {field} is {got.group(1) if got else 'not reported'}; the "
                                f"{deployment} deployment's is {value}")
        if not re.search(r"'mode': (3|<CompilationMode\.VLLM_COMPILE: 3>)[,}]", config):
            problems.append("the compilation mode is not VLLM_COMPILE (3), the recipe's default")
        if "'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE" not in config:
            problems.append("the CUDA-graph mode is not FULL_AND_PIECEWISE, the recipe's default")
    budget = re.findall(r"Chunked prefill is enabled with max_num_batched_tokens=(\d+)", text)
    if not budget or budget[-1] != "8192":
        problems.append(f"max_num_batched_tokens is {budget[-1] if budget else 'not reported'}; the recipe's is 8192")
    if "num_gpu_blocks_override" in want:
        override = re.findall(r"Overriding num_gpu_blocks=\d+ with num_gpu_blocks_override=(\d+)", text)
        if not override or any(v != str(want["num_gpu_blocks_override"]) for v in override):
            problems.append(f"the cache-block override is {sorted(set(override)) or 'not reported'}; the "
                            f"{deployment} deployment's is {want['num_gpu_blocks_override']}")

    compiled = re.findall(r"(?:Worker_DP(\d+)_TP\d+ pid=\d+\)[^\n]*?)?torch\.compile takes ([0-9.]+) s in total", text)
    compiled_replicas = {int(dp or 0) for dp, _ in compiled}
    if not compiled or (replicas > 1 and compiled_replicas != set(range(replicas))):
        problems.append("the engine did not compile the model on every replica (no `torch.compile takes ... s in "
                        "total`), so its mixed batches run eagerly rather than in the recipe's PIECEWISE graphs")

    backends: dict[tuple[int, int], set[str]] = {}
    for dp, tp, name in re.findall(r"Worker_(?:DP(\d+)_)?TP(\d+) pid=\d+\)[^\n]*?Using (.+?) backend on V1 engine",
                                   text):
        backends.setdefault((int(dp or 0), int(tp)), set()).add(name)
    wanted_workers = [(d, t) for d in range(replicas) for t in range(4)]
    if sorted(backends) != wanted_workers or any(names != {"FlashMLA"} for names in backends.values()):
        problems.append(f"the MLA attention backend by worker is {dict(sorted(backends.items()))}; the recipe's is "
                        f"FlashMLA on all {len(wanted_workers)}")

    summary = {"deployment": deployment, "args": {k: args.get(k) for k in want},
               "torch_compile_s": [float(v) for _, v in compiled],
               "backends": {f"dp{d}-tp{t}": sorted(v) for (d, t), v in sorted(backends.items())}}
    return problems, summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mode", choices=["accuracy", "api", "concurrency", "recipe"], required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="the running server")
    parser.add_argument("--fixture", type=Path, help="api/concurrency: the inputs (default: fixtures/<mode>.json)")
    parser.add_argument("--log", type=Path, help="recipe: the server's own output, e.g. launch.py --log")
    parser.add_argument("--deployment", default="tp4", choices=sorted(DEPLOYMENT_ARGS),
                        help="recipe: the launch.py deployment the log is from")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--output", type=Path, help="write the report as JSON here")
    args = parser.parse_args(argv)
    base = args.url.rstrip("/")

    def url_for(route: str) -> str:
        return f"{base}/{route}"

    report: dict = {"mode": args.mode, "verdict": "not determined here: the acceptance values are the grader's"}
    try:
        if args.mode == "accuracy":
            port = int(base.rsplit(":", 1)[1])
            report["result"] = accuracy(port, **GSM8K)
        elif args.mode == "api":
            probes = json.loads((args.fixture or HERE / "fixtures" / "api.json").read_text())
            report["batch_invariance"] = api_probes(url_for, probes, model=args.model)
        elif args.mode == "concurrency":
            fixture = json.loads((args.fixture or HERE / "fixtures" / "concurrency.json").read_text())
            report.update(concurrency(url_for, fixture["prompts"], fixture["bands"], model=args.model,
                                      max_tokens=fixture["max_tokens"], top_k=fixture["top_k"]))
        else:
            if args.log is None:
                parser.error("--mode recipe needs --log, the server's own output")
            problems, summary = recipe_problems(args.log.read_text(errors="replace"), deployment=args.deployment)
            report.update(problems=problems, summary=summary)
            if problems:
                raise AssertionError("the server does not serve the recipe as stated:\n  " + "\n  ".join(problems))
            report["verdict"] = "the served configuration is the recipe"
        report["failed"] = None
    except AssertionError as exc:
        report["failed"] = str(exc)
        report["malformed"] = isinstance(exc, MalformedResponse)
    body = json.dumps(report, indent=2, default=str)
    print(body)
    if args.output is not None:
        args.output.write_text(body + "\n")
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
