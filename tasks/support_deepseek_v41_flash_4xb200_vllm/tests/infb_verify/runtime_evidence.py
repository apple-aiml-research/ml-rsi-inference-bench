#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Runtime evidence for the deployment: actual CUDA-graph replay, real draft work and accepted draft tokens, on every rank.

Configuration facts are not runtime claims, so the
deployment node also observes what the engine does while it serves. This module is the task-local private observer;
it adds no helper the candidate must provide and uses only interfaces the engine has at the prepared base:

  * the engine's own torch-profiler support (`--profiler-config`, `POST /start_profile`, `POST /stop_profile`), which
    records every worker's CPU and CUDA activity (CUPTI through PyTorch's profiler) into one trace per rank;
  * the engine's Prometheus counters (`/metrics`): speculative-decoding drafts, draft tokens and accepted tokens, and
    generated tokens, read before and after each observed interval;
  * the ordinary completion API, whose responses are validated numerically.

The instrumented launch is separate from the timed efficiency session (the same committed tree, recipe and checkpoint;
the recipe's arguments plus the profiler's own configuration), so profiling never touches a timed measurement. On it,
warm C1 and C16 workloads at 4,096 input and 512 output tokens (drawn from the private pool with the efficiency client's
own `draw_workload`) run first unobserved, then observed. The record keeps configuration facts and runtime facts apart.

What must hold, on every one of the four ranks, for each observed interval (C1 and C16):
  * graph replay: `cudaGraphLaunch` (or `cuGraphLaunch`) calls during the interval, and kernels launched by those replays
    (correlated to a graph-launch call) that are useful model computation -- GEMM, attention and MoE kernels, not only
    copies or memsets -- so a detached no-op graph beside eager model computation does not pass;
And for the observed requests:
  * draft work: the speculative counters grew during the interval (drafts and draft tokens), with at least one accepted
    draft token, and the generated-token counter grew by exactly the tokens the responses returned;
  * output dependency: two distinct warm inputs, one per interval, return different generations, each structurally
    valid, and each generation's emitted log-probabilities agree with the server's own teacher-forced rescoring of that
    exact sequence (the self-consistency statistic of `test_text_history_fidelity`, mean k3 per response; the median of
    the two at most the same bar), so a stale replay that ignores the current input cannot pass.
Missing observability is never a pass: an absent, truncated or unparseable trace fails the node.
"""
from __future__ import annotations

import concurrent.futures
import contextlib
import gzip
import hashlib
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from infb_verify import _trusted, serve

TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
X = json.loads((TESTS_DIR / "config.json").read_text())["infb"]
R = X["runtime_evidence"]
OUT = Path(os.environ.get("INFB_RUNTIME_RECORD_DIR", "/logs/verifier/runtime"))
LAUNCH_LOG = OUT / "candidate-launch-instrumented.log"
#: The sparse-indexer observer's startup directory and record directory, for the instrumented launch only.
S = X["sparse_indexer"]
SPARSE_STARTUP = TESTS_DIR / "infb_verify" / "sparse_startup"
SPARSE_DIR = Path(S["records"])
SPARSE_AUDIT = SPARSE_DIR / "binding"

_FIDELITY = _trusted.load("fidelity/check.py")
_API = _trusted.load("tests/api_examples.py")

GRAPH_LAUNCH = ("cudaGraphLaunch", "cuGraphLaunch")
#: Kernel-name fragments of useful model computation (GEMMs, attention, MoE, normalization and quantization kernels the
#: engine's model path runs); a replayed graph that contains none of them is not the model's forward.
MODEL_KERNEL = re.compile(r"gemm|gemv|cutlass|cublas|matmul|mma|fmha|flash|mla|attn|attention|moe|expert|deep_?gemm"
                          r"|sm100|nvjet|tilelang|triton_|topk|rms_?norm|quant", re.I)


def profiler_args(trace_dir: Path) -> list[str]:
    """The engine's own profiler configuration for the instrumented launch (dotted CLI form, no JSON quoting)."""
    return [f"--profiler-config.profiler=torch", f"--profiler-config.torch_profiler_dir={trace_dir}",
            "--profiler-config.ignore_frontend=true", "--profiler-config.torch_profiler_with_stack=false",
            "--profiler-config.torch_profiler_record_shapes=false",
            "--profiler-config.torch_profiler_dump_cuda_time_total=false",
            f"--profiler-config.max_iterations={int(R['max_profiled_iterations'])}"]


@contextlib.contextmanager
def instrumented_server(trace_dir: Path):
    """The recipe, launched through upstream's own wrapper as `serve.launch` launches it, plus the profiler config."""
    from tests.utils import RemoteOpenAIServer  # noqa: PLC0415  (the upstream tree at /app)

    trace_dir.mkdir(parents=True, exist_ok=True)
    SPARSE_AUDIT.mkdir(parents=True, exist_ok=True)
    # only this diagnostic child gets the trusted startup directory, which
    # installs the sparse-indexer observer after the model loads; its records go to SPARSE_AUDIT
    env = {"INFB_SPARSE_AUDIT_DIR": str(SPARSE_AUDIT), "TESTS_DIR": str(TESTS_DIR),
           "PYTHONPATH": os.pathsep.join([str(SPARSE_STARTUP), *filter(None, [os.environ.get("PYTHONPATH")])])}
    with serve._tee_child_output(LAUNCH_LOG):  # noqa: SLF001  (the same launch-log capture as the session)
        server = RemoteOpenAIServer(serve.MODEL, serve.SERVER_ARGS + profiler_args(trace_dir), env_dict=env,
                                    max_wait_seconds=serve.STARTUP_MAX_WAIT_SECONDS)
    with server:
        yield server


def _post(url: str, timeout: float) -> int | str:
    """POST with an empty body; the status, or the error as text (a profiler that already stopped itself after its
    iteration limit ignores the stop, and the traces, not this status, are the evidence)."""
    req = urllib.request.Request(url, data=b"", method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status
    except (urllib.error.URLError, OSError) as exc:
        return f"{type(exc).__name__}: {exc}"[:300]


def counters(base_url: str) -> dict:
    """The speculative-decoding and generation counters, summed over every label set."""
    with urllib.request.urlopen(base_url.rstrip("/") + "/metrics", timeout=60) as r:
        text = r.read().decode()
    names = ("vllm:spec_decode_num_drafts", "vllm:spec_decode_num_draft_tokens",
             "vllm:spec_decode_num_accepted_tokens", "vllm:generation_tokens")
    out = {}
    for name in names:
        total = 0.0
        found = False
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            m = re.match(rf"^{re.escape(name)}(?:_total)?(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)$", line)
            if m:
                total += float(m.group(1))
                found = True
        out[name] = total if found else None
    return out


def workload() -> dict:
    """The observed prompts: the efficiency client's own draw from the private pool at C16, 4,096 input tokens."""
    client = _load_efficiency_client()
    tok = client.load_tokenizer(X["efficiency"]["tokenizer"])
    docs = client.load_texts(TESTS_DIR / "infb_efficiency" / "pool.jsonl.xz")
    wl = client.draw_workload(docs, tok, R["seed"], 16)
    c1 = [p.ids for p in wl.phase("c1_timed")]
    cc = [p.ids for p in wl.phase("cC_timed")]
    return {"warm_c1": c1[:1], "warm_c16": cc[:16], "observed_c1": c1[1:2], "observed_c16": cc[16:32],
            "digest": wl.digest}


def _load_efficiency_client():
    import importlib.util  # noqa: PLC0415

    name = "infb_dsv41_efficiency_client"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TESTS_DIR / "infb_efficiency" / "client.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def generate(base_url: str, ids: list[int], osl: int) -> dict:
    body = {"model": serve.MODEL, "prompt": ids, "max_tokens": osl, "temperature": 0.0, "seed": 0, "ignore_eos": True,
            "logprobs": 0, "return_token_ids": True, "skip_special_tokens": False,
            "cache_salt": "infb-runtime-" + hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]}
    status, reply, transport = _API.post(base_url, "/v1/completions", body, timeout=float(R["request_timeout_s"]))
    if transport:
        return {"errors": [f"the request failed: {transport}"], "transport_failed": True}
    errors, facts = _API.check_completion(status, reply, logprobs=True, token_ids=True)
    if not errors and (len(facts["token_ids"]) != osl or facts.get("finish_reason") != "length"):
        errors.append(f"{len(facts['token_ids'])} tokens with finish {facts.get('finish_reason')!r}, not {osl} by length")
    return {"errors": errors, "token_ids": facts.get("token_ids"), "logprobs": facts.get("token_logprobs"),
            "completion_tokens": facts.get("completion_tokens"), "transport_failed": False}


def rescore(base_url: str, prompt: list[int], gen: dict) -> dict:
    """The server's own teacher-forced reading of prompt + generation; mean k3 of emitted against rescored."""
    seq = prompt + gen["token_ids"]
    status, body, transport = _FIDELITY.request_prompt_logprobs(base_url.rstrip("/") + "/v1/completions", serve.MODEL,
                                                                seq, timeout=float(R["request_timeout_s"]))
    if transport or status != 200:
        return {"errors": [f"the rescoring request failed: {transport or f'HTTP {status}'}"]}
    values, errors, _ = _FIDELITY.parse_prompt_logprobs(body, seq)
    if errors:
        return {"errors": errors}
    rescored = values[len(prompt) - 1:]
    return {"errors": [], "k3_mean": _FIDELITY.score(rescored, gen["logprobs"])["mean"]}


def observe_interval(base_url: str, label: str, prompts: list[list[int]], trace_dir: Path, osl: int) -> dict:
    """One observed interval: counters, the profiler around the requests, the responses, then the traces written."""
    before = counters(base_url)
    t0 = time.time()
    started = _post(base_url.rstrip("/") + "/start_profile", 300)
    with concurrent.futures.ThreadPoolExecutor(len(prompts)) as ex:
        gens = list(ex.map(lambda ids: generate(base_url, ids, osl), prompts))
    stopped = _post(base_url.rstrip("/") + "/stop_profile", 900)
    t1 = time.time()
    after = counters(base_url)
    return {"label": label, "interval": [t0, t1], "counters_before": before, "counters_after": after,
            "profile_calls": {"start": started, "stop": stopped}, "generations": gens,
            "prompts_sha256": [hashlib.sha256(json.dumps(p).encode()).hexdigest() for p in prompts]}


def wait_for_traces(trace_dir: Path, expected: int, since: float, timeout_s: float) -> list[Path]:
    """The per-rank traces written after `since`, once their sizes stop changing."""
    deadline, last = time.time() + timeout_s, None
    while time.time() < deadline:
        files = sorted(p for p in trace_dir.rglob("*.pt.trace.json*") if p.stat().st_mtime >= since - 1)
        sizes = [p.stat().st_size for p in files]
        if len(files) >= expected and sizes == last:
            return files
        last = sizes
        time.sleep(10)
    return sorted(p for p in trace_dir.rglob("*.pt.trace.json*") if p.stat().st_mtime >= since - 1)


def inspect_trace(path: Path) -> dict:
    """One rank's trace: graph-launch calls, the kernels those replays launched (a kernel launched by a replay carries
    the launch call's correlation id and a nonzero `graph node id`), which of them are model kernels, the streams they
    ran on, and the engine's own per-iteration annotations. A truncated or unparseable file is an error, never partial
    evidence."""
    opener = gzip.open if path.suffix == ".gz" else open
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with opener(path, "rt") as fh:
        trace = json.load(fh)
    events = trace.get("traceEvents") or []
    launch_corr: set = set()
    for e in events:
        if e.get("cat") in ("cuda_runtime", "cuda_driver") and str(e.get("name", "")).startswith(GRAPH_LAUNCH):
            launch_corr.add((e.get("args") or {}).get("correlation"))
    kernels = 0
    graph_kernels: dict[str, int] = {}
    per_launch: dict = {}
    streams: dict = {}
    for e in events:
        if e.get("cat") != "kernel":
            continue
        kernels += 1
        args = e.get("args") or {}
        corr = args.get("correlation")
        if corr in launch_corr and args.get("graph node id") not in (None, 0):
            name = str(e.get("name", ""))
            graph_kernels[name] = graph_kernels.get(name, 0) + 1
            per_launch[corr] = per_launch.get(corr, 0) + 1
            streams[args.get("stream")] = streams.get(args.get("stream"), 0) + 1
    model = {k: v for k, v in graph_kernels.items() if MODEL_KERNEL.search(k)}
    steps: dict[str, int] = {}
    for e in events:
        if e.get("cat") == "user_annotation" and str(e.get("name", "")).startswith("execute_context_"):
            steps[e["name"]] = steps.get(e["name"], 0) + 1
    rank = re.search(r"_rank(\d+)\.", path.name) or re.search(r"rank[-_]?(\d+)", path.name)
    sizes = sorted(per_launch.values())
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest,
            "rank": int(rank.group(1)) if rank else None, "events": len(events), "kernels": kernels,
            "graph_launches": len(launch_corr), "replays_with_kernels": len(per_launch),
            "graph_kernels": sum(graph_kernels.values()), "graph_model_kernels": sum(model.values()),
            "kernels_per_replay": {"min": sizes[0], "median": sizes[len(sizes) // 2], "max": sizes[-1]} if sizes else None,
            "graph_model_kernel_names": sorted(model, key=lambda k: -model[k])[:12],
            "graph_streams": {str(k): v for k, v in sorted(streams.items(), key=lambda kv: -kv[1])[:4]},
            "engine_iterations": steps}


def delta(obs: dict, name: str) -> float | None:
    a, b = obs["counters_before"].get(name), obs["counters_after"].get(name)
    return None if a is None or b is None else b - a


def probe_prompts(lengths: list[int]) -> list[list[int]]:
    """Natural-text prompts of the given lengths, drawn by the task's client from its private pool with the sparse probe's
    own seed: 4,096-token draws of distinct documents, concatenated (the first keeps the tokenizer's special prefix)."""
    client = _load_efficiency_client()
    tok = client.load_tokenizer(X["efficiency"]["tokenizer"])
    docs = client.load_texts(TESTS_DIR / "infb_efficiency" / "pool.jsonl.xz")
    wl = client.draw_workload(docs, tok, f"infb-dsv41-sparse-probe-{S['seeds']['probe']}", 16)
    segs = [p.ids for p in wl.prompts]
    lead = len(list(tok.special_prefix()))
    out, i = [], 0
    for n in lengths:
        ids = list(segs[i])
        i += 1
        while len(ids) < n:
            ids += segs[i][lead:]
            i += 1
        out.append(ids[:n])
    return out


def persist(record: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "test_served_configuration.json").write_text(json.dumps(record, indent=1, default=str))


class Instrumented:
    """The instrumented launch's observation procedures: `observe()` (the deployment node's) and, after it,
    `observe_sparse_indexer()` (the serving-binding node's), each once, on the same launch."""

    def __init__(self, server, trace_dir: Path):
        self.server, self.trace_dir = server, trace_dir

    def observe(self) -> dict:
        base = self.server.url_root
        osl = int(R["osl"])
        bar = float(X["bars"]["fidelity_T_per_band"][str(R["consistency_bar_band"])])
        out: dict = {"trace_dir": str(self.trace_dir), "problems": [], "intervals": [], "bar": bar}
        try:
            wl = workload()
        except Exception as exc:  # noqa: BLE001  (the verifier's own data; a fault here is not the candidate's)
            out.update(problems=[f"verifier_fault: the observed workload could not be drawn: {exc}"], blocked_by="verifier")
            return out
        out["workload_digest"] = wl["digest"]
        # unobserved warm-up, both concurrencies (graphs captured, first-use compilation done)
        for label, prompts in (("warm_c1", wl["warm_c1"]), ("warm_c16", wl["warm_c16"])):
            with concurrent.futures.ThreadPoolExecutor(len(prompts)) as ex:
                gens = list(ex.map(lambda ids: generate(base, ids, osl), prompts))
            bad = [g for g in gens if g["errors"]]
            if bad:
                out["problems"].append(f"{label}: {len(bad)} warm-up request(s) failed: {bad[0]['errors']}")
                if all(g.get("transport_failed") for g in bad):
                    out["primary_cause"] = "the instrumented server stopped answering during the warm-up"
                return out
        for label, prompts in (("c1", wl["observed_c1"]), ("c16", wl["observed_c16"])):
            since = time.time()
            obs = observe_interval(base, label, prompts, self.trace_dir, osl)
            files = wait_for_traces(self.trace_dir, 4, since, float(R["trace_wait_s"]))
            ranks = []
            for f in files:
                try:
                    ranks.append(inspect_trace(f))
                except Exception as exc:  # noqa: BLE001
                    ranks.append({"path": str(f), "error": f"{type(exc).__name__}: {exc}"})
            obs["traces"] = ranks
            obs["returned_tokens"] = sum(len(g.get("token_ids") or []) for g in obs["generations"])
            obs["deltas"] = {k.split(":", 1)[1]: delta(obs, k) for k in obs["counters_before"]}
            first = obs["generations"][0]
            obs["self_consistency"] = (rescore(base, prompts[0], first) if not first["errors"]
                                       else {"errors": ["the first observed generation is invalid"]})
            out["intervals"].append(obs)
        verdict = judge(out["intervals"], bar)
        out["problems"] += verdict["problems"]
        out["self_consistency_median"] = verdict["self_consistency_median"]
        out["summary"] = {iv["label"]: {"deltas": iv["deltas"], "returned_tokens": iv["returned_tokens"],
                                        "graph": [(r.get("rank"), r.get("graph_launches"), r.get("graph_model_kernels"))
                                                  for r in iv.get("traces", [])],
                                        "self_consistency": (iv.get("self_consistency") or {}).get("k3_mean")}
                          for iv in out["intervals"]}
        return out


def _observe_sparse_indexer(self) -> dict:
    """The serving-binding node's observation (the sparse-indexer contract): the probe requests inside the observer's
    window, then an ordinary long request with the structural and self-consistency checks, then the judgement of the
    observer's records (`sparse_binding.judge`). The record is written to sparse-indexer/binding.json."""
    import json as _json  # noqa: PLC0415

    from infb_verify import sparse_binding  # noqa: PLC0415

    base = self.server.url_root
    probe = S["probe_workloads"]
    gen = int(probe["generate"])
    rec: dict = {"node": "test_sparse_indexer_serving_binding", "audit_dir": str(SPARSE_AUDIT), "probe": probe}
    control = SPARSE_AUDIT / "active"
    try:
        single, mixed = list(probe["single"]), list(probe["mixed"])
        prompts = probe_prompts(single + mixed + [max(mixed)])
    except Exception as exc:  # noqa: BLE001  (the verifier's own data)
        rec.update(status="blocked_by:verifier", why=f"verifier_fault: the probe prompts could not be drawn: {exc}")
        return rec
    rec["prompts_sha256"] = [hashlib.sha256(_json.dumps(p).encode()).hexdigest() for p in prompts]
    t0 = time.time()
    control.write_text(str(t0))
    gens = []
    try:
        for ids in prompts[: len(single)]:
            gens.append(generate(base, ids, gen))
        with concurrent.futures.ThreadPoolExecutor(len(mixed)) as ex:
            gens += list(ex.map(lambda ids: generate(base, ids, gen), prompts[len(single): len(single) + len(mixed)]))
        time.sleep(5)  # the last step's snapshots are written before its outputs return; a margin for the workers
    finally:
        control.unlink(missing_ok=True)
    rec["probe_seconds"] = round(time.time() - t0, 3)
    rec["probe_errors"] = [g["errors"] for g in gens if g["errors"]]
    if any(g.get("transport_failed") for g in gens):
        rec.update(status="blocked_by:server_unreachable",
                   why="the instrumented server stopped answering during the sparse-indexer probe")
        return rec
    long_ids = prompts[-1]
    g = generate(base, long_ids, int(R["osl"]))
    sc = rescore(base, long_ids, g) if not g["errors"] else {"errors": ["the long generation is invalid"]}
    bar = float(X["bars"]["fidelity_T_per_band"][str(R["consistency_bar_band"])])
    rec["long_request"] = {"input": len(long_ids), "errors": g["errors"], "self_consistency": sc, "bar": bar}
    try:
        result = sparse_binding.judge(sparse_binding.load(SPARSE_AUDIT), int(S["audit_layers"]["source"]),
                                      [int(c) for c in S["audit_layers"]["consumers"]])
    except Exception as exc:  # noqa: BLE001  (a record the judgement cannot read is the verifier's)
        result = {"status": "blocked_by:verifier", "why": f"verifier_fault: the observer's records: "
                                                          f"{type(exc).__name__}: {exc}"[:600]}
    rec["binding"] = sparse_binding.summary(result)
    problems = []
    if rec["probe_errors"]:
        problems.append(f"{len(rec['probe_errors'])} probe request(s) structurally invalid: {rec['probe_errors'][0]}")
    if g.get("transport_failed"):
        rec.update(status="blocked_by:server_unreachable", why="the instrumented server stopped answering")
        return rec
    if g["errors"]:
        problems.append(f"the ordinary long request is structurally invalid: {g['errors'][:2]}")
    elif sc.get("errors"):
        problems.append(f"rescoring the ordinary long request failed structurally: {sc['errors'][:2]}")
    elif not sc["k3_mean"] <= bar:
        problems.append(f"the ordinary long request disagrees with the server's own rescoring: mean k3 "
                        f"{sc['k3_mean']:.6g} over {bar:.6g}")
    if problems or result.get("status") == "candidate_fail":
        rec.update(status="candidate_fail", why="; ".join(problems + ([result.get("why")]
                                                                       if result.get("status") == "candidate_fail" else [])))
    else:
        rec.update(status=result.get("status", "blocked_by:verifier"), why=result.get("why", ""))
    SPARSE_DIR.mkdir(parents=True, exist_ok=True)
    full = {**rec, "binding_full": {k: v for k, v in result.items() if k in ("checked", "problems", "unobserved",
                                                                            "coverage", "installed", "replayed_decode",
                                                                            "observer_faults")}}
    (SPARSE_DIR / "binding.json").write_text(_json.dumps(full, indent=1, default=str))
    return rec


Instrumented.observe_sparse_indexer = _observe_sparse_indexer


def judge(intervals: list[dict], bar: float) -> dict:
    """The runtime claims over the observed intervals; a pure function of the records (the logic controls call it)."""
    problems: list[str] = []
    consistency = []
    for obs in intervals:
        label, ranks, gens, d = obs["label"], obs.get("traces") or [], obs["generations"], obs["deltas"]
        invalid = [g["errors"] for g in gens if g["errors"]]
        if invalid:
            problems.append(f"{label}: {len(invalid)} observed request(s) structurally invalid: {invalid[0]}")
        good = [r for r in ranks if "error" not in r]
        if len(good) != 4 or sorted(r.get("rank") for r in good) != [0, 1, 2, 3]:
            problems.append(f"{label}: one complete profiler trace from each of the four ranks is required; got "
                            f"{[(r.get('rank'), r.get('error')) for r in ranks]}")
        for r in good:
            if r["graph_launches"] < 1 or r["graph_model_kernels"] < 1:
                problems.append(f"{label}: rank {r.get('rank')}: no CUDA-graph replay of model kernels during the "
                                f"observed requests ({r['graph_launches']} graph launches, {r['graph_model_kernels']} "
                                f"model kernels inside replays, {r['kernels']} kernels in all)")
        if not (d.get("spec_decode_num_drafts") or 0) > 0 or not (d.get("spec_decode_num_draft_tokens") or 0) > 0:
            problems.append(f"{label}: no draft work during the observed requests (drafts "
                            f"{d.get('spec_decode_num_drafts')}, draft tokens {d.get('spec_decode_num_draft_tokens')})")
        if not (d.get("spec_decode_num_accepted_tokens") or 0) >= 1:
            problems.append(f"{label}: no accepted draft token during the observed requests "
                            f"({d.get('spec_decode_num_accepted_tokens')})")
        if not invalid and d.get("generation_tokens") != obs["returned_tokens"]:
            problems.append(f"{label}: the generated-token counter grew by {d.get('generation_tokens')}, the responses "
                            f"returned {obs['returned_tokens']} tokens")
        sc = obs.get("self_consistency") or {"errors": ["no rescoring"]}
        if sc.get("errors"):
            problems.append(f"{label}: rescoring the observed generation failed structurally: {sc['errors']}")
        else:
            consistency.append(sc["k3_mean"])
    median = None
    if len(consistency) == len(intervals) == 2:
        median = statistics.median(consistency)
        if intervals[0]["generations"][0].get("token_ids") == intervals[1]["generations"][0].get("token_ids"):
            problems.append("the two distinct observed inputs returned the same generation")
        if not median <= bar:
            problems.append(f"the observed generations disagree with the server's own rescoring: median mean k3 "
                            f"{median:.6g} over {bar:.6g}")
    return {"problems": problems, "self_consistency_median": median}
