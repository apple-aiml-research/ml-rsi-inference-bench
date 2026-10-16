#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The recipe's runtime witness, the private half of
`test_recipe_conformance`. The public `served_recipe.py` reads the served configuration and a short kernel trace; this
module observes what actually runs on each tensor-parallel rank while the server answers ordinary warm requests, with
the engine's own profiler (`/start_profile`, torch.profiler over CUPTI, installed with the engine's ordinary
dependencies) and the engine's own per-forward spans (`ModelRunner.forward` names each forward `step[<MODE> bs=<n>]`
while a profiler runs). No model-helper name is read: the observation is the forward modes the engine records, the
CUDA runtime calls inside them, and the kernels those calls launched.

The workload (`/tests/hidden/recipe_witness.json`) is fixed and private: one ordinary warm-up request, then two
distinct 4,096-token requests at concurrency 1 (A and B) and one batch of sixteen at concurrency 16, each generating
512 tokens greedily with `ignore_eos`. Every answer is checked structurally first (`check_generation`: one output
log-probability per generated token, the same token ids in order, a finite number each), and the two C1 answers must
differ, so a stale or detached replay cannot pass by returning a fixed answer.

Three profiles, each bound to its request interval:

  A  CPU activity over request A, start to finish (a few MB per rank). Each target-verify forward is one
     `step[TARGET_VERIFY bs=1]` span; GRAPH: every one of them issues a CUDA graph launch; SPECULATION: the returned
     tokens exceed one per executed verify step (the first token comes from the prefill), so at least one draft token
     was accepted, and the engine's own accounting agrees with what executed (`spec_verify_ct` equals the verify
     spans, or one fewer: the overlap scheduler has one step in flight when a request finishes; `completion_tokens`
     equals the tokens returned).
  B  CPU and GPU activity over the first `window_steps` scheduler steps of request B. USEFUL WORK: inside the
     target-verify spans the kernels the graph launches ran include the recipe's routed-expert and sparse-attention
     kernel families, and no kernel of those families was launched eagerly inside a target-verify span -- a detached
     graph beside eager model computation fails here; DRAFT WORK: the same kernel families ran outside the target's
     own forwards (its verify and prefill spans), in a replayed graph or eagerly -- the draft model's work.
  C  CPU activity over the batch of sixteen: every target-verify span, at whatever batch size the scheduler formed,
     issues a graph launch, and the batch's returned tokens exceed one per request per verify step it took part in.

Complete observation is all four TP ranks, each with its own trace file, for every profile. Configuration facts (the
server's reported arguments) are recorded separately from executed-work facts; a flag never stands in for an
observation. The witness runs after the timed efficiency nodes, on the same server. Each profile is one instrumented
runtime invocation under the public limit (1,800 s), timed as that limit's clock reads: dispatch (the profiler start)
through all ranks' records read back (every rank's trace file written); the warm-up request is one HTTP request. Reading
the traces is the verifier's own analysis, outside every operation: it runs one process per rank, and its cost grows
with the events the candidate's execution recorded (an eager forward records every kernel launch), never with the
candidate's own time limits.
"""
from __future__ import annotations

import collections
import concurrent.futures as cf
import contextlib
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import requests

SPAN = re.compile(r"^step\[(?P<mode>[A-Z_0-9]+) bs=(?P<bs>\d+)(?: toks=\d+)?\]$")
GRAPH_LAUNCH = ("cudaGraphLaunch", "cuGraphLaunch")
ACTION_S = 300
REQUEST_S = 900


class WitnessFault(RuntimeError):
    """The observation itself could not be made (a rank's trace missing or unreadable): verifier-side, recorded."""


def base_name(kernel: str) -> str:
    """A kernel's name without template arguments, parameter list or leading `void` (the public check.py's rule)."""
    n = kernel.strip()
    if n.startswith("void "):
        n = n[5:]
    out, i = [], 0
    while i < len(n):
        if n.startswith("(anonymous namespace)", i):
            out.append("(anonymous namespace)")
            i += len("(anonymous namespace)")
            continue
        if n[i] in "<(":
            break
        out.append(n[i])
        i += 1
    return "".join(out).strip()


class KernelSet:
    """One scoped role's operator identity: the base names of the kernels the candidate's own
    public calls launched in its component profile (the component map) -- matched against the kernels a served trace
    names. Specializations of one kernel function share a base name; no library or naming convention is fixed."""

    def __init__(self, names):
        self.names = frozenset(names)

    def search(self, kernel: str) -> bool:
        return base_name(kernel) in self.names

    def __repr__(self) -> str:
        return f"KernelSet({sorted(self.names)!r})"


#: the component map's roles each served execution role must show: the target's
#: verify graphs all four; the draft model (no compressor, COMPRESS_RATIO_NEXTN_LAYER = 0) the packed attention only
TARGET_ROLES = ("packed_attention", "compression_c4", "compression_c4_indexer", "compression_c128")
DRAFT_ROLES = ("packed_attention",)


def role_families(component_map: dict | None) -> tuple[dict, dict]:
    """(families, notes): a KernelSet per role the map attributes, and why any role could not be (a role whose profile
    shows no operator kernel is unattributable -- an attribution limit, recorded, never a numerical verdict)."""
    roles = (component_map or {}).get("roles") or {}
    fams, notes = {}, {}
    for role in TARGET_ROLES:
        names = (roles.get(role) or {}).get("base_names") or []
        if names:
            fams[role] = KernelSet(names)
        else:
            notes[role] = "no operator kernel in the component map" if component_map else "no component map"
    return fams, notes


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def load_workload(path: Path) -> dict:
    """The private workload, checked: a warm-up, two distinct C1 inputs and sixteen distinct C16 inputs, 4,096 token
    ids each."""
    w = json.loads(Path(path).read_text())
    n = int(w["input_tokens"])
    seqs = [w["warmup"], *w["c1"], *w["c16"]]
    if len(w["c1"]) != 2 or len(w["c16"]) != 16:
        raise WitnessFault("the witness workload needs two C1 inputs and sixteen C16 inputs")
    for ids in seqs:
        if not isinstance(ids, list) or len(ids) != n or not all(isinstance(t, int) and not isinstance(t, bool)
                                                                 and t >= 0 for t in ids):
            raise WitnessFault(f"a witness input is not {n} token ids")
    if len({json.dumps(s) for s in seqs}) != len(seqs):
        raise WitnessFault("the witness inputs are not all distinct")
    return w


def check_generation(ids: list[int], body, max_new: int, label: str) -> dict:
    """The task's structural response check applied to a generation: exactly `max_new` output ids (`ignore_eos`), one
    output log-probability entry per generated token carrying that token's id, each a finite number."""
    meta = body.get("meta_info") if isinstance(body, dict) else None
    out = body.get("output_ids") if isinstance(body, dict) else None
    lps = meta.get("output_token_logprobs") if isinstance(meta, dict) else None
    if not isinstance(out, list) or len(out) != max_new:
        raise AssertionError(f"{label}: {len(out) if isinstance(out, list) else 'no'} output ids, expected {max_new}")
    if not isinstance(lps, list) or len(lps) != len(out):
        raise AssertionError(f"{label}: {len(lps) if isinstance(lps, list) else 'no'} output log-probabilities for "
                             f"{len(out)} generated tokens")
    for j, (e, t) in enumerate(zip(lps, out)):
        if not isinstance(e, (list, tuple)) or len(e) != 3 or e[1] != t or not _is_number(e[0]):
            raise AssertionError(f"{label}: output position {j}: entry {e!r:.80} for token {t}")
    return {"tokens": len(out), "output_digest": hashlib.sha256(json.dumps(out).encode()).hexdigest(),
            "completion_tokens": meta.get("completion_tokens"), "spec_verify_ct": meta.get("spec_verify_ct"),
            "spec_accepted_drafts": meta.get("spec_accepted_drafts"), "cached_tokens": meta.get("cached_tokens")}


def generate(base_url: str, ids: list[int], max_new: int, label: str) -> dict:
    t0 = time.time()
    r = requests.post(f"{base_url}/generate", json={
        "input_ids": ids, "sampling_params": {"temperature": 0, "max_new_tokens": max_new, "ignore_eos": True},
        "return_logprob": True, "logprob_start_len": len(ids) - 1}, timeout=(REQUEST_S, REQUEST_S))
    t1 = time.time()
    if r.status_code != 200:
        raise AssertionError(f"{label}: HTTP {r.status_code}: {r.text[:300]}")
    got = check_generation(ids, r.json(), max_new, label)
    return {**got, "t0": t0, "t1": t1, "elapsed_s": round(t1 - t0, 3)}


def _profile(base_url: str, out: Path, activities: list[str], num_steps: int | None) -> None:
    body = {"output_dir": str(out), "activities": activities, "with_stack": False, "record_shapes": False,
            "profile_by_stage": False}
    if num_steps:
        body["num_steps"] = num_steps
    r = requests.post(f"{base_url}/start_profile", json=body, timeout=ACTION_S)
    if r.status_code != 200:
        raise AssertionError(f"/start_profile answered HTTP {r.status_code}: {r.text[:200]}")


def _stop(base_url: str) -> None:
    try:
        requests.post(f"{base_url}/stop_profile", timeout=ACTION_S)
    except requests.RequestException:
        pass  # a window with num_steps stops itself; the files decide


def _collect(out: Path, tp: int, wait_s: float) -> dict[int, Path]:
    deadline = time.time() + wait_s
    files: dict[int, Path] = {}
    while time.time() < deadline:
        files = {}
        for f in out.glob("*.trace.json.gz"):
            m = re.search(r"TP-(\d+)", f.name)
            if m:
                files[int(m.group(1))] = f
        if len(files) >= tp:
            time.sleep(3)  # let the last writer finish
            break
        time.sleep(2)
    return files


def read_trace(path: Path, families: dict[str, re.Pattern]) -> dict:
    """One rank's trace: its step spans, the runtime calls inside each, graph launches in and out of spans, and the
    kernels those calls ran, by family."""
    try:
        trace = json.load(gzip.open(path))
    except Exception as exc:  # noqa: BLE001 - an unreadable trace is the witness's own fault
        raise WitnessFault(f"{path.name}: unreadable trace ({type(exc).__name__}: {exc})") from exc
    events = trace.get("traceEvents") or []
    spans, runtime, kernels = [], collections.defaultdict(list), collections.defaultdict(list)
    for e in events:
        cat = e.get("cat")
        if cat == "user_annotation" and e.get("ph") == "X":
            m = SPAN.match(str(e.get("name", "")))
            if m:
                spans.append({"mode": m["mode"], "bs": int(m["bs"]), "ts": e["ts"], "end": e["ts"] + e.get("dur", 0),
                              "thread": (e.get("pid"), e.get("tid"))})
        elif cat in ("cuda_runtime", "cuda_driver"):
            runtime[(e.get("pid"), e.get("tid"))].append(e)
        elif cat == "kernel":
            kernels[(e.get("args") or {}).get("correlation")].append(e.get("name", ""))
    for v in runtime.values():
        v.sort(key=lambda e: e["ts"])
    in_span: set = set()
    modes: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    spans.sort(key=lambda s: s["ts"])
    for s in spans:
        c = modes[s["mode"]]
        c["spans"] += 1
        c["bs_sum"] += s["bs"]
        graph_here = 0
        for e in runtime.get(s["thread"], ()):
            if not (s["ts"] <= e["ts"] <= s["end"]):
                continue
            corr = (e.get("args") or {}).get("correlation")
            in_span.add(corr)
            is_graph = str(e.get("name", "")).startswith(GRAPH_LAUNCH)
            graph_here += is_graph
            for k in kernels.get(corr, ()):
                c[("graph_kernels" if is_graph else "eager_kernels")] += 1
                for fam, pat in families.items():
                    if pat.search(k):
                        c[("graph_" if is_graph else "eager_") + fam] += 1
        c["spans_with_graph_launch"] += bool(graph_here)
    outside = collections.Counter()
    for thread, calls in runtime.items():
        for e in calls:
            corr = (e.get("args") or {}).get("correlation")
            if corr in in_span:
                continue
            kind = "graph" if str(e.get("name", "")).startswith(GRAPH_LAUNCH) else "eager"
            outside[f"{kind}_launches"] += 1
            for k in kernels.get(corr, ()):
                outside[f"{kind}_kernels"] += 1
                for fam, pat in families.items():
                    if pat.search(k):
                        outside[f"{kind}_{fam}"] += 1
    pids = sorted({t[0] for t in runtime} | {s["thread"][0] for s in spans} - {None})
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "events": len(events), "pids": pids, "kernel_events": sum(len(v) for v in kernels.values()),
            "modes": {m: dict(c) for m, c in modes.items()}, "outside_spans": dict(outside)}


def _no_operation(_label: str):
    return contextlib.nullcontext()


def read_traces(files: dict[int, Path], families: dict[str, re.Pattern]) -> dict[int, dict]:
    """Every rank's trace read in its own freshly spawned process (the verifier's analysis, outside the candidate's
    operations); in this process, one after another, if no process pool can be made here."""
    try:
        with cf.ProcessPoolExecutor(max_workers=max(1, len(files)), mp_context=mp.get_context("spawn")) as ex:
            futures = {r: ex.submit(read_trace, f, families) for r, f in sorted(files.items())}
            out = {r: fut.result() for r, fut in futures.items()}
        for t in out.values():
            t["analysis"] = "parallel"
        return out
    except (OSError, cf.process.BrokenProcessPool):
        out = {r: read_trace(f, families) for r, f in sorted(files.items())}
        for t in out.values():
            t["analysis"] = "sequential"
        return out


def _ranks(base_url: str, activities: list[str], num_steps: int | None, send, tp: int, families, label: str,
           invocation=_no_operation) -> dict:
    """One profile: the instrumented invocation (profiler start, the requests, stop, every rank's trace written) under
    `invocation(label)`, then the traces read outside it."""
    out = Path(tempfile.mkdtemp(prefix=f"infb-witness-{label}-"))
    try:
        with invocation(f"recipe witness profile {label}"):
            t0 = time.time()
            _profile(base_url, out, activities, num_steps)
            sent = send()
            if not num_steps:
                _stop(base_url)
            files = _collect(out, tp, ACTION_S)
            t_read = time.time()
        if sorted(files) != list(range(tp)):
            raise WitnessFault(f"profile {label}: traces for ranks {sorted(files)}, expected {list(range(tp))}")
        ranks = read_traces(files, families)
        return {"sent": sent, "ranks": ranks, "invocation_s": round(t_read - t0, 3),
                "analysis_s": round(time.time() - t_read, 3)}
    finally:
        shutil.rmtree(out, ignore_errors=True)


def source_identity(app: Path) -> dict:
    """The served tree: its commit and whether its working tree is clean (the grader serves the committed tree)."""
    def git(*a):
        return subprocess.run(["git", "-C", str(app), *a], capture_output=True, text=True, timeout=ACTION_S).stdout
    return {"head": git("rev-parse", "HEAD").strip(), "status_lines": len(git("status", "--porcelain").splitlines())}


def witness(base_url: str, workload: dict, tp: int, families: dict, server_info: dict,
            window_steps: int = 16, invocation=_no_operation, request=_no_operation,
            draft_families: tuple[str, ...] | None = None) -> tuple[list[str], dict]:
    """(failures, facts) for the runtime witness; see the module docstring. `invocation(label)` and `request(label)`
    are the caller's operation contexts (the verifier passes its operation registry's): each profile runs under
    `invocation`, the warm-up under `request`; the traces are read outside both. `families` holds the routed-expert
    patterns and the candidate's own role kernel sets (`role_families`); the draft's work is required of
    `draft_families` only (the draft model has no compressor)."""
    draft_families = tuple(families) if draft_families is None else tuple(f for f in draft_families if f in families)
    max_new = int(workload["max_new_tokens"])
    failures: list[str] = []
    facts: dict = {"configuration": {k: server_info.get(k) for k in (
        "speculative_algorithm", "speculative_num_steps", "speculative_num_draft_tokens", "speculative_eagle_topk",
        "disable_cuda_graph", "cuda_graph_max_bs", "disable_overlap_schedule", "tp_size", "attention_backend",
        "moe_runner_backend")}}
    with request("recipe witness warm-up"):
        facts["warmup"] = generate(base_url, workload["warmup"], max_new, "warm-up")

    a = _ranks(base_url, ["CPU"], None, lambda: generate(base_url, workload["c1"][0], max_new, "request A"), tp,
               families, "A", invocation)
    b = _ranks(base_url, ["CPU", "GPU"], window_steps,
               lambda: generate(base_url, workload["c1"][1], max_new, "request B"), tp, families, "B", invocation)

    def batch():
        t0 = time.time()
        with cf.ThreadPoolExecutor(16) as ex:
            got = list(ex.map(lambda i: generate(base_url, workload["c16"][i], max_new, f"batch request {i}"),
                              range(16)))
        return {"t0": t0, "t1": time.time(), "requests": got}
    c = _ranks(base_url, ["CPU"], None, batch, tp, families, "C", invocation)
    facts["profiles"] = {"A": a, "B": b, "C": c}

    ra, rb = a["sent"], b["sent"]
    if ra["output_digest"] == rb["output_digest"]:
        failures.append("the two distinct C1 inputs returned the same 512 tokens: the output does not follow the input")
    if len({r["output_digest"] for r in c["sent"]["requests"]}) < 2:
        failures.append("the sixteen distinct batch inputs all returned the same tokens")

    for rank, t in sorted(a["ranks"].items()):
        v = t["modes"].get("TARGET_VERIFY", {})
        n = v.get("spans", 0)
        if n == 0:
            failures.append(f"A rank {rank}: no target-verify forward executed during request A (no speculative "
                            f"verification observed)")
            continue
        if v.get("spans_with_graph_launch", 0) != n:
            failures.append(f"A rank {rank}: {n - v.get('spans_with_graph_launch', 0)} of {n} target-verify forwards "
                            f"issued no CUDA graph launch (not replayed from a captured graph)")
        tokens = ra["tokens"]
        if not tokens - 1 > n:
            failures.append(f"A rank {rank}: {tokens} tokens returned over {n} executed verify steps: no draft token "
                            f"was accepted (one token per step and one from the prefill)")
        ct = ra.get("spec_verify_ct")
        if not (isinstance(ct, int) and n - 1 <= ct <= n):
            failures.append(f"A rank {rank}: the engine reports spec_verify_ct {ct!r} but {n} verify steps executed")
    if ra.get("completion_tokens") != ra["tokens"]:
        failures.append(f"A: completion_tokens {ra.get('completion_tokens')!r} but {ra['tokens']} tokens returned")

    for rank, t in sorted(b["ranks"].items()):
        v = t["modes"].get("TARGET_VERIFY", {})
        if not v.get("spans"):
            failures.append(f"B rank {rank}: no target-verify forward in the {window_steps}-step window")
            continue
        missing = [f for f in families if not v.get(f"graph_{f}")]
        if missing:
            failures.append(f"B rank {rank}: the replayed target-verify graphs ran no {', '.join(missing)} kernel (a "
                            f"graph launch without the model's work)")
        eager = {f: v.get(f"eager_{f}") for f in families if v.get(f"eager_{f}")}
        if eager:
            failures.append(f"B rank {rank}: model kernels launched eagerly inside target-verify forwards {eager} "
                            f"(the verify step's work is not the replayed graph's)")
        # the draft's work: the model's kernel families running outside the target's own forwards (verify and
        # prefill), in a replayed graph or eagerly, in a draft span or in none
        draft = collections.Counter()
        for m, cm in t["modes"].items():
            if m not in ("TARGET_VERIFY", "EXTEND"):
                for f in families:
                    draft[f] += cm.get(f"graph_{f}", 0) + cm.get(f"eager_{f}", 0)
        for f in families:
            draft[f] += t["outside_spans"].get(f"graph_{f}", 0) + t["outside_spans"].get(f"eager_{f}", 0)
        if not all(draft[f] for f in draft_families):
            failures.append(f"B rank {rank}: no draft-model work outside the target's forwards during decode "
                            f"({dict(draft)})")

    batch_reqs = c["sent"]["requests"]
    ct_sum = sum(int(r.get("spec_verify_ct") or 0) for r in batch_reqs)
    tok_sum = sum(r["tokens"] for r in batch_reqs)
    for rank, t in sorted(c["ranks"].items()):
        v = t["modes"].get("TARGET_VERIFY", {})
        n = v.get("spans", 0)
        if not n or v.get("spans_with_graph_launch", 0) != n:
            failures.append(f"C rank {rank}: {n - v.get('spans_with_graph_launch', 0) if n else 'no'} of {n} batch "
                            f"target-verify forwards without a CUDA graph launch")
        if not v.get("bs_sum", 0) >= ct_sum:
            failures.append(f"C rank {rank}: the verify spans cover {v.get('bs_sum', 0)} request-steps, fewer than the "
                            f"{ct_sum} the requests report")
    if not tok_sum - len(batch_reqs) > ct_sum:
        failures.append(f"C: {tok_sum} tokens returned for {ct_sum} request-verify-steps: no draft token accepted in "
                        f"the batch")
    for r in batch_reqs:
        if r.get("completion_tokens") != r["tokens"]:
            failures.append("C: a batch request's completion_tokens differs from the tokens it returned")
            break
    return failures, facts


def summary(facts: dict) -> dict:
    """The witness's facts in one line per profile and rank, for the node's printed record."""
    out = {"source": facts.get("source"), "server_pid": facts.get("server_pid")}
    for name, prof in (facts.get("profiles") or {}).items():
        sent = prof.get("sent") or {}
        req = sent.get("requests")
        out[name] = {"invocation_s": prof.get("invocation_s"), "analysis_s": prof.get("analysis_s"),
                     "tokens": sum(r["tokens"] for r in req) if req else sent.get("tokens"),
                     "spec_verify_ct": sum(int(r.get("spec_verify_ct") or 0) for r in req) if req else
                     sent.get("spec_verify_ct"),
                     "ranks": {r: {m: {k: v for k, v in c.items() if k in ("spans", "spans_with_graph_launch")}
                                   for m, c in t["modes"].items() if m == "TARGET_VERIFY"}
                               for r, t in (prof.get("ranks") or {}).items()}}
    return out
