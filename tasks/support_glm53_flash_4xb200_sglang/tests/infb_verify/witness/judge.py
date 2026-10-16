#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The witness verdicts: what the scored nodes of a graded class read from that class's (untimed) witness launch."""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

RECORDS = Path(os.environ.get("INFB_RECORD_DIR", "/logs/verifier/records")) / "witness"
EVENTS = Path(os.environ.get("INFB_WITNESS_ROOT", "/logs/verifier/witness"))
MODEL = "/model/zai-org/GLM-5.3-Flash"
N_GPU_RANKS = 4


def _load(cls: str, attempt: str) -> dict | None:
    try:
        return json.loads((RECORDS / cls / f"{attempt}.json").read_text())
    except (OSError, ValueError):
        return None


def events(cls: str) -> list[dict]:
    out = []
    for f in glob.glob(str(EVENTS / cls / "events" / "*.jsonl")):
        with open(f) as fh:
            for line in fh:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    return sorted(out, key=lambda e: (e.get("pid"), e.get("n")))


def _fail(reason: str, status: str = "candidate_fail", **evidence) -> dict:
    return {"status": status, "reason": reason, **evidence}


def _errors_before(cls: str, t_fail) -> list[dict]:
    """The observer's own error events of the class's witness launch recorded at or before the failure the driver
    observed at `t_fail` (an event a process flushed before it stopped precedes that; one without a time is kept)."""
    return [e for e in events(cls) if e.get("k") == "observer_error"
            and (not isinstance(t_fail, (int, float)) or not isinstance(e.get("t"), (int, float)) or e["t"] <= t_fail)]


def _launch_problem(cls: str) -> dict | None:
    """A witness launch that never became ready, or a server that exited during a phase, is the
    candidate's -- unless the observer recorded errors of its own before that failure. Then the observer may have
    caused it (an earlier observer that rebuilt every tuple as a plain tuple killed witness launches at decode-graph
    capture this way), and
    an observer that cannot read a conforming implementation is a verifier defect: `verifier_fault`, with samples. The
    failure's time is the driver's own observation of it (`launch_error_t`, `server_exited.t`), else the launch
    record's `written_at`."""
    launch = _load(cls, "launch")
    if launch is None:
        return _fail("the class's witness launch left no record", status="verifier_fault")
    exited = launch.get("server_exited")
    if launch.get("launch_error"):
        what = f"the witness launch of the class's own recipe did not become ready: {launch['launch_error']}"
        t_fail = launch.get("launch_error_t", launch.get("written_at"))
    elif exited:
        what = (f"the witness server exited during the {exited.get('after_phase')} phase "
                f"(returncode {exited.get('returncode')})")
        t_fail = exited.get("t", launch.get("written_at"))
    else:
        return None
    errors = _errors_before(cls, t_fail)
    if errors:
        return _fail(f"{what}, after the observer recorded {len(errors)} errors reading the implementation",
                     status="verifier_fault", observer_errors=len(errors),
                     observer_error_samples=[(e.get("where"), (e.get("error") or "")[:160]) for e in errors[:4]])
    return _fail(what)


def _phase(cls: str, name: str) -> tuple[dict | None, dict | None]:
    rec = _load(cls, f"phase-{name}")
    if rec is None:
        return None, _fail(f"the witness launch has no {name} phase record", status="verifier_fault")
    if not rec.get("completed"):
        return rec, _fail(f"the witness {name} phase did not complete: {rec.get('error')}")
    return rec, None


def _targets(evs: list[dict]) -> dict[int, dict]:
    """pid -> the target runner's rank facts (the draft runner shares the process and is reported separately)."""
    out = {}
    for e in evs:
        if e.get("k") == "rank_facts" and not e.get("is_draft_worker"):
            out[e["pid"]] = e
    return out


def _topology(argv: list[str]) -> dict:
    from glm53_protocols.recipe import recipe_flags
    f = recipe_flags(list(argv))
    tp, ep = int(f.get("tp_size", 1)), int(f.get("ep_size", 1))
    dp = int(f.get("dp_size", 1)) if f.get("enable_dp_attention") is True else 1
    return {"tp": tp, "ep": ep, "attn_dp": dp, "attn_tp": tp // dp, "flags": f}


def runtime(cls: str, recipe_argv: list[str], bar: float) -> dict:
    """R-RUNTIME: rank completeness, the recipe's configuration and native state on every rank, graph work with output
    dependency on two numerically valid warm requests, and (LowLatency) the draft's contribution."""
    problem = _launch_problem(cls)
    if problem:
        return problem
    rec, problem = _phase(cls, "runtime")
    if problem:
        return problem
    evs = events(cls)
    errors = [e for e in evs if e.get("k") == "observer_error"]
    topo = _topology(recipe_argv)
    ev: dict = {"topology": {k: v for k, v in topo.items() if k != "flags"}, "observer_errors": len(errors),
                "observer_error_samples": [(e.get("where"), (e.get("error") or "")[:160]) for e in errors[:4]]}
    targets = _targets(evs)
    ev["target_processes"] = len(targets)
    uuids = {t.get("device_uuid") for t in targets.values()}
    ranks = {name: sorted(t.get(field) for t in targets.values() if t.get(field) is not None)
             for name, field in (("tp", "tensor_model_parallel_rank"), ("ep", "moe_expert_parallel_rank"))}
    ps = [t.get("ps") or {} for t in targets.values()]
    ranks["attn_dp"] = sorted(p.get("attn_dp_rank") for p in ps if isinstance(p.get("attn_dp_rank"), int))
    ranks["attn_tp"] = sorted(p.get("attn_tp_rank") for p in ps if isinstance(p.get("attn_tp_rank"), int))
    sizes = {"tp": {t.get("tensor_model_parallel_world_size") for t in targets.values()},
             "ep": {t.get("moe_expert_parallel_world_size") for t in targets.values()},
             "attn_dp": {p.get("attn_dp_size") for p in ps}, "attn_tp": {p.get("attn_tp_size") for p in ps}}
    ev.update(ranks=ranks, sizes={k: sorted(v, key=str) for k, v in sizes.items()}, devices=len(uuids - {None}))
    want_ranks = {"tp": list(range(topo["tp"])), "ep": list(range(topo["ep"])),
                  "attn_dp": sorted(r for r in range(topo["attn_dp"]) for _ in range(topo["attn_tp"])),
                  "attn_tp": sorted(r for r in range(topo["attn_tp"]) for _ in range(topo["attn_dp"]))}
    if len(targets) != N_GPU_RANKS or len(uuids - {None}) != N_GPU_RANKS:
        return _fail(f"{len(targets)} target runner processes on {len(uuids - {None})} devices reported; the "
                     f"deployment runs on {N_GPU_RANKS} GPUs", **ev)
    for name in ("tp", "ep", "attn_dp", "attn_tp"):
        if sizes[name] != {topo[name]} or ranks[name] != want_ranks[name]:
            return _fail(f"the {name} ranks {ranks[name]} (sizes {sorted(sizes[name], key=str)}) are not the recipe's "
                         f"{name} {topo[name]} over {N_GPU_RANKS} GPUs", **ev)
    arm = (rec.get("arm") or {}).get("seq")
    natives = {}
    for e in evs:
        if e.get("k") == "native_facts" and not e.get("draft") and e["pid"] in targets:
            if e.get("cseq") == arm or e["pid"] not in natives:
                natives[e["pid"]] = e
    ev["native_facts"] = {str(pid): {k: v for k, v in e.items() if k not in ("k", "n", "t", "rank", "phase")}
                          for pid, e in list(natives.items())[:1]}
    if set(natives) != set(targets):
        return _fail(f"native facts from {len(natives)} of the {len(targets)} target ranks", **ev)
    want = {"kv_cache_dtype": "fp8_e4m3", "moe_runner_backend": topo["flags"].get("moe_runner_backend"),
            "moe_a2a_backend": topo["flags"].get("moe_a2a_backend", "none"),
            "dsa_prefill_backend": topo["flags"].get("dsa_prefill_backend"),
            "dsa_decode_backend": topo["flags"].get("dsa_decode_backend"),
            "enable_dp_attention": topo["flags"].get("enable_dp_attention") is True}
    if topo["flags"].get("speculative_algorithm"):
        want.update(speculative_algorithm=topo["flags"]["speculative_algorithm"],
                    speculative_adaptive=topo["flags"].get("speculative_adaptive") is True)
    for pid, nf in natives.items():
        for key, value in want.items():
            served = nf.get(f"server_args.{key}")
            if served is None or str(served).strip("'") != str(value):
                return _fail(f"rank pid {pid} runs {key}={served}, not the recipe's {value!r}", **ev)
        if "float8" not in str(nf.get("kv_cache_dtype")):
            return _fail(f"rank pid {pid} holds its KV cache as {nf.get('kv_cache_dtype')}, not FP8", **ev)
        if not nf.get("decode_graph_runner") or nf.get("server_args.disable_cuda_graph") == "True":
            return _fail(f"rank pid {pid} has no decode CUDA-graph runner", **ev)
        if nf.get("attn_backend") in (None, "NoneType"):
            return _fail(f"rank pid {pid} has no attention backend", **ev)
    fwd = [e for e in evs if e.get("k") == "forward" and not e.get("draft") and e.get("cseq") == arm]
    per_rank = {}
    for pid in targets:
        mine = [e for e in fwd if e["pid"] == pid]
        graph = [e for e in mine if e.get("can_run_graph")]
        per_rank[pid] = {"forwards": len(mine), "graph_forwards": len(graph),
                         "graph_layer_calls": sorted({e.get("layer_calls") for e in graph}, key=str),
                         "graph_logits_in_buffer": sorted({e.get("logits_in_graph_buffer") for e in graph}, key=str),
                         "decode_graph_logits_in_buffer": sorted({e.get("logits_in_graph_buffer") for e in graph
                                                                  if e.get("mode") in ("DECODE", "TARGET_VERIFY")},
                                                                 key=str),
                         "modes": sorted({e.get("mode") for e in graph}, key=str)}
    profiles = {e["pid"]: e.get("summary") or {} for e in evs if e.get("k") == "profile_stopped" and e["pid"] in targets}
    ev["graph"] = {str(pid): {**v, "profile": {k: profiles.get(pid, {}).get(k) for k in
                                               ("graph_launches", "kernels_in_graphs", "kernels_outside_graphs")}}
                   for pid, v in per_rank.items()}
    n_layers = json.loads(Path(MODEL, "config.json").read_text())["text_config"]["num_hidden_layers"]
    for pid, v in per_rank.items():
        if v["graph_forwards"] == 0:
            return _fail(f"rank pid {pid} ran no forward through a CUDA graph during the warm requests", **ev)
        if v["graph_layer_calls"] != [0]:
            return _fail(f"rank pid {pid}: decoder layers ran in Python during graph-path forwards "
                         f"({v['graph_layer_calls']} calls), so the graph is not what computed them", **ev)
        if v["decode_graph_logits_in_buffer"] not in ([], [True]):
            return _fail(f"rank pid {pid}: decode-path graph logits do not come from the graph's output buffer "
                         f"({v['decode_graph_logits_in_buffer']})", **ev)
        prof = profiles.get(pid)
        if not prof:
            return _fail(f"rank pid {pid} left no profile of its warm forwards", **ev)
        launches, inside = int(prof.get("graph_launches") or 0), int(prof.get("kernels_in_graphs") or 0)
        if launches < 1 or inside < n_layers * launches:
            return _fail(f"rank pid {pid}: {launches} graph launches ran {inside} kernels, fewer than one per decoder "
                         f"layer ({n_layers}) per launch", **ev)
    warm = rec.get("warm") or []
    ev["warm"] = [{k: w.get(k) for k in ("prompt_len", "generated", "k3_vs_teacher_forced")} for w in warm]
    if len(warm) != 2 or any(not isinstance(w.get("k3_vs_teacher_forced"), (int, float)) for w in warm):
        return _fail("the two warm requests were not both answered and scored", **ev)
    for w in warm:
        if not w["k3_vs_teacher_forced"] < bar:
            return _fail(f"a warm response's generation-time log-probabilities disagree with the same tokens' teacher-"
                         f"forced ones (k3 {w['k3_vs_teacher_forced']:.4g}, bar {bar})", **ev)
    if topo["flags"].get("speculative_algorithm"):
        drafts = [e for e in evs if e.get("k") == "spec_draft" and e.get("cseq") == arm]
        verifies = [e for e in evs if e.get("k") == "spec_verify" and e.get("cseq") == arm]
        accepted = [a for e in verifies for a in (e.get("accept_lens") or [])]
        ev["speculation"] = {"draft_calls": len(drafts), "verify_calls": len(verifies),
                             "draft_steps": sorted({e.get("steps") for e in drafts}, key=str),
                             "max_accept_len": max(accepted) if accepted else None}
        for pid in targets:
            d = [e for e in drafts if e["pid"] == pid]
            v = [e for e in verifies if e["pid"] == pid]
            acc = [a for e in v for a in (e.get("accept_lens") or [])]
            if not d or not all((e.get("steps") or 0) >= 1 for e in d):
                return _fail(f"rank pid {pid} did no draft work during the warm requests", **ev)
            if not acc or max(acc) < 2:
                return _fail(f"rank pid {pid}: no drafted token was verified and accepted (accept lengths "
                             f"{sorted(set(acc))[:8]})", **ev)
        generated = sum(int(w.get("generated") or 0) for w in warm)
        pid0 = min(targets)
        acc0 = sum(a for e in verifies if e["pid"] == pid0 for a in (e.get("accept_lens") or []))
        ev["speculation"].update(returned_tokens=generated, verified_tokens_rank0=acc0)
        if acc0 + len(warm) < generated:
            return _fail(f"the verify passes accepted {acc0} tokens, fewer than the {generated} returned (less one "
                         f"prefill token per request)", **ev)
    if errors:
        return _fail(f"the observer recorded {len(errors)} errors reading the implementation", status="verifier_fault",
                     **ev)
    return {"status": "pass", "reason": "every GPU rank of the recipe's topology runs its configuration with the FP8 KV "
                                        "cache, replays decode graphs that compute the model and produce its logits, on "
                                        "numerically valid warm responses" +
            (", with verified draft tokens" if topo["flags"].get("speculative_algorithm") else ""), **ev}


def _chains(evs: list[dict], pid: int, cseq) -> list[dict]:
    """Per batch on one rank: the controller's decision, the draft and verify work that followed it, and the acceptance
    feedback of that verify. The decision, draft and verify of a batch are recorded in that order; the feedback of a
    verify reaches the controller later (under the overlap scheduler, after the next batch's verify), so feedback and
    verifies are matched by their order: the k-th feedback is the k-th verify's."""
    mine = [e for e in evs if e["pid"] == pid and e.get("cseq") == cseq
            and e.get("k") in ("adaptive_choose", "spec_draft", "spec_verify", "adaptive_feedback")]
    batches, cur = [], None
    for e in mine:
        if e["k"] == "adaptive_choose":
            cur = {"choose": e}
            batches.append(cur)
        elif cur is not None and e["k"] == "spec_draft" and "draft" not in cur:
            cur["draft"] = e
        elif cur is not None and e["k"] == "spec_verify" and "verify" not in cur:
            cur["verify"] = e
    batches = [b for b in batches if {"draft", "verify"} <= set(b)]
    feedbacks = [e for e in mine if e["k"] == "adaptive_feedback"]
    for b, fb in zip(batches, feedbacks):
        b["feedback"] = fb
    return [b for b in batches if "feedback" in b]


def _chain_problem(c: dict, tier: int) -> str | None:
    if c["draft"].get("steps") != tier:
        return f"the draft ran {c['draft'].get('steps')} steps after the controller returned {tier}"
    if c["verify"].get("num_draft_tokens") != tier + 1:
        return f"the verify pass checked {c['verify'].get('num_draft_tokens')} tokens after the controller returned {tier}"
    acc = c["verify"].get("accept_lens") or []
    fb = c["feedback"].get("feedback") or []
    if fb != [a - 1 for a in acc]:
        return f"the controller's feedback {fb[:8]} is not the verified accepted drafts {[a - 1 for a in acc][:8]}"
    return None


def adaptive(cls: str) -> dict:
    problem = _launch_problem(cls)
    if problem:
        return problem
    rec, problem = _phase(cls, "adaptive")
    if problem:
        return problem
    evs = events(cls)
    targets = _targets(evs)
    ev: dict = {"target_processes": len(targets)}
    if len(targets) != N_GPU_RANKS:
        return _fail(f"{len(targets)} target ranks observed, {N_GPU_RANKS} expected", **ev)
    inits = [e for e in evs if e.get("k") == "adaptive_init" and e["pid"] in targets]
    if {e["pid"] for e in inits} != set(targets):
        return _fail("the adaptive controller was not constructed on every rank (adaptation is not active)", **ev)
    ordinary = (rec.get("ordinary_arm") or {}).get("seq")
    ev["ordinary"] = {k: rec.get(k) for k in ("ordinary_c1", "ordinary_c16")}
    for c in (1, 16):
        run = rec.get(f"ordinary_c{c}") or {}
        if run.get("rc") != 0:
            return _fail(f"the ordinary C{c} workload did not complete (rc {run.get('rc')})", **ev)
    counts = {}
    for pid in targets:
        chains = _chains(evs, pid, ordinary)
        counts[pid] = len(chains)
        if not chains:
            return _fail(f"rank pid {pid}: no controller decision was followed by draft, verify and feedback under the "
                         f"ordinary workloads", **ev)
        for ch in chains:
            tier = ch["choose"].get("returned")
            why = _chain_problem(ch, tier)
            if why:
                return _fail(f"rank pid {pid}, ordinary workloads: {why}", **ev)
            fb = ch["feedback"]
            if fb.get("before") == fb.get("after"):
                return _fail(f"rank pid {pid}: acceptance feedback left the controller's state unchanged "
                             f"({fb.get('before')})", **ev)
    ev["ordinary"]["chains_per_rank"] = sorted(counts.values())
    admitted = rec.get("admitted_c1") or []
    ev["admitted_c1"] = admitted
    if len(admitted) < 2:
        return _fail(f"the controller's slot for batch size 1 admits {admitted}: fewer than two positive tiers", **ev)
    for tier in (admitted[0], admitted[-1]):
        arm = (rec.get(f"substitute_{tier}_arm") or {}).get("seq")
        outs = rec.get(f"substitute_{tier}_outputs") or []
        ev[f"substitute_{tier}"] = {"outputs": outs}
        if not outs or any(not (isinstance(o, dict) and o.get("valid")) for o in outs):
            return _fail(f"with tier {tier} substituted the outputs were not all valid: {outs}", **ev)
        for pid in targets:
            chains = _chains(evs, pid, arm)
            if not chains:
                return _fail(f"rank pid {pid}: no batch ran with tier {tier} substituted", **ev)
            for ch in chains:
                if ch["choose"].get("returned") != tier:
                    return _fail(f"rank pid {pid}: the substituted return value was not {tier}", **ev)
                why = _chain_problem(ch, tier)
                if why:
                    return _fail(f"rank pid {pid}, tier {tier} substituted: {why}", **ev)
    restored = {e["pid"] for e in evs if e.get("k") == "adaptive_restored" and e["pid"] in targets}
    if restored != set(targets):
        return _fail(f"the controller's state was restored on {len(restored)} of {len(targets)} ranks", status=
                     "verifier_fault", **ev)
    replays = [e for e in evs if e.get("k") == "adaptive_replay" and e["pid"] in targets]
    ev["replay"] = {}
    differing = False
    for e in replays:
        if e.get("error"):
            return _fail(f"the controller could not be replayed: {e['error']}", status="verifier_fault", **ev)
        for bs, s in (e.get("slots") or {}).items():
            legal = set((s.get("config") or {}).get("candidate_steps") or [])
            z, a = (s.get("zero") or {}).get("final"), (s.get("all") or {}).get("final")
            ev["replay"].setdefault(bs, {"slot": s.get("slot"), "length": s.get("length"),
                                         "length_rule": s.get("length_rule"), "zero": z, "all": a})
            if legal and (z not in legal or a not in legal):
                return _fail(f"the replayed controller left its admitted tiers at batch size {bs} ({z}, {a})", **ev)
            if z != a:
                differing = True
    if not replays:
        return _fail("the controller replay produced no record", status="verifier_fault", **ev)
    if not differing:
        return _fail("replayed from its pre-use state, the controller reaches the same decision whether no draft or "
                     "every draft is accepted, at every occupied batch size: it does not adapt", **ev)
    return {"status": "pass", "reason": "acceptance feedback updates the active controller and its decisions drive the "
                                        "draft and verify work, substituted tiers are followed, and the replayed "
                                        "controller decides differently on opposite acceptance histories", **ev}


def attention(cls: str, recipe_argv: list[str]) -> dict:
    """R-ATTENTION: every designated tensor within its own allowance (`glm53_protocols.attention`: E <= sqrt(2) x
    max(E_rep, 2^-8), or the 1e-6 escape), and the connection control consumed at every probe, designated layer, rank
    and path (the within-forward rule of `assemble._connected`), with part (i) at the dense probe. What the witness
    could not measure (a missing child, branch or row) is `blocked`, never the candidate's failure."""
    from glm53_protocols.attention import ABS_ESCAPE, ALPHA, E_FLOOR, within_policy
    problem = _launch_problem(cls)
    if problem:
        return problem
    rec, problem = _phase(cls, "attention")
    if problem:
        return problem
    replay = _load(cls, "replay")
    if replay is None:
        return _fail("the witness launch left no replay record", status="verifier_fault")
    if replay.get("problem"):
        return _fail(replay["problem"], status="blocked")
    tensors = replay.get("tensors") or []
    ev: dict = {"alpha": ALPHA, "e_floor": E_FLOOR, "abs_escape": ABS_ESCAPE,
                "representation": replay.get("representation"), "readings": replay.get("readings"),
                "tensors": [{k: t.get(k) for k in ("input", "layer", "part", "status", "E", "E_rep", "allowance",
                                                    "ratio", "max_abs", "problem")} for t in tensors]}
    attn_tp = _topology(recipe_argv)["attn_tp"]
    n_inputs = len(rec.get("inputs") or [])
    if n_inputs == 0 or len(tensors) != n_inputs * 4:
        return _fail(f"{len(tensors)} compared tensors for {n_inputs} inputs; four per input expected",
                     status="blocked", **ev)
    for t in tensors:
        name = f"{t.get('input')} layer {t.get('layer')} {t.get('part')}"
        if t.get("status") != "valid" or t.get("E_rep") is None:
            return _fail(f"{name}: the witness could not compare this tensor ({t.get('problem')})", status="blocked",
                         **ev)
        if t.get("ranks_per_row") != [attn_tp]:
            return _fail(f"{name}: each row should be the sum of {attn_tp} attention tensor-parallel ranks' results; "
                         f"rows were observed on {t.get('ranks_per_row')} ranks", status="blocked", **ev)
        if not within_policy(t["E"], t["max_abs"], t["E_rep"]):
            return _fail(f"{name}: E {t['E']:.6f} exceeds its allowance {t['allowance']:.6f} (sqrt(2) x "
                         f"max(E_rep {t['E_rep']:.6f}, 2^-8)); maximum absolute difference {t['max_abs']:.3g}", **ev)
    cons = replay.get("consumption") or {}
    ev["consumption"] = cons
    if cons.get("status") != "valid":
        bad = [i for i in cons.get("inputs") or [] if i.get("status") != "valid"]
        return _fail(f"the connection control could not be measured "
                     f"({cons.get('problem') or (bad[0].get('problem') if bad else None)})", status="blocked", **ev)
    probes = cons.get("inputs") or []
    dense = [i for i in probes if "dense" in i]
    if len(dense) != 1 or len(probes) != n_inputs + 1:
        return _fail(f"{len(probes)} connection probes with {len(dense)} dense readings; {n_inputs + 1} and 1 expected",
                     status="blocked", **ev)
    for path in ("eager", "graph"):
        q = dense[0]["dense"].get(path) or {}
        if not q.get("consumed"):
            return _fail(f"at the dense-regime probe, replacing the designated layers' observed attention results by "
                         f"zeros moved the {path} path's logits by {q.get('zeroed_diff')}, not above "
                         f"{q.get('threshold')} (10 x the repeat difference and 1e-6)", **ev)
    for probe in probes:
        for L, per in (probe.get("layers") or {}).items():
            for pid, paths in per.items():
                for path, q in sorted(paths.items()):
                    if not q.get("consumed"):
                        return _fail(f"{probe['id']} layer {L} rank pid {pid} {path}: zeroing the observed attention "
                                     f"result moved the layer's MLP input by D {q.get('D')}, not above max(the floor "
                                     f"{q.get('floor')}, 3 x its largest re-execution difference {q.get('N')})", **ev)
    return {"status": "pass", "reason": f"all {len(tensors)} designated tensors within their allowances, and the "
                                        f"observed attention consumed at every probe, designated layer, rank and path",
            **ev}
