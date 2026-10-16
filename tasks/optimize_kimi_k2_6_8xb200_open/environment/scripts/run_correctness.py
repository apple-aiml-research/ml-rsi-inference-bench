#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Self-test correctness probe for the agent (NOT the scoring verifier).

Agent runs this BEFORE finalizing their patch to know whether their edits
break correctness. ALIGNED with the official verifier: it runs the SAME
teacher-forced next-token-logprob comparison + scorer (_aggregate_correctness)
against the SAME reference protocol, on a 20-prompt SUBSET of the official
100-prompt bank (correctness_prompts.json, shipped next to this script, drawn from
that bank with a fixed seed). So the self-check is PREDICTIVE of the official gate,
not identical to it: same protocol,
same scorer, same tolerances, one fifth of the prompts. A pass here on 20 prompts
is not a guarantee on 100 -- a real margin can sit just above the gate, close
enough that 20 prompts resolve it and 5 would not.
Per-task tolerances (atol/rtol/min_pass_rate) come from the same
env source the verifier reads (INFB_TASK_CORRECTNESS_*), not hardcoded.

Prompts are public by design — anti-gaming rests on the REFERENCE being
regenerated live from the pristine model each run, never on hiding prompts.

Engine paths:
  * sglang           — _run_sglang_teacher_forced: launch_server + HTTP
    /generate two-pass teacher forcing (pristine reference + candidate),
    the identical protocol as tests/_candidate_runner.py.
  * vllm (default here) — the in-process teacher-forced path below
    (INFB_ENGINE defaults to vllm for these vLLM tasks). TP is resolved to
    match the verifier (visible-GPU count / INFB_CORRECTNESS_TP).

Cost: dominated by two sequential server boots; the 10-prompt subsample keeps
the generate/teacher-force passes cheap.

Usage:
  python3 /app/scripts/run_correctness.py --output /logs/agent/correctness.json
  python3 /app/scripts/run_correctness.py --num-prompts 10 --max-tokens 100
"""
from __future__ import annotations

import os as _infb_os
for _infb_np in ("no_proxy", "NO_PROXY"):
    _infb_cur = _infb_os.environ.get(_infb_np, "")
    if "127.0.0.1" not in _infb_cur:
        _infb_os.environ[_infb_np] = "127.0.0.1,localhost,::1" + (
            f",{_infb_cur}" if _infb_cur else ""
        )
import argparse, json, os, subprocess, sys, time
from pathlib import Path

_BANK_SIZE = [0]

_ENGINE = os.environ.get("INFB_ENGINE", "vllm").lower()

TP_SIZE = 8

CORRECTNESS_CALIBRATION = {"atol": 0.2, "rtol": 0.1, "min_pass_rate": 0.95}


PROBE_PROMPTS = [
    "Explain how a transformer's self-attention layer works in 3 sentences.",
    "Write a Python function that computes the SHA256 of a file path.",
    "What's the time complexity of merge sort? Why?",
    "Translate to French: 'The quick brown fox jumps over the lazy dog.'",
    "Describe how MoE (Mixture of Experts) routing differs from dense FFN.",
]


def _resolve_workspace() -> Path:
    return Path(os.environ.get("INFB_WORKSPACE", "/app"))


def _resolve_venv_python() -> str:
    if env := os.environ.get("INFB_VENV"):
        p = Path(env) / "bin" / "python3"
        if p.exists():
            return str(p)
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _target_env import resolve_venv_python
    return resolve_venv_python()


def _resolve_model() -> str:
    if env := os.environ.get("INFB_MODEL_PATH"):
        if Path(env).is_dir():
            return env
    workspace = _resolve_workspace()
    if (workspace / "model").exists():
        return str((workspace / "model").resolve())
    raise RuntimeError("Cannot resolve model; set INFB_MODEL_PATH or "
                       "ensure $INFB_WORKSPACE/model symlink exists")


_INNER_RUNNER = '''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path

import numpy as np
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

# vLLM V1 boots EngineCore via multiprocessing "spawn", which re-executes
# this module (as __mp_main__) in the worker process. Without this guard the
# worker would re-run LLM(...) and crash ("Engine core initialization failed").
if __name__ == "__main__":
    mode = os.environ["INFB_PROBE_MODE"]               # "baseline" or "check"
    out_path = os.environ["INFB_PROBE_OUT"]
    prompts = json.loads(Path(os.environ["INFB_PROBE_PROMPTS_JSON"]).read_text())
    max_tokens = int(os.environ.get("INFB_PROBE_MAX_TOKENS", "50"))

    llm = LLM(
        model=os.environ["INFB_PROBE_MODEL"],
        tensor_parallel_size=int(os.environ.get("INFB_PROBE_TP", "1")), dtype="bfloat16", trust_remote_code=True,
        max_model_len=4096, gpu_memory_utilization=0.85,
        enforce_eager=False, seed=0,
        # Engine config MIRRORS the scored correctness gate: a probe that
        # loads the model into a different engine cannot predict that gate.
        kv_cache_dtype=os.environ.get("INFB_KV_CACHE_DTYPE", "fp8"),
        skip_mm_profiling=True,
    )

    if mode == "baseline":
        # Pass A: greedy decode → token_ids + prompt_token_ids
        gen = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=max_tokens, seed=0))
        N = len(prompts)
        token_ids = np.full((N, max_tokens), -1, dtype=np.int32)
        prompt_token_ids = []
        for i, out in enumerate(gen):
            ptoks = list(out.prompt_token_ids)
            gtoks = list(out.outputs[0].token_ids)
            token_ids[i, :len(gtoks)] = gtoks
            prompt_token_ids.append(np.asarray(ptoks, dtype=np.int32))

        # Pass B: teacher-forced self-pass for path-consistent logprobs
        full_seqs, n_prompt_tokens, n_gen_tokens = [], [], []
        for i in range(N):
            ptoks = prompt_token_ids[i].tolist()
            gtoks = [int(x) for x in token_ids[i] if x != -1]
            full_seqs.append(ptoks + gtoks)
            n_prompt_tokens.append(len(ptoks))
            n_gen_tokens.append(len(gtoks))

        tf = llm.generate(
            [TokensPrompt(prompt_token_ids=s) for s in full_seqs],
            SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1, seed=0),
        )
        logprobs = np.full((N, max_tokens), float("nan"), dtype=np.float32)
        for i, out in enumerate(tf):
            plp = out.prompt_logprobs or []
            for k in range(n_gen_tokens[i]):
                pos = n_prompt_tokens[i] + k
                if pos >= len(plp) or plp[pos] is None:
                    continue
                step = plp[pos]
                tok_id = int(token_ids[i, k])
                entry = step.get(tok_id) if isinstance(step, dict) else None
                if entry is not None:
                    logprobs[i, k] = float(getattr(entry, "logprob", entry))

        np.savez(out_path, token_ids=token_ids, logprobs=logprobs,
                 prompt_token_ids=np.array(prompt_token_ids, dtype=object))
        print(f"[probe-baseline] wrote {N} prompts to {out_path}")

    elif mode == "check":
        # Load baseline reference, run teacher-forced candidate
        ref = np.load(os.environ["INFB_PROBE_REF_NPZ"], allow_pickle=True)
        ref_token_ids = ref["token_ids"]
        ref_logprobs  = ref["logprobs"]
        ref_prompt_tokens = ref["prompt_token_ids"]

        full_seqs, n_prompt_tokens, target_token_ids = [], [], []
        for i in range(len(prompts)):
            ptoks = list(int(x) for x in ref_prompt_tokens[i])
            gen_len = int((ref_token_ids[i] != -1).sum())
            gtoks = [int(x) for x in ref_token_ids[i, :gen_len]]
            full_seqs.append(ptoks + gtoks)
            n_prompt_tokens.append(len(ptoks))
            target_token_ids.append(gtoks)

        tf = llm.generate(
            [TokensPrompt(prompt_token_ids=s) for s in full_seqs],
            SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1, seed=0),
        )

        atol = float(os.environ.get("INFB_PROBE_ATOL", "0.05"))
        rtol = float(os.environ.get("INFB_PROBE_RTOL", "0.05"))
        total_pos, total_pass = 0, 0
        max_drift = 0.0
        per_prompt = []
        for i, out in enumerate(tf):
            plp = out.prompt_logprobs or []
            n_prompt = n_prompt_tokens[i]
            targets = target_token_ids[i]
            ref_lp_arr = ref_logprobs[i]
            n_pass = n_total = 0
            prompt_max_drift = 0.0
            for k in range(len(targets)):
                rlp = float(ref_lp_arr[k])
                if rlp != rlp:  # NaN
                    continue
                pos = n_prompt + k
                if pos >= len(plp) or plp[pos] is None:
                    n_total += 1
                    continue
                entry = plp[pos].get(targets[k]) if isinstance(plp[pos], dict) else None
                if entry is None:
                    n_total += 1
                    continue
                clp = float(getattr(entry, "logprob", entry))
                n_total += 1
                drift = abs(clp - rlp)
                prompt_max_drift = max(prompt_max_drift, drift)
                max_drift = max(max_drift, drift)
                if drift <= atol + rtol * abs(rlp):
                    n_pass += 1
            total_pos += n_total
            total_pass += n_pass
            per_prompt.append({
                "prompt_idx": i,
                "n_positions": n_total,
                "n_passing": n_pass,
                "pass_rate": n_pass / n_total if n_total else 0.0,
                "max_drift": prompt_max_drift,
            })

        overall = total_pass / total_pos if total_pos else 0.0
        summary = {
            "comparison": "teacher_forced_logprobs",
            "num_prompts": len(prompts),
            "num_positions_total": total_pos,
            "num_positions_passing": total_pass,
            "overall_pass_rate": overall,
            "correctness_ok": overall >= 0.95,
            "max_drift": max_drift,
            "atol": atol, "rtol": rtol,
            "per_prompt": per_prompt,
        }
        Path(out_path).write_text(json.dumps(summary, indent=2))
        print(f"[probe-check] pass_rate={overall:.4f} ok={overall >= 0.95} max_drift={max_drift:.4f}")

    else:
        sys.exit(f"unknown INFB_PROBE_MODE: {mode}")
'''




def _sglang_http_gen(base_url, payload, timeout=1200):
    import urllib.request, json as _j
    req = urllib.request.Request(base_url + "/generate", data=_j.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return _j.loads(r.read().decode())


def _sglang_greedy_http(base, prompts, max_tokens):
    """Pass A over HTTP: greedy-decode each text prompt. token_ids from
    output_token_logprobs slot 1, prompt_token_ids from input_token_logprobs slot 1."""
    recs = []
    for p in prompts:
        o = _sglang_http_gen(base, {"text": p,
                                    "sampling_params": {"temperature": 0, "max_new_tokens": int(max_tokens)},
                                    "return_logprob": True, "logprob_start_len": 0})
        meta = o.get("meta_info") or {}
        out_lp = meta.get("output_token_logprobs") or []
        tids = [int(e[1]) for e in out_lp if isinstance(e, (list, tuple)) and len(e) >= 2 and e[1] is not None]
        in_lp = meta.get("input_token_logprobs") or []
        ptoks = [int(e[1]) for e in in_lp if isinstance(e, (list, tuple)) and len(e) >= 2 and e[1] is not None]
        recs.append({"token_ids": tids, "prompt_token_ids": ptoks})
    return recs


def _sglang_teacher_forced_http(base, full_seqs, n_prompt_tokens, n_gen_tokens):
    """Pass B over HTTP: feed (prompt+gen) as input_ids, read input_token_logprobs from
    logprob_start_len=n_prompt-1 → per-generated-position logprob (slot 0)."""
    results = []
    for i, seq in enumerate(full_seqs):
        start = max(0, n_prompt_tokens[i] - 1)
        o = _sglang_http_gen(base, {"input_ids": seq,
                                    "sampling_params": {"temperature": 0, "max_new_tokens": 1},
                                    "return_logprob": True,
                                    "logprob_start_len": start})
        raw = (o.get("meta_info") or {}).get("input_token_logprobs") or []
        seq_lp = []
        offset = n_prompt_tokens[i] - start
        for k in range(n_gen_tokens[i]):
            raw_index = offset + k
            if raw_index >= len(raw):
                seq_lp.append(None); continue
            entry = raw[raw_index]
            if entry is None:
                seq_lp.append(None)
            elif isinstance(entry, (list, tuple)) and len(entry) >= 1:
                seq_lp.append(float(entry[0]) if entry[0] is not None else None)
            else:
                seq_lp.append(float(entry))
        results.append(seq_lp)
    return results


def _aggregate_correctness(target_token_ids, ref_logprobs, cand_logprobs,
                           atol, rtol, min_pass_rate=0.95):
    results = []
    total_positions = 0
    total_passing = 0
    overall_max_drift = 0.0
    overall_first_fail = None

    for i in range(len(target_token_ids)):
        targets = target_token_ids[i]
        ref_lp_arr = ref_logprobs[i]
        cand_lps = cand_logprobs[i]

        n_pass = 0
        n_total = 0
        max_drift = 0.0
        first_fail = None
        for j in range(len(targets)):
            rlp = float(ref_lp_arr[j])
            if rlp != rlp:
                continue
            clp = cand_lps[j]
            if clp != clp:
                n_total += 1
                continue
            n_total += 1
            drift = abs(clp - rlp)
            if drift > max_drift:
                max_drift = drift
            threshold = atol + rtol * abs(rlp)
            if drift <= threshold:
                n_pass += 1
            elif first_fail is None:
                first_fail = {"position": j, "ref_lp": rlp, "cand_lp": clp,
                              "drift": drift, "threshold": threshold}

        prompt_pass_rate = n_pass / n_total if n_total > 0 else 0.0
        total_positions += n_total
        total_passing += n_pass
        if max_drift > overall_max_drift:
            overall_max_drift = max_drift
            if first_fail is not None and overall_first_fail is None:
                overall_first_fail = {"prompt_idx": i, **first_fail}

        results.append({
            "prompt_idx": i,
            "logprob_pass_rate": prompt_pass_rate,
            "max_drift": max_drift,
            "ref_len": len(targets),
            "n_positions_compared": n_total,
            "first_failure": first_fail,
        })

    overall_pass_rate = total_passing / total_positions if total_positions else 0.0
    return {
        "num_prompts": len(target_token_ids),
        "num_positions_total": total_positions,
        "num_positions_passing": total_passing,
        "overall_pass_rate": overall_pass_rate,
        "correctness_ok": overall_pass_rate >= min_pass_rate,
        "min_pass_rate": min_pass_rate,
        "atol": atol,
        "rtol": rtol,
        "max_drift": overall_max_drift,
        "first_failure": overall_first_fail,
        "per_prompt": results,
    }


def _run_sglang_teacher_forced(args, *, prompts, max_tokens) -> int:
    """Engine-conditional SGLang correctness gate — TEACHER-FORCED LOGPROB,
    aligned with the official verifier (same two-pass protocol + _aggregate_correctness).

    Boots the PRISTINE (git worktree @ inference_base) and CANDIDATE
    (workspace/repo) SGLang servers SEQUENTIALLY — never both at once (OOM).
    PRISTINE: two-pass reference (Pass A greedy → tokens; Pass B teacher-force
    → per-position logprobs). CANDIDATE: teacher-force the SAME token sequences
    → candidate logprobs. Then _aggregate_correctness compares per-position
    |lp_cand − lp_ref| ≤ atol + rtol·|lp_ref| and gates on pooled pass_rate ≥
    min_pass_rate — the IDENTICAL metric + scorer the official verifier uses.

    Non-blocking ONLY for genuine infra flakes: if the PRISTINE (unmodified
    base) server fails to come ready, that's a host/infra issue → prints "gate
    inconclusive" and returns 0 (a flaky host never falsely fails a good patch).
    But if the pristine server booted fine and only the CANDIDATE server fails
    to boot/probe, that is the agent's edit crashing serving → reported LOUDLY
    as a correctness FAIL (rc=1), because the official gate boots the same
    candidate server and scores it 0. Otherwise writes the aggregate summary
    (overall_pass_rate/max_drift/per_prompt) to args.output and returns
    0 / 1 on correctness_ok.

    The serve helpers are lazily imported from the sibling _run_benchmark_serve
    script (so a vllm-engine task never pays for the import); they inherit the
    task's TP/dtype/model/context-length, making this branch model-agnostic.
    """
    import shutil
    import subprocess as _sp
    import tempfile

    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    from _run_benchmark_serve import (
        _resolve_workspace,
        _resolve_venv_python,
        _resolve_model,
        _pick_free_port,
        _wait_for_server_ready,
        _start_engine_serve,
        _stop_engine_serve,
        _cleanup_stale_shm,
        _kill_stray_gpu_processes,
    )

    try:
        workspace = _resolve_workspace()
        cand_repo = workspace / "repo"
        if not cand_repo.exists():
            print(f"[sc-gate] gate inconclusive: workspace/repo missing at {cand_repo}",
                  flush=True)
            return 0
        venv_python = _resolve_venv_python()
        model = _resolve_model()
    except Exception as _e:
        print(f"[sc-gate] gate inconclusive: could not resolve target env "
              f"({type(_e).__name__}: {_e}) — non-blocking, returning 0", flush=True)
        return 0
    _vis = [d.strip() for d in (os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",") if d.strip()]
    if len(_vis) < TP_SIZE:
        _vis = [str(i) for i in range(TP_SIZE)]
    cuda_dev = ",".join(_vis[:TP_SIZE])
    log_dir = Path(args.workdir)
    log_dir.mkdir(parents=True, exist_ok=True)

    tmp_worktree = Path(tempfile.mkdtemp(prefix="sc_gate_pristine_"))
    pris_repo = tmp_worktree / "repo"
    wt_added = False
    try:
        r = _sp.run(
            ["git", "-C", str(cand_repo), "worktree", "add", "--detach",
             str(pris_repo), "inference_base"],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            print(f"[sc-gate] gate inconclusive: git worktree add failed "
                  f"(rc={r.returncode}): {r.stderr.strip()[:300]}", flush=True)
            return 0
        wt_added = True

        def _boot_and_gen_reference(repo: Path):
            """Boot PRISTINE, Pass A greedy + Pass B teacher-force → the reference.
            Returns dict(full_seqs,n_prompt,n_gen,token_ids,ref_logprobs) or None."""
            _cleanup_stale_shm()
            for _sd in cuda_dev.split(","):
                _kill_stray_gpu_processes(cuda_device=_sd)
            port = _pick_free_port()
            log_path = log_dir / "sc_gate_pristine.log"
            print(f"[sc-gate] booting pristine server (repo={repo}) on :{port}, "
                  f"log → {log_path}", flush=True)
            proc = _start_engine_serve(
                model=model, venv_python=venv_python, repo=repo, port=port,
                log_path=log_path, cuda_device=cuda_dev,
            )
            try:
                if not _wait_for_server_ready(port, 900, proc):
                    print(f"[sc-gate] pristine server failed to become ready "
                          f"(see {log_path})", flush=True)
                    return None
                base = f"http://127.0.0.1:{port}"
                try:
                    recs = _sglang_greedy_http(base, prompts, max_tokens)
                    full_seqs, n_prompt, n_gen = [], [], []
                    for r in recs:
                        p = [int(x) for x in r["prompt_token_ids"]]
                        g = [int(x) for x in r["token_ids"]][:max_tokens]
                        full_seqs.append(p + g); n_prompt.append(len(p)); n_gen.append(len(g))
                    ref_lp = _sglang_teacher_forced_http(base, full_seqs, n_prompt, n_gen)
                except Exception as _e:
                    print(f"[sc-gate] pristine reference gen failed "
                          f"({type(_e).__name__}: {_e})", flush=True)
                    return None
                tgt = [[int(x) for x in r["token_ids"]][:max_tokens] for r in recs]
                print(f"[sc-gate] pristine reference: {len(recs)} prompts, "
                      f"{sum(n_gen)} generated positions", flush=True)
                return {"full_seqs": full_seqs, "n_prompt": n_prompt, "n_gen": n_gen,
                        "token_ids": tgt, "ref_logprobs": ref_lp}
            finally:
                _stop_engine_serve(proc)
                _cleanup_stale_shm()
                for _sd in cuda_dev.split(","):
                    _kill_stray_gpu_processes(cuda_device=_sd)

        def _boot_and_teacher_force_candidate(repo: Path, ref):
            """Boot CANDIDATE, teacher-force the reference's token sequences →
            candidate logprobs (list[list]), or None if it didn't boot/probe."""
            _cleanup_stale_shm()
            for _sd in cuda_dev.split(","):
                _kill_stray_gpu_processes(cuda_device=_sd)
            port = _pick_free_port()
            log_path = log_dir / "sc_gate_candidate.log"
            print(f"[sc-gate] booting candidate server (repo={repo}) on :{port}, "
                  f"log → {log_path}", flush=True)
            proc = _start_engine_serve(
                model=model, venv_python=venv_python, repo=repo, port=port,
                log_path=log_path, cuda_device=cuda_dev,
            )
            try:
                if not _wait_for_server_ready(port, 900, proc):
                    print(f"[sc-gate] candidate server failed to become ready "
                          f"(see {log_path})", flush=True)
                    return None
                base = f"http://127.0.0.1:{port}"
                try:
                    return _sglang_teacher_forced_http(
                        base, ref["full_seqs"], ref["n_prompt"], ref["n_gen"])
                except Exception as _e:
                    print(f"[sc-gate] candidate teacher-forcing failed "
                          f"({type(_e).__name__}: {_e})", flush=True)
                    return None
            finally:
                _stop_engine_serve(proc)
                _cleanup_stale_shm()
                for _sd in cuda_dev.split(","):
                    _kill_stray_gpu_processes(cuda_device=_sd)

        ref = _boot_and_gen_reference(pris_repo)
        if ref is None:
            print("[sc-gate] gate inconclusive (pristine side) — non-blocking, "
                  "returning 0", flush=True)
            return 0
        cand_lp = _boot_and_teacher_force_candidate(cand_repo, ref)
        if cand_lp is None:
            cand_log = log_dir / "sc_gate_candidate.log"
            print(
                "[sc-gate] CANDIDATE server FAILED to boot/probe while the "
                "pristine server booted fine on this same host — this is a "
                "CORRECTNESS FAILURE, not a flake. Your edit almost certainly "
                f"crashes `sglang.launch_server` (read {cand_log}). The official "
                "correctness gate boots this same server and will score this "
                "candidate reward=0 regardless of speedup. Do NOT dismiss this "
                "as OOM and ship — fix the serve crash before finalizing.",
                flush=True)
            summary = {
                "comparison": "teacher_forced_logprobs",
                "num_prompts": len(prompts),
                "correctness_ok": False,
                "candidate_server_failed": True,
                "reason": ("candidate sglang server failed to boot/probe while "
                           "pristine booted OK — deterministic serve crash, "
                           "official gate scores this 0"),
                "candidate_log": str(cand_log),
            }
            out_path = Path(args.output) if args.output else log_dir / "result.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(summary, indent=2))
            return 1
    finally:
        if wt_added:
            _sp.run(["git", "-C", str(cand_repo), "worktree", "remove",
                     "--force", str(pris_repo)],
                    capture_output=True, text=True)
        shutil.rmtree(tmp_worktree, ignore_errors=True)

    def _nan_norm(rows):
        return [[(float(v) if v is not None else float("nan")) for v in (row or [])]
                for row in rows]
    result = _aggregate_correctness(
        ref["token_ids"], _nan_norm(ref["ref_logprobs"]), _nan_norm(cand_lp),
        atol=args.atol, rtol=args.rtol, min_pass_rate=args.min_pass_rate,
    )
    result["comparison"] = "teacher_forced_logprobs"
    result["num_prompts_sampled"] = len(prompts)
    if _BANK_SIZE[0]:
        result["num_prompts_bank"] = _BANK_SIZE[0]
    out_path = Path(args.output) if args.output else log_dir / "result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n=== run_correctness summary (sglang teacher-forced logprobs) ===")
    print(f"  overall_pass_rate: {result['overall_pass_rate']:.4f} "
          f"(>= {args.min_pass_rate})")
    print(f"  correctness_ok: {result['correctness_ok']}")
    print(f"  max_drift: {result['max_drift']:.4f}  "
          f"(atol={args.atol} rtol={args.rtol})")
    print(f"  positions: {result['num_positions_passing']}/{result['num_positions_total']}")
    _n_s = result.get("num_prompts_sampled")
    _n_b = result.get("num_prompts_bank") or result.get("num_prompts")
    if _n_s and _n_b and _n_s < _n_b:
        print(f"  COVERAGE: {_n_s}/{_n_b} prompts. A break that only some prompts expose can\n"
              f"            survive this check -- rerun with --num-prompts {_n_b} before finalizing.")
    print(f"  wrote: {out_path}")
    return 0 if result["correctness_ok"] else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-tokens", type=int, default=100,
                   help="Generated tokens per prompt (default 100 — matches the official verifier)")
    p.add_argument("--atol", type=float, default=CORRECTNESS_CALIBRATION["atol"],
                   help="Per-position logprob atol (default: env INFB_TASK_CORRECTNESS_ATOL, else this task's calibrated value)")
    p.add_argument("--rtol", type=float, default=CORRECTNESS_CALIBRATION["rtol"],
                   help="Per-position logprob rtol (default: env INFB_TASK_CORRECTNESS_RTOL, else this task's calibrated value)")
    p.add_argument("--min-pass-rate", type=float, default=CORRECTNESS_CALIBRATION["min_pass_rate"],
                   help="Pooled position pass-rate threshold (default: env INFB_TASK_CORRECTNESS_MIN_PASS_RATE, else this task's calibrated value)")
    p.add_argument("--num-prompts", type=int, default=50,
                   help="How many prompts to randomly sample from the official bank (default 50; a smaller value samples less of the bank and can miss a break that only some prompts expose)")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for the prompt subsample (default 0)")
    p.add_argument("--prompts", nargs="+", default=None,
                   help="Override: use these exact prompts instead of sampling the official bank")
    p.add_argument("--output", default=None,
                   help="Write summary JSON here (default: stdout)")
    p.add_argument("--workdir", default="/logs/agent/dsv2_correctness_probe")
    args = p.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    venv_python = _resolve_venv_python()
    model = _resolve_model()
    workspace = _resolve_workspace()
    repo = workspace / "repo"
    if not repo.exists():
        sys.exit(f"workspace/repo missing at {repo}; run prepare_target_env.sh first")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _gpu_cleanup import kill_stray_gpu_processes
    for _sd in [d.strip() for d in
                (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",") if d.strip()]:
        kill_stray_gpu_processes(cuda_device=_sd)

    if args.prompts:
        prompts = args.prompts
    else:
        _bank_path = Path(__file__).resolve().parent / "correctness_prompts.json"
        if _bank_path.exists():
            _bank = json.loads(_bank_path.read_text())
        else:
            print(f"[run_correctness] WARNING: official bank not found at {_bank_path}; "
                  f"falling back to the built-in probe set", flush=True)
            _bank = PROBE_PROMPTS
        import random as _random
        _n = min(args.num_prompts, len(_bank))
        prompts = _random.Random(args.seed).sample(list(_bank), _n)
        _BANK_SIZE[0] = len(_bank)
        print(f"[run_correctness] sampled {_n}/{len(_bank)} official prompts (seed={args.seed})",
              flush=True)

    args.atol = float(os.environ.get("INFB_TASK_CORRECTNESS_ATOL", args.atol))
    args.rtol = float(os.environ.get("INFB_TASK_CORRECTNESS_RTOL", args.rtol))
    args.min_pass_rate = float(
        os.environ.get("INFB_TASK_CORRECTNESS_MIN_PASS_RATE", args.min_pass_rate))

    if _ENGINE == "sglang":
        return _run_sglang_teacher_forced(args, prompts=prompts, max_tokens=args.max_tokens)
    prompts_json = workdir / "prompts.json"
    prompts_json.write_text(json.dumps(prompts))
    runner = workdir / "_inner.py"
    runner.write_text(_INNER_RUNNER)

    base_env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    base_env["INFB_PROBE_MODEL"] = model
    _tp = (os.environ.get("INFB_CORRECTNESS_TP") or "").strip() or str(TP_SIZE)
    base_env["INFB_PROBE_TP"] = _tp
    print(f"[run_correctness] vLLM in-process teacher-forced self-check, "
          f"tensor_parallel_size={_tp}", flush=True)
    base_env["INFB_PROBE_PROMPTS_JSON"] = str(prompts_json)
    base_env["INFB_PROBE_MAX_TOKENS"] = str(args.max_tokens)
    base_env["INFB_PROBE_ATOL"] = str(args.atol)
    base_env["INFB_PROBE_RTOL"] = str(args.rtol)
    base_env.setdefault("HF_HUB_OFFLINE", "1")
    base_env.setdefault("TRANSFORMERS_OFFLINE", "1")
    base_env.setdefault("CUDA_VISIBLE_DEVICES",
                        ",".join(str(i) for i in range(int(_tp))))
    _nccl_preload = base_env.get("INFB_NCCL_PRELOAD")
    if _nccl_preload and os.path.exists(_nccl_preload) and "LD_PRELOAD" not in base_env:
        base_env["LD_PRELOAD"] = _nccl_preload
        print(f"[run_correctness] LD_PRELOAD={_nccl_preload} (NCCL compat, mirrors serve)", flush=True)
    venv_bin = str(Path(venv_python).parent)
    base_env["PATH"] = f"{venv_bin}:" + base_env.get(
        "PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    )

    flashinfer_overlay = workspace / ".flashinfer_overlay"
    overlay_pp = str(flashinfer_overlay) if flashinfer_overlay.exists() else None

    ref_npz = workdir / "baseline.npz"
    print(f"[run_correctness] generating pristine baseline ({len(prompts)} prompts, "
          f"{args.max_tokens} max_tokens)...", flush=True)
    env_baseline = dict(base_env)
    env_baseline["INFB_PROBE_MODE"] = "baseline"
    env_baseline["INFB_PROBE_OUT"] = str(ref_npz)
    if overlay_pp:
        env_baseline["PYTHONPATH"] = overlay_pp
    t0 = time.time()
    rc = subprocess.run([venv_python, "-P", str(runner)], env=env_baseline)
    print(f"[run_correctness] baseline rc={rc.returncode} elapsed={time.time()-t0:.0f}s",
          flush=True)
    if rc.returncode != 0 and not ref_npz.exists():
        sys.exit("baseline pass failed")

    out_path = Path(args.output) if args.output else workdir / "result.json"
    print(f"[run_correctness] checking candidate (overlay)...", flush=True)
    env_check = dict(base_env)
    env_check["INFB_PROBE_MODE"] = "check"
    env_check["INFB_PROBE_REF_NPZ"] = str(ref_npz)
    env_check["INFB_PROBE_OUT"] = str(out_path)
    env_check["PYTHONPATH"] = ":".join([str(repo)] + ([overlay_pp] if overlay_pp else []))
    t0 = time.time()
    rc = subprocess.run([venv_python, "-P", str(runner)], env=env_check)
    print(f"[run_correctness] check rc={rc.returncode} elapsed={time.time()-t0:.0f}s",
          flush=True)
    if not out_path.exists():
        sys.exit("check pass produced no output")

    summary = json.loads(out_path.read_text())
    print(f"\n=== run_correctness summary ===")
    print(f"  pass_rate: {summary['overall_pass_rate']:.4f}")
    print(f"  correctness_ok: {summary['correctness_ok']}")
    print(f"  max_drift: {summary['max_drift']:.4f}")
    print(f"  positions: {summary['num_positions_passing']}/{summary['num_positions_total']}")
    print(f"  wrote: {out_path}")
    return 0 if summary["correctness_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
