#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Helper: run one vllm bench throughput invocation or correctness check.

Always called as a subprocess by verify_throughput_real.py.  Never imported.

Modes:
  python _candidate_runner.py throughput    --workload-json <wl.json> --out-json <out.json>
  python _candidate_runner.py serve_bench   --workload-json <wl.json> --out-json <out.json>
  python _candidate_runner.py gen_reference --prompts-json <p.json> --out-npz <ref.npz> --model <m>
  python _candidate_runner.py correctness   --prompts-json <p.json> --ref-npz <ref.npz> --out-json <out.json>

`gen_reference` is the LIVE, in-container producer of the teacher-forced reference
(the npz that `correctness` consumes). It is invoked by the verifier on the
PRISTINE side, in a clean worktree, so the reference is regenerated for every
run rather than carried with the task. See verify_throughput_real.py.

The caller sets PYTHONPATH=<workspace/repo> and CUDA_VISIBLE_DEVICES appropriately.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _set_offline():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")



def cmd_throughput(args) -> int:
    wl = json.loads(Path(args.workload_json).read_text())
    _set_offline()

    import subprocess, tempfile

    out_file = args.out_json
    python = sys.executable

    cmd = [
        python, "-P", "-m", "vllm.entrypoints.cli.main", "bench", "throughput",
        "--model", wl["model"],
        "--input-len",  str(wl["input_len"]),
        "--output-len", str(wl["output_len"]),
        "--num-prompts", str(wl["num_prompts"]),
        "--tensor-parallel-size", str(wl.get("tp", 2)),
        "--output-json", out_file,
        "--dataset-name", "random",
        "--random-input-len",  str(wl["input_len"]),
        "--random-output-len", str(wl["output_len"]),
    
        "--temperature", "1.0",
        "--top-p", "1.0",
        "--top-k", "0",
    ]

    env = dict(os.environ)
    t0 = time.time()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True,
                            start_new_session=True, timeout=2400)
    elapsed = time.time() - t0

    if result.returncode != 0:
        Path(out_file).write_text(json.dumps({
            "error": True, "rc": result.returncode,
            "stderr_tail": result.stderr[-3000:],
            "elapsed_sec": elapsed,
        }))
        return 1

    try:
        data = json.loads(Path(out_file).read_text())
    except Exception as e:
        Path(out_file).write_text(json.dumps({"error": True, "parse_error": str(e)}))
        return 1

    data["elapsed_sec"] = elapsed
    Path(out_file).write_text(json.dumps(data))
    return 0




def _dataset_args(wl: dict, num_prompts, out_dir) -> list:
    ds = wl.get("dataset")
    if not ds or ds == "random":
        return ["--dataset-name", "random",
                "--random-input-len", str(wl["input_len"]),
                "--random-output-len", str(wl["output_len"]),
                "--num-prompts", str(num_prompts)]
    if ds == "custom":
        bank_name = wl.get("prompt_bank")
        if not bank_name:
            raise ValueError(f"workload {wl.get('name')!r}: dataset=custom needs prompt_bank")
        bank_path = Path(__file__).resolve().parent / bank_name
        rows = [l for l in bank_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not rows:
            raise ValueError(f"{bank_name}: empty bank")
        n = int(num_prompts)
        picked = [rows[i % len(rows)] for i in range(n)]
        dst = Path(out_dir) / "mtp_longctx_prompts.jsonl"
        dst.write_text("\n".join(picked) + "\n", encoding="utf-8")
        out = ["--dataset-name", "custom", "--dataset-path", str(dst),
               "--custom-output-len", str(wl["output_len"]),
               "--num-prompts", str(n),
               "--disable-shuffle", "--no-oversample"]
        if wl.get("ignore_eos", True):
            out += ["--ignore-eos"]
        return out
    if ds != "sharegpt":
        raise ValueError(
            f"workload {wl.get('name')!r}: dataset={ds!r} is not supported here -- "
            "only random, sharegpt and custom are wired."
        )
    bank_name = wl.get("prompt_bank")
    if not bank_name:
        raise ValueError(f"workload {wl.get('name')!r}: dataset=sharegpt needs prompt_bank")
    bank_path = Path(__file__).resolve().parent / bank_name
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    items = bank["items"]
    if not items:
        raise ValueError(f"{bank_name}: empty bank")
    n = int(num_prompts)
    picked = [items[i % len(items)] for i in range(n)]
    conversations = [
        {"conversations": [
            {"from": "human", "value": it["prompt"]},
            {"from": "gpt", "value": it.get("turn2") or ""},
        ]}
        for it in picked
    ]
    dst = Path(out_dir) / "mtp_sharegpt_prompts.json"
    dst.write_text(json.dumps(conversations), encoding="utf-8")
    out = ["--dataset-name", "sharegpt", "--dataset-path", str(dst),
           "--sharegpt-output-len", str(wl["output_len"]),
           "--num-prompts", str(n),
           "--disable-shuffle", "--no-oversample"]
    if wl.get("ignore_eos", True):
        out += ["--ignore-eos"]
    return out
def cmd_serve_bench(args) -> int:
    """Run one `vllm bench serve --base-url http://127.0.0.1:$PORT` against
    a server already started by the verifier. Workload spec includes the
    port to hit. Output JSON is the standard vllm bench serve dump."""
    wl = json.loads(Path(args.workload_json).read_text())
    _set_offline()

    import subprocess

    out_file = args.out_json
    python = sys.executable
    port = wl["port"]
    request_rate = str(wl.get("request_rate", "inf"))
    max_concurrency = int(wl.get("max_concurrency", 128))
    burstiness = str(wl.get("burstiness", "1.0"))

    cmd = [
        python, "-P", "-m", "vllm.entrypoints.cli.main", "bench", "serve",
        "--backend", "vllm",
        "--model", wl["model"],
        "--host", "127.0.0.1",
        "--port", str(port),
        "--endpoint", "/v1/completions",
        *_dataset_args(wl, wl["num_prompts"], Path(out_file).parent),
        "--request-rate", request_rate,
        "--max-concurrency", str(max_concurrency),
        "--save-result",
        *(["--save-detailed"] if wl.get("save_detailed") else []),
        "--result-filename", out_file,
        "--metric-percentiles", "50,90,95,99",
        "--trust-remote-code",
    
        "--temperature", "1.0",
        "--top-p", "1.0",
        "--top-k", "0",
    ]
    seed = wl.get("seed")
    if seed is not None:
        cmd += ["--seed", str(int(seed))]
    if burstiness not in ("1.0", "1", ""):
        cmd += ["--burstiness", burstiness]

    env = dict(os.environ)
    env["NO_PROXY"] = "127.0.0.1,localhost"
    env["no_proxy"] = "127.0.0.1,localhost"
    t0 = time.time()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True,
                            start_new_session=True, timeout=int(os.environ.get("INFB_BENCH_SUBPROCESS_TIMEOUT_SEC", "1800")))
    elapsed = time.time() - t0

    if result.returncode != 0:
        Path(out_file).write_text(json.dumps({
            "error": True, "rc": result.returncode,
            "stderr_tail": result.stderr[-3000:],
            "stdout_tail": result.stdout[-3000:],
            "elapsed_sec": elapsed,
        }))
        return 1

    try:
        data = json.loads(Path(out_file).read_text())
    except Exception as e:
        Path(out_file).write_text(json.dumps({
            "error": True, "parse_error": str(e),
            "stderr_tail": result.stderr[-3000:],
            "stdout_tail": result.stdout[-3000:],
        }))
        return 1

    completed = data.get("completed", data.get("successful_requests", -1))
    failed = data.get("failed_requests", data.get("failed", -1))
    if completed == 0:
        data["server_likely_dead"] = True
        data["stdout_tail"] = result.stdout[-3000:]
        data["stderr_tail"] = result.stderr[-3000:]
    elif failed and failed > 0:
        data["partial_failures"] = {"completed": completed, "failed": failed}
        data["stdout_tail"] = result.stdout[-3000:]

    data["elapsed_sec"] = elapsed
    Path(out_file).write_text(json.dumps(data))
    return 0




def _build_llm(model: str, tp: int):
    from vllm import LLM
    import json as _json

    _kw_env = os.environ.get("INFB_CORRECTNESS_LLM_KWARGS_JSON")
    if _kw_env:
        kwargs = _json.loads(_kw_env)
        kwargs.setdefault("model", model)
        kwargs.setdefault("tensor_parallel_size", tp)
        return LLM(**kwargs)

    gpu_mem = float(os.environ.get("INFB_CORRECTNESS_GPU_MEM_UTIL", "0.85"))
    return LLM(
        model=model,
        tensor_parallel_size=tp,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=gpu_mem,
        enforce_eager=False,
        seed=0,
        kv_cache_dtype=os.environ.get("INFB_KV_CACHE_DTYPE", "auto"),
    )
def cmd_correctness(args) -> int:
    _set_offline()

    import numpy as np

    prompts = json.loads(Path(args.prompts_json).read_text())
    ref = np.load(args.ref_npz, allow_pickle=True)
    ref_token_ids = ref["token_ids"]
    ref_logprobs  = ref["logprobs"]
    ref_prompt_token_ids = ref["prompt_token_ids"]

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    model = args.model
    tp    = int(args.tp)

    t0 = time.time()
    llm = _build_llm(model, tp)
    load_sec = time.time() - t0

    full_seqs = []
    n_prompt_tokens = []
    target_token_ids = []
    for i in range(len(prompts)):
        ptoks = list(int(x) for x in ref_prompt_token_ids[i])
        gen_len = int((ref_token_ids[i] != -1).sum())
        gtoks = [int(x) for x in ref_token_ids[i, :gen_len]]
        full_seqs.append(ptoks + gtoks)
        n_prompt_tokens.append(len(ptoks))
        target_token_ids.append(gtoks)

    tokens_prompts = [TokensPrompt(prompt_token_ids=seq) for seq in full_seqs]

    t0 = time.time()
    outputs = llm.generate(
        tokens_prompts,
        sampling_params=SamplingParams(
            temperature=0,
            max_tokens=1,
            prompt_logprobs=1,
            seed=0,
        ),
    )
    gen_sec = time.time() - t0

    atol = float(args.atol)
    rtol = float(args.rtol)

    cand_logprobs = []
    for i, out in enumerate(outputs):
        plp = out.prompt_logprobs or []
        n_prompt = n_prompt_tokens[i]
        targets = target_token_ids[i]
        cand_lps = []
        for k in range(len(targets)):
            pos = n_prompt + k
            if pos >= len(plp) or plp[pos] is None:
                cand_lps.append(float("nan"))
                continue
            step_dict = plp[pos]
            tok_id = targets[k]
            entry = step_dict.get(tok_id) if isinstance(step_dict, dict) else None
            if entry is None:
                cand_lps.append(float("nan"))
            else:
                cand_lps.append(float(getattr(entry, "logprob", entry)))
        cand_logprobs.append(cand_lps)

    min_pass_rate = float(args.min_pass_rate)
    summary = _aggregate_correctness(
        target_token_ids, ref_logprobs, cand_logprobs, atol, rtol, min_pass_rate,
    )
    summary["load_sec"] = load_sec
    summary["gen_sec"] = gen_sec
    summary["comparison"] = "teacher_forced_logprobs"
    Path(args.out_json).write_text(json.dumps(summary, indent=2))
    return 0


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



def cmd_gen_reference(args) -> int:
    _set_offline()

    import numpy as np
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    prompts = json.loads(Path(args.prompts_json).read_text())
    model = args.model
    tp    = int(args.tp)
    L     = int(args.max_tokens)
    N     = len(prompts)

    t0 = time.time()
    llm = _build_llm(model, tp)
    load_sec = time.time() - t0

    tA = time.time()
    gen_outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0, max_tokens=L, seed=0),
    )
    genA_sec = time.time() - tA

    token_ids = np.full((N, L), -1, dtype=np.int32)
    prompt_token_ids_list = [None] * N
    for i, out in enumerate(gen_outputs):
        ptoks = list(out.prompt_token_ids)
        gtoks = list(out.outputs[0].token_ids)
        token_ids[i, : len(gtoks)] = gtoks
        prompt_token_ids_list[i] = np.asarray(ptoks, dtype=np.int32)

    full_seqs = []
    n_prompt_tokens = []
    n_gen_tokens = []
    for i in range(N):
        ptoks = prompt_token_ids_list[i].tolist()
        gtoks = [int(x) for x in token_ids[i] if x != -1]
        full_seqs.append(ptoks + gtoks)
        n_prompt_tokens.append(len(ptoks))
        n_gen_tokens.append(len(gtoks))

    tB = time.time()
    tf_outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=seq) for seq in full_seqs],
        sampling_params=SamplingParams(
            temperature=0, max_tokens=1, prompt_logprobs=1, seed=0,
        ),
    )
    genB_sec = time.time() - tB

    logprobs = np.full((N, L), float("nan"), dtype=np.float32)
    for i, out in enumerate(tf_outputs):
        plp = out.prompt_logprobs or []
        n_prompt = n_prompt_tokens[i]
        n_gen = n_gen_tokens[i]
        for k in range(n_gen):
            pos = n_prompt + k
            if pos >= len(plp) or plp[pos] is None:
                continue
            step = plp[pos]
            tok_id = int(token_ids[i, k])
            entry = step.get(tok_id) if isinstance(step, dict) else None
            if entry is not None:
                logprobs[i, k] = float(getattr(entry, "logprob", entry))

    np.savez(
        args.out_npz,
        token_ids=token_ids,
        logprobs=logprobs,
        prompt_token_ids=np.array(prompt_token_ids_list, dtype=object),
    )
    print(f"[gen_reference] wrote {args.out_npz} "
          f"({N} prompts × {L} tokens; load={load_sec:.1f}s "
          f"passA={genA_sec:.1f}s passB={genB_sec:.1f}s)", flush=True)
    return 0



def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    tp = sub.add_parser("throughput")
    tp.add_argument("--workload-json", required=True)
    tp.add_argument("--out-json", required=True)

    sb = sub.add_parser("serve_bench")
    sb.add_argument("--workload-json", required=True)
    sb.add_argument("--out-json", required=True)

    cr = sub.add_parser("correctness")
    cr.add_argument("--prompts-json", required=True)
    cr.add_argument("--ref-npz", required=True)
    cr.add_argument("--out-json", required=True)
    cr.add_argument("--model", required=True)
    cr.add_argument("--tp", default="1")
    cr.add_argument("--max-tokens", default="100")
    cr.add_argument("--atol", default="0.05")
    cr.add_argument("--rtol", default="0.05")
    cr.add_argument("--min-pass-rate", default="0.95")

    gr = sub.add_parser("gen_reference")
    gr.add_argument("--prompts-json", required=True)
    gr.add_argument("--out-npz", required=True)
    gr.add_argument("--model", required=True)
    gr.add_argument("--tp", default="1")
    gr.add_argument("--max-tokens", default="100")

    args = parser.parse_args()
    if args.cmd == "throughput":
        return cmd_throughput(args)
    elif args.cmd == "serve_bench":
        return cmd_serve_bench(args)
    elif args.cmd == "correctness":
        return cmd_correctness(args)
    elif args.cmd == "gen_reference":
        return cmd_gen_reference(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
