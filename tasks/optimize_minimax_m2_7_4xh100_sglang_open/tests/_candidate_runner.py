#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Helper: run one bench throughput invocation or correctness check.

Always called as a subprocess by verify_throughput_real.py. The only importer is
tests/test_sglang_unit.py, which imports it to exercise module-level state.

ENGINE (INFB_ENGINE env, default 'sglang' for this task):
  * sglang — throughput via `sglang.Engine.async_generate` (offline, uvloop-safe);
             serve_bench via `python -m sglang.bench_serving` HTTP client against
             a server the verifier launched (`sglang.launch_server`); correctness
             uses live teacher-forced logprobs through launch_server HTTP.
  * vllm   — legacy `vllm bench throughput` / `vllm bench serve` / teacher-forced
             `vllm.LLM` correctness. Reachable when INFB_ENGINE=vllm.

Modes:
  python _candidate_runner.py throughput    --workload-json <wl.json> --out-json <out.json>
  python _candidate_runner.py serve_bench   --workload-json <wl.json> --out-json <out.json>
  python _candidate_runner.py gen_reference --prompts-json <p.json> --out-npz <ref.npz> --model <m>
  python _candidate_runner.py correctness   --prompts-json <p.json> --ref-npz <ref.npz> --out-json <out.json>

`gen_reference` is the LIVE, in-container producer of the teacher-forced reference
(the npz that `correctness` consumes). It is invoked by the verifier on the
PRISTINE side, in a clean worktree, so the reference is regenerated for every
run rather than carried with the task. See verify_throughput_real.py.
Under ENGINE=sglang both correctness modes use the live HTTP path.

The caller sets PYTHONPATH=<workspace/repo> and CUDA_VISIBLE_DEVICES appropriately.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ENGINE = os.environ.get("INFB_ENGINE", "sglang").lower()

LIVE_CORRECTNESS_SUPPORTED = True


def _scrub_hf_offline(env: dict) -> dict:
    """Remove HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE from a subprocess env."""
    if ENGINE == "sglang":
        env.pop("HF_HUB_OFFLINE", None)
        env.pop("TRANSFORMERS_OFFLINE", None)
    return env


def _set_offline():
    if ENGINE == "sglang":
        return
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _derive_total_token_throughput(data: dict):
    """Map a sglang.bench_serving --output-file dict -> total tokens/s (input+output / wall)."""
    def _num(*keys):
        for k in keys:
            v = data.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
                return float(v)
        return None

    tps = _num("total_token_throughput", "total_throughput")
    if tps is None:
        t_in = _num("total_input_tokens", "total_prompt_tokens")
        t_out = _num("total_output_tokens", "total_completion_tokens")
        dur = _num("duration", "benchmark_duration", "elapsed_time", "elapsed")
        if t_in is not None and t_out is not None and dur:
            tps = (t_in + t_out) / dur
    if tps is None:
        tps = _num("output_throughput")
    return tps



def cmd_throughput(args) -> int:
    wl = json.loads(Path(args.workload_json).read_text())
    _set_offline()

    if ENGINE == "sglang":
        return _cmd_throughput_sglang(wl, args)

    import subprocess, tempfile

    out_file = args.out_json
    python = sys.executable

    cmd = [
        python, "-P", "-m", "vllm.entrypoints.cli.main", "bench", "throughput",
        "--model", wl["model"],
        "--input-len",  str(wl["input_len"]),
        "--output-len", str(wl["output_len"]),
        "--num-prompts", str(wl["num_prompts"]),
        "--tensor-parallel-size", str(wl.get("tp", 1)),
        "--output-json", out_file,
        "--dataset-name", "random",
        "--random-input-len",  str(wl["input_len"]),
        "--random-output-len", str(wl["output_len"]),
    
        "--temperature", "1.0",
        "--top-p", "1.0",
        "--top-k", "0",
    ]

    env = dict(os.environ)
    _scrub_hf_offline(env)
    env["NO_PROXY"] = "127.0.0.1,localhost"
    env["no_proxy"] = "127.0.0.1,localhost"
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



def _cmd_throughput_sglang(wl: dict, args) -> int:
    import signal
    import subprocess

    out_file = args.out_json
    python = sys.executable

    model = wl["model"]
    input_len = int(wl["input_len"])
    output_len = int(wl["output_len"])
    num_prompts = int(wl["num_prompts"])
    tp = int(wl.get("tp", 1))
    context_length = max(input_len + output_len + 64, 4096)

    engine_kwargs = {
        "model_path": model,
        "tp_size": tp,
        "trust_remote_code": True,
        "mem_fraction_static": 0.85,
        "context_length": context_length,
    }

    script = f'''
import asyncio, json, random, time
from sglang import Engine

engine_kwargs = {engine_kwargs!r}
input_len = {input_len}
output_len = {output_len}
num_prompts = {num_prompts}

random.seed(42)
prompts_ids = [
    [random.randint(0, 32000 - 1) for _ in range(input_len)]
    for _ in range(num_prompts)
]

async def _run():
    engine = Engine(**engine_kwargs)
    try:
        sp = {{"temperature": 1.0, "top_p": 1.0, "top_k": 1073741824, "max_new_tokens": output_len, "ignore_eos": True}}
        t0 = time.time()
        coros = [
            engine.async_generate(input_ids=p, sampling_params=sp,
                                  return_logprob=False)
            for p in prompts_ids
        ]
        results = await asyncio.gather(*coros)
        elapsed = time.time() - t0
        return results, elapsed
    finally:
        engine.shutdown()

results, elapsed = asyncio.run(_run())

total_input_tokens = sum(r["meta_info"].get("prompt_tokens", 0) for r in results)
total_output_tokens = sum(r["meta_info"].get("completion_tokens", 0) for r in results)
total_tokens = total_input_tokens + total_output_tokens
output_throughput = total_output_tokens / elapsed if elapsed > 0 else 0.0
total_throughput = total_tokens / elapsed if elapsed > 0 else 0.0

out = {{
    # total_token_throughput is the harness-canonical field _paired_helper.
    # _extract_tps reads. Set it to total_throughput (in+out tokens / wall) to
    # match vllm `bench throughput` semantics across backends.
    "total_token_throughput": total_throughput,
    "tokens_per_second": total_throughput,
    "total_throughput": total_throughput,
    "output_throughput": output_throughput,
    "total_input_tokens": total_input_tokens,
    "total_output_tokens": total_output_tokens,
    "total_latency": elapsed,
    "successful_requests": len(results),
    "num_prompts": num_prompts,
    "input_len": input_len,
    "output_len": output_len,
}}
print("RESULT_JSON:" + json.dumps(out))
'''

    env = dict(os.environ)
    _scrub_hf_offline(env)
    env["NO_PROXY"] = "127.0.0.1,localhost"
    env["no_proxy"] = "127.0.0.1,localhost"

    t0 = time.time()
    proc = subprocess.Popen(
        [python, "-P", "-c", script],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=env, start_new_session=True,
    )
    pgid = os.getpgid(proc.pid)
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=2400)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            stdout, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", "subprocess timed out and could not be drained"
    finally:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    elapsed = time.time() - t0
    rc = -1 if timed_out else proc.returncode

    if rc != 0:
        Path(out_file).write_text(json.dumps({
            "error": True, "rc": rc,
            "stderr_tail": (stderr or "")[-3000:],
            "stdout_tail": (stdout or "")[-3000:],
            "elapsed_sec": elapsed,
        }))
        return 1

    for line in (stdout or "").split("\n"):
        if line.startswith("RESULT_JSON:"):
            try:
                data = json.loads(line[len("RESULT_JSON:"):])
            except json.JSONDecodeError as e:
                Path(out_file).write_text(json.dumps({
                    "error": True, "parse_error": str(e),
                    "stdout_tail": (stdout or "")[-3000:],
                }))
                return 1
            data["elapsed_sec"] = elapsed
            Path(out_file).write_text(json.dumps(data))
            return 0

    Path(out_file).write_text(json.dumps({
        "error": True, "parse_error": "no RESULT_JSON line",
        "stdout_tail": (stdout or "")[-3000:],
        "stderr_tail": (stderr or "")[-3000:],
        "elapsed_sec": elapsed,
    }))
    return 1



def cmd_serve_bench(args) -> int:
    """Run one `vllm bench serve --base-url http://127.0.0.1:$PORT` against
    a server already started by the verifier. Workload spec includes the
    port to hit. Output JSON is the standard vllm bench serve dump."""
    wl = json.loads(Path(args.workload_json).read_text())
    _set_offline()

    if ENGINE == "sglang":
        return _cmd_serve_bench_sglang(wl, args)

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
        "--dataset-name", "random",
        "--random-input-len", str(wl["input_len"]),
        "--random-output-len", str(wl["output_len"]),
        "--num-prompts", str(wl["num_prompts"]),
        "--request-rate", request_rate,
        "--max-concurrency", str(max_concurrency),
        "--save-result",
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
    _scrub_hf_offline(env)
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



def _cmd_serve_bench_sglang(wl: dict, args) -> int:
    import signal
    import subprocess

    out_file = args.out_json
    python = sys.executable
    model = wl["model"]
    port = wl["port"]
    in_len = int(wl["input_len"])
    out_len = int(wl["output_len"])
    n = int(wl["num_prompts"])
    rate = str(wl.get("request_rate", "inf"))

    cmd = [
        python, "-P", "-m", "sglang.bench_serving",
        "--backend", "sglang",
        "--model", model,
        "--base-url", f"http://127.0.0.1:{port}",
        "--dataset-name", "random-ids",
        "--random-input-len", str(in_len),
        "--random-output-len", str(out_len),
        "--random-range-ratio", "1.0",
        "--tokenize-prompt",
        "--num-prompts", str(n),
        "--request-rate", str(rate),
        "--output-file", out_file,
    ]
    seed = wl.get("seed")
    if seed is not None:
        cmd += ["--seed", str(int(seed))]
    mc = wl.get("max_concurrency")
    if isinstance(mc, (int, float)) and not isinstance(mc, bool) and int(mc) > 0:
        cmd += ["--max-concurrency", str(int(mc))]
    if wl.get("save_detailed"):
        cmd += ["--output-details"]

    env = dict(os.environ)
    _scrub_hf_offline(env)
    env["NO_PROXY"] = "127.0.0.1,localhost"
    env["no_proxy"] = "127.0.0.1,localhost"

    t0 = time.time()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=env, start_new_session=True,
    )
    pgid = os.getpgid(proc.pid)
    timed_out = False
    try:
        stdout, stderr = proc.communicate(
            timeout=int(os.environ.get("INFB_BENCH_SUBPROCESS_TIMEOUT_SEC", "1800")))
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            stdout, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", "client timed out and could not be drained"
    finally:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    elapsed = time.time() - t0
    rc = -1 if timed_out else proc.returncode

    if rc != 0:
        Path(out_file).write_text(json.dumps({
            "error": True, "rc": rc,
            "stderr_tail": (stderr or "")[-3000:],
            "stdout_tail": (stdout or "")[-3000:],
            "elapsed_sec": elapsed,
        }))
        return 1

    try:
        data = json.loads(Path(out_file).read_text())
    except Exception as e:
        Path(out_file).write_text(json.dumps({
            "error": True, "parse_error": str(e),
            "stderr_tail": (stderr or "")[-3000:],
            "stdout_tail": (stdout or "")[-3000:],
        }))
        return 1

    remapped_tps = _derive_total_token_throughput(data)
    if isinstance(remapped_tps, (int, float)):
        data["total_token_throughput"] = float(remapped_tps)

    completed = data.get("completed", data.get("successful_requests", -1))
    failed = data.get("failed_requests", data.get("failed", -1))
    if completed == 0:
        data["server_likely_dead"] = True
        data["stdout_tail"] = (stdout or "")[-3000:]
        data["stderr_tail"] = (stderr or "")[-3000:]
    elif failed and failed > 0:
        data["partial_failures"] = {"completed": completed, "failed": failed}
        data["stdout_tail"] = (stdout or "")[-3000:]

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
_SGLANG_CTX_LEN = 8192


def _sglang_http_gen(base_url, payload, timeout=1200):
    import urllib.request, json as _j
    req = urllib.request.Request(base_url + "/generate", data=_j.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return _j.loads(r.read().decode())


def _sglang_pick_free_port() -> int:
    """Choose an HTTP port whose derived gRPC port is also valid and free."""
    import random
    import socket

    for _ in range(64):
        port = random.randint(20000, 32767)
        try:
            with (
                socket.socket(socket.AF_INET, socket.SOCK_STREAM) as http_socket,
                socket.socket(socket.AF_INET, socket.SOCK_STREAM) as grpc_socket,
            ):
                http_socket.bind(("127.0.0.1", port))
                grpc_socket.bind(("127.0.0.1", port + 10000))
                return port
        except OSError:
            continue
    raise RuntimeError("could not reserve safe SGLang HTTP/gRPC ports")


def _flashinfer_overlay_dir():
    """Return the writable flashinfer_cubin overlay dir (built by prepare_target_env.sh), or None."""
    _cands = []
    _ws = os.environ.get("INFB_WORKSPACE")
    if _ws:
        _cands.append(os.path.join(_ws, ".flashinfer_overlay"))
    _cands += ["/app/.flashinfer_overlay", "/logs/artifacts/.flashinfer_overlay"]
    for _c in _cands:
        if os.path.isdir(os.path.join(_c, "flashinfer_cubin")):
            return _c
    return None


def _sglang_launch_and_wait(model: str, tp, log_path: str, ready_timeout: float | None = None):
    """Launch `sglang.launch_server` (the PROVEN sgl_kernel-loading path) + wait /health.
    Returns (proc, pgid, base_url, log_handle). The bare in-process `from sglang import Engine` can't
    dlopen sgl_kernel (libnuma.so.1) in this env; launch_server resolves it, so golden-gen +
    candidate teacher-forcing go over HTTP /generate instead."""
    if ready_timeout is None:
        ready_timeout = float(os.environ.get("INFB_SERVE_READY_TIMEOUT_SEC", "1500"))
    import subprocess, time as _t, urllib.request, glob as _glob
    port = _sglang_pick_free_port()
    env = dict(os.environ); _scrub_hf_offline(env)
    env["NO_PROXY"] = "127.0.0.1,localhost"; env["no_proxy"] = "127.0.0.1,localhost"
    env.setdefault("CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(int(tp))))
    _syslibs, _sp = [], []
    for _ev in ("INFB_VLLM_TARGET_ENV_ARTIFACT", "INFB_SGLANG_TARGET_ENV_ARTIFACT"):
        _b = os.environ.get(_ev)
        if not _b:
            continue
        _d = os.path.join(_b, "syslibs")
        if os.path.isdir(_d) and _d not in _syslibs:
            _syslibs.append(_d)
        for _c in _glob.glob(os.path.join(_b, "venv", "lib", "python*", "site-packages")):
            if os.path.isdir(_c) and (os.path.isdir(os.path.join(_c, "torch")) or
                                       os.path.isdir(os.path.join(_c, "sgl_kernel"))) and _c not in _sp:
                _sp.append(_c)
    if _syslibs:
        _prev = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(_syslibs + ([_prev] if _prev else []))
    _fi_ov = _flashinfer_overlay_dir()
    if _sp or _fi_ov:
        _pp = env.get("PYTHONPATH", "")
        _rest = [p for p in (_pp.split(":") if _pp else []) if p and p != _fi_ov]
        env["PYTHONPATH"] = ":".join(([_fi_ov] if _fi_ov else []) + _sp + _rest)
    print(f"[sglang-corr] syslibs={_syslibs} site-packages={_sp}", flush=True)
    cmd = [sys.executable, "-P", "-m", "sglang.launch_server",
           "--model-path", model, "--host", "127.0.0.1", "--port", str(port),
           "--tp-size", str(int(tp)), "--mem-fraction-static", "0.85",
           "--trust-remote-code", "--context-length", str(_SGLANG_CTX_LEN)]
    _lf = open(log_path, "wb")
    try:
        proc = subprocess.Popen(cmd, env=env, start_new_session=True, stdout=_lf, stderr=_lf)
    except BaseException:
        _lf.close()
        raise
    try:
        pgid = os.getpgid(proc.pid)
    except Exception:
        pgid = proc.pid
    base = f"http://127.0.0.1:{port}"
    deadline = _t.time() + ready_timeout
    try:
        while _t.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"sglang server died (rc={proc.returncode}); see {log_path}")
            try:
                with urllib.request.urlopen(base + "/health", timeout=3) as r:
                    if 200 <= r.status < 300:
                        print(f"[sglang-corr] server ready on {base}", flush=True)
                        return proc, pgid, base, _lf
            except Exception:
                pass
            _t.sleep(3)
        raise RuntimeError(f"sglang server /health not ready in {ready_timeout}s; see {log_path}")
    except BaseException:
        _sglang_stop(proc, pgid, _lf)
        raise


def _sglang_stop(proc, pgid, log_handle, wait_timeout: float = 30.0):
    import signal, subprocess
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=wait_timeout)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=5)
        except (subprocess.TimeoutExpired, ChildProcessError):
            pass
    except ChildProcessError:
        pass
    finally:
        log_handle.close()


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




def _gen_reference_sglang(args) -> int:
    """SGLang golden via launch_server + HTTP /generate (the proven sgl_kernel-loading path).
    Pass A greedy → token_ids+prompt_token_ids; Pass B teacher-forced → reference logprobs.
    Same reference.npz schema as the vLLM cmd_gen_reference."""
    import numpy as np
    prompts = json.loads(Path(args.prompts_json).read_text())
    max_tokens = int(args.max_tokens)
    _log = str(Path(args.out_npz).parent / "sglang_refserver.log")
    proc, pgid, base, log_handle = _sglang_launch_and_wait(args.model, args.tp, _log)
    try:
        recs = _sglang_greedy_http(base, prompts, max_tokens)
        full_seqs, n_prompt, n_gen = [], [], []
        for r in recs:
            p = [int(x) for x in r["prompt_token_ids"]]
            g = [int(x) for x in r["token_ids"]][:max_tokens]
            full_seqs.append(p + g); n_prompt.append(len(p)); n_gen.append(len(g))
        tf = _sglang_teacher_forced_http(base, full_seqs, n_prompt, n_gen)
    finally:
        _sglang_stop(proc, pgid, log_handle)
    N, L = len(recs), max_tokens
    token_ids = np.full((N, L), -1, dtype=np.int32)
    logprobs = np.full((N, L), np.nan, dtype=np.float32)
    ptoks_obj = [None] * N
    for i, r in enumerate(recs):
        g = [int(x) for x in r["token_ids"]][:L]
        token_ids[i, :len(g)] = g
        for k, v in enumerate((tf[i] or [])[:L]):
            logprobs[i, k] = float(v) if v is not None else np.nan
        ptoks_obj[i] = np.asarray([int(x) for x in r["prompt_token_ids"]], dtype=np.int32)
    np.savez(args.out_npz, token_ids=token_ids, logprobs=logprobs,
             prompt_token_ids=np.array(ptoks_obj, dtype=object))
    print(f"[gen_reference sglang-http] wrote {args.out_npz} (N={N}, L={L})", flush=True)
    return 0


def _correctness_sglang(args) -> int:
    """SGLang candidate correctness via launch_server + HTTP: teacher-force the ref's
    (prompt+gen) on the candidate server, compare to ref logprobs via the shared scorer."""
    import numpy as np
    ref = np.load(args.ref_npz, allow_pickle=True)
    ref_token_ids = ref["token_ids"]
    ref_logprobs = ref["logprobs"]
    ref_ptoks = ref["prompt_token_ids"]
    full_seqs, n_prompt, n_gen, target_token_ids = [], [], [], []
    for i in range(len(ref_ptoks)):
        p = [int(x) for x in ref_ptoks[i]]
        glen = int((ref_token_ids[i] != -1).sum())
        g = [int(x) for x in ref_token_ids[i, :glen]]
        full_seqs.append(p + g); n_prompt.append(len(p)); n_gen.append(len(g))
        target_token_ids.append(g)
    t0 = time.time()
    _log = str(Path(args.out_json).parent / "sglang_candserver.log")
    proc, pgid, base, log_handle = _sglang_launch_and_wait(args.model, args.tp, _log)
    try:
        tf = _sglang_teacher_forced_http(base, full_seqs, n_prompt, n_gen)
    finally:
        _sglang_stop(proc, pgid, log_handle)
    cand_logprobs = [[(float(v) if v is not None else float("nan")) for v in (row or [])]
                     for row in tf]
    summary = _aggregate_correctness(
        target_token_ids, ref_logprobs, cand_logprobs,
        float(args.atol), float(args.rtol), float(args.min_pass_rate),
    )
    summary["gen_sec"] = time.time() - t0
    summary["comparison"] = "teacher_forced_logprobs_sglang_http"
    Path(args.out_json).write_text(json.dumps(summary, indent=2))
    return 0


def cmd_correctness(args) -> int:
    if ENGINE == "sglang":
        return _correctness_sglang(args)

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
    if ENGINE == "sglang":
        return _gen_reference_sglang(args)

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
