#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Self-test throughput probe for the agent (NOT the sealed verifier).

Runs candidate (your edits) and pristine (reset to base) sides under
`vllm serve` + `vllm bench serve --request-rate inf` (concurrent
serving mode), both with `PYTHONPATH=$repo` overlay so the only
difference between sides is patch content.  One vllm serve starts per
side and all workloads' bench runs hit the same server (cheaper than
offline mode's per-run model reload).

Defaults to running ALL workloads listed in `public/workloads.json`
(same shapes the sealed verifier uses) and reporting per-workload
speedup + geometric-mean speedup.  Override with --workload to run
a subset, or --input-len/--output-len/--num-prompts for a single
custom workload.

Reports trimmed mean, stddev, 95% CI half-width, and a CI-overlap
flag.  If `completed=0` on any measurement run, the script flags it
loudly — your patch crashed EngineCore under concurrent load (CUDA
SIGABRT, no Python traceback).  `run_correctness.py` sends prompts
sequentially and won't catch that; only this serve probe will.

Cost on 1×H100, all 3 workloads: ~30–60 min (one cold start per side,
HTTP-client runs across workloads are fast).

Usage:
  python3 scripts/run_benchmark.py --output /logs/agent/bench.json
  python3 scripts/run_benchmark.py --workload decode_heavy
  python3 scripts/run_benchmark.py --workload decode_heavy --workload high_load
  python3 scripts/run_benchmark.py --input-len 64 --output-len 128 --num-prompts 200
"""
from __future__ import annotations

import os as _infb_os
for _infb_np in ("no_proxy", "NO_PROXY"):
    _infb_cur = _infb_os.environ.get(_infb_np, "")
    if "127.0.0.1" not in _infb_cur:
        _infb_os.environ[_infb_np] = "127.0.0.1,localhost,::1" + (
            f",{_infb_cur}" if _infb_cur else ""
        )
import argparse, contextlib, json, math, os, shutil, signal, socket, statistics, subprocess, sys, time
from pathlib import Path


"""Mirrors writes to two streams simultaneously (stdout + progress log)."""

class _TeeWriter:
    def __init__(self, primary, secondary):
        self._p = primary
        self._s = secondary

    def write(self, data: str) -> int:
        n = self._p.write(data)
        try:
            self._s.write(data)
            self._s.flush()
        except Exception:
            pass
        return n

    def flush(self) -> None:
        self._p.flush()
        try:
            self._s.flush()
        except Exception:
            pass

    def fileno(self):
        return self._p.fileno()



def _resolve_workspace() -> Path:
    return Path(os.environ.get("INFB_WORKSPACE", "/app"))


def _resolve_venv_python() -> str:
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
    raise RuntimeError("Cannot resolve model")


def _load_default_workloads() -> list[dict]:
    """Load all workloads from public/workloads.json (same shapes the sealed
    verifier evaluates).  Each entry: name, input_len, output_len, num_prompts.
    """
    path = _resolve_workspace() / "public" / "workloads.json"
    if not path.exists():
        raise RuntimeError(f"workloads file missing at {path}")
    return json.loads(path.read_text())["workloads"]



def _workload_flag(speedup, *, paired, ci95_overlap=None):
    """Verdict for a workload that HAS a measured ratio on both sides.

    The no-measurement case is handled by the caller, which prints how many runs each side
    completed instead of a verdict. That split is deliberate. Three earlier versions of this
    function tried to fold the two together, and each one ended up asserting whose fault a
    zero was — which the record does not establish:

    * `_run_side` appends a measurement only under `if tps > 0`, so a bench that reported
      completed=0 is discarded exactly like a server that never booted.
    * the paired bench runs the candidate leg first and skips the pristine leg when the
      candidate's server never comes up, so in every real all-zero bench BOTH
      `candidate_runs` and `pristine_runs` are 0. A pristine mean of 0.0 therefore carries
      no information about the baseline at all.

    What the caller's execution line does give the agent, which no verdict did: how many runs
    each side actually completed.
    """
    if paired and ci95_overlap:
        return "  ~ 95% CIs overlap (likely noise)"
    if speedup is not None and speedup < 1.0:
        return "  \u26a0 regression"
    if speedup is not None and speedup >= 1.05:
        return "  \u2713 promising"
    return ""
def _stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": 0.0, "trimmed_mean": 0.0, "stddev": 0.0,
                "ci95_halfwidth": 0.0, "min": 0.0, "max": 0.0}
    # n>=3, not n>=5. At n=3 "drop min and max" is the median -- the cheapest estimator one
    # outlier cannot move. This copy matters as much as the scored one: `trimmed_mean` here is
    # what the AGENT reads as its own speedup, so leaving it at 5 would show the agent a
    # different number from the one it is scored on. Keep it byte-identical to
    # _paired_helper.trimmed_mean.
    if len(values) >= 3:
        trimmed = sorted(values)[1:-1]
    else:
        trimmed = list(values)
    mean = statistics.mean(values)
    trimmed_mean = statistics.mean(trimmed)
    stddev = statistics.stdev(values) if len(values) > 1 else 0.0
    ci95 = 1.96 * stddev / (len(values) ** 0.5) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "mean": mean,
        "trimmed_mean": trimmed_mean,
        "stddev": stddev,
        "ci95_halfwidth": ci95,
        "min": min(values),
        "max": max(values),
    }


def _geomean(values: list[float]) -> float:
    if not values:
        return 0.0
    return math.exp(sum(math.log(v) for v in values) / len(values))


_VERIFIER_NUM_RUNS = {"prefill_heavy": 5, "decode_heavy": 1, "high_load": 1}
_VERIFIER_BOOTS    = {"prefill_heavy": 1, "decode_heavy": 1, "high_load": 1}
_DEFAULT_NUM_RUNS  = 5


def _wl_num_runs(name: str) -> int:
    """Per-workload measurement reps: env INFB_WORKLOAD_NUM_RUNS_<WL> wins, else
    the verifier-aligned map above, else _DEFAULT_NUM_RUNS."""
    return int(os.environ.get(f"INFB_WORKLOAD_NUM_RUNS_{name.upper()}",
                              _VERIFIER_NUM_RUNS.get(name, _DEFAULT_NUM_RUNS)))


def _wl_boots(name: str) -> int:
    """Per-workload K-reboot count: env INFB_WORKLOAD_BOOTS_<WL> wins, else the
    verifier-aligned map above, else 1."""
    return max(1, int(os.environ.get(f"INFB_WORKLOAD_BOOTS_{name.upper()}",
                                     _VERIFIER_BOOTS.get(name, 1))))



def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _dump_crash_diag(tag: str) -> None:
    """On a serve death, capture the definitive killer evidence into the SYNCED
    artifact dir so we can settle WHY the EngineCore gets externally SIGKILL'd.
    The decisive signal is cgroup v2
    memory.events `oom_kill` counter: if it increments at a death, the container
    cgroup OOM-killer (host-RAM exhaustion) is the killer → real fix = raise the
    container memory limit / reduce host-RAM footprint (baseline-neutral). dmesg is best-effort
    (often namespaced-empty in a container). Gated by INFB_PAIRED_KEEP_SERVER_LOGS.
    """
    if os.environ.get("INFB_PAIRED_KEEP_SERVER_LOGS") != "1":
        return
    try:
        outdir = Path(os.environ.get("INFB_ARTIFACTS_DIR", "/logs/artifacts")) / "paired_server_logs"
        outdir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%H%M%S")
        lines = [f"=== crash_diag tag={tag} t={ts} ==="]
        for f in ("memory.events", "memory.current", "memory.max", "memory.peak",
                  "memory.swap.current", "memory.swap.events"):
            p = Path("/sys/fs/cgroup") / f
            try:
                lines.append(f"--- /sys/fs/cgroup/{f} ---\n{p.read_text().strip()}")
            except Exception as e:
                lines.append(f"--- {f}: {e}")
        for pf in ("/sys/fs/cgroup/memory.pressure", "/proc/pressure/memory"):
            try:
                lines.append(f"--- {pf} ---\n{Path(pf).read_text().strip()}")
            except Exception as e:
                lines.append(f"--- {pf}: {e}")
        try:
            ps = subprocess.run(["ps", "-eo", "pid,ppid,rss,vsz,comm,args", "--sort=-rss"],
                                capture_output=True, text=True, timeout=10)
            top = "\n".join((ps.stdout or "").splitlines()[:20])
            lines.append(f"--- ps top-20 by RSS (kB) ---\n{top}")
        except Exception as e:
            lines.append(f"--- ps: {e}")
        for src in (["dmesg"], None):
            try:
                if src is None:
                    txt = "".join(open("/dev/kmsg", errors="replace").readlines()[-40:]) if os.path.exists("/dev/kmsg") else ""
                    label = "/dev/kmsg tail"
                else:
                    r = subprocess.run(src, capture_output=True, text=True, timeout=10)
                    txt = "\n".join((r.stdout or "").splitlines()[-40:]); label = f"dmesg (rc={r.returncode})"
                oom = [l for l in txt.splitlines() if any(k in l.lower() for k in ("oom","kill","memory","sigkill"))]
                lines.append(f"--- {label} ---\n{(chr(10).join(oom) if oom else txt) or '(empty / permission-denied — expected in container)'}")
            except Exception as e:
                lines.append(f"--- kmsg/dmesg: {e}")
        (outdir / f"crash_diag_{tag}_{ts}.txt").write_text("\n".join(lines) + "\n")
        print(f"  [crash-diag] wrote crash_diag_{tag}_{ts}.txt "
              f"(check cgroup memory.events oom_kill)", flush=True)
    except Exception as e:
        print(f"  [crash-diag] WARN: {e}", flush=True)


def _wait_for_server_ready(port: int, timeout: float, server_proc) -> bool:
    import urllib.request, urllib.error
    url = f"http://127.0.0.1:{port}/v1/models"
    http_to = float(os.environ.get("INFB_SERVE_READY_HTTP_TIMEOUT_SEC", "10"))
    deadline = time.time() + timeout
    while time.time() < deadline:
        if server_proc.poll() is not None:
            print(f"  [serve] server died (rc={server_proc.returncode})", flush=True)
            _dump_crash_diag(f"startup_died_rc{server_proc.returncode}")
            return False
        try:
            with urllib.request.urlopen(url, timeout=http_to) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            pass
        time.sleep(2)
    return False




def _start_vllm_serve(*, model: str, venv_python: str, repo: Path, port: int,
                      log_path: Path, cuda_device: str):
    venv_bin = str(Path(venv_python).parent)
    _max_num_seqs = os.environ.get("INFB_DIAG_MAX_NUM_SEQS", "128")
    _gpu_mem_util = os.environ.get("INFB_DIAG_GPU_MEM_UTIL", "0.85")
    _enforce_eager = os.environ.get("INFB_DIAG_ENFORCE_EAGER", "0") == "1"
    cmd = [
        venv_python, "-P", "-m", "vllm.entrypoints.cli.main", "serve", model,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--tensor-parallel-size", "1",
        "--dtype", "bfloat16",
        "--trust-remote-code",
        "--max-model-len", "16384",
        "--gpu-memory-utilization", _gpu_mem_util,
        "--no-enable-prefix-caching",
        "--max-num-seqs", _max_num_seqs,
        "--limit-mm-per-prompt", '{"image": 0, "video": 0}',
    
    ]
    if _enforce_eager:
        cmd.append("--enforce-eager")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    pp_parts = [str(repo)]
    overlay_dir = _resolve_workspace() / ".flashinfer_overlay"
    if overlay_dir.exists():
        pp_parts.insert(0, str(overlay_dir))
    env["PYTHONPATH"] = ":".join(pp_parts)
    env["PATH"] = f"{venv_bin}:" + env.get(
        "PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    )
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env["CUDA_VISIBLE_DEVICES"] = cuda_device

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = open(log_path, "w")
    print(f"  [serve] starting vllm serve on :{port}, log → {log_path}", flush=True)
    proc = subprocess.Popen(cmd, stdout=log_fp, stderr=subprocess.STDOUT,
                            env=env, start_new_session=True)
    proc._log_fp = log_fp  # type: ignore[attr-defined]
    return proc


def _stop_vllm_serve(proc) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    finally:
        try:
            fp = getattr(proc, "_log_fp", None)
            if fp:
                fp.close()
        except Exception:
            pass


def _cleanup_stale_shm() -> None:
    """Best-effort cleanup of /dev/shm/vllm_* leftovers.
    vLLM V1 may leak POSIX shm segments on hard kill / OOM-kill of the
    EngineCore subprocess. Accumulating segments cause `ENOMEM` on the
    next spawn's mmap. Safe no-op if /dev/shm has nothing to clean."""
    import glob as _glob
    for _p in _glob.glob("/dev/shm/vllm_*"):
        try:
            os.unlink(_p)
        except OSError:
            pass


def _kill_stray_gpu_processes(cuda_device: str = "0", grace_sec: float = 5.0) -> None:
    """vllm V1 EngineCore runs in a new session; SIGKILL on APIServer pgid
    misses it.  nvidia-smi-based cleanup ensures the next vllm serve can
    allocate GPU memory.

    ONE DEVICE PER CALL: this queries a single `--id`, so a caller on a TP>1
    task must loop — a straggler holding memory on device 1..N-1 OOMs the next
    boot exactly like one on device 0. Handing the
    whole comma-joined list to `cuda_device` is not a substitute — see the --id
    note in the sibling _gpu_cleanup.kill_stray_gpu_processes; and unlike that
    one, this helper has no /proc fallback, so a query that returns no rows is
    the end of the sweep.
    """
    try:
        r = subprocess.run(
            ["nvidia-smi", f"--id={cuda_device}",
             "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode != 0:
            return
        pids = [int(p.strip()) for p in r.stdout.splitlines() if p.strip().isdigit()]
        if not pids:
            return
        print(f"  [gpu-cleanup] killing {len(pids)} stray GPU pid(s): {pids}", flush=True)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        time.sleep(grace_sec)
    except Exception as e:
        print(f"  [gpu-cleanup] WARNING: {e}", flush=True)




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
def _run_one_bench_serve(*, label: str, model: str, venv_python: str, port: int,
                         workload: dict, workdir: Path,
                         seed: int, num_prompts_override: int | None = None,
                         profile: dict | None = None) -> tuple[float, int]:
    """Returns (tps, completed_requests).

    num_prompts_override: if set, run this many prompts instead of
    workload["num_prompts"]. Used to make WARMUP cheap (warmup only needs to
    populate cudagraph/Triton/cache state, not a statistically-meaningful sample)
    — a full-num warmup of a long-decode workload doubles wall-clock for no
    measurement value. Measurement passes always use the full num.
    """
    out_file = workdir / f"bench_{label}.json"
    out_file.unlink(missing_ok=True)
    if num_prompts_override is not None:
        _num_prompts = num_prompts_override
    elif profile is not None and (profile.get("num_requests") or profile.get("num_prompts")):
        _num_prompts = profile.get("num_requests", profile.get("num_prompts"))
    else:
        _num_prompts = workload["num_prompts"]
    cmd = [
        venv_python, "-P", "-m", "vllm.entrypoints.cli.main", "bench", "serve",
        "--backend", "vllm",
        "--model", model,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--endpoint", "/v1/completions",
        *_dataset_args(workload, _num_prompts, workdir),
        "--seed", str(seed),
        "--max-concurrency", str(profile["concurrency"] if profile else workload.get("max_concurrency", 128 if workload.get("profiles") else 1)),
        "--request-rate", (("inf" if profile.get("pattern", "burst") in ("burst", "closed_loop") else str(profile["rate_per_s"])) if profile else "inf"),
        "--burstiness", ("inf" if (profile and profile.get("pattern") == "constant") else "1.0"),
        "--save-result",
        *(["--save-detailed"] if workload.get("save_detailed") else []),
        "--result-filename", str(out_file),
        "--metric-percentiles", "50,90,95,99",
        "--trust-remote-code",
    
        "--temperature", "1.0",
        "--top-p", "1.0",
        "--top-k", "0",
    ]
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")

    print(f"  [bench-serve] {label}", flush=True)
    t0 = time.time()
    try:
        rc = subprocess.run(cmd, env=env, text=True,
                            start_new_session=True, timeout=int(os.environ.get("INFB_BENCH_SUBPROCESS_TIMEOUT_SEC", "1800")))
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT", flush=True)
        return 0.0, 0
    elapsed = time.time() - t0
    print(f"    rc={rc.returncode}  elapsed={elapsed:.0f}s", flush=True)
    if rc.returncode != 0:
        _dump_crash_diag(f"bench_rc{rc.returncode}")

    if not out_file.exists():
        return 0.0, 0
    try:
        data = json.loads(out_file.read_text())
    except Exception:
        return 0.0, 0

    completed = data.get("completed", data.get("successful_requests", 0))
    if completed == 0:
        print(f"    *** completed=0 — server crashed mid-bench (likely "
              f"EngineCore SIGABRT under load) ***", flush=True)
        _dump_crash_diag("midbench_completed0")
        return 0.0, 0
    for k in ("total_token_throughput", "tokens_per_second", "output_throughput"):
        v = data.get(k)
        if isinstance(v, (int, float)) and v > 0:
            return float(v), int(completed)
    return 0.0, int(completed)


def _run_workloads_against_server(*, side: str, workloads: list[dict], model: str,
                                  venv_python: str, port: int, workdir: Path,
                                  num_warmup: int,
                                  server_proc, seed: int) -> dict[str, list[float]]:
    per_wl_tps: dict[str, list[float]] = {w["name"]: [] for w in workloads}
    for wl in workloads:
        _wl_nr = _wl_num_runs(wl["name"])
        _clock_sensitive = bool(wl.get("profiles"))
        if _clock_sensitive:
            _warmup_np = wl["num_prompts"]
            _wl_nw = max(num_warmup, 1)
        else:
            _wu = os.environ.get(f"INFB_WORKLOAD_WARMUP_NUM_PROMPTS_{wl['name'].upper()}",
                                 os.environ.get("INFB_WORKLOAD_WARMUP_NUM_PROMPTS", "1"))
            _warmup_np = (wl["num_prompts"] if str(_wu).lower() in ("full", "-1")
                          else min(int(_wu), wl["num_prompts"]))
            _wl_nw = num_warmup
        if wl.get("profiles"):
            from _paired_helper import measure_workload_multi_profile as _mwmp
            def _rb(_prof, _label, _npo):
                _t, _c = _run_one_bench_serve(
                    label=_label, model=model, venv_python=venv_python, port=port,
                    workload=wl, workdir=workdir, seed=seed, profile=_prof,
                    num_prompts_override=_npo,
                )
                if _t <= 0:
                    return None
                try:
                    return json.loads((workdir / f"bench_{_label}.json").read_text())
                except Exception:
                    return None
            per_wl_tps[wl["name"]].extend(_mwmp(
                wl, _rb, num_runs=_wl_nr, warmup_runs=_wl_nw,
                side_label=side, warmup_num_prompts=_warmup_np))
        else:
            from _paired_helper import compute_primary_score as _cps
            for i in range(_wl_nw):
                _run_one_bench_serve(
                    label=f"{side}_{wl['name']}_warmup{i}", model=model,
                    venv_python=venv_python, port=port, workload=wl, workdir=workdir,
                    seed=seed, num_prompts_override=_warmup_np,
                )
            for i in range(_wl_nr):
                _lbl = f"{side}_{wl['name']}_meas{i}"
                tps, completed = _run_one_bench_serve(
                    label=_lbl, model=model,
                    venv_python=venv_python, port=port, workload=wl, workdir=workdir,
                    seed=seed,
                )
                if tps > 0:
                    try:
                        _sc = _cps(json.loads((workdir / f"bench_{_lbl}.json").read_text()), wl["name"])
                    except Exception:
                        _sc = None
                    if _sc and _sc > 0:
                        per_wl_tps[wl["name"]].append(_sc)
    return per_wl_tps


def _run_side(*, side: str, workloads: list[dict], model: str, venv_python: str,
              repo: Path, workdir: Path, num_warmup: int,
              cuda_device: str, boot_timeout: int, seed: int) -> dict[str, list[float]]:
    per_wl_tps: dict[str, list[float]] = {w["name"]: [] for w in workloads}
    max_boots = max((_wl_boots(w["name"]) for w in workloads), default=1)
    for _boot in range(max_boots):
        wls_this_boot = [w for w in workloads if _wl_boots(w["name"]) > _boot]
        if not wls_this_boot:
            break
        _bs = "" if max_boots == 1 else f"_boot{_boot}"
        port = _pick_free_port()
        log_path = workdir / f"vllm_serve_{side}{_bs}_{port}.log"
        if os.environ.get("INFB_PAIRED_KEEP_SERVER_LOGS") == "1":
            _kdir = Path(os.environ.get("INFB_ARTIFACTS_DIR", "/logs/artifacts")) / "paired_server_logs"
            _kdir.mkdir(parents=True, exist_ok=True)
            log_path = _kdir / f"vllm_serve_{side}{_bs}_{port}.log"
        server = _start_vllm_serve(model=model, venv_python=venv_python, repo=repo,
                                   port=port, log_path=log_path, cuda_device=cuda_device)
        try:
            print(f"  [serve] waiting for {side}{_bs} server (timeout={boot_timeout}s)...",
                  flush=True)
            if not _wait_for_server_ready(port, boot_timeout, server):
                print(f"  [serve] {side}{_bs} server FAILED to start — see {log_path}",
                      flush=True)
                continue
            print(f"  [serve] {side}{_bs} ready on :{port}", flush=True)
            _got = _run_workloads_against_server(
                side=side, workloads=wls_this_boot, model=model, venv_python=venv_python,
                port=port, workdir=workdir, num_warmup=num_warmup,
                server_proc=server, seed=seed,
            )
            for _name, _tps in _got.items():
                per_wl_tps[_name].extend(_tps)
        finally:
            _stop_vllm_serve(server)
            print(f"  [serve] {side}{_bs} server stopped", flush=True)
            _kill_stray_gpu_processes(cuda_device=cuda_device)
            _cleanup_stale_shm()
        if max_boots > 1 and _boot < max_boots - 1:
            time.sleep(10)   # match the verifier's inter-boot settle
    return per_wl_tps




def _measured_tree_provenance(repo: Path) -> dict:
    """Content provenance of the tree this bench measured.

    The candidate side benches the LIVE repo tree, which may differ from HEAD, and
    the agent may commit before or after a bench. Recording the git TREE hash
    (content, not commit identity) lets the harness's candidate finalizer tell
    whether the `gate=PASS` commit it anchors on is the tree that was actually
    measured. Best-effort: this must never fail a bench.
    """
    def _git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                              text=True, check=True).stdout.strip()

    try:
        head = _git("rev-parse", "HEAD")
        dirty = bool(_git("status", "--porcelain"))
        if dirty:
            index = Path(repo) / ".git" / "infb-bench-provenance.index"
            env = {**os.environ, "GIT_INDEX_FILE": str(index)}
            for args in (["read-tree", "HEAD"], ["add", "-A"]):
                subprocess.run(["git", "-C", str(repo), *args], env=env,
                               capture_output=True, text=True, check=True)
            tree = subprocess.run(["git", "-C", str(repo), "write-tree"], env=env,
                                  capture_output=True, text=True,
                                  check=True).stdout.strip()
            index.unlink(missing_ok=True)
        else:
            tree = _git("rev-parse", "HEAD^{tree}")
        return {"head_sha": head, "tree_sha": tree, "worktree_clean": not dirty}
    except Exception as exc:                                   # pragma: no cover
        return {"error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workload", action="append", default=None,
                   help="Run only the named workload(s) from public/workloads.json. "
                        "Repeatable.  If omitted, runs ALL workloads.")
    p.add_argument("--input-len", type=int, default=None,
                   help="Override: define a single custom workload (combined with "
                        "--output-len and --num-prompts). Ignores public/workloads.json.")
    p.add_argument("--output-len", type=int, default=None)
    p.add_argument("--num-prompts", type=int, default=None)
    p.add_argument("--num-warmup", type=int, default=2)
    p.add_argument("--output", default=None)
    p.add_argument("--workdir", default="/logs/agent/dsv2_bench_probe")
    p.add_argument("--boot-timeout", type=int, default=2400,
                   help="Seconds to wait for vllm serve to become ready.")
    p.add_argument("--cuda-device", default=None,
                   help="CUDA_VISIBLE_DEVICES override (default: inherit env or '0').")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed for the random-prompt dataset. Default: auto-picked per "
                        "invocation (env INFB_BENCH_SEED, else int(time.time())) so each "
                        "run_benchmark.py call uses different prompts and the agent can't "
                        "overfit to seed=0's specific token sequence. Pass a fixed value to "
                        "reproduce a specific run. Within one invocation, all warmup + "
                        "measurement bench calls share the same seed (so per-run variance "
                        "is hardware noise, not prompt jitter).")
    p.add_argument("--paired", dest="paired", action="store_true", default=True,
                   help="(DEFAULT) drift-controlled comparison: candidate side first, "
                        "then `git reset --hard inference_base` and re-measure "
                        "pristine LIVE on the same GPU state, then restore HEAD. "
                        "Refuses to run on a dirty tree (the reset would clobber "
                        "uncommitted edits). ~2× wall time vs --solo but cancels "
                        "cross-session GPU drift.")
    p.add_argument("--solo", dest="paired", action="store_false",
                   help="Fast candidate-only mode: reports RAW candidate TPS per workload only "
                        "— NO speedup (dividing by the stored baseline_metrics.json scalar is "
                        "not verifier-comparable). Accepted so a caller that passes --solo "
                        "still works; run --paired to score.")
    args = p.parse_args()

    if args.seed is None:
        env_seed = os.environ.get("INFB_BENCH_SEED")
        if env_seed is not None:
            try:
                args.seed = int(env_seed)
            except ValueError:
                sys.exit(f"INFB_BENCH_SEED is not an integer: {env_seed!r}")
        else:
            args.seed = int(time.time()) & 0x7FFFFFFF

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    if args.output:
        _progress_log_path = Path(args.output).with_suffix(".progress.log")
    else:
        _progress_log_path = workdir / "bench_progress.log"
    _progress_log_path.parent.mkdir(parents=True, exist_ok=True)
    _progress_fh = _progress_log_path.open("w", buffering=1, encoding="utf-8")
    sys.stdout = _TeeWriter(sys.__stdout__, _progress_fh)
    sys.stderr = _TeeWriter(sys.__stderr__, _progress_fh)
    print(f"[run_benchmark] progress log: {_progress_log_path}", flush=True)

    venv_python = _resolve_venv_python()
    model = _resolve_model()
    workspace = _resolve_workspace()
    repo = workspace / "repo"
    cuda_device = (args.cuda_device or
                   os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])

    if not repo.exists():
        sys.exit(f"workspace/repo missing at {repo}; run prepare_target_env.sh first")

    _kill_stray_gpu_processes(cuda_device=cuda_device)
    _cleanup_stale_shm()

    if args.input_len is not None:
        if args.output_len is None or args.num_prompts is None:
            sys.exit("--input-len requires both --output-len and --num-prompts")
        workloads = [{
            "name":        "custom",
            "input_len":   args.input_len,
            "output_len":  args.output_len,
            "num_prompts": args.num_prompts,
        }]
    else:
        all_workloads = _load_default_workloads()
        if args.workload:
            wanted = set(args.workload)
            workloads = [w for w in all_workloads if w["name"] in wanted]
            missing = wanted - {w["name"] for w in workloads}
            if missing:
                sys.exit(f"unknown workload(s): {missing}; "
                         f"available: {[w['name'] for w in all_workloads]}")
        else:
            workloads = all_workloads

    print(f"[run_benchmark] workloads: {[w['name'] for w in workloads]}", flush=True)
    print(f"[run_benchmark] {args.num_warmup} warmup + verifier-aligned per-workload "
          f"reps={ {w['name']: _wl_num_runs(w['name']) for w in workloads} } "
          f"boots={ {w['name']: _wl_boots(w['name']) for w in workloads} } × 2 sides "
          f"per workload (one server per side; "
          f"env INFB_WORKLOAD_NUM_RUNS/BOOTS_<WL> override)", flush=True)
    print(f"[run_benchmark] random-prompt seed: {args.seed}  "
          f"(set INFB_BENCH_SEED=<int> or pass --seed <int> to reproduce)",
          flush=True)


    saved_head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if args.paired:
        print(f"[run_benchmark] HEAD={saved_head[:8]} (PAIRED via disposable worktree; main /app/repo HEAD NEVER mutated; CANDIDATE benches the LIVE tree and reboots mid-run -- do NOT edit /app/repo during a bench; develop in a separate worktree and apply after)",
              flush=True)
    else:
        print(f"[run_benchmark] HEAD={saved_head[:8]}", flush=True)

    def _do_prewarm_inline(prewarm_repo: Path, label: str):
        _prewarm_port = _pick_free_port()
        _prewarm_log = workdir / f"vllm_serve_{label}_{_prewarm_port}.log"
        if os.environ.get("INFB_PAIRED_KEEP_SERVER_LOGS") == "1":
            _kdir = Path(os.environ.get("INFB_ARTIFACTS_DIR", "/logs/artifacts")) / "paired_server_logs"
            _kdir.mkdir(parents=True, exist_ok=True)
            _prewarm_log = _kdir / f"vllm_serve_{label}_{_prewarm_port}.log"
        print(f"  [{label}] starting throwaway vllm serve on :{_prewarm_port} (repo={prewarm_repo})", flush=True)
        _prewarm_server = _start_vllm_serve(model=model, venv_python=venv_python,
                                             repo=prewarm_repo, port=_prewarm_port,
                                             log_path=_prewarm_log,
                                             cuda_device=cuda_device)
        try:
            _prewarm_t0 = time.time()
            if _wait_for_server_ready(_prewarm_port, args.boot_timeout, _prewarm_server):
                print(f"  [{label}] ready in {time.time()-_prewarm_t0:.1f}s; killing "
                      f"(disk caches now populated)", flush=True)
            else:
                print(f"  [{label}] WARN: server didn\'t reach ready; proceeding anyway", flush=True)
        finally:
            _stop_vllm_serve(_prewarm_server)
            _kill_stray_gpu_processes(cuda_device=cuda_device)
            _cleanup_stale_shm()
            time.sleep(8)

    @contextlib.contextmanager
    def _disposable_pristine_worktree(_repo: Path, _workdir: Path,
                                      _base_ref: str = "inference_base"):
        _wt_dir = _workdir / "_paired_pristine_worktree"
        subprocess.run(["git", "-C", str(_repo), "worktree", "remove", "--force", str(_wt_dir)],
                       capture_output=True, text=True, check=False)
        subprocess.run(["git", "-C", str(_repo), "worktree", "prune"],
                       capture_output=True, text=True, check=False)
        if _wt_dir.exists():
            shutil.rmtree(_wt_dir, ignore_errors=True)
        subprocess.run(
            ["git", "-C", str(_repo), "worktree", "add", "--detach", str(_wt_dir), _base_ref],
            capture_output=True, text=True, check=True,
        )
        try:
            _scripts_dir = Path(__file__).resolve().parent
            if str(_scripts_dir) not in sys.path:
                sys.path.insert(0, str(_scripts_dir))
            try:
                from _paired_helper import rsync_build_artifacts as _rsync_build_artifacts
            except ImportError:
                _rsync_build_artifacts = None
            if _rsync_build_artifacts is not None:
                _so_count = _rsync_build_artifacts(_repo, _wt_dir)
            else:
                _so_count = 0
                _vllm_dir = _repo / "vllm"
                if _vllm_dir.exists():
                    for src in _vllm_dir.rglob("*.so"):
                        rel = src.relative_to(_repo)
                        dst = _wt_dir / rel
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, dst)
                        _so_count += 1
            if _rsync_build_artifacts is None:
                print("[paired_helper] WARNING: _paired_helper is not importable from "
                      f"{_scripts_dir}; falling back to a raw .so copy.", flush=True)
                print("[paired_helper] WARNING: this fallback does NOT restore the venv "
                      "baseline .so, so if you have rebuilt any kernel BOTH legs will run it "
                      "and the paired ratio will read ~1.00 whatever you changed. Treat any "
                      "result from this run as unmeasured.", flush=True)
            if _so_count:
                _src = "rsync_build_artifacts" if _rsync_build_artifacts else ".so-only fallback"
                print(f"[paired_helper] copied {_so_count} build artifact(s) into {_wt_dir} ({_src})",
                      flush=True)
        except Exception as e:
            print(f"[paired_helper] WARN: rsync build artifacts failed: {e}", flush=True)
        try:
            yield _wt_dir
        finally:
            try:
                subprocess.run(["git", "-C", str(_repo), "worktree", "remove", "--force", str(_wt_dir)],
                               capture_output=True, text=True, check=False)
                subprocess.run(["git", "-C", str(_repo), "worktree", "prune"],
                               capture_output=True, text=True, check=False)
            except Exception as e:
                print(f"[paired_helper] WARN: worktree cleanup failed: {e}", flush=True)
            if _wt_dir.exists():
                shutil.rmtree(_wt_dir, ignore_errors=True)

    _exp_prewarm = os.environ.get("INFB_PAIRED_PREWARM", "1") == "1"
    _exp_prewarm_2side = os.environ.get("INFB_PAIRED_PREWARM_2SIDE", "1") == "1"

    pris: dict[str, list[float]] = {}
    if args.paired:
        with _disposable_pristine_worktree(repo, workdir) as pris_repo:
            if _exp_prewarm:
                _side_str = "BOTH cand+pris paths" if _exp_prewarm_2side else "cand path only"
                print(f"[run_benchmark] EXPERIMENT: INFB_PAIRED_PREWARM=1 — throwaway init spawn(s) to "
                      f"populate inductor/Triton/FlashInfer disk caches ({_side_str})",
                      flush=True)
                _do_prewarm_inline(repo, label="prewarm_cand")
                if _exp_prewarm_2side:
                    _do_prewarm_inline(pris_repo, label="prewarm_pris")
                print(f"  [prewarm] done — proceeding to cand+pris measurement spawns",
                      flush=True)

            cand = _run_side(side="candidate", workloads=workloads, model=model,
                             venv_python=venv_python, repo=repo, workdir=workdir,
                             num_warmup=args.num_warmup,
                             cuda_device=cuda_device, boot_timeout=args.boot_timeout,
                             seed=args.seed)
            print(f"\n[run_benchmark] === pristine bench in worktree {pris_repo} ===",
                  flush=True)
            pris = _run_side(side="pristine", workloads=workloads, model=model,
                             venv_python=venv_python, repo=pris_repo, workdir=workdir,
                             num_warmup=args.num_warmup,
                             cuda_device=cuda_device, boot_timeout=args.boot_timeout,
                             seed=args.seed)
    else:
        if _exp_prewarm:
            print(f"[run_benchmark] EXPERIMENT: INFB_PAIRED_PREWARM=1 — cand prewarm "
                  f"(SOLO mode, no pris spawn)", flush=True)
            _do_prewarm_inline(repo, label="prewarm_cand")
            print(f"  [prewarm] done — proceeding to cand-only bench", flush=True)
        cand = _run_side(side="candidate", workloads=workloads, model=model,
                         venv_python=venv_python, repo=repo, workdir=workdir,
                         num_warmup=args.num_warmup,
                         cuda_device=cuda_device, boot_timeout=args.boot_timeout,
                         seed=args.seed)

    per_workload_results = {}
    for wl in workloads:
        cand_tps = cand.get(wl["name"], [])
        cand_stats = _stats(cand_tps)
        cand_mean = cand_stats["trimmed_mean"]
        if args.paired:
            pris_tps = pris.get(wl["name"], [])
            pris_stats = _stats(pris_tps)
            baseline_source = "live_pristine"
            pris_mean = pris_stats["trimmed_mean"]
            speedup = cand_mean / pris_mean if pris_mean > 0 else 0.0
        else:
            pris_tps = []
            pris_stats = _stats([])
            baseline_source = "solo_raw_no_denominator"
            pris_mean = 0.0
            speedup = None
        if args.paired:
            cand_lo = cand_mean - cand_stats["ci95_halfwidth"]
            cand_hi = cand_mean + cand_stats["ci95_halfwidth"]
            pris_lo = pris_mean - pris_stats["ci95_halfwidth"]
            pris_hi = pris_mean + pris_stats["ci95_halfwidth"]
            cis_overlap = not (cand_lo > pris_hi or pris_lo > cand_hi)
        else:
            cis_overlap = None
        per_workload_results[wl["name"]] = {
            "workload":        wl,
            "candidate_stats": cand_stats,
            "candidate_runs":  cand_tps,
            "pristine_stats":  pris_stats,
            "pristine_runs":   pris_tps,
            "speedup":         speedup,
            "ci95_overlap":    cis_overlap,
            "baseline_source": baseline_source,
        }

    speedups = [r["speedup"] for r in per_workload_results.values()
                if r["speedup"] is not None and r["speedup"] > 0]
    geomean = _geomean(speedups) if speedups else None

    summary = {
        "num_runs":            {w["name"]: _wl_num_runs(w["name"]) for w in workloads},
        "boots":               {w["name"]: _wl_boots(w["name"]) for w in workloads},
        "num_warmup":          args.num_warmup,
        "per_workload":        per_workload_results,
        "candidate_provenance": _measured_tree_provenance(repo),
        "speedup_geomean":     geomean,
        "speedup_per_workload": {k: r["speedup"] for k, r in per_workload_results.items()},
    }
    out_path = Path(args.output) if args.output else workdir / "result.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"\n=== run_benchmark summary ===", flush=True)
    if not args.paired:
        print("  (SOLO mode: RAW candidate TPS only — no speedup reported. The stored "
              "baseline_metrics.json denominator is NOT verifier-representative and is "
              "suppressed here. Run --paired for a scored-comparable speedup.)", flush=True)
    for name, r in per_workload_results.items():
        cs = r["candidate_stats"]
        ps = r["pristine_stats"]
        cand_pct = 100.0 * cs["stddev"] / cs["trimmed_mean"] if cs["trimmed_mean"] > 0 else 0.0
        pris_pct = 100.0 * ps["stddev"] / ps["trimmed_mean"] if ps["trimmed_mean"] > 0 else 0.0
        _n_cand = len(r.get("candidate_runs") or [])
        _n_pris = len(r.get("pristine_runs") or [])
        if args.paired:
            _nothing = cs["trimmed_mean"] == 0 or ps["trimmed_mean"] == 0
            _ran = (f"candidate measured {_n_cand} run(s), "
                    f"pristine measured {_n_pris} run(s)")
        else:
            _nothing = cs["trimmed_mean"] == 0
            _ran = f"candidate measured {_n_cand} run(s)"
        if _nothing:
            print(f"  {name}:  *** NO MEASUREMENT — {_ran}; no speedup can be computed "
                  f"for this workload. The [serve] lines above name the side that "
                  f"stopped. ***", flush=True)
            continue
        flag = _workload_flag(r["speedup"], paired=args.paired,
                              ci95_overlap=r.get("ci95_overlap"))
        if args.paired:
            print(f"  {name}:  {r['speedup']:.4f}×  "
                  f"(cand σ={cand_pct:.2f}%, pris σ={pris_pct:.2f}%){flag}", flush=True)
        else:
            print(f"  {name}:  raw {cs['trimmed_mean']:.2f} tps  "
                  f"(cand σ={cand_pct:.2f}%; SOLO=no scored speedup){flag}",
                  flush=True)
    if speedups:
        _dropped = len(per_workload_results) - len(speedups)
        _partial = "" if not _dropped else (
            f"\n  [!! PARTIAL: {_dropped} of {len(per_workload_results)} workloads were "
            f"EXCLUDED from this geomean (failed / non-positive speedup). It is therefore "
            f"NOT comparable to the scored geomean, which requires every workload.]")
        print(f"\n  speedup geomean across {len(speedups)} workloads: {geomean:.4f}×{_partial}", flush=True)
    elif args.paired:
        print(f"\n  (PAIRED: no scored speedup — all {len(per_workload_results)} workloads had non-positive/failed pristine TPS; check for serve/bench crashes)", flush=True)
    else:
        print(f"\n  (SOLO: no scored speedup — {len(per_workload_results)} workloads measured RAW; run --paired to score)", flush=True)
    print(f"  wrote: {out_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
