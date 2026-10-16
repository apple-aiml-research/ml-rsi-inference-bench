#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Scored bench launcher for regression_multimodal_prefill_ttft: one served engine per invocation.

`tests/_bench_rounds.py` invokes this once per (round, arm). One invocation boots ONE
`sglang.launch_server` from the tree under measurement and drives the whole workload matrix of
tests/workloads.json against it with tests/_mm_ttft_client.py -- the anchor first, then the other
scored points, then the guards -- so every point of an arm-round shares one engine session and the
repeat guard replays the anchor's images into a session that has already seen them by construction.

WHAT IS SCORED. Mean time to first token per request, stamped by the streaming client on the first
chunk of a `stream=true` /generate call, LOWER IS BETTER (the verifier divides pristine / candidate
and takes the geometric mean over the scored points). The record is in `sglang.bench_serving`'s
shape so `_verifier_lib/metrics.py` maps `mean_ttft_ms` to `summary.latency_ms.ttft.mean`.

WHY A SERVED ENGINE AND NOT THE OFFLINE `sgl.Engine` THIS TASK USED. The offline bench read TTFT
through a proxy -- generated tok/s at one output token -- and the agent's self-test and the scorer
had drifted onto different sampling flags. One client, posting to one server, measures first-token
latency directly and is the same program on both surfaces (the public copy under environment/public/
is byte-identical). The engine flags below are the offline Engine's kwargs, spelled as server flags.

THE SESSION, in order:
  1. a fresh HOME for the server (the on-disk state that outlives a process lives under it; wiped
     before the boot so the first arm cannot pay the other's setup), the regime probe installed on
     the server's PYTHONPATH, the tree assertion (`import sglang` resolves inside --repo);
  2. boot, wait for /health, dump the server log if the process dies first;
  3. the ANCHOR point at the raw root: its record carries the regime block (counted inside the
     scheduler process by the probe), the cubin counts at window-open and after the point (the
     pre-staging gate reads the first) and the output-agreement pass
     (tests/_mm_logprob_pass.py, sixteen of the anchor's own requests, greedy, prompt logprobs);
  4. every other point under wl/<name>/, each with its own record and manifest; a failed point
     keeps no record, so the rounds driver reads it as unpairable rather than as a number;
  5. the server's process group is killed, on every path.

THE SEED. One value per verifier process, shared by both arms and every point: INFB_BENCH_SEED,
exported by tests/test.sh (time-based unless pinned). Falls back to a time-based value with a warning
when run by hand without it, in which case two invocations draw different images.

Writes sglang_bench_serving.json + sglang_bench_manifest.json into --raw-dir, the contract
verify_regression_real.py checks. Never fails the shell -- the verifier reads the manifest returncode.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _verifier_lib.bench_stats import load_matrix_spec, point_raw_dir  # noqa: E402

CONFIG = {
    "port": 30117,
    "server_ready_timeout_sec": 900,
    "invocation_budget_sec": 1380,
    "anchor_budget_sec": 600,
    "request_timeout_sec": 300,
    "logprob_pass_budget_sec": 180,
    "log_level": "warning",
}

RECORD_NAME = "sglang_bench_serving.json"
MANIFEST_NAME = "sglang_bench_manifest.json"
_MIN_SCRATCH_ENTRIES = 24


def _say(msg: str) -> None:
    print(f"[_candidate_runner] {msg}", flush=True)


def _workspace() -> Path:
    w = os.environ.get("INFB_WORKSPACE")
    return Path(w) if w and Path(w).is_dir() else Path("/app")


def _venv_python() -> str:
    v = os.environ.get("INFB_VLLM_PYTHON")
    if v and Path(v).is_file():
        return v
    venv = os.environ.get("INFB_VENV")
    if venv and (Path(venv) / "bin" / "python").is_file():
        return str(Path(venv) / "bin" / "python")
    if Path("/opt/venv/bin/python").is_file():
        return "/opt/venv/bin/python"
    return sys.executable


def _model(ws: Path) -> str:
    v = os.environ.get("INFB_MODEL_PATH")
    if v and Path(v).exists():
        return v
    link = ws / "model"
    if link.exists():
        return str(link)
    raise SystemExit("model not resolved (set INFB_MODEL_PATH or <workspace>/model)")


def _bench_seed() -> tuple[int, str]:
    """The per-eval seed every arm and point draws from, and where it came from."""
    raw = os.environ.get("INFB_BENCH_SEED")
    if raw:
        try:
            return int(raw), "INFB_BENCH_SEED"
        except ValueError:
            _say(f"INFB_BENCH_SEED={raw!r} is not an integer; drawing a time-based seed instead")
    seed = int(time.time()) % 2_000_000_000
    _say(f"WARNING no INFB_BENCH_SEED in the environment: seed {seed} is time-based, so another "
         f"invocation draws different images and its numbers are not paired with these")
    return seed, "time"


def _select_extra_points(selection: str, spec: dict) -> list[str]:
    """The non-anchor points this invocation benches, scored ones first, then guards."""
    if selection in ("", "anchor"):
        return []
    candidates = [n for n in spec["scored"] if n != spec["anchor"]] + list(spec["guards"])
    if selection != "all":
        wanted = [s.strip() for s in selection.split(",") if s.strip()]
        unknown = sorted(set(wanted) - set(spec["points"]))
        if unknown:
            raise SystemExit(f"--workloads names points the spec does not declare: {unknown}")
        candidates = [n for n in candidates if n in wanted]
    return [n for n in candidates if not isinstance(spec["points"][n].get("from"), str)]


def _serve_argv(python: str, model: str, port: int) -> list[str]:
    """The offline Engine's kwargs, as server flags. The public self-test boots the identical list."""
    return [
        python, "-m", "sglang.launch_server",
        "--model-path", model,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--tp-size", "1",
        "--mem-fraction-static", "0.85",
        "--context-length", "8192",
        "--chunked-prefill-size", "8192",
        "--cuda-graph-max-bs-decode", "4",
        "--disable-radix-cache",
        "--random-seed", "1234",
        "--trust-remote-code",
        "--log-level", "warning",
    ]


def _client_argv(python: str, client_py: Path, base_url: str, tokenizer: str, wl: dict, seed: int,
                 out_json: Path, budget_sec: float) -> list[str]:
    """One point's client invocation. The public self-test builds the same list."""
    return [
        python, str(client_py),
        "--base-url", base_url,
        "--tokenizer", tokenizer,
        "--input-len", str(int(wl["input_len"])),
        "--output-len", str(int(wl["output_len"])),
        "--num-prompts", str(int(wl["num_prompts"])),
        "--image-count", str(int(wl.get("image_count", 0))),
        "--image-format", str(wl.get("image_format", "jpeg")),
        "--image-min-hw", f"{int(wl.get('image_min_height', 512))}x{int(wl.get('image_min_width', 512))}",
        "--image-max-hw", f"{int(wl.get('image_max_height', 768))}x{int(wl.get('image_max_width', 1024))}",
        "--seed", str(int(seed)),
        "--request-timeout-sec", str(int(CONFIG["request_timeout_sec"])),
        "--time-budget-sec", str(int(budget_sec)),
        "--out-json", str(out_json),
    ]


def _fresh_home() -> Path:
    """A per-boot HOME for the server, wiped and proven empty; `~/.cache` stays shared."""
    root = Path(tempfile.mkdtemp(prefix="infb_engine_home_"))
    home = root / "a"
    home.mkdir()
    if list(home.rglob("*")):
        raise SystemExit(f"scratch dir {home} is not empty after creation")
    shared = Path(os.environ.get("HOME", "/root")) / ".cache"
    if shared.is_dir():
        (home / ".cache").symlink_to(shared, target_is_directory=True)
    return home


def _scratch_counts(home: Path) -> tuple[int, int]:
    """(entries, cubins) under the server's HOME. `rglob` does not follow the .cache symlink."""
    entries = sum(1 for _ in home.rglob("*"))
    cubins = sum(1 for _ in home.rglob("*.cubin"))
    return entries, cubins


def _assert_scratch_used(home: Path) -> None:
    """The wipe is only worth anything if the redirect took: prove the server wrote its state here."""
    entries, _ = _scratch_counts(home)
    if entries < _MIN_SCRATCH_ENTRIES:
        raise SystemExit(
            f"scratch root {home} holds {entries} entries after the anchor, want at least "
            f"{_MIN_SCRATCH_ENTRIES}. The server wrote its state somewhere else, so it inherited state "
            "from an earlier boot and its timing is not comparable to another arm's."
        )


class _CacheAtWindowOpen:
    """One-shot `_scratch_counts(home)` when the anchor client drops its `.t0` sentinel; the
    verifier's pre-staging gate reads it, so a warm-up through any function is counted."""

    def __init__(self, home: Path, sentinel: Path, poll_sec: float = 0.02) -> None:
        self._home, self._sentinel, self._poll = home, sentinel, poll_sec
        self._stop, self._done = threading.Event(), threading.Event()
        self.files: int | None = None
        self.cubins: int | None = None
        self._thread = threading.Thread(target=self._run, name="cache-at-window-open", daemon=True)

    def start(self) -> "_CacheAtWindowOpen":
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._sentinel.exists():
                for _ in range(10):
                    try:
                        self.files, self.cubins = _scratch_counts(self._home)
                        break
                    except OSError:
                        time.sleep(0.01)
                self._done.set()
                return
            self._stop.wait(self._poll)

    def wait(self, timeout: float) -> bool:
        return self._done.wait(timeout)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)


def _install_probe(probe_json: Path, sentinel: Path, env: dict) -> Path:
    """A sitecustomize that counts l2norm_fwd calls in every process the server spawns."""
    probe_dir = probe_json.parent / (probe_json.stem + ".probe")
    probe_dir.mkdir(parents=True, exist_ok=True)
    (probe_dir / "sitecustomize.py").write_text(
        "import atexit, importlib.abc, importlib.util, json, os, sys\n"
        f"_OUT = {str(probe_json)!r}\n"
        f"_SENTINEL = {str(sentinel)!r}\n"
        "_TARGET = 'sglang.kernels.ops.attention.fla.l2norm'\n"
        "_S = {'calls': 0, 'distinct_T': set(), 'first_sight_in_window': 0,\n"
        "      'callers': set(), 'scored': False}\n"
        "\n"
        "def _dump():\n"
        "    try:\n"
        "        d = {'calls': _S['calls'], 'distinct_T': len(_S['distinct_T']),\n"
        "             'first_sight_in_window': _S['first_sight_in_window'],\n"
        "             'min_T': min(_S['distinct_T']) if _S['distinct_T'] else None,\n"
        "             'max_T': max(_S['distinct_T']) if _S['distinct_T'] else None,\n"
        "             'callers': sorted(_S['callers'])[:8], 'pid': os.getpid()}\n"
        "        with open(_OUT + '.' + str(os.getpid()), 'w') as fh:\n"
        "            json.dump(d, fh)\n"
        "    except Exception:\n"
        "        pass\n"
        "atexit.register(_dump)\n"
        "\n"
        "def _patch(mod):\n"
        "    real = getattr(mod, 'l2norm_fwd', None)\n"
        "    if real is None:\n"
        "        return\n"
        "    seen = _S['distinct_T']\n"
        "    def wrapped(x, *a, **kw):\n"
        "        t = x.shape[0] if x.ndim == 2 else x.numel() // x.shape[-1]\n"
        "        _S['calls'] += 1\n"
        "        if t not in seen:\n"
        "            seen.add(t)\n"
        "            if not _S['scored']:\n"
        "                _S['scored'] = os.path.exists(_SENTINEL)\n"
        "            if _S['scored']:\n"
        "                _S['first_sight_in_window'] += 1\n"
        "            if _S['calls'] <= 4:\n"
        "                try:\n"
        "                    f = sys._getframe(1)\n"
        "                    _S['callers'].add(f.f_globals.get('__name__', '?'))\n"
        "                except Exception:\n"
        "                    pass\n"
        "            _dump()\n"
        "        elif _S['calls'] % 500 == 0:\n"
        "            _dump()\n"
        "        return real(x, *a, **kw)\n"
        "    mod.l2norm_fwd = wrapped\n"
        "\n"
        "class _Hook(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname != _TARGET:\n"
        "            return None\n"
        "        sys.meta_path.remove(self)\n"
        "        try:\n"
        "            spec = importlib.util.find_spec(fullname)\n"
        "        finally:\n"
        "            sys.meta_path.insert(0, self)\n"
        "        if spec is None or spec.loader is None:\n"
        "            return None\n"
        "        real_exec = spec.loader.exec_module\n"
        "        def exec_module(module, _r=real_exec):\n"
        "            _r(module)\n"
        "            _patch(module)\n"
        "        spec.loader.exec_module = exec_module\n"
        "        return spec\n"
        "\n"
        "if not any(type(h).__name__ == '_Hook' for h in sys.meta_path):\n"
        "    sys.meta_path.insert(0, _Hook())\n",
        encoding="utf-8",
    )
    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{probe_dir}{os.pathsep}{old}" if old else str(probe_dir)
    return probe_dir


def _collect_probe(probe_json: Path) -> dict:
    """Merge the per-pid counter files; the maxima are what the gates read."""
    merged = {"calls": 0, "distinct_T": 0, "first_sight_in_window": 0,
              "min_T": None, "max_T": None, "callers": [], "pids": 0}
    for p in sorted(probe_json.parent.glob(probe_json.name + ".*")):
        try:
            d = json.loads(p.read_text())
        except Exception:  # noqa: BLE001 -- a half-written counter must not lose the measurement
            continue
        merged["pids"] += 1
        merged["calls"] += int(d.get("calls") or 0)
        merged["distinct_T"] = max(merged["distinct_T"], int(d.get("distinct_T") or 0))
        merged["first_sight_in_window"] = max(merged["first_sight_in_window"],
                                              int(d.get("first_sight_in_window") or 0))
        for k, fn in (("min_T", min), ("max_T", max)):
            v = d.get(k)
            if v is not None:
                merged[k] = v if merged[k] is None else fn(merged[k], v)
        merged["callers"] = sorted(set(merged["callers"]) | set(d.get("callers") or []))
    return merged


def _pdeathsig_preexec() -> None:
    """Own process group, and SIGKILL if this runner dies: an orphaned server holds the GPU for the
    NEXT arm, which corrupts the very ratio this rig exists to measure."""
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGKILL)
    except Exception:  # noqa: BLE001 -- best effort; explicit teardown still runs
        pass
    os.setpgrp()


def _port_busy(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def _http_ok(url: str, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 -- fixed localhost
            return 200 <= r.status < 300
    except (urllib.error.URLError, OSError):
        return False


def _tail(path: Path, n: int = 40) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return "(no log)"


def _wait_ready(proc: subprocess.Popen, url: str, timeout: float, log: Path) -> float:
    """Block until the server answers, or die with the server's own tail."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise SystemExit(f"server exited rc={proc.returncode} before becoming ready\n"
                             f"--- server log ---\n{_tail(log)}")
        if _http_ok(url):
            return time.time() - t0
        time.sleep(1.0)
    raise SystemExit(f"server not ready after {timeout:.0f}s\n--- server log ---\n{_tail(log)}")


def _stop_server(proc: subprocess.Popen, port: int) -> None:
    """Kill the server's whole process group (scheduler and detokenizer are separate processes)."""
    if proc.poll() is None:
        for sig, grace in ((signal.SIGTERM, 60.0), (signal.SIGKILL, 30.0)):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                break
            try:
                proc.wait(timeout=grace)
                break
            except subprocess.TimeoutExpired:
                continue
    deadline = time.time() + 60
    while _port_busy(port) and time.time() < deadline:
        time.sleep(1.0)


def _wait_for_free_gpu(timeout_s: float = 180.0, floor_mib: int = 4096) -> str:
    """Block until the GPU is actually free, or say why we gave up."""
    if not shutil.which("nvidia-smi"):
        return "nvidia-smi absent; not waiting"
    deadline = time.time() + timeout_s
    last = -1
    while time.time() < deadline:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=30).stdout
            last = max(int(v.strip()) for v in out.split() if v.strip().isdigit())
        except (OSError, ValueError, subprocess.SubprocessError):
            return "nvidia-smi unreadable; not waiting"
        if last <= floor_mib:
            return f"free after {timeout_s - (deadline - time.time()):.0f}s ({last} MiB used)"
        time.sleep(5)
    return f"STILL BUSY after {timeout_s:.0f}s ({last} MiB used) -- benching anyway"


def _assert_benched_tree(python: str, repo: Path, env: dict) -> str:
    """`import sglang` under the server's environment must resolve inside --repo, or nothing here
    measures that tree. Returns the resolved package path for the record."""
    proc = subprocess.run(
        [python, "-c", "import sglang, sys; sys.stdout.write(sglang.__file__)"],
        env=env, capture_output=True, text=True, timeout=600,
    )
    resolved = proc.stdout.strip()
    if proc.returncode != 0 or not resolved:
        raise SystemExit(f"`import sglang` failed under PYTHONPATH={env.get('PYTHONPATH')}:\n"
                         f"{proc.stderr[-2000:]}")
    if not Path(resolved).resolve().is_relative_to(repo.resolve()):
        raise SystemExit(f"`import sglang` resolved to {resolved}, outside the tree under test {repo}")
    return resolved


def _run_point(python: str, client_py: Path, client_env: dict, base_url: str, tokenizer: str,
               name: str, wl: dict, pdir: Path, seed: int, budget_sec: float,
               log_stem: str, serve_argv: list[str]) -> dict:
    """One point against the running server; its record and manifest under `pdir`."""
    pdir.mkdir(parents=True, exist_ok=True)
    out_json = pdir / RECORD_NAME
    for stale in (out_json, out_json.with_suffix(".failed.json"), out_json.with_suffix(".t0")):
        stale.unlink(missing_ok=True)
    argv = _client_argv(python, client_py, base_url, tokenizer, wl, seed, out_json, budget_sec)
    _say(f"point {name}: n={wl['num_prompts']} images={wl.get('image_count', 0)} "
         f"in={wl['input_len']} out={wl['output_len']} budget={budget_sec:.0f}s")
    t0 = time.time()
    timed_out = False
    with (pdir / f"{log_stem}_stdout.txt").open("w") as outf, \
         (pdir / f"{log_stem}_stderr.txt").open("w") as errf:
        try:
            rc = subprocess.run(argv, env=client_env, stdout=outf, stderr=errf,
                                timeout=budget_sec + 2 * CONFIG["request_timeout_sec"] + 60).returncode
        except subprocess.TimeoutExpired:
            rc, timed_out = 124, True
    elapsed = time.time() - t0
    error = None
    record: dict | None = None
    if timed_out:
        error = "client exceeded its wall-clock cap"
    elif rc != 0:
        error = f"client exited rc={rc}"
    elif not out_json.is_file():
        error = "client wrote no record"
    else:
        try:
            record = json.loads(out_json.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            error = f"record unreadable: {exc}"
    if record is not None and error is None:
        if int(record.get("completed") or 0) != int(wl["num_prompts"]) or int(record.get("failed") or 0):
            error = (f"completed {record.get('completed')} of {wl['num_prompts']} requests, "
                     f"failed {record.get('failed')}")
        elif not record.get("mean_ttft_ms"):
            error = f"no mean_ttft_ms in the record; keys {sorted(record)[:12]}"
    if error:
        if out_json.is_file():
            out_json.rename(out_json.with_suffix(".failed.json"))
        record = None
        _say(f"point {name}: FAILED -- {error}")
    else:
        _say(f"point {name}: mean TTFT {record['mean_ttft_ms']:.1f} ms over "
             f"{record['completed']} requests in {record['duration']:.1f}s "
             f"({record.get('distinct_prompt_token_counts')} distinct prompt token counts)")
    (pdir / MANIFEST_NAME).write_text(json.dumps({
        "mode": "serve", "returncode": rc if rc else (1 if error else 0), "error": error,
        "timed_out": timed_out, "point": name, "workload": wl, "seed": seed,
        "command": "sglang launch_server + _mm_ttft_client bench: " + shlex.join(argv),
        "serve_command": shlex.join(serve_argv), "elapsed_sec": round(elapsed, 2),
    }, indent=2) + "\n", encoding="utf-8")
    return {"name": name, "returncode": rc, "error": error, "record": record,
            "elapsed_sec": round(elapsed, 1),
            "mean_ttft_ms": record.get("mean_ttft_ms") if record else None}


def _logprob_pass(python: str, here: Path, client_env: dict, base_url: str, wl: dict, seed: int,
                  out_json: Path) -> dict:
    """The output-agreement pass, after the anchor's window and while the server is still up."""
    argv = [
        python, str(here / "_mm_logprob_pass.py"),
        "--base-url", base_url,
        "--input-len", str(int(wl["input_len"])),
        "--num-prompts", str(int(wl["num_prompts"])),
        "--image-count", str(int(wl.get("image_count", 0))),
        "--image-format", str(wl.get("image_format", "jpeg")),
        "--image-min-hw", f"{int(wl.get('image_min_height', 512))}x{int(wl.get('image_min_width', 512))}",
        "--image-max-hw", f"{int(wl.get('image_max_height', 768))}x{int(wl.get('image_max_width', 1024))}",
        "--seed", str(int(seed)),
        "--n", "16",
        "--request-timeout-sec", str(int(CONFIG["request_timeout_sec"])),
        "--budget-sec", str(int(CONFIG["logprob_pass_budget_sec"])),
        "--out-json", str(out_json),
    ]
    try:
        with out_json.with_suffix(".log").open("w") as log:
            subprocess.run(argv, env=client_env, stdout=log, stderr=subprocess.STDOUT,
                           timeout=CONFIG["logprob_pass_budget_sec"] + CONFIG["request_timeout_sec"])
        return json.loads(out_json.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 -- recorded; the verifier fails closed on None
        return {"logprobs": None, "error": f"{type(exc).__name__}: {exc}"[:500]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["serve"])
    ap.add_argument("--repo", required=True, help="engine tree to overlay via PYTHONPATH")
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument(
        "--workloads", default="anchor",
        help="`anchor` (the primary workload only, the warmup default), `all` (every scored and guard "
             "point of tests/workloads.json against one server), or a comma-separated list",
    )
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    ws = _workspace()
    repo = Path(args.repo).resolve()
    raw = Path(args.raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    out_json = raw / RECORD_NAME
    manifest = raw / MANIFEST_NAME
    server_log = raw / "sglang_server.log"
    probe_json = raw / "regime_probe.json"
    logprob_json = raw / "output_agreement.json"
    for stale in [out_json, out_json.with_suffix(".failed.json"), out_json.with_suffix(".t0"),
                  logprob_json, *probe_json.parent.glob(probe_json.name + ".*")]:
        stale.unlink(missing_ok=True)

    spec = load_matrix_spec(Path(__file__).resolve().parent / "workloads.json")
    wl = spec["points"][spec["anchor"]]
    extra_points = _select_extra_points(args.workloads, spec)
    seed, seed_source = _bench_seed()
    python = _venv_python()
    model = _model(ws)
    port = int(os.environ.get("INFB_SERVE_PORT") or CONFIG["port"])
    base_url = f"http://127.0.0.1:{port}"
    client_py = here / "_mm_ttft_client.py"
    started = time.time()
    deadline = started + CONFIG["invocation_budget_sec"]

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{repo}:{env.get('PYTHONPATH', '')}".rstrip(":")
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "TRITON_CACHE_DIR",
                "TORCHINDUCTOR_CACHE_DIR", "CUDA_CACHE_PATH"):
        env.pop(key, None)
    client_env = dict(os.environ)
    client_env["PYTHONPATH"] = f"{repo}:{client_env.get('PYTHONPATH', '')}".rstrip(":")
    home = _fresh_home()
    env["HOME"] = str(home)
    sentinel = out_json.with_suffix(".t0")
    _install_probe(probe_json, sentinel, env)
    _say(f"seed {seed} ({seed_source}); server HOME {home}; points: {[spec['anchor']] + extra_points}")

    serve_argv = _serve_argv(python, model, port)
    rc = 1
    error: str | None = None
    engine_package = None
    boot = None
    regime: dict = {}
    correctness: dict = {}
    points_benched: list[dict] = []
    anchor_result: dict | None = None
    gpu_wait = ""
    proc: subprocess.Popen | None = None
    try:
        engine_package = _assert_benched_tree(python, repo, env)
        _say(f"engine package resolves to {engine_package}")
        gpu_wait = _wait_for_free_gpu()
        _say(f"gpu: {gpu_wait}")
        if _port_busy(port):
            raise SystemExit(f"port {port} already in use before this arm started; refusing to bench "
                             f"against someone else's server")
        _say(f"serve: {shlex.join(serve_argv)}")
        with server_log.open("w") as slog:
            proc = subprocess.Popen(serve_argv, env=env, stdout=slog, stderr=subprocess.STDOUT,
                                    preexec_fn=_pdeathsig_preexec)
        boot = _wait_ready(proc, f"{base_url}/health",
                           float(os.environ.get("INFB_BOOT_TIMEOUT_SEC") or CONFIG["server_ready_timeout_sec"]),
                           server_log)
        _say(f"server ready in {boot:.0f}s")

        anchor_budget = min(CONFIG["anchor_budget_sec"], max(60.0, deadline - time.time() - 60.0))
        sentinel.unlink(missing_ok=True)
        at_t0 = _CacheAtWindowOpen(home, sentinel).start()
        try:
            anchor_result = _run_point(python, client_py, client_env, base_url, model, spec["anchor"],
                                       wl, raw, seed, anchor_budget, "sglang_bench", serve_argv)
        finally:
            at_t0.stop()
        if anchor_result["error"]:
            raise SystemExit(f"anchor point failed: {anchor_result['error']}\n--- server log ---\n"
                             f"{_tail(server_log)}")
        _assert_scratch_used(home)
        counters = _collect_probe(probe_json)
        cache_files, cubins = _scratch_counts(home)
        rec = anchor_result["record"]
        sizes = rec.get("image_sizes") or []
        regime = {
            "distinct_prompt_token_counts": rec.get("distinct_prompt_token_counts"),
            "min_prompt_tokens": rec.get("min_prompt_tokens"),
            "max_prompt_tokens": rec.get("max_prompt_tokens"),
            "distinct_image_size_tuples": len({tuple(tuple(s) for s in per) for per in sizes}),
            "l2norm_calls": counters["calls"],
            "l2norm_distinct_T": counters["distinct_T"],
            "l2norm_first_sight_in_scored_window": counters["first_sight_in_window"],
            "l2norm_min_T": counters["min_T"],
            "l2norm_max_T": counters["max_T"],
            "l2norm_callers": counters["callers"],
            "probe_processes": counters["pids"],
            "triton_cache_files": cache_files,
            "triton_cubins": cubins,
            "triton_cache_files_at_t0": at_t0.files,
            "triton_cubins_at_t0": at_t0.cubins,
            "triton_cubins_in_window": cubins - at_t0.cubins if at_t0.cubins is not None else None,
        }
        _say(f"REGIME: l2norm calls={counters['calls']} distinct_T={counters['distinct_T']} "
             f"first_sight_in_window={counters['first_sight_in_window']} cubins={cubins} "
             f"cache_files={cache_files}")
        _say(f"CACHE: cubins at window-open={at_t0.cubins} after the anchor={cubins} "
             f"compiled inside the window={regime['triton_cubins_in_window']}")
        if counters["calls"] == 0:
            _say("WARNING the linear-attention l2-norm helper was never called; both arms execute the "
                 "same code on this path and the ratio is meaningless")
        correctness = _logprob_pass(python, here, client_env, base_url, wl, seed, logprob_json)
        _say("output agreement: " + (f"{len(correctness.get('lengths') or [])} requests, "
                                     f"{sum(correctness.get('lengths') or [])} positions"
                                     if not correctness.get("error") else
                                     f"UNAVAILABLE -- {correctness['error']}"))
        rec["regime"] = regime
        rec["correctness_check"] = correctness
        rec["engine_package"] = engine_package
        rec["engine_args"] = serve_argv[3:]
        rec["server_boot_sec"] = round(boot, 2)
        rec["workload"] = wl
        out_json.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for name in extra_points:
            remaining = deadline - time.time() - 30.0
            pdir = point_raw_dir(raw, name, spec["anchor"])
            if remaining < 60.0:
                pdir.mkdir(parents=True, exist_ok=True)
                (pdir / MANIFEST_NAME).write_text(json.dumps({
                    "mode": "serve", "returncode": 1, "point": name,
                    "error": "not benched: the invocation's time budget is exhausted"}, indent=2) + "\n")
                points_benched.append({"name": name, "returncode": 1, "mean_ttft_ms": None,
                                       "error": "not benched: the invocation's time budget is exhausted"})
                _say(f"point {name}: SKIPPED -- out of time")
                continue
            result = _run_point(python, client_py, client_env, base_url, model, name,
                                spec["points"][name], pdir, seed, remaining, "client", serve_argv)
            result.pop("record", None)
            points_benched.append(result)
        rc = 0
    except SystemExit as exc:
        error = str(exc)
        _say(f"FAILED: {error}")
    except Exception as exc:  # noqa: BLE001 -- recorded in the manifest, never a silent zero
        error = f"{type(exc).__name__}: {exc}"
        _say(f"FAILED: {error}")
    finally:
        if proc is not None:
            _stop_server(proc, port)
        session_files, session_cubins = _scratch_counts(home)
        session_counters = _collect_probe(probe_json)
        shutil.rmtree(home.parent, ignore_errors=True)
        if proc is not None:
            _say(f"gpu after stop: {_wait_for_free_gpu(timeout_s=120.0)}")
    elapsed = time.time() - started

    manifest.write_text(json.dumps({
        "mode": "serve",
        "returncode": rc,
        "error": error,
        "timed_out": bool(anchor_result and anchor_result.get("returncode") == 124),
        "command": "sglang launch_server + _mm_ttft_client bench: " + shlex.join(serve_argv),
        "serve_command": shlex.join(serve_argv),
        "repo": str(repo),
        "engine_package": engine_package,
        "model": model,
        "seed": seed,
        "seed_source": seed_source,
        "server_boot_sec": round(boot, 2) if boot is not None else None,
        "gpu_wait": gpu_wait,
        "metric_key": "latency_ms.ttft.mean",
        "lower_is_better": True,
        "workload": wl,
        "regime": regime,
        "regime_session": {"l2norm_calls": session_counters["calls"],
                           "l2norm_distinct_T": session_counters["distinct_T"],
                           "triton_cubins": session_cubins, "triton_cache_files": session_files},
        "anchor": {"name": spec["anchor"], "returncode": anchor_result["returncode"] if anchor_result else None,
                   "error": anchor_result["error"] if anchor_result else error,
                   "mean_ttft_ms": anchor_result["mean_ttft_ms"] if anchor_result else None},
        "workloads_benched": ([spec["anchor"]] if rc == 0 else [])
                             + [pt["name"] for pt in points_benched if not pt["error"]],
        "points": points_benched,
        "elapsed_sec": round(elapsed, 1),
        "result_json": str(out_json),
    }, indent=2) + "\n", encoding="utf-8")
    _say(f"repo={repo} rc={rc} elapsed={elapsed:.0f}s seed={seed} "
         f"benched={[spec['anchor']] + [pt['name'] for pt in points_benched if not pt['error']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
