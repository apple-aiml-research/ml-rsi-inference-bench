#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The reference phase: the human reference, measured before any candidate code runs.

`tests/test.sh` runs it before any candidate code:

    "$PY" -I /tests/infb_efficiency/reference.py --config /tests/config.json

  1. **Identity.** The reference tree was built into the verifier image by `snapshot.py`, before any trial: an
     independent Git copy of the pristine `/app`, `tests/reference/solution.patch` applied, the tree checked against
     the closure tree `oracle.lock` records, `/app/.deps` copied in and the editable build run offline in the tree.
     Here the snapshot record is checked against `config.json` and the tree and the products against the snapshot.
  2. **Component timing**: the verifier-owned Triton arm (`infb_verify/msa_triton_arm.py`, the contract's
     entry points over the reference tree's Triton operators) timed by the public protocol on GPU 0 with no server
     resident, every family, into `/logs/verifier/msa/reference-timing.json`.
  3. **The launches**, one per deployment (`infb.efficiency.recipes`, in order): the public recipe's argv from the
     trusted `/tests/eval` copy, with the verifier's two launch additions exactly as the candidate's
     (`VLLM_SERVER_DEV_MODE=1`, an idle torch profiler), `PYTHONPATH=<isolation>:<tree>`, arm-private build and JIT
     roots, a working directory outside `/app`, its own port and process group, and the engine's `__file__` asserted
     inside the tree. Healthy within the startup limit, or it is a host fault; a reference that crashes once is
     retried after the IPC reap (the candidate's launch is never retried).
  4. **The measurement**: the task's client, in a clean process, on the hidden draw -- three draws of S1 and of S_C,
     the median draw kept -- into each deployment's core record.
  5. **Teardown**: the benchmark's `teardown_and_confirm_free`, `cleanup_stale_ipc`, and the GPU-free and `/dev/shm`
     checks before the next launch.

Any failure here is the host's, not the candidate's: the step exits 3, `test.sh` exits without a reward, and the
frame writes the -1 sentinel (preflight semantics).
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if __name__ == "__main__":
    _cwd = os.getcwd()
    sys.path[:] = [p for p in sys.path if p == str(HERE) or (p and os.path.realpath(p) != _cwd)]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import client as effclient  # noqa: E402
import nodes  # noqa: E402 - the one client invocation, shared with the candidate's nodes

EXIT_OK = 0
EXIT_HOST_FAULT = 3
RECORD_DIR = Path(os.environ.get("INFB_EFFICIENCY_DIR", "/logs/verifier/efficiency"))
# The host-level reap -- stray GPU processes, /dev/shm segments by name, GPU-free waits -- is right in
# a verifier container; INFB_EFFICIENCY_HOST_REAP=0 switches it off where /dev/shm is shared with
# other workloads.
HOST_REAP = os.environ.get("INFB_EFFICIENCY_HOST_REAP", "1") != "0"


class HostFault(Exception):
    pass


def log(msg: str) -> None:
    print(f"[reference] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    return effclient.sha256_file(path)


def protocol_sha256(protocol_dir: Path = HERE) -> str:
    """Digest the task-owned measurement implementation and its inputs."""
    h = hashlib.sha256()
    for p in sorted(protocol_dir.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            h.update(str(p.relative_to(protocol_dir)).encode())
            h.update(b"\0")
            h.update(sha256_file(p).encode())
    return h.hexdigest()


def server_env(tree: Path, ref: dict, extra: dict | None = None) -> dict:
    """The reference server's environment: the verifier's own (test.sh's exports and the image's ENV,
    which the candidate's server also inherits: the CUDA compat library path, the model cache dir),
    with PYTHONPATH pointing at the reference tree instead of anything the candidate's run sets."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "INFB_DEADLINE_"))}
    env["PYTHONPATH"] = os.pathsep.join([str(HERE / "isolation"),
                                        *(str(tree / p) for p in ref.get("pythonpath", ["."]))])
    env["INFB_REFERENCE_TREE"] = str(tree.resolve())
    env["INFB_CANDIDATE_TREE"] = str(Path(ref.get("app", "/app")).resolve())
    env["VLLM_SERVER_DEV_MODE"] = "1"
    env.update(extra or {})
    return env


#: Both servers run as one user in one container and
#: so shared `$HOME/.cache`. FlashInfer keys its compiled ops by NAME under
#: `$FLASHINFER_WORKSPACE_BASE/.cache/flashinfer/<version>/<arch>/cached_ops/<op>`; tvm-ffi (SGLang's JIT kernels) keys by name plus
#: a sha256 of the inline include stubs (which carry the csrc files' absolute paths), the file lists, the compiler flags and the
#: include paths under `$TVM_FFI_CACHE_DIR` (default `~/.cache/tvm-ffi`), so two trees never share a build directory but the
#: directory itself is shared; DeepGEMM and Triton hash their generated code. A shared directory lets the second server start warm
#: on what the first compiled, and a name-keyed cache would let one server run another's kernels. The reference therefore gets its
#: own directories (the candidate keeps the image's defaults, cold in the fresh verifier container). Cubins and AOT modules come
#: from their packages where installed; where FlashInfer would look for cubins under the redirected base, the default location is
#: named explicitly so nothing is fetched. Environment setup, not grading.
JIT_CACHE_VARS = ("TVM_FFI_CACHE_DIR", "FLASHINFER_WORKSPACE_BASE", "FLASHINFER_CUBIN_DIR")


def private_jit_caches(env: dict, root: Path) -> dict:
    """Point the reference server's JIT caches at `root`; the record of what was set and what it replaced."""
    root.mkdir(parents=True, exist_ok=True)
    home = Path(env.get("HOME") or os.environ.get("HOME") or "/root")
    rec: dict = {"previous": {k: env.get(k) for k in JIT_CACHE_VARS}}
    env["TVM_FFI_CACHE_DIR"] = rec["tvm_ffi_cache_dir"] = str(root / "tvm-ffi")
    env["FLASHINFER_WORKSPACE_BASE"] = rec["flashinfer_workspace_base"] = str(root / "flashinfer")
    default_cubins = home / ".cache" / "flashinfer" / "cubins"
    if not env.get("FLASHINFER_CUBIN_DIR") and default_cubins.is_dir():
        env["FLASHINFER_CUBIN_DIR"] = rec["flashinfer_cubin_dir"] = str(default_cubins)
    return rec


def container_identity() -> dict:
    """What the container runtime says about this container, where it says anything."""
    out: dict = {}
    ce = Path("/run/.containerenv")
    if ce.is_file():
        out["runtime"] = "podman"
        with contextlib.suppress(OSError):
            for line in ce.read_text(errors="replace").splitlines():
                k, _, v = line.partition("=")
                if k in ("engine", "name", "image", "imageid"):
                    out[k] = v.strip().strip('"')
    elif Path("/.dockerenv").exists():
        out["runtime"] = "docker"
    return out


def gpu_indices() -> list[str]:
    if not HOST_REAP:
        return []
    smi = shutil.which("nvidia-smi")
    if not smi:
        return []
    p = subprocess.run([smi, "--query-gpu=index", "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
    if p.returncode != 0:  # its complaint is on stdout, and is not a device list
        return []
    return [line.strip() for line in p.stdout.splitlines() if line.strip().isdigit()]


# ── 1. identity ──────────────────────────────────────────────────────────────────────────────────


def check_isolated(app: Path) -> list[str]:
    """Nothing of the candidate's tree on this step's path or among its modules (`python -I`).

    Drops such path entries (returning them) and raises HostFault if a module already came from there.
    """
    root = os.path.realpath(app)

    def inside(path: str) -> bool:
        real = os.path.realpath(path or os.getcwd())
        return real == root or real.startswith(root + os.sep)

    def entry_inside(entry: str) -> bool:
        # A relative entry that names no directory is a path hook's token (setuptools' editable
        # finder puts `__editable__.<dist>.finder.__path_hook__` there), not a place modules load from.
        if entry and not os.path.isabs(entry) and not os.path.isdir(entry):
            return False
        return inside(entry)

    dropped = [p or "(the working directory)" for p in sys.path if entry_inside(p)]
    sys.path[:] = [p for p in sys.path if not entry_inside(p)]
    loaded = sorted(name for name, mod in list(sys.modules.items())
                    if isinstance(getattr(mod, "__file__", None), str) and inside(mod.__file__))
    if loaded:
        raise HostFault(f"the reference step imported {loaded[:10]} from the candidate's tree {root}, so candidate "
                        f"code has run in it. Run it as `python -I /tests/infb_efficiency/reference.py`")
    return dropped


def check_snapshot(ref: dict) -> dict:
    root = Path(ref["root"])
    snap_path = root / "snapshot.json"
    if not snap_path.is_file():
        raise HostFault(f"no reference snapshot at {snap_path}: the verifier image was built without the snapshot step")
    snap = json.loads(snap_path.read_text())
    patch = Path(ref["patch"])
    problems = []
    if sha256_file(patch) != ref["patch_sha256"]:
        problems.append(f"{patch} is not the reference patch config.json names")
    if snap.get("patch_sha256") != ref["patch_sha256"]:
        problems.append("the snapshot was built from a different patch")
    if snap.get("tree") != ref["target_tree"] or snap.get("base") != ref["base"]:
        problems.append(f"the snapshot is {snap.get('base')}+patch = {snap.get('tree')}, config.json wants "
                        f"{ref['base']}+patch = {ref['target_tree']}")
    tree = root / "tree"
    diff = subprocess.run(["git", "-C", str(tree), "diff", "--quiet"], capture_output=True)
    if diff.returncode != 0:
        problems.append("the reference tree differs from its index")
    wt = subprocess.run(["git", "-C", str(tree), "write-tree"], capture_output=True, text=True)
    if wt.stdout.strip() != ref["target_tree"]:
        problems.append(f"the reference tree hashes to {wt.stdout.strip() or wt.stderr.strip()!r}, not {ref['target_tree']}")
    for prod in snap.get("products", []):
        p = tree / prod["path"]
        if not p.is_file() or sha256_file(p) != prod["sha256"]:
            problems.append(f"build product {prod['path']} is missing or changed")
    if problems:
        raise HostFault("reference identity: " + "; ".join(problems))
    log(f"identity ok: tree {ref['target_tree'][:12]} at {tree}, {len(snap.get('products', []))} build product(s)")
    return snap


# ── 2. the argv, from the vendored class ─────────────────────────────────────────────────────────

_CAPTURE = r'''
import importlib, importlib.util, inspect, json, os, re, shlex, subprocess, sys
spec = json.loads(sys.argv[1])
class _Captured(BaseException):
    pass
seen = {}
real_exec = subprocess.Popen._execute_child
def fake_exec(self, *a, **kw):
    # Every way a harness starts a process -- subprocess.Popen, a name imported from it, run() --
    # reaches this method; the launch is the first process whose argv names the engine.
    bound = inspect.signature(real_exec).bind(self, *a, **kw).arguments
    args = bound.get("args")
    argv = shlex.split(os.fsdecode(args)) if isinstance(args, (str, bytes)) else [os.fsdecode(x) for x in args]
    if not re.search(spec["match"], " ".join(argv)):
        return real_exec(self, *a, **kw)
    seen["argv"] = argv
    env = bound.get("env")
    if env is not None:
        env = {os.fsdecode(k): os.fsdecode(v) for k, v in env.items()}
        seen["env"] = env
        seen["env_changed"] = {k: v for k, v in env.items() if os.environ.get(k) != v}
        seen["env_removed"] = sorted(k for k in os.environ if k not in env)
    raise _Captured()
m_spec = importlib.util.spec_from_file_location("infb_reference_registered", spec["module"])
mod = importlib.util.module_from_spec(m_spec)
sys.modules[m_spec.name] = mod
m_spec.loader.exec_module(mod)
cls = getattr(mod, spec["class"])
subprocess.Popen._execute_child = fake_exec
try:
    cls.setUpClass()
except _Captured:
    pass
finally:
    subprocess.Popen._execute_child = real_exec
engine = importlib.import_module(spec["engine_module"])
seen["engine_file"] = os.path.realpath(getattr(engine, "__file__", "") or "")
seen["class_attrs"] = {k: getattr(cls, k) for k in ("model", "base_url") if isinstance(getattr(cls, k, None), str)}
print("INFB_CAPTURE " + json.dumps(seen))
'''
CAPTURE_TIMEOUT_S = 600.0


_ENGINE_FILE = r'''
import importlib, os, sys
print("INFB_ENGINE_FILE " + os.path.realpath(getattr(importlib.import_module(sys.argv[1]), "__file__", "") or ""))
'''


def engine_file(ref: dict, py: str) -> str:
    """Where the engine resolves for the reference server's interpreter and environment."""
    tree = Path(ref["root"]) / "tree"
    p = subprocess.run([py, "-c", _ENGINE_FILE, ref.get("engine_module", "vllm")], capture_output=True, text=True,
                       env=server_env(tree, ref), cwd=str(Path(ref["root"])), timeout=CAPTURE_TIMEOUT_S)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("INFB_ENGINE_FILE ")]
    if p.returncode != 0 or not lines:
        raise HostFault(f"could not import the engine for the reference: rc {p.returncode}\n{p.stderr[-3000:]}")
    return lines[-1][len("INFB_ENGINE_FILE "):]


def capture_argv(ref: dict, launch: dict, py: str) -> dict:
    tree = Path(ref["root"]) / "tree"
    if "argv" in launch:
        found = engine_file(ref, py)
        if not found.startswith(str(tree.resolve())):
            raise HostFault(f"the engine resolved to {found!r}, outside the reference tree {tree.resolve()}")
        return {"argv": list(launch["argv"]), "source": "config.json (a static argv the task declares)",
                "engine_file": found}
    engine = ref.get("engine_module", "vllm")
    spec = {"module": str(tree / launch["module"]), "class": launch["class"], "engine_module": engine,
            "match": launch.get("match") or re.escape(engine)}
    env = server_env(tree, ref, ref.get("capture_env"))
    proc = subprocess.Popen([py, "-c", _CAPTURE, json.dumps(spec)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, env=env, cwd=str(Path(ref["root"])), start_new_session=True)
    try:
        out, err = proc.communicate(timeout=CAPTURE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        stop_group(proc, grace_s=5.0)
        raise HostFault(f"capturing the launch argv from {launch['module']}::{launch['class']} took over "
                        f"{CAPTURE_TIMEOUT_S:g} s") from None
    finally:
        # whatever the class started besides the launch dies with the capture's process group
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)
    lines = [ln for ln in out.splitlines() if ln.startswith("INFB_CAPTURE ")]
    if proc.returncode != 0 or not lines:
        raise HostFault(f"could not capture the launch argv from {launch['module']}::{launch['class']}: "
                        f"rc {proc.returncode}\n{out[-2000:]}\n{err[-4000:]}")
    seen = json.loads(lines[-1][len("INFB_CAPTURE "):])
    if "argv" not in seen:
        raise HostFault(f"the vendored class's setUpClass started no process matching {spec['match']!r}: "
                        f"nothing to capture")
    root = str(tree.resolve())
    if not seen.get("engine_file", "").startswith(root):
        raise HostFault(f"the engine resolved to {seen.get('engine_file')!r}, outside the reference tree {root}")
    seen["source"] = f"{launch['module']}::{launch['class']}.setUpClass"
    return seen


def with_port(argv: list[str], port: int, host: str = "127.0.0.1") -> list[str]:
    out = list(argv)
    for flag, value in (("--port", str(port)), ("--host", host)):
        if flag in out:
            out[out.index(flag) + 1] = value
        else:
            out += [flag, value]
    return out


# ── 3. the launch ────────────────────────────────────────────────────────────────────────────────


def wait_healthy(proc: subprocess.Popen, url: str, deadline: float) -> tuple[bool, str]:
    import urllib.request  # noqa: PLC0415

    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False, f"exited with code {proc.returncode}"
        with contextlib.suppress(Exception):
            with urllib.request.urlopen(url, timeout=5) as r:  # noqa: S310 - a local server
                if r.status == 200:
                    return True, ""
        time.sleep(5)
    return False, "not healthy within the launch bound"


_LOAD_END = (re.compile(r"Load weight end\. elapsed=([0-9.]+) s"),                  # SGLang, one line per TP rank
             re.compile(r"Model loading took .*? and ([0-9.]+) seconds"))          # vLLM, one line per worker
_RANK_TAG = re.compile(r"\bTP(\d+)\b|\brank=(\d+)|Worker_TP(\d+)")
#: The interim classification: the slowest rank's first load is at least 5x the fastest and at
#: least 600 s. The fallback, when no per-rank lines exist, is a time to healthy of at least 970 s.
SLOW_RANK_RATIO, SLOW_RANK_S, SLOW_HEALTHY_S = 5.0, 600.0, 970.0


def _rank_load_times(log_path: Path, offset: int) -> dict[str, list[float]]:
    """Per rank, every load this launch logged, in order (a speculative draft loads after the target)."""
    out: dict[str, list[float]] = {}
    with open(log_path, "rb") as fh:
        fh.seek(offset)
        text = fh.read().decode("utf-8", "replace")
    for line in text.splitlines():
        m = _LOAD_END[0].search(line) or _LOAD_END[1].search(line)
        if not m:
            continue
        tag = _RANK_TAG.search(line)
        rank = next((g for g in tag.groups() if g is not None), "?") if tag else "?"
        out.setdefault(f"rank {rank}", []).append(float(m.group(1)))
    return out


def _process_tree(root: int) -> list[dict]:
    """The server's process tree at healthy: pid, comm and read_bytes for each, parents before children."""
    children: dict[int, list[int]] = {}
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        with contextlib.suppress(OSError, ValueError, IndexError):
            ppid = int((d / "stat").read_text().rsplit(")", 1)[1].split()[1])
            children.setdefault(ppid, []).append(int(d.name))
    out, todo = [], [root]
    while todo:
        pid = todo.pop(0)
        entry: dict = {"pid": pid}
        with contextlib.suppress(OSError):
            entry["comm"] = Path(f"/proc/{pid}/comm").read_text().strip()
        try:
            io = dict(line.split(": ", 1) for line in Path(f"/proc/{pid}/io").read_text().splitlines())
            entry["read_bytes"] = int(io["read_bytes"])
        except (OSError, ValueError, KeyError) as exc:
            entry["read_bytes_error"] = type(exc).__name__
        out.append(entry)
        todo += sorted(children.get(pid, []))
    return out


def loading_record(proc: subprocess.Popen, log_path: Path, offset: int, time_to_healthy_s: float) -> dict:
    record: dict = {"time_to_healthy_s": round(time_to_healthy_s, 3)}
    try:
        ranks = _rank_load_times(log_path, offset)
        record["rank_load_s"] = ranks
        first = [loads[0] for loads in ranks.values() if loads]
        if first:
            hi, lo = max(first), min(first)
            record["first_load_max_s"], record["first_load_min_s"] = hi, lo
            record["first_load_max_over_min"] = round(hi / lo, 2) if lo > 0 else None
            record["slow_rank_signature"] = bool(lo > 0 and hi >= SLOW_RANK_RATIO * lo and hi >= SLOW_RANK_S)
        else:
            record["slow_rank_signature"] = time_to_healthy_s >= SLOW_HEALTHY_S
            record["slow_rank_signature_basis"] = "time to healthy (no per-rank load lines in the log)"
    except OSError as exc:
        record["rank_load_error"] = f"{type(exc).__name__}: {exc}"[:300]
    try:
        record["processes"] = _process_tree(proc.pid)
    except OSError as exc:
        record["processes_error"] = f"{type(exc).__name__}: {exc}"[:300]
    return record


def launch_reference(argv: list[str], env: dict, cwd: Path, health_url: str, bound_s: float,
                     log_path: Path) -> tuple[subprocess.Popen | None, dict]:
    import _paired_helper as ph  # noqa: PLC0415

    info: dict = {"attempts": []}
    state: dict = {"proc": None, "timed_out": False}

    def attempt(i: int) -> int:
        if state["timed_out"]:
            info["attempts"].append({"attempt": i + 1, "skipped": "the first attempt timed out; a retry is for crashes"})
            return 2
        fh = open(log_path, "ab")  # noqa: SIM115 - held by the server for its lifetime
        offset = fh.tell()  # this attempt's lines start here; an earlier attempt's stay out of its loading record
        t0 = time.monotonic()
        proc = subprocess.Popen(argv, env=env, cwd=str(cwd), stdout=fh, stderr=subprocess.STDOUT,
                                start_new_session=True)
        proc._log_fp = fh  # noqa: SLF001 - closed by _paired_helper.teardown_and_confirm_free
        ok, why = wait_healthy(proc, health_url, t0 + bound_s)
        took = time.monotonic() - t0
        info["attempts"].append({"attempt": i + 1, "healthy": ok, "seconds": round(took, 3), "why": why or None})
        if ok:
            state["proc"] = proc
            info["time_to_healthy_s"] = round(took, 3)
            try:
                info["loading"] = loading_record(proc, log_path, offset, took)
            except Exception as exc:  # noqa: BLE001 - a diagnostic never fails the launch it describes
                info["loading"] = {"time_to_healthy_s": round(took, 3), "error": f"{type(exc).__name__}: {exc}"[:300]}
            return 0
        state["timed_out"] = why.startswith("not healthy")
        stop_group(proc)
        if HOST_REAP:
            ph.teardown_and_confirm_free(proc, cuda_devices=",".join(gpu_indices()) or "0", timeout_sec=120)
        return 1

    rc = ph.spawn_with_retry(attempt, max_retries=1, cleanup_between=HOST_REAP, grace_sec=5.0,
                             label="reference launch")
    return (state["proc"] if rc == 0 else None), info


# ── 4. the measurement, through the shared client ───────────────────────────────────────────────


def run_host_checks(checks: list[dict], tree: Path, py: str, base_url: str, env: dict, cwd: Path,
                    engine: str = "vllm") -> list[dict]:
    """Upstream nodes run unchanged on the reference: the class's `setUpClass` runs with its launcher
    (`popen_launch_server` unless the check names another) returning a stand-in, so the node meets the
    reference server already up."""
    results = []
    driver = r'''
import importlib.util, inspect, json, os, re, shlex, subprocess, sys, unittest
spec = json.loads(sys.argv[1])
m = importlib.util.spec_from_file_location("infb_reference_host_check", spec["module"])
mod = importlib.util.module_from_spec(m); sys.modules[m.name] = mod; m.loader.exec_module(mod)
class _Dummy:
    pid = -1
    def poll(self): return 0
setattr(mod, spec["launcher"], lambda *a, **k: _Dummy())
real_exec = subprocess.Popen._execute_child
def no_second_server(self, *a, **kw):
    args = inspect.signature(real_exec).bind(self, *a, **kw).arguments.get("args")
    argv = shlex.split(os.fsdecode(args)) if isinstance(args, (str, bytes)) else [os.fsdecode(x) for x in args]
    if re.search(spec["match"], " ".join(argv)):
        raise RuntimeError("the host check's class launched a server by another path than " + spec["launcher"])
    return real_exec(self, *a, **kw)
subprocess.Popen._execute_child = no_second_server
cls = getattr(mod, spec["class"])
cls.setUpClass()
cls.base_url = spec["base_url"]
case = cls(spec["method"])
result = unittest.TestResult()
case.run(result)
bad = result.failures + result.errors
print("INFB_HOST_CHECK " + json.dumps({"ok": not bad and not result.skipped, "failures": [t[1][-1500:] for t in bad],
      "skipped": [s[1] for s in result.skipped]}))
'''
    for chk in checks:
        spec = {"module": str(tree / chk["module"]), "class": chk["class"], "method": chk["method"],
                "launcher": chk.get("launcher", "popen_launch_server"), "base_url": base_url,
                "match": chk.get("match") or re.escape(engine)}
        t0 = time.monotonic()
        try:
            p = subprocess.run([py, "-c", driver, json.dumps(spec)], capture_output=True, text=True, env=env,
                               cwd=str(cwd), timeout=float(chk.get("deadline_s", 300)))
            print(p.stdout[-3000:], flush=True)
            lines = [ln for ln in p.stdout.splitlines() if ln.startswith("INFB_HOST_CHECK ")]
            got = json.loads(lines[-1][len("INFB_HOST_CHECK "):]) if lines else {"ok": False, "failures": [p.stderr[-2000:]]}
        except subprocess.TimeoutExpired:
            got = {"ok": False, "failures": [f"exceeded {chk.get('deadline_s', 300)} s"]}
        got.update({"check": f"{chk['class']}.{chk['method']}", "seconds": round(time.monotonic() - t0, 3)})
        results.append(got)
    return results


# ── 5. teardown ──────────────────────────────────────────────────────────────────────────────────


def stop_group(proc: subprocess.Popen | None, grace_s: float = 10.0) -> None:
    """SIGTERM the server's own process group, then SIGKILL what is left of it."""
    if proc is None or proc.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=10)


#: The CUDA probe of every visible device: one context and one allocation each (a device the previous server still
#: holds refuses it with cudaErrorDevicesUnavailable).
DEVICE_PROBE = ("import torch\n"
                "for i in range(torch.cuda.device_count()):\n"
                "    torch.zeros(1, device=f'cuda:{i}')\n"
                "torch.cuda.synchronize()\n")


def wait_devices_ready(devices: str, timeout_s: float, py: str | None = None) -> dict:
    """Before the next reference launch: every device shows no compute process of anyone else and takes a new
    CUDA context. The previous server's memory can read free while a force-killed process of it still holds a device,
    and the next launch's rank then dies at set_device; so this polls, every 2 s up to ``timeout_s``, rather than
    trusting the memory reading alone."""
    nvsmi = shutil.which("nvidia-smi")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices)
    t0, tries, last = time.monotonic(), 0, None
    while True:
        tries += 1
        apps: list[str] = []
        if nvsmi:
            with contextlib.suppress(subprocess.SubprocessError, OSError):
                q = subprocess.run([nvsmi, "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
                                   capture_output=True, text=True, timeout=30)
                apps = [line.strip() for line in q.stdout.splitlines()
                        if line.strip() and line.split(",")[0].strip() != str(os.getpid())]
        try:
            probe = subprocess.run([py or sys.executable, "-c", DEVICE_PROBE], env=env, capture_output=True, text=True,
                                   timeout=120)
            probe_rc, probe_err = probe.returncode, probe.stderr[-300:]
        except (subprocess.SubprocessError, OSError) as exc:
            probe_rc, probe_err = None, f"{type(exc).__name__}: {exc}"[:300]
        if probe_rc == 0 and not apps:
            return {"ready": True, "tries": tries, "seconds": round(time.monotonic() - t0, 3)}
        last = {"compute_apps": apps[:8], "probe_rc": probe_rc, "probe_error": probe_err}
        if time.monotonic() - t0 > timeout_s:
            return {"ready": False, "tries": tries, "seconds": round(time.monotonic() - t0, 3), "last": last}
        time.sleep(2)


def teardown(proc: subprocess.Popen | None, eff: dict, deadlines: dict, py: str | None = None) -> dict:
    import _paired_helper as ph  # noqa: PLC0415

    t0 = time.monotonic()
    if not HOST_REAP:
        stop_group(proc)
        return {"host_reap": False, "freed": proc is None or proc.poll() is not None, "gpu_free": None,
                "ok": proc is None or proc.poll() is not None, "seconds": round(time.monotonic() - t0, 3)}
    devices = ",".join(gpu_indices()) or "0"
    out: dict = {"devices": devices}
    out["freed"] = ph.teardown_and_confirm_free(proc, cuda_devices=devices,
                                                gpu_memory_utilization=float(eff.get("gpu_free_fraction", 0.95)),
                                                timeout_sec=float(deadlines.get("teardown_s", 120)))
    out["ipc"] = ph.cleanup_stale_ipc(verbose=True)
    shm = Path("/dev/shm")
    leftovers = sorted(p.name for p in shm.iterdir()
                       if p.name.startswith(("psm_", "nccl-", "vllm_", "sglang"))) if shm.is_dir() else []
    st = os.statvfs("/dev/shm") if shm.is_dir() else None
    free_gib = (st.f_bavail * st.f_frsize) / (1 << 30) if st else None
    out["dev_shm"] = {"free_gib": round(free_gib, 1) if free_gib is not None else None, "leftovers": leftovers[:50]}
    try:
        ph.wait_for_gpu_free_memory(cuda_devices=devices, gpu_memory_utilization=float(eff.get("gpu_free_fraction", 0.95)),
                                    timeout_sec=float(deadlines.get("teardown_s", 120)))
        out["gpu_free"] = True
    except TimeoutError as exc:
        out["gpu_free"] = False
        out["gpu_error"] = str(exc)[:300]
    out["devices_ready"] = wait_devices_ready(devices, float(deadlines.get("teardown_s", 120)), py)
    out["seconds"] = round(time.monotonic() - t0, 3)
    min_shm = float(eff.get("shm_min_gib", 1))
    # Leftover segments are recorded, not fatal. What must hold for the candidate is free GPUs, devices
    # that take a context, and room.
    out["ok"] = bool(out["freed"] and out["gpu_free"] and out["devices_ready"]["ready"]
                     and (free_gib is None or free_gib >= min_shm))
    return out


# ── the step ─────────────────────────────────────────────────────────────────────────────────────



def provenance(cfg: dict, eff: dict, config_path: Path) -> dict:
    ident = container_identity()
    model_patch = Path("/logs/artifacts/model.patch")
    return {
        "task": cfg["infb"].get("task"), "score_profile": eff.get("score_profile"),
        "verifier_version": cfg["infb"].get("verifier_version"), "protocol_sha256": protocol_sha256(),
        "config_sha256": sha256_file(config_path), "reference_patch_sha256": eff["reference"]["patch_sha256"],
        "model_patch_sha256": sha256_file(model_patch) if model_patch.is_file() else None,
        "base_commit": cfg.get("base_commit"), "engine_image": os.environ.get("INFB_ENGINE_IMAGE"),
        "verifier_image": ident.get("image") or ident.get("imageid"),
        "container": ident, "trial": os.environ.get("PIER_TRIAL_NAME") or os.environ.get("TRIAL_NAME"),
        "route": ident.get("runtime", "unknown"),
        "host": os.uname().nodename,
    }


def _fresh_record_dir(name: str) -> Path:
    rdir = nodes.record_dir(name)
    rdir.mkdir(parents=True, exist_ok=True)
    for stale in ("efficiency.json", "requests.jsonl.gz"):
        with contextlib.suppress(FileNotFoundError):
            (rdir / stale).unlink()
    return rdir


def serve_recipe(name: str, recipe: dict, eff: dict, snap: dict, py: str, shared: dict) -> str | None:
    """Serve and measure one recipe's reference into its own record; returns the host fault, or None."""
    ref = eff["reference"]
    rdir = _fresh_record_dir(name)
    record = rdir / "efficiency.json"
    prov_path = rdir / "provenance.json"
    prov_path.write_text(json.dumps({**shared, "recipe": name, "C": recipe["C"]}, indent=1))
    deadlines = nodes.resolved_deadlines(eff, name)
    phase: dict = {"side": "reference", "recipe": name, "status": "running",
                   "snapshot": {"tree": snap.get("tree"), "products": len(snap.get("products", []))}}
    proc = None
    fault = None
    try:
        # The reference serves exactly as the candidate does: the public recipe of this deployment from the trusted
        # /tests/eval copy, plus the verifier's two launch additions (dev mode, an idle profiler), made identically
        # for the candidate by infb_verify/serve.py.
        for root in (str(HERE.parent / "eval"), str(HERE.parent)):
            if root not in sys.path:
                sys.path.insert(0, root)
        from minimax_m3.serving import recipes  # noqa: PLC0415 - the trusted copy
        from infb_verify import serve  # noqa: PLC0415
        tree = Path(ref["root"]) / "tree"
        found = engine_file(ref, py)
        if not found.startswith(str(tree.resolve())):
            raise HostFault(f"the engine resolved to {found!r}, outside the reference tree {tree.resolve()}")
        port = int(ref.get("port", 31000))
        trace_dir = Path(ref["root"]) / "traces" / name
        trace_dir.mkdir(parents=True, exist_ok=True)
        argv = with_port(recipes.argv(recipe["deployment"]) + serve.profiler_args(trace_dir), port)
        phase["launch"] = {"argv": argv, "argv_source": f"minimax_m3.serving.recipes.argv({recipe['deployment']!r})",
                           "engine_file": found, "port": port}
        log(f"{name}: argv: {' '.join(argv)}")
        env = server_env(tree, ref)
        env.update(serve.arm_env("reference"))
        env.update(ref.get("launch_env", {}))
        env.update(recipe.get("launch_env", {}))
        # isolation first, then the trusted protocol copy ahead of the tree's own, the verifier package last
        first, *rest = env["PYTHONPATH"].split(os.pathsep)
        env["PYTHONPATH"] = os.pathsep.join([first, str(HERE.parent / "eval"), *rest, str(HERE.parent)])
        phase["launch"]["jit_cache"] = private_jit_caches(env, Path(ref["root"]) / "jit-cache")
        cwd = Path(ref["root"])
        base_url = f"http://127.0.0.1:{port}"
        proc, info = launch_reference(argv, env, cwd, base_url + ref.get("health_path", "/health"),
                                      float(deadlines["launch_s"]), rdir / "reference-server.log")
        phase["launch"].update(info)
        if proc is None:
            raise HostFault(f"the reference did not become healthy: {info['attempts']}")
        log(f"{name}: healthy in {info['time_to_healthy_s']} s on {base_url}")
        draws = nodes.reference_draws(eff)
        for axis, key in (("decode_c1", "reference_decode_c1_s"), ("throughput", "reference_throughput_s")):
            limit = float(deadlines[key])
            kept: list[dict] = []
            for i in range(draws):
                rc, out, killed = nodes.run_client(
                    nodes.client_argv(eff, recipe, base_url, axis, "reference", limit, record, prov_path,
                                      python=py),
                    limit, rdir)
                print(out[-4000:], flush=True)
                if rc != effclient.EXIT_OK or killed:
                    raise HostFault(f"the reference's {axis} measurement failed (client exit {rc}, killed {killed})"
                                    + (f" on draw {i + 1} of {draws}" if draws > 1 else ""))
                this = dict(effclient.read_record(record)["axes"]["reference"][axis])
                if draws > 1:
                    kept.append(this)
            if draws > 1:
                entry = nodes.keep_median_draw(kept)

                def put(rec: dict, axis: str = axis, entry: dict = entry) -> None:
                    rec["axes"]["reference"][axis] = entry
                    effclient.summarize(rec)
                effclient.update_record(record, put)
                log(f"{name}: {axis}: {draws} reference draws {[e.get('value') for e in kept]} tok/s; "
                    f"kept draw {entry['draw_index']} (the median)")
        checks = recipe.get("host_checks") or []
        if checks:
            phase["host_checks"] = run_host_checks(checks, tree, py, base_url, env, cwd,
                                                   ref.get("engine_module", "vllm"))
            bad = [c for c in phase["host_checks"] if not c.get("ok")]
            if bad:
                raise HostFault(f"the reference failed upstream's own check on this host: {bad}")
        phase["status"] = "complete"
    except HostFault as exc:
        phase["status"] = "failed"
        phase["error"] = str(exc)[:4000]
        fault = f"{name}: {exc}"
    except Exception as exc:  # noqa: BLE001 - anything else here is the host's too, and is recorded
        phase["status"] = "failed"
        phase["error"] = f"{type(exc).__name__}: {exc}"[:4000]
        fault = f"{name}: {phase['error']}"
    finally:
        td = teardown(proc, eff, deadlines, py)
        phase["teardown"] = td
        if not td["ok"] and fault is None:
            phase["status"] = "failed"
            phase["error"] = f"teardown left the host unready for the candidate: {td}"
            fault = f"{name}: {phase['error']}"
        if fault:
            log(f"HOST FAULT: {fault}")

        def mutate(rec: dict) -> None:
            rec["provenance"].update({**shared, "recipe": name, "C": recipe["C"]})
            rec["phases"] = [p for p in rec["phases"] if p.get("side") != "reference"] + [phase]
            if fault:
                rec["status"] = "failed"
            effclient.summarize(rec)
        effclient.update_record(record, mutate)
    return fault


MSA_DIR = Path(os.environ.get("INFB_MSA_DIR", "/logs/verifier/msa"))


def component_phase(eff: dict, py: str) -> str | None:
    """The Triton arm's component timing, every family, on GPU 0 with no server resident; returns the
    host fault, or None. Runs in the same worker the candidate's efficiency nodes use, in the reference arm: the
    reference tree first on the path through the isolation hook, arm-private build and JIT roots."""
    for root in (str(HERE.parent / "eval"), str(HERE.parent)):
        if root not in sys.path:
            sys.path.insert(0, root)
    from infb_verify import test_msa  # noqa: PLC0415 - the verifier's own worker runner
    ref = eff["reference"]
    tree = Path(ref["root"]) / "tree"
    families = [f for fams in test_msa.EFFICIENCY.values() for f in fams]
    t0 = time.monotonic()
    rows = test_msa.run_units("measure", families, contract=HERE.parent / "infb_verify" / "msa_triton_arm.py",
                              arm="reference", tag="reference-timing", python=py,
                              env=server_env(tree, ref), isolated=False)
    record = {"schema": "infb-m3-reference-timing/1", "arm": "infb_verify/msa_triton_arm.py", "tree": str(tree),
              "families": dict(zip(families, rows)), "seconds": round(time.monotonic() - t0, 3)}
    MSA_DIR.mkdir(parents=True, exist_ok=True)
    (MSA_DIR / "reference-timing.json").write_text(json.dumps(record, indent=1, default=str))
    bad = {f: r.get("status") for f, r in record["families"].items() if r.get("status") != "measured"}
    if bad:
        return f"the Triton arm's component timing did not complete: {bad}"
    timings = ", ".join(f"{f} {r['gpu_ms_median']:.3f} ms" for f, r in record["families"].items())
    log(f"component timing: {timings}")
    return None


def record_unserved(name: str, recipe: dict, shared: dict, why: str) -> None:
    """A recipe whose reference was never served still gets a valid record that says why."""
    record = _fresh_record_dir(name) / "efficiency.json"

    def mutate(rec: dict) -> None:
        rec["provenance"].update({**shared, "recipe": name})
        if isinstance(recipe.get("C"), int):
            rec["provenance"]["C"] = recipe["C"]
        rec["phases"] = [{"side": "reference", "recipe": name, "status": "failed", "error": f"not served: {why}"[:4000]}]
        rec["status"] = "failed"
        effclient.summarize(rec)
    effclient.update_record(record, mutate)


def run(config_path: Path, py: str) -> int:
    cfg = json.loads(config_path.read_text())
    eff = cfg["infb"]["efficiency"]
    recipes = eff.get("recipes") or {}
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    shared = provenance(cfg, eff, config_path)
    (RECORD_DIR / "provenance.json").write_text(json.dumps(shared, indent=1))
    fault = None if recipes else "config.json infb.efficiency.recipes names no recipe"
    snap: dict = {}
    if fault is None:
        try:
            dropped = check_isolated(Path(eff["reference"].get("app", "/app")))
            if dropped:
                log(f"dropped {dropped} from sys.path: they lie in the candidate's tree (start this step with python -I)")
            snap = check_snapshot(eff["reference"])
        except HostFault as exc:
            fault = str(exc)
            log(f"HOST FAULT: {fault}")
    if fault is None:
        fault = component_phase(eff, py)
        if fault:
            log(f"HOST FAULT: {fault}")
    for name, recipe in recipes.items():
        if fault is None:
            fault = serve_recipe(name, recipe, eff, snap, py, shared)
        else:
            record_unserved(name, recipe, shared, fault)
    log(f"done: {'ok' if fault is None else 'host fault'}; records under {RECORD_DIR}/<recipe>/")
    return EXIT_OK if fault is None else EXIT_HOST_FAULT


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="/tests/config.json")
    ap.add_argument("--python", default=sys.executable, help="the engine's interpreter")
    args = ap.parse_args(argv)
    return run(Path(args.config), args.python)


if __name__ == "__main__":
    sys.exit(main())
