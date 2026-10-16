#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Task-local SD3 execution clock. Stdlib only; Linux is the runtime contract.

The outer Pier timeout remains authoritative. We finish before it, or, if owned
process quiescence cannot be proved, wait for that outer cancellation rather
than report a clean exit. This module never changes Pier's publication guard.
Each command has its own child session and subreaper. Cleanup addresses only
observed descendants through pidfds, including double-fork/setsid descendants.
Gate files owned by an interrupted command are retained as raw evidence and
marked failed: an incomplete writer cannot leave a full passing verdict.
"""
from __future__ import annotations

import argparse
import ctypes
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import uuid


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def record(root: Path, **fields) -> None:
    event = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
             "monotonic": time.monotonic(), "pid": os.getpid(), **fields}
    with (root / "events.jsonl").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(event) + "\n")
        f.flush()
    print("[sd3-clock] " + json.dumps(event), flush=True)


def subreaper() -> None:
    if sys.platform != "linux" or not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise RuntimeError("SD3 execution clock requires Linux /proc and pidfds")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER")


def identity(pid: int) -> str | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        return text[text.rfind(")") + 2:].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def descendants() -> dict[int, str]:
    """Census only this process's child tree; never scan or signal shared jobs."""
    todo, seen = [os.getpid()], {}
    while todo:
        parent = todo.pop()
        try:
            threads = list(Path(f"/proc/{parent}/task").iterdir())
        except FileNotFoundError:
            continue
        for thread in threads:
            try:
                children = (thread / "children").read_text().split()
            except FileNotFoundError:
                continue
            for value in children:
                pid = int(value)
                birth = identity(pid)
                if birth is not None and pid not in seen:
                    seen[pid] = birth
                    todo.append(pid)
    return seen


def signal_owned(owned: dict[int, str], sig: int) -> int:
    count = 0
    for pid, birth in owned.items():
        if identity(pid) != birth:
            continue
        try:
            fd = os.pidfd_open(pid)
        except ProcessLookupError:
            continue
        try:
            if identity(pid) == birth:
                signal.pidfd_send_signal(fd, sig)
                count += 1
        except ProcessLookupError:
            pass
        finally:
            os.close(fd)
    return count


def reap(proc: subprocess.Popen) -> list[dict]:
    reaped = []
    proc.poll()


    children = set()
    for thread in Path(f"/proc/{os.getpid()}/task").iterdir():
        try:
            children.update(int(x) for x in (thread / "children").read_text().split())
        except FileNotFoundError:
            continue
    for pid in children - {proc.pid}:
        try:
            found, status = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            continue
        if found:
            reaped.append(dict(pid=found, exit_code=os.waitstatus_to_exitcode(status)))
    return reaped


def process_snapshot(owned: dict[int, str]) -> list[dict]:
    """Bounded process identity/state only; never capture command arguments."""
    rows = []
    for pid, birth in sorted(owned.items())[:32]:
        row = dict(pid=pid, birth=birth)
        try:
            text = Path(f"/proc/{pid}/stat").read_text(errors="replace")
            fields = text[text.rfind(")") + 2:].split()
            row.update(state=fields[0], ppid=int(fields[1]), observed_birth=fields[19],
                       comm=text[text.find("(") + 1:text.rfind(")")][:80])
            status = Path(f"/proc/{pid}/status").read_text(errors="replace")
            for line in status.splitlines():
                if line.startswith("CoreDumping:"):
                    row["core_dumping"] = line.split(":", 1)[1].strip()
            row["wchan"] = Path(f"/proc/{pid}/wchan").read_text().strip()[:128]
        except (FileNotFoundError, ProcessLookupError):
            row["gone"] = True
        except (PermissionError, ValueError, IndexError) as exc:
            row["read_error"] = type(exc).__name__
        rows.append(row)
    return rows


def child_limits() -> None:


    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def clean(proc: subprocess.Popen, term_s: float, kill_s: float,
          recovery_s: float = 0.0) -> dict:
    """Signal each owned identity once; wait for proved exit within the reserve.

    Delayed driver/process teardown after SIGKILL is observed during recovery;
    it never permits another GPU producer to overlap the old tree.
    """
    sent = {"term": 0, "kill": 0}
    history = []
    recovered = False
    reaped = []
    signalled = {"term": set(), "kill": set()}
    def census():
        reaped.extend(reap(proc) or [])
        owned = descendants()
        if proc.returncode is None:
            birth = identity(proc.pid)
            if birth is not None:
                owned.setdefault(proc.pid, birth)
            elif proc.poll() is None:
                raise RuntimeError("direct child exit could not be proved")
        return owned
    def send_once(owned, key, sig):
        fresh = {p: b for p, b in owned.items() if (p, b) not in signalled[key]}
        sent[key] += signal_owned(fresh, sig) if fresh else 0
        signalled[key].update(fresh.items())
    def result(owned):
        return dict(quiescent=not owned, remaining=sorted(owned), **sent,
                    remaining_processes=process_snapshot(owned), cleanup_observations=history,
                    reaped=reaped[:128], recovered_after_kill=recovered)
    for key, sig, grace in (("term", signal.SIGTERM, term_s), ("kill", signal.SIGKILL, kill_s)):
        until = time.monotonic() + grace


        if key == "term" and proc.returncode == 0:
            passive_until = time.monotonic() + min(.25, grace / 4)
            while time.monotonic() < passive_until:
                owned = census()
                if not owned:
                    return result(owned)
                time.sleep(min(.05, max(0, passive_until - time.monotonic())))
        owned = census()
        history.append(dict(phase=key, processes=process_snapshot(owned)))
        while True:
            owned = census()
            if not owned:
                return result(owned)
            send_once(owned, key, sig)
            if time.monotonic() >= until:
                break
            time.sleep(min(.05, max(0, until - time.monotonic())))
    owned = census()
    if owned and recovery_s > 0:
        history.append(dict(phase="recovery", processes=process_snapshot(owned)))
        until = time.monotonic() + recovery_s
        while owned and time.monotonic() < until:

            send_once(owned, "kill", signal.SIGKILL)
            time.sleep(min(.05, max(0, until - time.monotonic())))
            owned = census()
        recovered = not owned
    return result(owned)


def owned_gate_ids(script: str, env: dict) -> set[str]:
    """Writer ownership, so one concurrent queue cannot invalidate the other."""
    profile = env.get("SD3_PRECISION_PROFILE", "mixed")
    if script == "verify_callsite_main.py":
        bases = ("sd3_registered", "sd3_dit_importable", "components_loaded", "native_text_encoders",
                 "native_vae", "native_transformer", "pipeline_runs", "image_similarity_smoke",
                 "image_similarity_full", "prompt_conditioning", "native_dit_forward",
                 "runs_through_sglang_engine", "stage_step_count", "precision_recorded")
        result = {f"{b}_{profile}" for b in bases}
        if profile == "mixed":
            result.update(("native_joint_attention", "attention_backend_selection"))
        return result
    if script == "verify_component_alignment.py":
        return {f"component_alignment_{profile}"}
    if script == "verify_lora_semantics.py":
        if env.get("SD3_DAY0_LORA_OFFLOAD", "0") == "1":
            return {"offload_x_lora"}
        suffix = ("_tp2" if env.get("SD3_DAY0_LORA_TP_SIZE", "1") == "2" else
                  "_sp2" if env.get("SD3_DAY0_LORA_SP_DEGREE", "1") == "2" else "")
        return {x + suffix for x in ("lora_reference_agreement", "lora_state_isolation", "lora_multi_adapter_composition")}
    if script == "verify_fp8_execution.py":
        return {"fp8_dit_execution" + ("_tp2" if env.get("SD3_DAY0_FP8_TP_SIZE", "1") == "2" else "")}
    if script == "verify_dynamic_batching.py":
        suffix = "_tp2" if env.get("SD3_BATCH_TP_SIZE", "1") == "2" else ""
        return {x + suffix for x in ("dynamic_batching_is_real", "batching_x_multioutput_isolation")}
    if script == "verify_parallel.py" and env.get("SD3_PARALLEL_MODE"):
        return set({"dense": (), "tp2": ("tensor_parallel_dit", "lifecycle_under_parallel_tp2"),
                    "sp2": ("sequence_parallel_dit", "sp2_x_cfg_x_nondivisible", "lifecycle_under_parallel_sp2")}
                   .get(env["SD3_PARALLEL_MODE"], ("tensor_parallel_dit", "sequence_parallel_dit",
                        "sp2_x_cfg_x_nondivisible", "lifecycle_under_parallel_tp2", "lifecycle_under_parallel_sp2")))
    return set({
        "verify_t2_lifecycle.py": ("request_isolation_abab", "error_recovery", "in_request_batching",
                                    "cfg1_single_row", "prompt_semantics", "request_cancellation"),
        "verify_parallel.py": ("tensor_parallel_dit", "sequence_parallel_dit", "sp2_x_cfg_x_nondivisible",
                               "lifecycle_under_parallel_tp2", "lifecycle_under_parallel_sp2"),
        "verify_cfg_parallel.py": ("cfg_parallel_is_real",),
        "verify_clip_encoder_parallel.py": ("clip_encoder_parallel_is_real",),
        "verify_dit_layerwise_offload.py": ("dit_layerwise_offload_is_real",),
        "verify_vae_tiled_decode.py": ("vae_tiled_decode_is_real",),
        "verify_no_diffusion_regression.py": ("flux_text_encoding",),
        "verify_shared_regressions.py": ("shared_loader_selection",),
        "verify_anti_cheat.py": ("anti_cheat",),
        "verify_no_hf_evaluated_path.py": ("no_hf_evaluated_path",),
    }.get(script, ()))


def gate_snapshot(verifier: Path, gate_ids: set[str]) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (verifier / "gates").glob("*.gate.json")
            if p.is_file() and p.name.removesuffix(".gate.json") in gate_ids}


def invalidate_changed(root: Path, segment: str, verifier: Path, before: dict, status: str,
                       gate_ids: set[str]) -> list[str]:
    affected = []


    for gate in sorted(gate_ids):
        p = verifier / 'gates' / (gate + '.gate.json')
        if not p.exists():
            atomic_json(p, dict(gate=gate, passed=False, score=0.0,
                error=f'verifier command {status}; gate was not produced',
                details={'not_evaluated': True, 'execution_clock': {
                    'status': status, 'segment': segment, 'missing_raw_gate': True,
                    'raw_evidence': None, 'raw_sha256': None}}))
            affected.append(p.name)
    for p in (verifier / "gates").glob("*.gate.json"):
        if p.name.removesuffix(".gate.json") not in gate_ids:
            continue
        raw = p.read_bytes()
        try:
            marker = json.loads(raw).get('details', {}).get('execution_clock', {})
            if marker.get('missing_raw_gate') and marker.get('segment') == segment:
                continue
        except (ValueError, AttributeError):
            pass
        if before.get(p.name) == hashlib.sha256(raw).hexdigest():
            continue
        dest = root / "interrupted_gate_evidence" / segment / p.name
        dest.parent.mkdir(parents=True, exist_ok=True)


        try:
            with dest.open('xb') as saved:
                saved.write(raw)
        except FileExistsError:
            pass
        original_sha = hashlib.sha256(dest.read_bytes()).hexdigest()
        try:
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                raise ValueError("gate is not an object")
        except (ValueError, UnicodeDecodeError):
            obj = {"gate": p.name.removesuffix(".gate.json")}
        obj.update(passed=False, score=0.0, error=f"verifier command {status}; incomplete reading")
        if not isinstance(obj.get("details"), dict):
            obj["details"] = {}
        obj["details"]["execution_clock"] = {"status": status, "raw_evidence": str(dest),
                                               "raw_sha256": original_sha}
        atomic_json(p, obj)
        affected.append(p.name)
    return affected


def execute(root: Path, command: list[str], *, label: str, phase: str,
            cap: float, deadline: float, term_s: float, kill_s: float,
            verifier: Path | None = None, env: dict | None = None,
            gate_ids: set[str] | None = None) -> dict:
    subreaper()
    if not command or not all(math.isfinite(x) for x in (cap, deadline, term_s, kill_s)):
        raise ValueError("command and finite clock limits required")
    if cap <= 0 or term_s <= 0 or kill_s <= 0:
        raise ValueError("all clock limits must be positive")
    segment = uuid.uuid4().hex
    start = time.monotonic()
    allowed = min(cap, deadline - start - term_s - kill_s)


    recovery_s = min(30.0, max(0.0, deadline - start - allowed - term_s - kill_s)) if phase == "work" else 0.0
    common = dict(segment=segment, label=label, phase=phase,
                  owned_gate_ids=sorted(gate_ids or set()))
    if allowed <= 0:
        result = {**common, "status": "not_started", "returncode": 124,
                  "quiescent": True, "elapsed_s": 0.0, "reason": "global_deadline"}
        record(root, event="end", **result)
        return result
    gate_ids = gate_ids or set()
    try:
        child_env = dict(os.environ if env is None else env,
                         SD3_COMMAND_DEADLINE=str(start + allowed))
        proc = subprocess.Popen(command, start_new_session=True, env=child_env, preexec_fn=child_limits)
    except (OSError, subprocess.SubprocessError) as exc:
        result = {**common, "status": "launch_error", "returncode": 127, "quiescent": True,
                  "elapsed_s": time.monotonic() - start, "error": f"{type(exc).__name__}: {exc}"}
        record(root, event="end", **result)
        return result
    stop = start + allowed
    interrupted = [None]
    def on_signal(signum, _frame):
        interrupted[0] = signum
    old, problem, timed_out = {}, None, False
    try:
        record(root, event="start", **common, child_pid=proc.pid, cap_s=cap,
               allocated_s=allowed, cleanup_recovery_s=recovery_s, deadline=deadline,
               child_core_limit_bytes=0)
        old = {s: signal.signal(s, on_signal) for s in (signal.SIGTERM, signal.SIGINT)}
        while proc.poll() is None and not interrupted[0] and time.monotonic() < stop:
            time.sleep(min(.05, max(0, stop - time.monotonic())))
        timed_out = proc.poll() is None and interrupted[0] is None
    except Exception as exc:
        problem = f"{type(exc).__name__}: {exc}"
    finally:
        try:


            recovery_s = (min(30.0, max(0.0, deadline - time.monotonic() - term_s - kill_s))
                          if phase == "work" else 0.0)
            cleanup = clean(proc, term_s, kill_s, recovery_s=recovery_s)
        except Exception as exc:
            cleanup = {"quiescent": False, "cleanup_error": f"{type(exc).__name__}: {exc}"}
        for sig, handler in old.items():
            signal.signal(sig, handler)
    if not cleanup["quiescent"]:
        status, code = "cleanup_unproven", 75
    elif problem:
        status, code = "instrument_error", 75
    elif interrupted[0]:
        status, code = "cancelled", 128 + interrupted[0]
    elif timed_out:
        status, code = "timed_out", 124
    elif proc.returncode == 0 and (cleanup.get("term", 0) or cleanup.get("kill", 0)):


        status, code = "residual_processes", 70
    else:
        code = proc.returncode
        status = "completed" if code == 0 else "failed"
    result = {**common, "status": status, "returncode": code, "child_returncode": proc.returncode,
              "elapsed_s": time.monotonic() - start, "cleanup_recovery_s": recovery_s, **cleanup}
    if problem:
        result["error"] = problem

    if verifier and code != 0 and cleanup["quiescent"]:


        result["invalidated"] = invalidate_changed(root, segment, verifier, {}, status, gate_ids)
    try:
        record(root, event="end", **result)
    except OSError as exc:
        result.update(returncode=75, journal_error=str(exc))
        print("[sd3-clock] journal write failed; stop this verifier", file=sys.stderr, flush=True)
    return result


def run(args) -> int:
    total, tail, outer = args.total, args.tail, args.outer
    if not 0 < tail + outer + args.term + args.kill < total:
        raise ValueError("total must include tail, outer, TERM and KILL reserves")
    artifacts = Path(os.environ.get("INFB_ARTIFACTS_DIR", "/logs/artifacts"))
    verifier = Path(os.environ.get("INFB_VERIFIER_DIR", "/logs/verifier"))
    root = artifacts / "verifier_clock" / uuid.uuid4().hex
    root.mkdir(parents=True)
    start = time.monotonic()
    clock = {"schema": "sd3-clock-v1", "start": start, "total_s": total,
             "work_deadline": start + total - tail - outer - args.term - args.kill,
             "tail_deadline": start + total - outer - args.term - args.kill,
             "term_s": args.term, "kill_s": args.kill,
             "root_pid": os.getpid(), "root_birth": identity(os.getpid())}
    atomic_json(root / "clock.json", clock)
    env = dict(os.environ, SD3_CLOCK_ROOT=str(root), PYTHONUNBUFFERED="1")
    finalizer = getattr(args, 'finalizer', None)
    trusted = {}
    if finalizer:


        tests = Path(finalizer).resolve().parent
        trusted = {
            str(p.relative_to(tests)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in tests.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
        env['ANTI_CHEAT_PRE_HASHES_JSON'] = json.dumps(trusted)
    result = execute(root, args.command, label="whole_verifier", phase="outer",
                     cap=total, deadline=start + total - outer - (tail if finalizer else 0), term_s=args.term,
                     kill_s=args.kill, env=env)
    if not result["quiescent"]:


        try:
            atomic_json(root / "result.json", result)
            record(root, event="awaiting_outer_cancellation", reason="owned_cleanup_unproven")
        except OSError:
            print("[sd3-clock] cleanup unproven; awaiting outer cancellation", file=sys.stderr, flush=True)
        while True:
            time.sleep(1)
    if result["returncode"] == 0 and result.get("term", 0):
        result.update(status="residual_processes_after_body", returncode=75)
    if finalizer:


        atomic_json(root / 'body-result.json', result)
        tail_env = dict(env, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
        tail_env.pop('PYTHONPATH', None)
        tail_env.pop('PYTHONHOME', None)
        current = {str(p.relative_to(tests)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in tests.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
        if current != trusted:
            publication = dict(status='trusted_source_changed', returncode=75, quiescent=True,
                               mismatched_files=sorted(k for k in trusted.keys() | current.keys()
                                                       if trusted.get(k) != current.get(k)))
        else:
            publication = execute(root, [sys.executable, '-I', '-S', finalizer,
                              '--body-result', str(root / 'body-result.json')],
                label='publication_after_quiescence', phase='tail',
                cap=tail - args.term - args.kill, deadline=start + total - outer,
                term_s=args.term, kill_s=args.kill, env=tail_env)
        if not publication['quiescent']:
            atomic_json(root / 'result.json', dict(body=result, publication=publication, quiescent=False))
            record(root, event='awaiting_outer_cancellation', reason='publication_cleanup_unproven')
            while True:
                time.sleep(1)
        result = dict(status='graded_after_quiescence' if publication['returncode'] == 0 else 'publication_failed',
                      returncode=publication['returncode'], quiescent=True,
                      body=result, publication=publication)
    if result["returncode"] != 0:

        for name in ("reward.json", "reward.txt"):
            p = verifier / name
            if p.exists():
                (root / ("incomplete_" + name)).write_bytes(p.read_bytes())
                p.unlink()
        verifier.mkdir(parents=True, exist_ok=True)
        (verifier / "reward.txt").write_text("-1\n")
    atomic_json(root / "result.json", result)
    return result["returncode"]


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("run")
    for name, value in (("total", 3600), ("tail", 120), ("outer", 30), ("term", 5), ("kill", 5)):
        r.add_argument("--" + name, type=float, default=value)
    r.add_argument('--finalizer', default=None)
    r.add_argument("command", nargs=argparse.REMAINDER)
    s = sub.add_parser("segment")
    s.add_argument("--root", type=Path, required=True)
    s.add_argument("--label", required=True)
    s.add_argument("--phase", choices=("work", "tail"), default="work")
    s.add_argument("--cap", type=float, required=True)
    s.add_argument("--owner-script", default="")
    s.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if args.mode == "run":
        return run(args)
    c = json.loads((args.root / "clock.json").read_text())
    if identity(c["root_pid"]) != c["root_birth"]:
        raise RuntimeError("execution clock owner is no longer alive")
    verifier = Path(os.environ.get("INFB_VERIFIER_DIR", "/logs/verifier")) if args.owner_script else None
    return execute(args.root, args.command, label=args.label, phase=args.phase,
                   cap=args.cap, deadline=c[args.phase + "_deadline"],
                   term_s=c["term_s"], kill_s=c["kill_s"], verifier=verifier,
                   gate_ids=owned_gate_ids(args.owner_script, os.environ))["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
