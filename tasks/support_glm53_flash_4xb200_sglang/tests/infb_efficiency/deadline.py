#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Deadlines for every generation-driving node: a per-node alarm, and a process-group backstop.

Two levels, because `signal.alarm` in `setUp` alone covers
neither a class's launch in `setUpClass` nor a node whose worker threads retry forever:

  * **The pytest plugin** (`-p infb_efficiency.deadline`, with `/tests` on `PYTHONPATH`). Around each
    node's setup, call and teardown it arms a real-time alarm with that node's deadline, read from
    `config.json` (`infb.efficiency.deadlines`). On expiry it raises `DeadlineExceeded`, a
    `BaseException`, so a retry loop's `except Exception` cannot swallow it, and the node fails. The
    setup phase of a class's first node is where `setUpClass` launches the server, so the launch
    bound applies there. Every report is also appended to a JSONL file the moment pytest makes it,
    so a run that is killed later still says which nodes finished.

  * **The group runner** (`python -I /tests/infb_efficiency/deadline.py run ... -- <command>`). It runs one group --
    a class together with its launch -- as its own process group under a wall-clock budget, the
    group's launch bound plus the sum of its node deadlines. On expiry it kills the whole group
    (pytest and every server it started), reaps what escaped by GPU residency, marks the efficiency
    record censored, and writes the group's junit report from what the plugin recorded: a node that
    never reported is a failing case that names the budget. A hang ends at its deadline as a failure,
    and the trial still produces a reward instead of voiding at the verifier's cap.

Only this module and `_paired_helper.py` (the benchmark's own teardown, IPC-reaping and GPU-free
functions, byte for byte) are used; nothing here imports the engine.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape, quoteattr

HERE = Path(__file__).resolve().parent

RESULTS_ENV = "INFB_DEADLINE_RESULTS"
CONFIG_ENV = "INFB_DEADLINE_CONFIG"
EXIT_BUDGET_EXCEEDED = 124


class DeadlineExceeded(BaseException):
    """Raised by the per-node alarm. A BaseException, so `except Exception` retry loops let it through."""


# ── deadlines from config.json ───────────────────────────────────────────────────────────────────


def load_deadlines(config_path: str | os.PathLike | None) -> dict:
    if not config_path or not Path(config_path).is_file():
        return {}
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return ((cfg.get("infb") or {}).get("efficiency") or {}).get("deadlines") or {}


def resolve_deadline(deadlines: dict, classname: str, name: str, when: str) -> tuple[float | None, str]:
    """(seconds a phase of a node may take, or None for no alarm; where that number came from).

    `setup` gets the launch bound (the first node's setup runs `setUpClass`), `teardown` the teardown
    bound, and `call` the node's own entry -- `nodes["Class.name"]`, then `nodes["name"]` -- or the
    default for generation-driving nodes. The source is recorded with every report, and the runner lists
    the nodes that fell back to the default.
    """
    if not deadlines:
        return None, "none"
    if when == "setup":
        return deadlines.get("launch_s"), "launch_s"
    if when == "teardown":
        return deadlines.get("teardown_s"), "teardown_s"
    nodes = deadlines.get("nodes") or {}
    short = classname.rsplit(".", 1)[-1]
    for key in (f"{short}.{name}", name):
        if key in nodes:
            return nodes[key], f"nodes[{key!r}]"
    return deadlines.get("node_default_s"), "node_default_s"


def node_deadline(deadlines: dict, classname: str, name: str, when: str) -> float | None:
    """The seconds a phase of a node may take, or None for no alarm (see resolve_deadline)."""
    return resolve_deadline(deadlines, classname, name, when)[0]


def junit_names(nodeid: str) -> tuple[str, str]:
    """(classname, name) exactly as pytest's junitxml derives them from a node id
    (`_pytest.junitxml.mangle_test_address`, restated so the runner needs no pytest)."""
    path, bracket, params = nodeid.partition("[")
    names = path.split("::")
    names[0] = re.sub(r"\.py$", "", names[0].replace("/", "."))
    names[-1] += bracket + params
    return ".".join(names[:-1]), names[-1]


def _siblings() -> None:
    """The runner half imports its siblings; the plugin half, loaded into pytest, never does."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))


# ── the alarm ────────────────────────────────────────────────────────────────────────────────────

_armed: dict[str, Any] = {}


def _on_alarm(signum, frame):  # noqa: ARG001
    if _armed.get("label"):
        label, seconds = _armed["label"], _armed["seconds"]
        _armed.clear()
        raise DeadlineExceeded(f"{label}: exceeded its deadline of {seconds:g} s")


@contextlib.contextmanager
def alarm(seconds: float | None, label: str):
    if not seconds or seconds <= 0 or not hasattr(signal, "setitimer"):
        yield
        return
    previous = signal.signal(signal.SIGALRM, _on_alarm)
    _armed.update({"label": label, "seconds": float(seconds)})
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        _armed.clear()
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


# ── the pytest plugin ────────────────────────────────────────────────────────────────────────────

try:
    import pytest
except ImportError:  # the runner half works without pytest
    pytest = None

if pytest is not None:
    _DEADLINES: dict = {}
    _APPLIED: dict[tuple[str, str], tuple[float | None, str]] = {}

    def pytest_configure(config):  # noqa: ARG001
        global _DEADLINES
        _DEADLINES = load_deadlines(os.environ.get(CONFIG_ENV))

    def _wrap(item, when: str):
        classname, name = junit_names(item.nodeid)
        seconds, source = resolve_deadline(_DEADLINES, classname, name, when)
        _APPLIED[(item.nodeid, when)] = (seconds, source)
        return alarm(seconds, f"{item.nodeid} ({when})")

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_setup(item):
        with _wrap(item, "setup"):
            yield

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(item):
        with _wrap(item, "call"):
            yield

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_teardown(item):
        with _wrap(item, "teardown"):
            yield

    def pytest_runtest_logreport(report):
        classname, name = junit_names(report.nodeid)
        path = os.environ.get(RESULTS_ENV)
        if not path:
            return
        message = ""
        if report.failed or report.skipped:
            message = str(getattr(report, "longreprtext", "") or report.longrepr or "")[-2000:]
        seconds, source = _APPLIED.get((report.nodeid, report.when), (None, "none"))
        line = {"nodeid": report.nodeid, "classname": classname, "name": name, "when": report.when,
                "outcome": report.outcome, "duration": round(float(report.duration or 0.0), 3),
                "message": message, "deadline_s": seconds, "deadline_from": source, "t": time.time()}
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(line) + "\n")
            fh.flush()


# ── the group runner ─────────────────────────────────────────────────────────────────────────────


def expected_ids(config_path: str | None, prefixes: list[str]) -> list[str]:
    if not config_path or not prefixes:
        return []
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    ids = list(cfg.get("f2p_node_ids", [])) + list(cfg.get("p2p_node_ids", []))
    ids += list(((cfg.get("infb") or {}).get("diagnostic_node_ids") or []))
    seen, out = set(), []
    for i in ids:
        if any(i.startswith(p) for p in prefixes) and i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            with contextlib.suppress(json.JSONDecodeError):
                out.append(json.loads(line))
    return out


def recorded_results(path: Path) -> dict[str, dict]:
    """classname.name -> the worst phase outcome the plugin recorded (failed > skipped > passed)."""
    rank = {"passed": 0, "skipped": 1, "failed": 2}
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        with contextlib.suppress(json.JSONDecodeError):
            r = json.loads(line)
            nid = f"{r['classname']}.{r['name']}" if r["classname"] else r["name"]
            cur = out.get(nid)
            if r["when"] != "call" and r["outcome"] == "passed":
                if cur is None:
                    out[nid] = {**r, "outcome": "pending"}
                continue
            if cur is None or cur["outcome"] == "pending" or rank[r["outcome"]] > rank.get(cur["outcome"], -1):
                out[nid] = r
    return out


def write_junit(path: Path, cases: list[dict], suite: str) -> None:
    fails = sum(1 for c in cases if c["outcome"] == "failed")
    skips = sum(1 for c in cases if c["outcome"] == "skipped")
    lines = ['<?xml version="1.0" encoding="utf-8"?>', "<testsuites>",
             f'<testsuite name={quoteattr(suite)} tests="{len(cases)}" failures="{fails}" errors="0" skipped="{skips}">']
    for c in cases:
        head = f"<testcase classname={quoteattr(c['classname'])} name={quoteattr(c['name'])} time=\"{c.get('duration', 0):.3f}\""
        if c["outcome"] == "passed":
            lines.append(head + "/>")
        elif c["outcome"] == "skipped":
            lines.append(head + f"><skipped message={quoteattr(c.get('message', '')[:500])}/></testcase>")
        else:
            msg = c.get("message", "")
            lines.append(head + f"><failure message={quoteattr(msg.splitlines()[0][:500] if msg else 'failed')}>"
                         f"{escape(msg)}</failure></testcase>")
    lines += ["</testsuite>", "</testsuites>", ""]
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(lines), encoding="utf-8")
    os.replace(tmp, path)


def junit_is_readable(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        import xml.etree.ElementTree as ET  # noqa: PLC0415
        ET.parse(path)
        return True
    except Exception:  # noqa: BLE001
        return False


def _kill_group(pgid: int, grace_s: float = 15.0) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, signal.SIGTERM)
    end = time.monotonic() + grace_s
    while time.monotonic() < end:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.5)
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, signal.SIGKILL)


def reap_after_group(cuda_devices: str | None, patterns: list[str]) -> dict:
    """What escaped the process group: server processes by name, then GPU holders, then IPC, then a
    wait for the memory -- the benchmark's own `_paired_helper` functions for the last three."""
    out: dict[str, Any] = {"killed_by_pattern": [], "gpu_free": None}
    if os.environ.get("INFB_EFFICIENCY_HOST_REAP", "1") == "0":
        out["host_reap"] = False  # on a shared host the group kill is all
        return out
    me = {os.getpid(), os.getppid()}
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit() or int(pid_dir.name) in me:
            continue
        with contextlib.suppress(OSError):
            cmd = (pid_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            if any(re.search(p, cmd) for p in patterns):
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.kill(int(pid_dir.name), signal.SIGKILL)
                    out["killed_by_pattern"].append(int(pid_dir.name))
    _siblings()
    try:
        import _paired_helper as ph  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        out["helper_error"] = repr(exc)
        return out
    devices = [d for d in (cuda_devices or "").split(",") if d.strip()]
    for dev in devices:
        with contextlib.suppress(Exception):
            ph.kill_stray_gpu_processes(cuda_device=dev, grace_sec=3.0)
    with contextlib.suppress(Exception):
        for pid in ph._gpu_pids_via_proc():  # noqa: SLF001 - the helper's own container-safe scan
            if pid not in me:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.kill(pid, signal.SIGKILL)
    with contextlib.suppress(Exception):
        out["ipc"] = ph.cleanup_stale_ipc(verbose=True)
    if devices:
        try:
            ph.wait_for_gpu_free_memory(cuda_devices=",".join(devices), gpu_memory_utilization=0.9,
                                        timeout_sec=120.0)
            out["gpu_free"] = True
        except Exception as exc:  # noqa: BLE001
            out["gpu_free"] = False
            out["gpu_error"] = str(exc)[:300]
    return out


def censor_record(record: Path, group: str, budget_s: float) -> None:
    _siblings()
    import client as effclient  # noqa: PLC0415 - sibling module

    def mutate(rec: dict) -> None:
        rec["status"] = "censored"
        rec["summary"].setdefault("deadlines_hit", []).append({"group": group, "budget_s": budget_s})
        for side in rec.get("axes", {}).values():
            for axis in side.values():
                if axis.get("status") == "running":
                    axis["status"] = "censored"
                    axis["censored"] = True
        effclient.summarize(rec)
    if record.exists():
        effclient.update_record(record, mutate)


def close_unmeasured_record(record: Path, group: str, recorded: dict[str, dict]) -> None:
    """A record the group left `in_progress` has a candidate axis that was never measured: the class
    never reached that efficiency node, most often because the candidate's launch in `setUpClass`
    failed. Closed as `failed`, the unmeasured axis failing, with the efficiency
    nodes' own outcomes, so the record says what the junit says; `in_progress` read as a run cut off."""
    if not record.exists():
        return
    _siblings()
    import client as effclient  # noqa: PLC0415 - sibling module

    nodes = {nid: {"outcome": r["outcome"], "when": r.get("when"), "message": (r.get("message") or "")[-500:]}
             for nid, r in recorded.items() if r.get("name", "").startswith("test_0_efficiency")}

    def mutate(rec: dict) -> None:
        if rec.get("status") != "in_progress":
            return
        measured = rec.get("axes", {}).get("candidate", {})
        unmeasured = [a for a in effclient.AXES if a not in measured]
        rec["status"] = "failed"
        effclient.summarize(rec)
        for axis in unmeasured:
            rec["summary"]["ratios"][axis] = None
            rec["summary"]["verdicts"][axis] = "fail"
            rec["summary"]["flags"].append({"side": "candidate", "axis": axis, "flag": "candidate_not_measured",
                                            "detail": "no efficiency node measured the candidate on this axis"})
        rec["summary"]["closed_by"] = {"runner": "deadline", "group": group, "unmeasured_axes": unmeasured,
                                       "efficiency_nodes": nodes}
    effclient.update_record(record, mutate)


def run_group(args) -> int:
    results = Path(args.results)
    results.parent.mkdir(parents=True, exist_ok=True)
    results.write_text("", encoding="utf-8")
    junit = Path(args.junit)
    with contextlib.suppress(FileNotFoundError):
        junit.unlink()
    env = dict(os.environ, **{RESULTS_ENV: str(results)})
    if args.config:
        env[CONFIG_ENV] = args.config
    t0 = time.monotonic()
    wait_s = args.budget_s
    bound = f"wall-clock budget of {args.budget_s:g} s"
    print(f"[deadline] group {args.group}: budget {args.budget_s:g} s: {' '.join(args.command)}", flush=True)
    killed = False
    proc = subprocess.Popen(args.command, env=env, start_new_session=True)
    try:
        rc = proc.wait(timeout=wait_s)
    except subprocess.TimeoutExpired:
        killed = True
        print(f"[deadline] group {args.group}: {bound} reached; killing its process group", flush=True)
        _kill_group(proc.pid)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=30)
        rc = EXIT_BUDGET_EXCEEDED
    elapsed = time.monotonic() - t0
    reaped = reap_after_group(args.cuda_devices, args.reap_pattern) if (killed or args.always_reap) else {}
    expect = expected_ids(args.config, args.expect_prefix)
    if killed or not junit_is_readable(junit):
        recorded = recorded_results(results)
        cases = []
        for nid, r in recorded.items():
            outcome = r["outcome"]
            msg = r.get("message", "")
            if outcome == "pending":
                outcome = "failed"
                msg = (f"the group was killed at its {bound} while this node was running"
                       if killed else "the node started and never reported")
            cases.append({"classname": r["classname"], "name": r["name"], "outcome": outcome,
                          "duration": r.get("duration", 0.0), "message": msg})
        have = {f"{c['classname']}.{c['name']}" for c in cases}
        for nid in expect:
            if nid not in have:
                classname, _, name = nid.rpartition(".")
                cases.append({"classname": classname, "name": name, "outcome": "failed", "duration": 0.0,
                              "message": (f"deadline: the group {args.group!r} reached its {bound} before this "
                                          f"node reported" if killed else
                                          f"the group {args.group!r} ended (rc {rc}) without a report for this node")})
        write_junit(junit, cases, f"{args.group} (written by the deadline runner)")
        print(f"[deadline] group {args.group}: wrote {junit} with {len(cases)} case(s) from the plugin's record",
              flush=True)
    if killed and args.record:
        censor_record(Path(args.record), args.group, args.budget_s)
    elif args.record:
        close_unmeasured_record(Path(args.record), args.group, recorded_results(results))
    defaulted = sorted({f"{r['classname']}.{r['name']}" for r in _read_jsonl(results)
                        if r.get("when") == "call" and r.get("deadline_from") == "node_default_s"})
    summary = {"group": args.group, "budget_s": args.budget_s, "elapsed_s": round(elapsed, 3), "killed": killed,
               "rc": rc, "reaped": reaped, "nodes_on_node_default_s": defaulted}
    print("INFB_DEADLINE " + json.dumps(summary, sort_keys=True), flush=True)
    if args.summary:
        Path(args.summary).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="run a group of nodes under a wall-clock budget")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--group", required=True)
    r.add_argument("--budget-s", type=float, required=True)
    r.add_argument("--junit", required=True, help="the junit report the grader reads for this group")
    r.add_argument("--results", required=True, help="the plugin's per-report JSONL")
    r.add_argument("--config", help="config.json: the deadlines and the expected node ids")
    r.add_argument("--expect-prefix", action="append", default=[], help="node ids of this group")
    r.add_argument("--record", help="the efficiency record to mark censored on expiry")
    r.add_argument("--cuda-devices", help="devices to reap and check free on expiry, e.g. 0,1,2,3")
    r.add_argument("--reap-pattern", action="append",
                   default=[r"sglang\.launch_server", r"sglang serve", r"sglang::", r"vllm serve",
                            r"vllm\.entrypoints", r"VLLM::"])
    r.add_argument("--always-reap", action="store_true")
    r.add_argument("--summary", help="write the runner's summary JSON here")
    r.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        ap.error("no command after --")
    return run_group(args)


if __name__ == "__main__":
    sys.exit(main())
