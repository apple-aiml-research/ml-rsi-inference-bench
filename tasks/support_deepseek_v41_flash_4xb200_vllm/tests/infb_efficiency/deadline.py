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

Only this module and `_paired_helper.py` (the teardown, IPC-reaping and GPU-free functions) are used;
nothing here imports the engine.

The plugin also orders the efficiency nodes. With `infb.efficiency.relaunch` set in config.json, within each class or
module that holds them (or the whole session, with its `order` "session"), `test_00_efficiency_measure` runs first,
and `test_0_efficiency_relaunch` and then the two scored nodes run after every other node (nodes.py says why); unset,
the two relaunch nodes are deselected, and the order is pytest's own. It keeps each graded node's worst outcome in
this session for the relaunch's reward-0 skip, and the environment the session started with, for the reference's
second launch. The runner, killing a group at its budget, also kills the reference's second launch, which runs in a
process group of its own.

With the relaunch set, the two scored nodes only judge the record, so the plugin arms them with `node_default_s`.
Their `deadlines.nodes` entries stay what they are in every mode: each axis's measurement limit, which
`nodes.run_axis` reads in the measurement node and for the candidate's second launch in the relaunch node. Arming a
scored node with its own entry instead would cut its axis's measurements short with its alarm.

The verifier clock. The verifier's cap is the harness's, and nothing inside the verifier knows it, so test.sh states
it: `infb.verifier_clock` in config.json gives `cap_s` (the cap), `before_s` (what the cap spends before test.sh's
first line: the verifier environment's build bound and 300 s to start it) and `tail_s` (what test.sh runs after its
last bounded group). test.sh takes `INFB_VERIFIER_T0=$(date +%s)` first, and for the session that carries the
relaunch asks `end-by` for END_BY = T0 + cap_s - before_s - (the bounds of the groups after the session) - tail_s,
and passes it to `run --end-by`. The runner is done by END_BY: it hands its child the session's end, END_BY less what
the runner spends after a kill (`INFB_SESSION_END_BY`), records the clock in the efficiency record
(`verifier_clock`), and kills the session there if its budget would run past it. The relaunch node starts only what
ends before that with room for what the session runs after it, and brings its own alarm forward (`shorten_alarm`).
The clock assumes the verifier environment builds within `before_s`. Each trial's record says which clock it ran on.
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
#: The epoch second by which the session must end. test.sh computes END_BY, when the
#: runner must be done, from the verifier clock (`end-by`) and passes it (`run --end-by`); the runner hands its child
#: END_BY less what the runner spends after a kill (`runner_after_kill_s`), and the relaunch node reads it.
END_BY_ENV = "INFB_SESSION_END_BY"
#: test.sh's first line: `INFB_VERIFIER_T0=$(date +%s)`, recorded beside END_BY.
T0_ENV = "INFB_VERIFIER_T0"
EXIT_BUDGET_EXCEEDED = 124
#: The efficiency nodes the plugin orders (their names are nodes.py's; restated so the runner needs no nodes).
MEASURE_NODE = "test_00_efficiency_measure"
RELAUNCH_NODE = "test_0_efficiency_relaunch"
SCORED_NODES = ("test_0_efficiency_decode_c1", "test_0_efficiency_throughput")
_RANK = {"passed": 0, "skipped": 1, "failed": 2}
#: This session's worst outcome per node (classname.name), as junit will carry it; nodes.reward_bound reads it.
_OUTCOMES: dict[str, str] = {}
#: The environment when the session was configured, before any class's setUpClass could change it.
_ENV_AT_START: dict[str, str] = {}
#: Whether the plugin ordered this session's nodes for the relaunch; the scored nodes refuse to judge without it.
_REORDERED = False
#: The relaunch node runs the reference's second launch in a process group of its own, so that it can stop that launch
#: without stopping itself, and names the group in this file beside the record while it runs: a group killed at its
#: budget kills that one too (`_kill_relaunch`), which the kill of pytest's own group does not reach.
RELAUNCH_PGID_FILE = "relaunch-reference.pgid"
#: The session's items in the order they run, (nodeid, classname, name, parent),
#: as pytest finished collecting them, and the node whose call is running; the relaunch node reads what runs after it.
_ORDER: list[tuple[str, str, str, int]] = []
_CURRENT: dict[str, str] = {}
#: What the runner can spend once it ends a group: its kill's grace (15 s), the wait for the group (30 s) and the kill of
#: the reference's second launch (15 s); the reap adds 3 s per device and 120 s for the memory. `--end-by` is when the
#: runner itself is done, so the session is killed that much before it (`session_end_by` in the record).
RUNNER_KILL_S = 60.0


class DeadlineExceeded(BaseException):
    """Raised by the per-node alarm. A BaseException, so `except Exception` retry loops let it through."""


# ── deadlines from config.json ───────────────────────────────────────────────────────────────────


def load_deadlines(config_path: str | os.PathLike | None) -> dict:
    if not config_path or not Path(config_path).is_file():
        return {}
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return ((cfg.get("infb") or {}).get("efficiency") or {}).get("deadlines") or {}


def resolve_deadline(deadlines: dict, classname: str, name: str, when: str,
                     relaunch: bool = False) -> tuple[float | None, str]:
    """(seconds a phase of a node may take, or None for no alarm; where that number came from).

    `setup` gets the launch bound (the first node's setup runs `setUpClass`), `teardown` the teardown
    bound, and `call` the node's own entry -- `nodes["Class.name"]`, then `nodes["name"]` -- or the
    default for generation-driving nodes. The source is recorded with every report, and the runner lists
    the nodes that fell back to the default.
    
    With `relaunch` (the session ordered for `infb.efficiency.relaunch`), a scored node's call gets
    `node_default_s` whatever its entry says (source `node_default_s: relaunch`, which the runner does not list as an
    unnamed default): it judges the record and draws nothing, and its entry is its axis's measurement limit inside
    the measurement and relaunch nodes.
    """
    if not deadlines:
        return None, "none"
    if when == "setup":
        return deadlines.get("launch_s"), "launch_s"
    if when == "teardown":
        return deadlines.get("teardown_s"), "teardown_s"
    if relaunch and when == "call" and name.split("[", 1)[0] in SCORED_NODES:
        return deadlines.get("node_default_s"), "node_default_s: relaunch"
    nodes = deadlines.get("nodes") or {}
    short = classname.rsplit(".", 1)[-1]
    for key in (f"{short}.{name}", name):
        if key in nodes:
            return nodes[key], f"nodes[{key!r}]"
    return deadlines.get("node_default_s"), "node_default_s"


def node_deadline(deadlines: dict, classname: str, name: str, when: str, relaunch: bool = False) -> float | None:
    """The seconds a phase of a node may take, or None for no alarm (see resolve_deadline)."""
    return resolve_deadline(deadlines, classname, name, when, relaunch)[0]


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


def relaunch_order(config_path: str | os.PathLike | None) -> str | None:
    """None where config.json leaves `infb.efficiency.relaunch` unset; else where to order the nodes, its `order`:
    `parent` (the default) or `session`. nodes.relaunch_config checks what it says, and refuses any other value."""
    try:
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return None
    c = ((cfg.get("infb") or {}).get("efficiency") or {}).get("relaunch")
    if c in (None, False):
        return None
    return "session" if isinstance(c, dict) and c.get("order") == "session" else "parent"


def verifier_clock(config_path: str | os.PathLike | None) -> dict | None:
    """`infb.verifier_clock` from config.json, `cap_s`, `before_s` and `tail_s`,
    each a number of seconds; None where it is unset."""
    try:
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return None
    c = (cfg.get("infb") or {}).get("verifier_clock")
    if c is None:
        return None
    for key in ("cap_s", "before_s", "tail_s"):
        v = (c if isinstance(c, dict) else {}).get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
            raise ValueError(f"config.json infb.verifier_clock.{key} must be a number of seconds, got {v!r}")
    return c


def session_end_by(clock: dict, t0: float, after_s: float) -> int:
    """END_BY = T0 + cap_s - before_s - after_s - tail_s: the epoch second by which a session must end for everything
    test.sh runs after it, and its tail, to end before the cap. `after_s` is the sum of the bounds test.sh hands to
    `timeout` for the groups after the session, each with its kill-after."""
    return int(float(t0) + clock["cap_s"] - clock["before_s"] - float(after_s) - clock["tail_s"])


def runner_after_kill_s(cuda_devices: str | None) -> float:
    """The most the runner spends after it ends a group: RUNNER_KILL_S, and the reap's bound on its devices."""
    devices = [d for d in (cuda_devices or "").split(",") if d.strip()]
    return RUNNER_KILL_S + (3.0 * len(devices) + 120.0 if devices else 0.0)


def session_after(nodeid: str | None) -> dict | None:
    """What the session runs after `nodeid`: each item, in order, with the alarm
    the plugin will arm for its call (`items`), and how many classes or modules other than `nodeid`'s own those items
    belong to, each still to set up and tear down (`other_parents`). None where the plugin recorded no order, or
    `nodeid` is not in it."""
    ids = [o[0] for o in _ORDER]
    if not nodeid or nodeid not in ids:
        return None
    i = ids.index(nodeid)
    deadlines = globals().get("_DEADLINES") or {}
    later = _ORDER[i + 1:]
    return {"items": [{"nodeid": nid, "call_s": resolve_deadline(deadlines, cn, nm, "call", relaunch=_REORDERED)[0]}
                      for nid, cn, nm, _ in later],
            "other_parents": len({par for *_, par in later} - {_ORDER[i][3]})}


def order_nodes(items: list, name=lambda it: it.name, parent=lambda it: it.parent) -> list:
    """The node order, within each run of items that share a parent (a class, or a module of functions) and hold
    an efficiency node: the measurement first, every other node in pytest's order, then the relaunch, then the scored
    nodes. Items of other parents keep their places. With `parent` giving every item the same one, the whole session
    is one run (`order: session`)."""
    ours = (MEASURE_NODE, RELAUNCH_NODE, *SCORED_NODES)
    out, i = [], 0
    while i < len(items):
        j = i + 1
        while j < len(items) and parent(items[j]) is parent(items[i]):
            j += 1
        group = items[i:j]
        names = [name(it).split("[", 1)[0] for it in group]
        if any(n in ours for n in names):
            head = [it for it, n in zip(group, names) if n == MEASURE_NODE]
            tail = [it for k in ours[1:] for it, n in zip(group, names) if n == k]
            group = head + [it for it, n in zip(group, names) if n not in ours] + tail
        out += group
        i = j
    return out


# ── the alarm ────────────────────────────────────────────────────────────────────────────────────

_armed: dict[str, Any] = {}


def _on_alarm(signum, frame):  # noqa: ARG001
    if _armed.get("label"):
        label, seconds, cut = _armed["label"], _armed["seconds"], _armed.get("shortened_to")
        _armed.clear()
        raise DeadlineExceeded(f"{label}: exceeded its deadline of {seconds:g} s" if cut is None else
                               f"{label}: reached the verifier clock's end, which cut its alarm to {cut:.0f} s from "
                               f"the cut (its own deadline was {seconds:g} s)")


def shorten_alarm(seconds: float) -> float | None:
    """Bring the node's armed alarm forward to `seconds` from now, never back.
    Returns the seconds the alarm had left before; None when no alarm is armed, and then nothing is armed."""
    if not _armed.get("label") or not hasattr(signal, "getitimer"):
        return None
    left = signal.getitimer(signal.ITIMER_REAL)[0]
    if left <= 0:
        return None
    if seconds < left:
        _armed["shortened_to"] = max(float(seconds), 0.0)
        signal.setitimer(signal.ITIMER_REAL, max(float(seconds), 0.001))
    return left


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
        _ENV_AT_START.clear()
        _ENV_AT_START.update(os.environ)

    def pytest_collection_modifyitems(session, config, items):  # noqa: ARG001
        """With `infb.efficiency.relaunch` set, the efficiency nodes in their order (`order_nodes`);
        unset, the measurement and relaunch nodes deselected, so the session runs and reports the scored nodes
        alone."""
        global _REORDERED
        path = os.environ.get(CONFIG_ENV) or Path(os.environ.get("TESTS_DIR", "/tests")) / "config.json"
        order = relaunch_order(path)
        if order:
            items[:] = order_nodes(items, parent=(lambda it: None) if order == "session" else (lambda it: it.parent))
            _REORDERED = True
            return
        drop = [it for it in items if it.name.split("[", 1)[0] in (MEASURE_NODE, RELAUNCH_NODE)]
        if drop:
            items[:] = [it for it in items if it.name.split("[", 1)[0] not in (MEASURE_NODE, RELAUNCH_NODE)]
            config.hook.pytest_deselected(items=drop)

    def pytest_collection_finish(session):
        """The session's final order, after every plugin's selection, for `session_after`."""
        _ORDER[:] = [(it.nodeid, *junit_names(it.nodeid), id(it.parent)) for it in session.items]

    def _wrap(item, when: str):
        classname, name = junit_names(item.nodeid)
        seconds, source = resolve_deadline(_DEADLINES, classname, name, when, relaunch=_REORDERED)
        _APPLIED[(item.nodeid, when)] = (seconds, source)
        _CURRENT["nodeid"] = item.nodeid
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
        nid = f"{classname}.{name}" if classname else name
        if (report.when == "call" or report.outcome != "passed") and \
                _RANK.get(report.outcome, -1) >= _RANK.get(_OUTCOMES.get(nid), -1):
            _OUTCOMES[nid] = report.outcome
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


def _kill_relaunch(record_dir: Path) -> int | None:
    """The reference's second launch, if the relaunch node was running one, killed with its group; the
    group's pgid, or None. Only a group whose leader is still that launch (`reference.py ... --relaunch`) is killed, so
    a stale file never reaches a process that took the number since."""
    try:
        pgid = int((record_dir / RELAUNCH_PGID_FILE).read_text(encoding="utf-8").strip())
        cmd = Path(f"/proc/{pgid}/cmdline").read_bytes()
    except (OSError, ValueError):
        return None
    if b"reference.py" not in cmd or b"--relaunch" not in cmd:
        return None
    _kill_group(pgid)
    return pgid


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


def censor_record(record: Path, group: str, budget_s: float, end_by: float | None = None) -> None:
    _siblings()
    import client as effclient  # noqa: PLC0415 - sibling module

    def mutate(rec: dict) -> None:
        rl = rec.get("relaunch") or {}
        if rl.get("decision") == "running":
            # killed inside the relaunch, the record keeps the first launch, as a scored node would have
            import nodes as effnodes  # noqa: PLC0415 - sibling module

            effnodes._put_launch(rec, rl["launches"][0], rl.get("axes") or [])  # noqa: SLF001
            rl.update({"decision": "interrupted", "axes_restored": rl.get("axes") or [],
                       "by": "the group's budget" if end_by is None else "the verifier clock's end"})
            rec["summary"]["relaunch"] = {"decision": "interrupted"}
        rec["status"] = "censored"
        rec["summary"].setdefault("deadlines_hit", []).append(
            {"group": group, "budget_s": budget_s, **({"end_by": end_by} if end_by is not None else {})})
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
    failed (as on every run of the unmodified base tree). Closed as `failed`, the unmeasured axis failing, with the
    efficiency nodes' own outcomes, so the record says what the junit says; `in_progress` read as a run cut off."""
    if not record.exists():
        return
    _siblings()
    import client as effclient  # noqa: PLC0415 - sibling module

    nodes = {nid: {"outcome": r["outcome"], "when": r.get("when"), "message": (r.get("message") or "")[-500:]}
             for nid, r in recorded.items() if r.get("name", "").startswith(("test_0_efficiency", MEASURE_NODE))}

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


def record_verifier_clock(record: Path, clock: dict) -> None:
    """The efficiency record carries the clock its session ran on, top-level."""
    _siblings()
    import client as effclient  # noqa: PLC0415 - sibling module

    if record.exists():
        effclient.update_record(record, lambda rec: rec.__setitem__("verifier_clock", clock))


def _clock_of(args, started: float, wait_s: float, session_end: float) -> dict:
    """The record's `verifier_clock` for a session run with `--end-by`: T0 and END_BY, the config's clock, and the
    session's two bounds. `after_s`, the later groups' bounds test.sh counted, is what END_BY implies."""
    try:
        cfg = verifier_clock(args.config) or {}
    except ValueError:
        cfg = {}
    t0 = os.environ.get(T0_ENV)
    t0 = int(t0) if t0 and t0.strip().isdigit() else None
    out = {"clock": "file", "t0": t0, "end_by": args.end_by, "session_end_by": round(session_end),
           "runner_after_kill_s": round(args.end_by - session_end, 3),
           **{k: cfg.get(k) for k in ("cap_s", "before_s", "tail_s")},
           "after_s": (round(t0 + cfg["cap_s"] - cfg["before_s"] - cfg["tail_s"] - args.end_by, 3)
                       if t0 is not None and cfg else None),
           "session": {"group": args.group, "started": round(started, 3), "budget_s": args.budget_s,
                       "session_end_by_left_s": round(session_end - started, 3), "wait_s": round(wait_s, 3),
                       "bound": "end_by" if wait_s < args.budget_s else "budget"}}
    return out


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
    started = time.time()
    # With --end-by the runner is done by END_BY, so the session ends by END_BY less
    # what the runner spends after a kill (`session_end_by`), or at its budget, whichever comes first
    session_end = None if args.end_by is None else args.end_by - runner_after_kill_s(args.cuda_devices)
    wait_s = args.budget_s if session_end is None else min(args.budget_s, session_end - started)
    by_clock = session_end is not None and wait_s < args.budget_s
    bound = (f"wall-clock budget of {args.budget_s:g} s" if not by_clock else
             f"verifier clock's end (the session's end {session_end:.0f}, {max(wait_s, 0):.0f} s after the group "
             f"started, inside its budget of {args.budget_s:g} s; END_BY {args.end_by:.0f})")
    clock = None
    if session_end is not None:
        env[END_BY_ENV] = f"{session_end:.0f}"
        clock = _clock_of(args, started, wait_s, session_end)
        if args.record:
            record_verifier_clock(Path(args.record), clock)
    print(f"[deadline] group {args.group}: budget {args.budget_s:g} s"
          + (f", END_BY {args.end_by:.0f} ({args.end_by - started:.0f} s from now)" if args.end_by is not None else "")
          + f": {' '.join(args.command)}", flush=True)
    killed = False
    if wait_s <= 0:
        killed = True
        print(f"[deadline] group {args.group}: the verifier clock's end has passed; the group is not started", flush=True)
        rc = EXIT_BUDGET_EXCEEDED
    else:
        proc = subprocess.Popen(args.command, env=env, start_new_session=True)
        try:
            rc = proc.wait(timeout=wait_s)
        except subprocess.TimeoutExpired:
            killed = True
            print(f"[deadline] group {args.group}: {bound} reached; killing its process group", flush=True)
            _kill_group(proc.pid)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=30)
            if args.record and _kill_relaunch(Path(args.record).parent) is not None:
                print(f"[deadline] group {args.group}: killed the reference's second launch too", flush=True)
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
        censor_record(Path(args.record), args.group, args.budget_s, args.end_by if by_clock else None)
    elif args.record:
        close_unmeasured_record(Path(args.record), args.group, recorded_results(results))
    if clock is not None and args.record:
        clock["session"].update({"elapsed_s": round(elapsed, 3), "ended_by": "exit" if not killed else
                                 ("end_by" if by_clock else "budget")})
        record_verifier_clock(Path(args.record), clock)
    defaulted = sorted({f"{r['classname']}.{r['name']}" for r in _read_jsonl(results)
                        if r.get("when") == "call" and r.get("deadline_from") == "node_default_s"})
    summary = {"group": args.group, "budget_s": args.budget_s, "elapsed_s": round(elapsed, 3), "killed": killed,
               "rc": rc, "reaped": reaped, "nodes_on_node_default_s": defaulted,
               **({"end_by": args.end_by, "ended_by": clock["session"].get("ended_by") or
                   ("exit" if not killed else "end_by" if by_clock else "budget")} if clock is not None else {})}
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
    r.add_argument("--end-by", type=float,
                   help="the epoch second by which this runner must be done (`end-by`); the session's "
                        f"end, that less what the runner spends after a kill, is exported to the child as {END_BY_ENV}, "
                        "and the group is killed there if its budget would run past it")
    r.add_argument("command", nargs=argparse.REMAINDER)
    e = sub.add_parser("end-by", help="print END_BY for the session that carries the relaunch")
    e.add_argument("--config", required=True, help="config.json, with infb.verifier_clock")
    e.add_argument("--t0", type=float, help=f"test.sh's first `date +%%s` (default: ${T0_ENV})")
    e.add_argument("--after", type=float, required=True,
                   help="the bounds test.sh hands to `timeout` for the groups after the session, kill-after included")
    args = ap.parse_args(argv)
    if args.cmd == "end-by":
        t0 = args.t0 if args.t0 is not None else os.environ.get(T0_ENV)
        clock = verifier_clock(args.config)
        if clock is None or t0 in (None, ""):
            print(f"deadline end-by: {'no infb.verifier_clock in ' + args.config if clock is None else 'no T0'}",
                  file=sys.stderr)
            return 2
        print(session_end_by(clock, float(t0), args.after))
        return 0
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        ap.error("no command after --")
    return run_group(args)


if __name__ == "__main__":
    sys.exit(main())
