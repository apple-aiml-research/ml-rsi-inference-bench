#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Run one bounded verifier step so that its WHOLE process tree ends at the bound.

    python3 -I /tests/bounded_step.py --group <name> --bound-s <s> [--grace-s 10] [--reap-s 20] -- <command...>

WHY. `timeout --kill-after=10` reached only the processes that stayed in its own process group. Upstream's
fork-per-test decorator (tests/utils.py, fork_new_process_for_each_test) calls os.setpgrp() in pytest and then
forks the test body. In one measured case the bound therefore signalled
pytest alone: the forked body kept running. It also held the write end of run_log's `| tee` pipe, so the log
never reached end-of-file, test.sh waited on the pipeline, and the verifier would have run to its cap with no
verdict.

WHAT THIS RUNNER DOES.
  * It becomes a child subreaper (prctl PR_SET_CHILD_SUBREAPER). A process of the step whose parent exits is
    re-parented to this runner instead of the container's init. Every process the step starts therefore stays a
    descendant, whatever process group or session it moves to.
  * It leads a process group of its own, as `timeout` does, and runs the step as its child in that group. It
    does NOT start the step in a new session: os.setpgrp() fails in a session leader, which would break the very
    decorator above.
  * The step writes to the stdout and stderr this runner was given. test.sh redirects them to run.log, which is
    a file, so no pipe exists for an orphan to hold open.
  * At the bound it sends TERM to every live process of the tree, found by walking /proc from this runner. After
    --grace-s it sends KILL to whatever is left (the 10 s grace of `timeout --kill-after=10`). It then waits up to
    --reap-s for the tree to be gone, and names any survivor.
  * When the step's main process ends before the bound, anything it left running is ended the same way. So no
    process of one step, and no GPU memory such a process holds, reaches the next step.
  * It exits as `timeout` does. It returns the step's own status. At the bound it returns 124, or 137 when the
    main process had to be killed. It returns 127 when the command is not found, 126 when it cannot be executed,
    and 125 for any other failure to start it.
  * It prints one `INFB_BOUNDED {json}` line. The line gives the times, the processes signalled, any survivor,
    and which processes hold a GPU device once the tree has ended.

Standard library only. Run it with `python3 -I`, so that nothing on the candidate's paths reaches it.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time

PR_SET_CHILD_SUBREAPER = 36
EXIT_BOUND = 124
EXIT_CANNOT_RUN = 125
EXIT_NOT_EXECUTABLE = 126
EXIT_NOT_FOUND = 127
POLL_S = 0.2
LISTED = 40  # at most this many pids in a console line; the summary keeps every one


def say(group: str, text: str) -> None:
    print(f"[bounded] {group}: {text}", flush=True)


def become_subreaper() -> str | None:
    """None when this process is now a child subreaper, else why not."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
            return os.strerror(ctypes.get_errno())
    except (OSError, AttributeError) as exc:
        return repr(exc)
    return None


def processes() -> dict[int, tuple[int, str]]:
    """pid -> (ppid, state) for every process visible here."""
    out: dict[int, tuple[int, str]] = {}
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat", "rb") as fh:
                data = fh.read()
        except OSError:
            continue
        fields = data[data.rfind(b")") + 2:].split()  # the command name may hold spaces and parentheses
        out[int(name)] = (int(fields[1]), fields[0].decode())
    return out


def live_tree(root: int) -> list[int]:
    """Every live (not yet exited) descendant of `root`, parents before children."""
    procs = processes()
    children: dict[int, list[int]] = {}
    for pid, (ppid, _state) in procs.items():
        children.setdefault(ppid, []).append(pid)
    out: list[int] = []
    queue = [root]
    while queue:
        for child in sorted(children.get(queue.pop(0), [])):
            queue.append(child)
            if procs[child][1] not in ("Z", "X"):
                out.append(child)
    return out


def reap(main_pid: int, state: dict) -> None:
    """Collect every exited child without blocking; keep the main process's wait status."""
    while True:
        try:
            pid, status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return
        if pid == main_pid:
            state["main_status"] = status
            state["main_ended_s"] = time.monotonic()


def send(pids: list[int], sig: int) -> list[int]:
    sent = []
    for pid in pids:
        try:
            os.kill(pid, sig)
            sent.append(pid)
        except ProcessLookupError:
            pass
    return sent


def listing(pids: list[int]) -> str:
    shown = " ".join(str(p) for p in pids[:LISTED])
    return shown + (f" ... ({len(pids)} in all)" if len(pids) > LISTED else "")


def describe(pid: int) -> dict:
    out: dict = {"pid": pid}
    for key, name in (("state", "stat"), ("wchan", "wchan"), ("cmd", "cmdline")):
        try:
            with open(f"/proc/{pid}/{name}", "rb") as fh:
                raw = fh.read()
        except OSError:
            continue
        if key == "state":
            out[key] = raw[raw.rfind(b")") + 2:].split()[0].decode()
        else:
            out[key] = raw.replace(b"\0", b" ").decode(errors="replace").strip()[:200]
    return out


def end_tree(group: str, me: int, main_pid: int, state: dict, grace_s: float, reap_s: float) -> dict:
    """TERM to the whole tree; KILL to what is left after `grace_s`; up to `reap_s` for it to be gone."""
    termed: list[int] = []
    grace_end = time.monotonic() + grace_s
    while True:
        reap(main_pid, state)
        left = live_tree(me)
        fresh = [p for p in left if p not in termed]
        if fresh:
            termed += send(fresh, signal.SIGTERM)
        if not left or time.monotonic() >= grace_end:
            break
        time.sleep(POLL_S)
    killed: list[int] = []
    left = live_tree(me)
    if left:
        say(group, f"{len(left)} process(es) still running {grace_s:g} s after TERM; KILL: {listing(left)}")
        reap_end = time.monotonic() + reap_s
        while True:
            killed += [p for p in send(live_tree(me), signal.SIGKILL) if p not in killed]
            reap(main_pid, state)
            left = live_tree(me)
            if not left or time.monotonic() >= reap_end:
                break
            time.sleep(POLL_S)
    reap(main_pid, state)
    survivors = [describe(p) for p in live_tree(me)]
    if survivors:
        say(group, f"{len(survivors)} process(es) survived KILL for {reap_s:g} s and are left behind: "
                   + "; ".join(f"{s['pid']} {s.get('state', '?')} {s.get('wchan', '?')} {s.get('cmd', '')[:80]}"
                               for s in survivors[:LISTED]))
    return {"termed": termed, "killed": killed, "survivors": survivors}


def gpu_holders(me: int) -> list[int]:
    """Processes that hold a GPU device open (the container-safe reading: /proc/<pid>/fd, no driver tool)."""
    holders = []
    for name in os.listdir("/proc"):
        if not name.isdigit() or int(name) == me:
            continue
        try:
            fds = os.listdir(f"/proc/{name}/fd")
        except OSError:
            continue
        for fd in fds:
            try:
                target = os.readlink(f"/proc/{name}/fd/{fd}")
            except OSError:
                continue
            if re.fullmatch(r"/dev/nvidia\d+", target):
                holders.append(int(name))
                break
    return sorted(holders)


def gpu_memory_used() -> list[list[int]] | None:
    """[[index, MiB used], ...] from nvidia-smi when it is present, else None."""
    tool = shutil.which("nvidia-smi")
    if not tool:
        return None
    try:
        done = subprocess.run([tool, "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    rows = []
    for line in done.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            rows.append([int(parts[0]), int(parts[1])])
    return rows or None


def status_code(status: int) -> int:
    if os.WIFEXITED(status):
        return os.WEXITSTATUS(status)
    if os.WIFSIGNALED(status):
        return 128 + os.WTERMSIG(status)
    return 1


def run(args: argparse.Namespace, command: list[str]) -> int:
    group = args.group
    me = os.getpid()
    refused = become_subreaper()
    if refused:
        say(group, f"cannot become a child subreaper ({refused}): a process that leaves the step's tree "
                   f"cannot be ended")
    try:
        os.setpgid(0, 0)  # a group of its own, as `timeout` makes; the step is its child in that group
    except OSError:
        pass
    stop: dict[str, int] = {}

    def on_signal(signum, _frame):
        stop.setdefault("signal", signum)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, on_signal)
    say(group, f"bound {args.bound_s:g} s; then TERM to the step's whole process tree, KILL {args.grace_s:g} s "
               f"later, up to {args.reap_s:g} s for it to be gone")
    t0 = time.monotonic()
    try:
        proc = subprocess.Popen(command, close_fds=False)  # inherits stdin, stdout and stderr, as under timeout
    except FileNotFoundError as exc:
        say(group, f"cannot run the step: {exc}")
        return EXIT_NOT_FOUND
    except PermissionError as exc:
        say(group, f"cannot run the step: {exc}")
        return EXIT_NOT_EXECUTABLE
    except OSError as exc:
        say(group, f"cannot run the step: {exc}")
        return EXIT_CANNOT_RUN
    state: dict = {}
    deadline = t0 + args.bound_s
    reason = "exited"
    while "main_status" not in state:
        reap(proc.pid, state)
        if "main_status" in state:
            break
        if stop:
            reason = f"signal {signal.Signals(stop['signal']).name}"
            break
        if time.monotonic() >= deadline:
            reason = "bound"
            break
        time.sleep(POLL_S)
    ended: dict = {"termed": [], "killed": [], "survivors": []}
    if reason == "exited":
        rc = status_code(state["main_status"])
        left = live_tree(me)
        if left:
            say(group, f"the step's main process ended (exit {rc}) after {state['main_ended_s'] - t0:.1f} s and "
                       f"left {len(left)} process(es) running; TERM: {listing(left)}")
            ended = end_tree(group, me, proc.pid, state, args.grace_s, args.reap_s)
    else:
        what = (f"the bound of {args.bound_s:g} s was reached" if reason == "bound"
                else f"this runner received {reason.split()[-1]}")
        tree = live_tree(me)
        say(group, f"{what} after {time.monotonic() - t0:.1f} s; TERM to the step's whole tree "
                   f"({len(tree)} process(es)): {listing(tree)}")
        ended = end_tree(group, me, proc.pid, state, args.grace_s, args.reap_s)
        main = state.get("main_status")
        if reason != "bound":
            rc = status_code(main) if main is not None else 128 + stop["signal"]
        elif main is not None and os.WIFSIGNALED(main) and os.WTERMSIG(main) == signal.SIGKILL:
            rc = 128 + signal.SIGKILL  # as `timeout` reports a command it had to KILL
        else:
            rc = EXIT_BOUND
    if "main_status" in state:
        proc.returncode = status_code(state["main_status"])  # reaped here: Popen must never wait on it
    holders = gpu_holders(me)
    summary = {
        "group": group, "bound_s": args.bound_s, "grace_s": args.grace_s, "reap_s": args.reap_s,
        "reason": reason, "rc": rc, "elapsed_s": round(time.monotonic() - t0, 3),
        "main_ended_s": round(state["main_ended_s"] - t0, 3) if "main_ended_s" in state else None,
        "termed": ended["termed"], "killed": ended["killed"], "survivors": ended["survivors"],
        "subreaper": refused is None, "gpu_holders_after": holders, "gpu_memory_used_mib_after": gpu_memory_used(),
    }
    if reason != "exited" or ended["termed"]:
        say(group, f"the step's tree has ended: exit {rc} after {summary['elapsed_s']:.1f} s; "
                   f"GPU device holders now: {listing(holders) if holders else 'none'}")
    print("INFB_BOUNDED " + json.dumps(summary, sort_keys=True), flush=True)
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="run one verifier step so that its whole process tree ends at the bound")
    ap.add_argument("--group", required=True, help="the group's name, as in config.json's group_bounds_s")
    ap.add_argument("--bound-s", type=float, required=True)
    ap.add_argument("--grace-s", type=float, default=10.0, help="TERM to KILL, as timeout's --kill-after")
    ap.add_argument("--reap-s", type=float, default=20.0, help="how long to wait for a KILLed tree to be gone")
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        ap.error("no command after --")
    return run(args, command)


if __name__ == "__main__":
    sys.exit(main())
