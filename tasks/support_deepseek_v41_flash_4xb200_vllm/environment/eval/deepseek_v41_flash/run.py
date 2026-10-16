#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Run this task's checks. A convenience: each command runs one of the scripts in this directory, in its own process.

    python run.py native                      the native-support checks (tests/native.py)
    python run.py build                       rebuild after changing compiled sources (build.sh)
    python run.py upstream                    the engine's existing tests that must keep passing (tests/selection.json)
    python run.py serve [--port 8000]         start the recipe's server in the background, or reuse a healthy one
    python run.py stop [--port 8000]          stop the server `serve` started
    python run.py configuration [--log F]     the served-configuration check on the server's launch log
    python run.py quality [--limit N | --all] GSM8K through the engine's evaluator (tests/quality.py)
    python run.py fidelity                    numerical fidelity on the examples (fidelity/check.py)
    python run.py efficiency [--axis A]       decode speed and throughput (serving/client.py)
    python run.py api                         API development examples (tests/api_examples.py)
    python run.py served                      serve (or reuse), then configuration, quality, fidelity, efficiency, api

The served checks run against one server at the recipe, as the evaluation's served checks share one server. Each
command prints what its script prints and exits with its code; `served` prints a summary. Work from any directory.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import recipe  # noqa: E402

WORKSPACE = Path(os.environ.get("INFB_WORKSPACE", "/app"))
PY = recipe.PYTHON if os.path.exists(recipe.PYTHON) else sys.executable
STATE = Path(os.environ.get("INFB_EVAL_STATE", "/tmp/infb-eval"))


def base_url(port: int) -> str:
    return f"http://127.0.0.1:{port}"


def healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(base_url(port) + "/health", timeout=5) as r:
            return r.status == 200
    except OSError:
        return False


def call(argv: list[str], cwd: Path | None = None) -> int:
    print("+ " + " ".join(str(a) for a in argv), flush=True)
    return subprocess.call([str(a) for a in argv], cwd=str(cwd) if cwd else None)


def serve(port: int, log: Path) -> int:
    if healthy(port):
        print(f"reusing the healthy server on port {port}")
        return 0
    STATE.mkdir(parents=True, exist_ok=True)
    out = open(STATE / f"serve-{port}.out", "wb")  # noqa: SIM115 - the server's own descriptor
    proc = subprocess.Popen([PY, str(HERE / "recipe.py"), "serve", "--port", str(port), "--log", str(log)],
                            stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    (STATE / f"server-{port}.pid").write_text(str(proc.pid))
    print(f"started the recipe on port {port} (pid {proc.pid}); its log: {log}", flush=True)
    t0 = time.monotonic()
    while time.monotonic() - t0 < recipe.STARTUP_MAX_WAIT_SECONDS:
        if proc.poll() is not None:
            print(f"the server exited with {proc.returncode} before it was healthy; see {log}")
            return 1
        if healthy(port):
            print(f"healthy after {time.monotonic() - t0:.0f} s")
            return 0
        time.sleep(5)
    print(f"not healthy within {recipe.STARTUP_MAX_WAIT_SECONDS} s; see {log}")
    return 1


def stop(port: int) -> int:
    pid_file = STATE / f"server-{port}.pid"
    if not pid_file.is_file():
        print(f"no server started by run.py on port {port}")
        return 0
    pid = int(pid_file.read_text())
    try:
        os.killpg(pid, signal.SIGTERM)
        print(f"stopped the server's process group {pid}")
    except ProcessLookupError:
        print(f"process group {pid} had already exited")
    pid_file.unlink()
    return 0


def upstream() -> int:
    """The engine's existing tests listed in tests/selection.json: each file's listed tests in its own pytest process,
    from the workspace, with the workspace as the rootdir."""
    sel = json.loads((HERE / "tests" / "selection.json").read_text())
    by_file: dict[str, list[str]] = {}
    for test in sel["tests"]:
        by_file.setdefault(test.split("::", 1)[0], []).append(test)
    print(f"{len(sel['tests'])} existing engine test(s) in {len(by_file)} file(s) (tests/selection.json)", flush=True)
    rcs = [call([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--rootdir", WORKSPACE, *tests], cwd=WORKSPACE)
           for _, tests in sorted(by_file.items())]
    return max(rcs, default=0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("native", "build", "upstream", "api", "fidelity"):
        sub.add_parser(name)
    for name in ("serve", "stop", "configuration", "quality", "efficiency", "served", "api", "fidelity"):
        p = sub.choices.get(name) or sub.add_parser(name)
        p.add_argument("--port", type=int, default=8000)
    sub.choices["serve"].add_argument("--log", help="the server's launch log (default: under /tmp/infb-eval)")
    sub.choices["configuration"].add_argument("--log")
    q = sub.choices["quality"].add_mutually_exclusive_group()
    q.add_argument("--limit", type=int)
    q.add_argument("--all", action="store_true")
    sub.choices["efficiency"].add_argument("--axis", choices=("decode_c1", "throughput", "both"), default="both")
    a = ap.parse_args(argv)
    port = getattr(a, "port", 8000)
    log = Path(getattr(a, "log", None) or STATE / f"server-{port}.log")
    url = base_url(port)
    if a.cmd == "native":
        return call([PY, HERE / "tests" / "native.py", "--workspace", WORKSPACE])
    if a.cmd == "build":
        return call(["bash", HERE / "build.sh"], cwd=WORKSPACE)
    if a.cmd == "upstream":
        return upstream()
    if a.cmd == "serve":
        return serve(port, log)
    if a.cmd == "stop":
        return stop(port)
    if a.cmd == "configuration":
        return call([PY, HERE / "tests" / "served_configuration.py", log])
    if a.cmd == "quality":
        extra = ["--all"] if a.all else (["--limit", str(a.limit)] if a.limit else [])
        return call([PY, HERE / "tests" / "quality.py", "--base-url", url, "--app", WORKSPACE, *extra])
    if a.cmd == "fidelity":
        return call([PY, HERE / "fidelity" / "check.py", "--base-url", url])
    if a.cmd == "efficiency":
        return call([PY, "-I", HERE / "serving" / "client.py", "measure", "--engine", "vllm", "--base-url", url,
                     "--tokenizer", recipe.MODEL + "/tokenizer.json", "--texts", HERE / "serving" / "sample.jsonl.xz",
                     "--concurrency", "16", "--axis", a.axis])
    if a.cmd == "api":
        return call([PY, HERE / "tests" / "api_examples.py", "--base-url", url])
    if a.cmd == "served":
        if serve(port, log) != 0:
            return 1
        results = {}
        for name in ("configuration", "quality", "fidelity", "efficiency", "api"):
            results[name] = main([name, "--port", str(port)] + (["--log", str(log)] if name == "configuration" else []))
        print("summary (exit codes): " + json.dumps(results))
        return max(results.values())
    return 2


if __name__ == "__main__":
    sys.exit(main())
