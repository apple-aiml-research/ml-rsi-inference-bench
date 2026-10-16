#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Launch the served model at the recipe, through upstream's own server wrapper."""

from __future__ import annotations

import contextlib
import itertools
import json
import os
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

from infb_verify import _trusted

TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
X = json.loads((TESTS_DIR / "config.json").read_text())["infb"]

#: The recipe the agent is given (`recipe.py`, the grading copy) is the one declarative source of the deployment: the
#: private launch takes its arguments from it, and config.json's `infb.recipe` (which the efficiency gate's reference
#: launch reads) is a derived record that must agree with it.
_RECIPE = _trusted.load("recipe.py")
MODEL = _RECIPE.MODEL
SERVER_ARGS = list(_RECIPE.SERVER_ARGS)
STARTUP_MAX_WAIT_SECONDS = int(_RECIPE.STARTUP_MAX_WAIT_SECONDS)
assert (X["model_id"], list(X["recipe"]["server_args"]), int(X["recipe"]["startup_max_wait_seconds"])) == (
    MODEL, SERVER_ARGS, STARTUP_MAX_WAIT_SECONDS), "config.json infb.recipe is not the recipe's record"

#: The session server's own launch log, every rank's lines, kept for `test_served_configuration`.
#: The verifier's run log keeps every line as before.
LAUNCH_LOG = Path(os.environ.get("INFB_LAUNCH_LOG_DIR", "/logs/verifier")) / "candidate-launch.log"


@contextlib.contextmanager
def _tee_child_output(path: Path) -> Iterator[None]:
    """While the server process is created, route the descriptors it inherits through a pipe.

    `RemoteOpenAIServer` hands its child `sys.stdout` and `sys.stderr`. For the length of this block both
    point at a pipe, and a thread copies everything from it to this process's own stdout (so the run log
    keeps every line) and to `path`. The engine's workers inherit the same descriptors, so `path` holds
    every rank's launch log. Only this process's Python-level streams are swapped, only while the child is
    created and becomes healthy, and they are restored before any node runs.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    read_end, write_end = os.pipe()
    sink = open(path, "wb", buffering=0)  # noqa: SIM115  (closed by the pump when the child exits)
    console = os.dup(1)

    def pump() -> None:
        try:
            while chunk := os.read(read_end, 1 << 16):
                os.write(console, chunk)
                sink.write(chunk)
        finally:
            os.close(read_end)
            os.close(console)
            sink.close()

    threading.Thread(target=pump, name="infb-launch-log", daemon=True).start()
    writer = os.fdopen(write_end, "w", buffering=1)
    saved = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = writer
    try:
        yield
    finally:
        sys.stdout, sys.stderr = saved
        writer.close()  # the child holds its own copies; the pump sees EOF when every rank has exited


def launch() -> Iterator:
    """Yield a healthy RemoteOpenAIServer at the recipe; shut it down afterwards."""
    from tests.utils import RemoteOpenAIServer  # noqa: PLC0415  (the upstream tree at /app)

    with _tee_child_output(LAUNCH_LOG):
        server = RemoteOpenAIServer(MODEL, SERVER_ARGS, max_wait_seconds=STARTUP_MAX_WAIT_SECONDS)
    with server:
        yield server


RELAUNCH_LOG = LAUNCH_LOG.with_name("candidate-launch-2.log")


def stop(server) -> None:
    """The relaunch node's stop: the session server shut down exactly as the fixture shuts it down at the session's
    end (upstream's `_shutdown`: the process tree killed, then a wait for the GPU memory it held)."""
    server.__exit__(None, None, None)


def down(server, timeout_s: float = 60.0) -> str | None:
    """None while the server answers its health endpoint; otherwise why not. A served node asks this first: a node that finds the server already down is `blocked_by:server_unreachable`, a check after a
    dead server rather than another model defect, and the dead server is the primary cause where it was first seen --
    the node during which it stopped answering, a candidate failure."""
    import urllib.error  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    proc = getattr(server, "proc", None)
    rc = proc.poll() if proc is not None else None
    if rc is not None:
        return f"the server process exited with code {rc}"
    try:
        with urllib.request.urlopen(server.url_for("health"), timeout=timeout_s) as r:
            return None if r.status == 200 else f"GET /health answered HTTP {r.status}"
    except (urllib.error.URLError, OSError) as exc:
        return f"GET /health failed: {type(exc).__name__}: {exc}"[:300]


#: The message a served node fails with when `down` answers: the status first, so the reward's disposition (and a reader
#: of the junit report) sees the block without reading the record.
def blocked_message(why: str) -> str:
    return (f"blocked_by:server_unreachable: {why} when this node started -- a check after a dead server, not another "
            f"model defect; the primary cause is the node during which the server stopped answering")


def relaunch(server) -> str:
    """The relaunch node's start: the same RemoteOpenAIServer object launched again at the recipe, exactly as
    `launch()` launched it -- upstream's own `__init__`, which starts a fresh server on a fresh free port and waits for
    its health -- with every rank's launch log in RELAUNCH_LOG. The fixture's own exit stops this launch at the
    session's end. Returns the new base URL."""
    with _tee_child_output(RELAUNCH_LOG):
        type(server).__init__(server, MODEL, SERVER_ARGS, max_wait_seconds=STARTUP_MAX_WAIT_SECONDS)
    return server.url_root


_SC = _trusted.load("tests/served_configuration.py")
STATED_CONFIGURATION = _SC.STATED_CONFIGURATION
ATTENTION_BACKEND_FAMILY = _SC.ATTENTION_BACKEND_FAMILY
served_configuration = _SC.served_configuration


def launch_argv(server) -> list[str]:
    """The argv upstream's wrapper started this server with (`vllm serve <model> <args> --port <free> --seed 0`), its
    free port masked: every launch draws its own."""
    argv = [str(a) for a in server.proc.args]
    return ["<port>" if i and argv[i - 1] == "--port" else a for i, a in enumerate(argv)]


def relaunch_check(server, first_argv: list[str]) -> dict:
    """The check of the session server's second launch: RELAUNCH_LOG read exactly as
    `test_served_configuration` reads LAUNCH_LOG, beside that same reading of the first launch, and the argv against the
    first launch's. `ok` only when the second launch shows no departure from the recipe (no expert parallelism among
    them) and serves what the first served -- the same engine configuration lines and the same attention backend
    on every rank -- from the same argv. Otherwise its draws do not count, and the relaunch is flagged."""
    def read(path: Path) -> dict:
        return served_configuration(path.read_text(errors="replace") if path.exists() else "")
    first, second = read(LAUNCH_LOG), read(RELAUNCH_LOG)
    argv = launch_argv(server)
    argv_diff = [f"{i}: {a!r} -> {b!r}" for i, (a, b) in enumerate(itertools.zip_longest(first_argv, argv)) if a != b]
    same = {k: first[k] == second[k] for k in ("engine_config_lines", "attention_backends")}
    ok = bool(second["engine_config_lines"]) and not second["departures"] and all(same.values()) and not argv_diff
    return {"ok": ok, "argv": argv, "argv_diff": argv_diff, "same_as_first_launch": same,
            "departures": second["departures"], "engine_config_lines": second["engine_config_lines"],
            "attention_backends": second["attention_backends"], "breakable_lines": second["breakable_lines"],
            "expert_parallel_lines": second["expert_parallel_lines"],
            "first_launch": {k: first[k] for k in ("departures", "attention_backends", "breakable_lines")}}
