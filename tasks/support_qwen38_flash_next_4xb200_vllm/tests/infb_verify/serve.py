#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Task-owned launches at the two public recipes.

The grading launch serves the recipe with the measurement worker extension (the private
`GradingWorker`: the public `MeasurementWorker` with independent expected outputs); nothing
else changes the recipe's argv or environment, and the reference phase launches the same way.
The MTP3 production/state diagnostic is a separate launch: `DiagnosticWorker`, the native
interventions' mapping, iteration logging and a fresh compile cache, removed afterwards.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
X = json.loads((TESTS_DIR / "config.json").read_text())["infb"]
for _path in (str(TESTS_DIR / "eval"), str(TESTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from qwen38.serving.recipes import MTP_ARGS, SERVER_ARGS  # noqa: E402
from qwen38.serving.server import Server  # noqa: E402
from infb_verify import budgets  # noqa: E402

# The port is chosen when a Server is constructed and bound by vLLM seconds later, so another socket can take it in
# between. Only that failure (an exit before health whose launch log says the address is in use) repeats the
# launch on a fresh port, at most LAUNCH_ATTEMPTS times. Each such log is kept beside the launch log, and
# <launch log stem>.attempts.json records every launch's attempt count and outcome. No measurement is repeated.
PORT_IN_USE = "Address already in use"
LAUNCH_ATTEMPTS = 3
GRADING_ARGS = ["--worker-extension-cls", "infb_verify.qwen38_private.worker.GradingWorker"]
DIAGNOSTIC_ARGS = ["--worker-extension-cls", "infb_verify.qwen38_private.worker.DiagnosticWorker",
                   "--enable-logging-iteration-details"]
LAUNCH_LOG = Path(os.environ.get("INFB_LAUNCH_LOG_DIR", "/logs/verifier")) / "candidate-launch.log"


def environment(tree: str = "/app") -> dict:
    """The measurement interface's environment: the trusted protocol copy first, then the tree, then
    the private verifier package (for the worker extension)."""
    return {"VLLM_SERVER_DEV_MODE": "1", "QWEN38_ENGINE_ROOT": tree,
            "PYTHONPATH": os.pathsep.join([str(TESTS_DIR / "eval"), tree, str(TESTS_DIR)])}


def launch(extra_args=(), launch_log=LAUNCH_LOG, native_mapping=None):
    """One server at a recipe (`extra_args` = `MTP_ARGS` for MTP3); a native mapping makes it the
    separate production diagnostic launch."""
    env = environment()
    args = list(GRADING_ARGS)
    cache = None
    if native_mapping is not None:
        # An AOT artifact traced before observer installation omits its device operators, so the
        # diagnostic launch compiles into its own cache.
        cache = tempfile.mkdtemp(prefix="qwen38-native-compile-")
        env.update(QWEN38_NATIVE_MAPPING=str(native_mapping), VLLM_CACHE_ROOT=cache)
        args = list(DIAGNOSTIC_ARGS)
    log, kept = Path(launch_log), []

    def record(attempts, outcome):
        log.parent.mkdir(parents=True, exist_ok=True)
        log.with_name(f"{log.stem}.attempts.json").write_text(
            json.dumps({"attempts": attempts, "outcome": outcome, "port_in_use_logs": kept}) + "\n")

    try:
        for attempt in range(1, LAUNCH_ATTEMPTS + 1):
            started = False
            try:
                with Server(X["model_id"], SERVER_ARGS + list(extra_args) + args,
                            max_wait_seconds=budgets.seconds("server_launch"),
                            launch_log=launch_log, env_dict=env) as server:
                    started = True
                    record(attempt, "healthy")
                    yield server
                return
            except BaseException as exc:
                if started:
                    raise
                if not (isinstance(exc, RuntimeError) and attempt < LAUNCH_ATTEMPTS and log.is_file()
                        and PORT_IN_USE in log.read_text(errors="replace")):
                    record(attempt, f"failed: {exc!r}"[:500])
                    raise
                kept.append(f"{log.stem}.port-in-use-{attempt}{log.suffix}")
                shutil.copyfile(log, log.with_name(kept[-1]))
    finally:
        if cache is not None:
            shutil.rmtree(cache, ignore_errors=True)


from qwen38.serving.configuration import served_configuration  # noqa: E402


_METRIC = re.compile(r"^(?P<name>[a-zA-Z_:][\w:]*)(?P<labels>\{[^}]*\})?\s+(?P<value>[-+0-9.eE]+)\s*$")


def counter_total(metrics_text: str, name: str) -> float:
    """Sum a Prometheus counter over every label set.

    Label-aware on purpose: vLLM emits these per engine and per model, so a bare-name lookup finds
    nothing and reads as "no hits". Both the bare name and the `_total` spelling are accepted,
    because which one appears depends on the exporter.
    """
    wanted = {name, f"{name}_total"}
    total = 0.0
    seen = False
    for line in metrics_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _METRIC.match(line)
        if m and m.group("name") in wanted:
            total += float(m.group("value"))
            seen = True
    return total if seen else float("nan")
