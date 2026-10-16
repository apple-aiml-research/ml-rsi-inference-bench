#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Task-owned launches of the two public deployments, and the dead-server rule.

The verifier serves exactly the public recipe (`serving/recipes.py`, from the trusted copy under /tests/eval) with
two additions made identically for every implementation it serves, reference included: `VLLM_SERVER_DEV_MODE=1`
(for `/reset_prefix_cache`) and an idle torch profiler (`--profiler-config`, for the execution witness). Nothing else
changes the recipe's argv or environment.

A node that finds the server dead is blocked by the node that first saw it die, never failed as a second,
independent model defect.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
for _path in (str(TESTS_DIR / "eval"), str(TESTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from minimax_m3.serving.server import Server  # noqa: E402
from infb_verify import deadline_dag  # noqa: E402
from infb_verify.records import Blocked  # noqa: E402

LOGS = Path(os.environ.get("INFB_LAUNCH_LOG_DIR", "/logs/verifier"))
ARMS = Path(os.environ.get("INFB_ARMS_DIR", "/tmp/infb-arms"))
#: Build and JIT roots set arm-private before either arm imports a runtime; each name was confirmed
#: against the pinned package's source.
ARM_VARIABLES = ("TRITON_CACHE_DIR", "TORCH_EXTENSIONS_DIR", "TILELANG_CACHE_DIR", "TILELANG_TMP_DIR",
                 "MM_SPARSE_ATTN_AOT_CACHE", "CUTE_DSL_CACHE_DIR", "FLASHINFER_WORKSPACE_BASE", "VLLM_CACHE_ROOT",
                 "XDG_CACHE_HOME")


def arm_env(arm: str) -> dict[str, str]:
    """{variable: an arm-private directory} for `arm` ("reference" or "candidate"), created."""
    root = ARMS / arm
    out = {}
    for name in ARM_VARIABLES:
        path = root / name.lower()
        path.mkdir(parents=True, exist_ok=True)
        out[name] = str(path)
    return out


def profiler_args(trace_dir: Path) -> list[str]:
    return ["--profiler-config", json.dumps({"profiler": "torch", "torch_profiler_dir": str(trace_dir),
                                              "torch_profiler_with_stack": False})]


class Session:
    """One launch at a deployment, and what later nodes need to know about it."""

    def __init__(self, deployment: str):
        self.deployment = deployment
        self.trace_dir = Path(tempfile.mkdtemp(prefix=f"infb-m3-{deployment}-traces-"))
        self.launch_log = LOGS / deployment / "candidate-launch.log"
        self.server = Server(deployment, extra_args=profiler_args(self.trace_dir),
                             env={"VLLM_SERVER_DEV_MODE": "1", **arm_env("candidate")},
                             launch_log=self.launch_log,
                             max_wait_seconds=deadline_dag.limits()["startup_s"])
        self.dead_at: str | None = None
        self.launch_error: str | None = None

    @property
    def url(self) -> str:
        return self.server.url_root

    def alive(self) -> bool:
        proc = self.server.proc
        if proc is None or proc.poll() is not None:
            return False
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(self.server.url_for("health"), timeout=30) as response:
                return response.status == 200
        except OSError:
            return False

    def require(self, node: str) -> None:
        """Before a node's first request. A failed launch is the first node's failure (the candidate's server did not
        become healthy within the startup limit); every later node is blocked by that one, as by a server that an
        earlier node saw die."""
        if self.launch_error is not None and self.dead_at is None:
            self.dead_at = f"infb_verify.test_{self.deployment}.{node} (launch failed)"
            raise AssertionError(f"the {self.deployment} server did not become healthy: {self.launch_error[:3000]}")
        if self.dead_at is not None:
            raise Blocked(self.dead_at)
        if not self.alive():
            self.dead_at = f"infb_verify.test_{self.deployment}.{node} (server dead before the node began)"
            raise Blocked(self.dead_at)

    def after(self, node: str) -> None:
        """After a node: if the server died during it, later nodes are blocked by this one."""
        if self.dead_at is None and not self.alive():
            self.dead_at = f"infb_verify.test_{self.deployment}.{node}"


def launch(deployment: str):
    session = Session(deployment)
    try:
        session.server.__enter__()
    except BaseException as exc:  # a launch failure fails the first node, and blocks the rest
        session.launch_error = f"{type(exc).__name__}: {exc}"
        yield session
        return
    try:
        yield session
    finally:
        session.server.close()
