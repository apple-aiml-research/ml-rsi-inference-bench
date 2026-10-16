#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Bounded local launch of a deployment, its raw log and cleanup, shared by the public protocols."""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path

from . import recipes

LIMITS = json.loads((Path(__file__).resolve().parent.parent / "limits.json").read_text())


class Server:
    def __init__(self, deployment: str = "bf16kv", *, extra_args=(), env=None, launch_log=None, cwd="/app",
                 max_wait_seconds=None):
        self.deployment = deployment
        self.args = [*recipes.server_args(deployment), *extra_args]
        self.env = dict(os.environ)
        self.env.update(env or {})
        self.cwd = cwd
        self.max_wait_seconds = max_wait_seconds or LIMITS["startup_s"]
        self.launch_log = Path(launch_log or f"/tmp/minimax-m3-server-{uuid.uuid4().hex}.log")
        self.proc = None
        self.log = None
        if "--port" in self.args:
            self.port = int(self.args[self.args.index("--port") + 1])
        else:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                self.port = sock.getsockname()[1]
            self.args.extend(["--port", str(self.port)])
        if "--seed" not in self.args:
            self.args.extend(["--seed", "0"])
        self.argv = ["vllm", "serve", recipes.MODEL, *self.args]
        self.url_root = f"http://127.0.0.1:{self.port}"
        self.healthy_s = None

    def url_for(self, path: str) -> str:
        return self.url_root + "/" + path.lstrip("/")

    def __enter__(self):
        self.launch_log.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.launch_log.open("w")
        try:
            self.proc = subprocess.Popen(self.argv, cwd=self.cwd, env=self.env, stdout=self.log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
            started = time.monotonic()
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            while time.monotonic() - started < self.max_wait_seconds:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"server exited {self.proc.returncode}; log: {self.launch_log}")
                try:
                    with opener.open(self.url_for("health"), timeout=5) as response:
                        if response.status == 200:
                            self.healthy_s = time.monotonic() - started
                            return self
                except (OSError, ValueError):
                    pass
                time.sleep(2)
            raise TimeoutError(f"server startup exceeded {self.max_wait_seconds}s; log: {self.launch_log}")
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.proc is not None:
            import psutil
            try:
                children = psutil.Process(self.proc.pid).children(recursive=True)
            except psutil.NoSuchProcess:
                children = []
            if self.proc.poll() is None:
                try:
                    os.killpg(self.proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                    self.proc.wait(timeout=10)
            _, alive = psutil.wait_procs(children, timeout=10)
            for child in alive:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(alive, timeout=10)
            if alive:
                raise RuntimeError(f"server cleanup left PIDs {[p.pid for p in alive]}")
            self.proc = None
        if self.log is not None:
            self.log.close()
            self.log = None

    def __exit__(self, *_):
        self.close()
