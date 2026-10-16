#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""A served DiffusionGemma recipe as a context manager: the launch the verifier uses, public."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import recipes

NOPROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))
ARM_DIRS = {
    "TRITON_CACHE_DIR": "triton", "TORCHINDUCTOR_CACHE_DIR": "inductor", "VLLM_CACHE_ROOT": "vllm",
    "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": "fa-cute", "CUTE_DSL_CACHE_DIR": "cute-dsl",
    "TVM_FFI_CACHE_DIR": "tvm-ffi", "TORCH_EXTENSIONS_DIR": "torch-extensions",
    "FLASHINFER_WORKSPACE_BASE": "flashinfer", "XDG_CACHE_HOME": "xdg", "TMPDIR": "tmp",
}


def arm_environment(arm_root: str | os.PathLike) -> dict[str, str]:
    """The arm-private roots (created on first use)."""
    env = {}
    for var, sub in ARM_DIRS.items():
        path = Path(arm_root) / sub
        path.mkdir(parents=True, exist_ok=True)
        env[var] = str(path)
    return env


def profiler_args(trace_dir: str | os.PathLike) -> list[str]:
    """An idle torch profiler writing CUDA kernel traces to trace_dir. Python stacks and the CUDA-time
    table are off: the witness reads kernel names only, and with stacks one window's export took
    443.7 s on the reference (a 187.6 MB trace)."""
    return ["--profiler-config", json.dumps({"profiler": "torch", "torch_profiler_dir": str(trace_dir),
                                             "torch_profiler_with_stack": False,
                                             "torch_profiler_dump_cuda_time_total": False})]


class Client:
    """Requests to a running server at `url` (the HTTP half of Server; run.py uses it directly)."""

    def __init__(self, url: str, log_path: str | os.PathLike | None = None) -> None:
        self.url = url.rstrip("/")
        self.log_path = Path(log_path) if log_path else None

    def post(self, path: str, body: dict, timeout: float = 1800.0) -> dict:
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.monotonic()
        try:
            with NOPROXY.open(req, timeout=timeout) as r:
                raw = r.read()
                out = json.loads(raw) if raw else {}
                out["_status"] = r.status
        except urllib.error.HTTPError as e:
            out = {"_status": e.code, "_error": e.read().decode(errors="replace")[:4000]}
        except Exception as e:  # noqa: BLE001 -- recorded, never raised
            out = {"_status": None, "_error": f"{type(e).__name__}: {e}"[:4000]}
        out["_wall_s"] = round(time.monotonic() - t0, 4)
        return out

    def stream(self, path: str, body: dict, timeout: float = 1800.0) -> tuple[int | None, list[dict], float]:
        """A streaming request: (status, the decoded SSE chunks, wall)."""
        body = dict(body, stream=True)
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.monotonic()
        chunks: list[dict] = []
        try:
            with NOPROXY.open(req, timeout=timeout) as r:
                status = r.status
                for line in r:
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        break
                    chunks.append(json.loads(data))
        except urllib.error.HTTPError as e:
            return e.code, [{"_error": e.read().decode(errors="replace")[:4000]}], time.monotonic() - t0
        except Exception as e:  # noqa: BLE001
            return None, [{"_error": f"{type(e).__name__}: {e}"}], time.monotonic() - t0
        return status, chunks, time.monotonic() - t0

    def get(self, path: str, timeout: float = 30.0) -> tuple[int | None, str]:
        try:
            with NOPROXY.open(self.url + path, timeout=timeout) as r:
                return r.status, r.read().decode(errors="replace")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode(errors="replace")
        except Exception as e:  # noqa: BLE001
            return None, f"{type(e).__name__}: {e}"

    def reset_prefix_cache(self) -> int | None:
        req = urllib.request.Request(self.url + "/reset_prefix_cache", data=b"", method="POST")
        try:
            with NOPROXY.open(req, timeout=60) as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code
        except Exception:  # noqa: BLE001
            return None

    def metric(self, name: str) -> float:
        """The sum of a Prometheus counter or gauge over its label sets."""
        status, text = self.get("/metrics")
        total = 0.0
        for line in text.splitlines():
            if line.startswith(name) and not line.startswith("#"):
                head, _, value = line.rpartition(" ")
                if head.split("{")[0] == name:
                    try:
                        total += float(value)
                    except ValueError:
                        pass
        return total


class Server(Client):
    def __init__(self, recipe: str, *, port: int = 8000, arm_root: str | os.PathLike = "/tmp/dgemma-arm",
                 gencfg_root: str | os.PathLike | None = None, log_path: str | os.PathLike | None = None,
                 extra_args: list[str] | tuple = (), env: dict | None = None, max_wait_s: float = 1800.0,
                 devices: str | None = None, profile_dir: str | os.PathLike | None = None,
                 cwd: str = "/app") -> None:
        self.recipe, self.port, self.max_wait_s, self.cwd = recipe, port, max_wait_s, cwd
        self.arm_root = Path(arm_root)
        self.gencfg_root = gencfg_root
        self.log_path = Path(log_path) if log_path else self.arm_root / "server.log"
        self.url = f"http://127.0.0.1:{port}"
        args = recipes.argv(recipe, gencfg_root=gencfg_root, port=port) + list(extra_args)
        if profile_dir is not None:
            args += profiler_args(profile_dir)
        self.argv = args
        self.env = dict(os.environ)
        self.env.update(arm_environment(self.arm_root))
        self.env.update({"VLLM_SERVER_DEV_MODE": "1", "CUDA_VISIBLE_DEVICES": devices or recipes.devices(recipe)})
        if env:
            self.env.update(env)
        self.proc: subprocess.Popen | None = None
        self.ready_s: float | None = None

    def __enter__(self) -> "Server":
        if self.gencfg_root is not None:
            recipes.write_gencfg(self.gencfg_root)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "w")
        t0 = time.monotonic()
        self.proc = subprocess.Popen(self.argv, cwd=self.cwd, env=self.env, stdout=self._log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        while time.monotonic() - t0 < self.max_wait_s:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited with {self.proc.returncode} before becoming healthy "
                                   f"(log {self.log_path})")
            try:
                with NOPROXY.open(self.url + "/health", timeout=5) as r:
                    if r.status == 200:
                        self.ready_s = round(time.monotonic() - t0, 1)
                        return self
            except Exception:  # noqa: BLE001 -- not up yet
                pass
            time.sleep(2)
        self.__exit__(None, None, None)
        raise TimeoutError(f"server not healthy within {self.max_wait_s} s (log {self.log_path})")

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def __exit__(self, *exc) -> None:
        if self.proc is not None and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=10)  # budgets.json server_cleanup: TERM, KILL after 10 s, reap
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(timeout=30)
        if getattr(self, "_log", None):
            self._log.close()
