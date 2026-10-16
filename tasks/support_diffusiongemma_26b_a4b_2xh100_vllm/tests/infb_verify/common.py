#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Shared machinery of the DiffusionGemma verifier: paths, hidden inputs, bars, records, sessions.

A session is one launch of one recipe: S-F1 (tp1_fixed), S-A (tp1), S-F2 (tp2_fixed), S-AR
(ar_smoke), in series. `session(recipe)` returns the live server, stopping any other recipe's
first; a server that dies makes every later node of its session fail as `blocked_by` its death, and
no node relaunches a server.

Calibration mode (`INFB_CALIBRATION=1`, or no frozen policy): the statistic nodes record their
measurements for the bar construction and pass; graded runs read the frozen bars.
"""
from __future__ import annotations

import json
import lzma
import os
import sys
import time
from pathlib import Path

TESTS = Path(os.environ.get("TESTS_DIR", "/tests"))
EVAL_ROOT = Path(os.environ.get("INFB_EVAL_ROOT", str(TESTS)))  # holds eval/diffusion_gemma (trusted copy)
HIDDEN = Path(os.environ.get("INFB_HIDDEN_DIR", str(TESTS / "hidden")))
RECORDS = Path(os.environ.get("INFB_RECORD_DIR", "/logs/verifier/records"))
ARM_ROOT = Path(os.environ.get("INFB_ARM_ROOT", "/tmp/infb-candidate-arm"))
GENCFG_ROOT = Path(os.environ.get("INFB_GENCFG_ROOT", "/opt/infb/gencfg"))
for _p in (str(EVAL_ROOT), str(TESTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval.diffusion_gemma.serving import recipes  # noqa: E402
from eval.diffusion_gemma.serving.server import Server  # noqa: E402

CALIBRATION = os.environ.get("INFB_CALIBRATION") == "1"
PORT = int(os.environ.get("INFB_SERVE_PORT", "8000"))


def hidden(name: str):
    path = HIDDEN / name
    raw = path.read_bytes()
    return json.loads(lzma.decompress(raw) if path.suffix == ".xz" else raw)


def verifier_version() -> str | None:
    """config.json's infb.verifier_version: the revision of the bodies this verifier sends; None without config.json."""
    try:
        return (json.loads((TESTS / "config.json").read_text()).get("infb") or {}).get("verifier_version")
    except (OSError, ValueError, AttributeError):
        return None


def policy() -> dict | None:
    """The frozen bars, or None in calibration mode."""
    path = HIDDEN / "policy.json"
    if CALIBRATION or not path.exists():
        return None
    return json.loads(path.read_text())


def record(node: str, data: dict) -> Path:
    RECORDS.mkdir(parents=True, exist_ok=True)
    path = RECORDS / (node.replace("/", "_").replace("[", "_").replace("]", "") + ".json")
    path.write_text(json.dumps(data, indent=1, default=str))
    return path


class _Sessions:
    def __init__(self) -> None:
        self.recipe: str | None = None
        self.server: Server | None = None
        self.dead: dict[str, str] = {}
        self.launch_s: dict[str, float] = {}

    def get(self, recipe: str, max_wait_s: float | None = None, extra_args=(), profile_dir=None) -> Server:
        import pytest

        if recipe in self.dead:
            pytest.fail(f"blocked_by: the {recipe} server {self.dead[recipe]}", pytrace=False)
        if self.recipe == recipe and self.server is not None:
            if not self.server.alive():
                self.dead[recipe] = f"died (exit {self.server.proc.returncode}); see {self.server.log_path}"
                pytest.fail(f"blocked_by: the {recipe} server {self.dead[recipe]}", pytrace=False)
            return self.server
        self.stop()
        from infb_verify import budgets

        wait = max_wait_s if max_wait_s is not None else budgets.seconds("server_launch")
        # Every verifier launch carries an idle torch profiler identically in both arms.
        prof = profile_dir or self.profile_root(recipe)
        srv = Server(recipe, port=PORT, arm_root=ARM_ROOT / recipe, gencfg_root=GENCFG_ROOT,
                     log_path=Path("/logs/verifier") / f"server-{recipe}.log", max_wait_s=wait,
                     extra_args=list(extra_args), profile_dir=prof)
        t0 = time.monotonic()
        try:
            srv.__enter__()
        except Exception as exc:  # noqa: BLE001
            self.dead[recipe] = f"failed to launch: {exc}"
            pytest.fail(f"launch of {recipe} failed: {exc}", pytrace=False)
        self.launch_s[recipe] = round(time.monotonic() - t0, 1)
        self.recipe, self.server = recipe, srv
        return srv

    @staticmethod
    def profile_root(recipe: str) -> Path:
        path = Path(os.environ.get("INFB_PROFILE_ROOT", "/logs/verifier/profiles")) / recipe
        path.mkdir(parents=True, exist_ok=True)
        return path

    def stop(self) -> None:
        if self.server is not None:
            self.server.__exit__(None, None, None)
        self.recipe, self.server = None, None


SESSIONS = _Sessions()
