#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Continued service: one server per deployment class, and what a server that died in one node blocks in the next."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

try:
    from glm53_protocols.limits import ACTION_S
except ImportError:  # pragma: no cover - a module imported outside the verifier image
    ACTION_S = 300

RECORD_DIR = Path(os.environ.get("INFB_SERVICE_DIR", "/logs/verifier/service"))


def read_server(process, base_url: str, timeout: float = ACTION_S) -> dict:
    """Is the class's server alive: its launch process has not exited and its HTTP front answers `/health` with 200."""
    out: dict = {"t": time.time(), "process_alive": None, "returncode": None, "health": None}
    if process is not None:
        rc = process.poll()
        out["process_alive"], out["returncode"] = rc is None, rc
    try:
        r = requests.get(base_url + "/health", timeout=timeout)
        out["health"] = r.status_code
    except requests.RequestException as exc:
        out["health"] = f"error: {type(exc).__name__}: {str(exc)[:200]}"
    out["alive"] = out["process_alive"] is not False and out["health"] == 200
    return out


class ServiceTracker:
    def __init__(self, label: str):
        self.label = label
        self.died_in: str | None = None
        self.readings: list[dict] = []

    def _record(self) -> None:
        try:
            RECORD_DIR.mkdir(parents=True, exist_ok=True)
            (RECORD_DIR / f"{self.label}.json").write_text(json.dumps(
                {"label": self.label, "died_in": self.died_in, "readings": self.readings}, indent=1, default=str))
        except OSError:
            pass

    def before(self, node: str, process, base_url: str) -> str | None:
        """None when the node may run; otherwise its status: `blocked_by:<node>` after a recorded death, or
        `candidate_fail` when this node is the first to find the server gone."""
        if self.died_in is not None:
            self.readings.append({"node": node, "when": "before", "skipped_read": True, "blocked_by": self.died_in})
            self._record()
            return f"blocked_by:{self.died_in}"
        reading = read_server(process, base_url)
        self.readings.append({"node": node, "when": "before", **reading})
        if not reading["alive"]:
            self.died_in = node
            self.readings[-1]["primary"] = True
            self._record()
            return "candidate_fail"
        self._record()
        return None

    def after(self, node: str, process, base_url: str) -> None:
        if self.died_in is not None:
            return
        reading = read_server(process, base_url)
        self.readings.append({"node": node, "when": "after", **reading})
        if not reading["alive"]:
            self.died_in = node
        self._record()
