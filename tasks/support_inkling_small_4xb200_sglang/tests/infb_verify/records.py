#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Attempt records for the hidden nodes: every request a node sends is written
here as it completes, before any verdict is raised, with its identities, expected and observed spans, structural
validity, numerical result, elapsed time and status.

  * `<record dir>/<check>.jsonl` holds one line per attempt; the raw served entries go to
    `<record dir>/<check>-raw.jsonl.gz` (one line per attempt, keyed by check, id and draw), so a malformed answer can
    be read back position by position.
  * Statuses: `measured` (a structurally valid answer and its statistic), `candidate_fail` (a malformed answer, a
    refused request, or an operation over its limit), `reference_fault`, and, for a server that stopped answering,
    `candidate_fail` on the first node that saw it and `blocked_by:<that node>` on every later one: a dead server is
    one finding, not a new model defect per node.

The record directory is `INFB_RECORD_DIR`, by default `/logs/verifier/records`.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import time
from pathlib import Path

_FIRST_UNREACHABLE: dict[str, str] = {}


def record_dir() -> Path:
    return Path(os.environ.get("INFB_RECORD_DIR", "/logs/verifier/records"))


def _finite_or_text(value):
    """JSON has no infinities: +inf is written as the string "inf" (a numerical +infinity, not a malformed value)."""
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else ("-inf" if value < 0 else "nan")
    if isinstance(value, dict):
        return {k: _finite_or_text(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_or_text(v) for v in value]
    return value


def _config_digest() -> str | None:
    path = Path(os.environ.get("TESTS_DIR", "/tests")) / "config.json"
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


class AttemptLog:
    """The `on_attempt` callback for one node: `AttemptLog(node, check, server_pid=...)`."""

    def __init__(self, node: str, check: str, server_pid: int | None = None):
        self.node, self.check, self.server_pid = node, check, server_pid
        self.attempts: list[dict] = []

    def __call__(self, attempt: dict) -> None:
        attempt = dict(attempt)
        raw = attempt.pop("raw", None)
        for side in ("cold", "warm"):
            if isinstance(attempt.get(side), dict) and "raw" in attempt[side]:
                attempt[side] = dict(attempt[side])
                raw = {**(raw if isinstance(raw, dict) else {}), side: attempt[side].pop("raw")}
        if attempt.get("status") == "unreachable":
            first = _FIRST_UNREACHABLE.setdefault("node", self.node)
            attempt["status"] = "candidate_fail" if first == self.node else f"blocked_by:{first}"
            attempt["cause"] = "the server stopped answering"
        attempt.update({"node": self.node, "recorded_at": time.time(), "verifier_pid": os.getpid(),
                        "server_pid": self.server_pid, "config_sha256": _config_digest()})
        self.attempts.append(attempt)
        d = record_dir()
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{self.check}.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(_finite_or_text(attempt), sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        if raw is not None:
            with gzip.open(d / f"{self.check}-raw.jsonl.gz", "at", encoding="utf-8") as fh:
                fh.write(json.dumps({"node": self.node, "id": attempt.get("id"), "draw": attempt.get("draw"),
                                     "raw": _finite_or_text(raw)}) + "\n")
