#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The verifier's attempt records (the common evidence contract): every attempt is persisted before its verdict."""
from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from pathlib import Path
from typing import Any

TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
RECORD_ROOT = Path(os.environ.get("INFB_RECORD_DIR", "/logs/verifier/records"))
STATUSES = ("pass", "candidate_fail", "reference_fault", "environment_fault", "verifier_fault", "blocked")

_SOURCES = ("config.json", "infb_verify/records.py", "infb_verify/service.py", "infb_verify/fidelity.py",
            "infb_verify/test_hidden.py", "glm53_protocols/fidelity.py", "glm53_protocols/limits.py")


def valid_status(status: str) -> bool:
    return status in STATUSES or (status.startswith("blocked_by:") and len(status) > len("blocked_by:"))


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def identities(extra_sources: tuple[str, ...] = ()) -> dict:
    """What every record names: the task, its label and base, and the bytes of the verifier code that judged it."""
    cfg: dict = {}
    try:
        cfg = json.loads((TESTS_DIR / "config.json").read_text())
    except (OSError, ValueError):
        pass
    infb = cfg.get("infb") or {}
    return {"task": infb.get("task"), "score_profile": (infb.get("efficiency") or {}).get("score_profile"),
            "base_commit": cfg.get("base_commit"), "model_revision": infb.get("model_revision"),
            "sources": {rel: _sha256(TESTS_DIR / rel) for rel in (*_SOURCES, *extra_sources)},
            "host": socket.gethostname(), "pid": os.getpid()}


def write_record(check: str, label: str, attempt: str, payload: dict) -> str:
    """Persist one attempt's record and return its path (or the reason it could not be written)."""
    status = payload.get("status")
    if status is not None and not valid_status(status):
        payload = {**payload, "status": "verifier_fault",
                   "status_error": f"a record was written with the unknown status {status!r}"}
    rec = {"check": check, "label": label, "attempt": attempt, "written_at": time.time(),
           "identities": identities(), **payload}
    try:
        out = RECORD_ROOT / check / label
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{attempt}.json"
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(rec, indent=1, sort_keys=True, default=str))
        os.replace(tmp, path)
        return str(path)
    except Exception as exc:  # noqa: BLE001 - a record, never a verdict
        return f"not written: {exc!r}"[:300]


def summarize_status(statuses: list[str]) -> str:
    """The node's status from its attempts' statuses, in order: the first non-pass decides, unless a later attempt is
    allowed to pass (the caller decides that and passes only the attempts that count)."""
    for s in statuses:
        if s != "pass":
            return s
    return "pass" if statuses else "verifier_fault"


def compact(value: Any, limit: int = 4000) -> Any:
    """A raw output small enough for a record: text is cut at `limit` characters and marked."""
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[:limit] + f"... [{len(text) - limit} more characters]"
