#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Attempt records, persisted before any verdict.

Every attempt carries the task, configuration and source identities, the request or workload identity, what was
expected and what was observed (raw outputs, or their digests and file paths when large), structural validity, the
numerical result, the elapsed time, process identities and any `blocked_by` cause. Status is one of `pass`,
`candidate_fail`, `reference_fault`, `environment_fault`, `verifier_fault`, `blocked_by:<id>`; missing evidence never
passes. A node reads its records back only to decide; it never edits one.
"""
from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
RECORDS = Path(os.environ.get("INFB_RECORDS_DIR", "/logs/verifier/records"))
STATUSES = ("pass", "candidate_fail", "reference_fault", "environment_fault", "verifier_fault")


class Blocked(Exception):
    """A prerequisite record is unavailable for a reason other than this node's own candidate failure."""

    def __init__(self, cause: str):
        super().__init__(f"blocked_by:{cause}")
        self.cause = cause


class VerifierFault(Exception):
    """The verifier could not observe (e.g. a conforming implementation it cannot attribute); never the candidate's."""


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@functools.cache
def identities() -> dict:
    """Task, configuration and source identities, read once per process."""
    out = {"config_sha256": sha256_file(TESTS_DIR / "config.json")}
    with contextlib.suppress(Exception):
        out["task"] = json.loads((TESTS_DIR / "config.json").read_text())["infb"]["task_id"]
    app = Path(os.environ.get("APP_DIR", "/app"))
    with contextlib.suppress(Exception):
        index = Path("/tmp") / f"infb-records-index-{os.getpid()}"
        env = dict(os.environ, GIT_INDEX_FILE=str(index))
        subprocess.run(["git", "-C", str(app), "read-tree", "HEAD"], env=env, check=True, capture_output=True)
        subprocess.run(["git", "-C", str(app), "add", "-A"], env=env, check=True, capture_output=True)
        out["tree"] = subprocess.run(["git", "-C", str(app), "write-tree"], env=env, check=True, capture_output=True,
                                     text=True).stdout.strip()
        out["base_commit"] = subprocess.run(["git", "-C", str(app), "rev-parse", "HEAD"], check=True,
                                            capture_output=True, text=True).stdout.strip()
        index.unlink(missing_ok=True)
    return out


class Attempt:
    def __init__(self, recorder: "Recorder", node: str, n: int, workload: dict):
        self.row = {"node": node, "attempt": n, "group": recorder.group, "started": utc(), "workload": workload,
                    "identities": identities(), "pid": os.getpid(), "status": None}
        self._t0 = time.monotonic()
        self._recorder = recorder

    def set(self, **fields) -> None:
        self.row.update(fields)

    def close(self, status: str, **fields) -> dict:
        if status not in STATUSES and not status.startswith("blocked_by:"):
            raise ValueError(f"unknown status {status!r}")
        self.row.update(fields, status=status, ended=utc(), elapsed_s=round(time.monotonic() - self._t0, 3))
        self._recorder.write(self.row)
        return self.row


class Recorder:
    """One JSONL file per group: every attempt appended the moment it ends (and a start marker when it begins)."""

    def __init__(self, group: str):
        self.group = group
        self.path = RECORDS / f"{group}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._counts: dict[str, int] = {}

    def write(self, row: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str, allow_nan=True) + "\n")
            fh.flush()

    @contextlib.contextmanager
    def attempt(self, node: str, **workload):
        """Yields an Attempt; an exception closes it as the candidate's failure unless it names another cause."""
        n = self._counts[node] = self._counts.get(node, 0) + 1
        a = Attempt(self, node, n, workload)
        self.write({"node": node, "attempt": n, "group": self.group, "event": "start", "started": a.row["started"]})
        try:
            yield a
        except Blocked as exc:
            a.close(f"blocked_by:{exc.cause}", error=str(exc))
            raise
        except VerifierFault as exc:
            a.close("verifier_fault", error=f"{type(exc).__name__}: {exc}"[:4000])
            raise
        except BaseException as exc:
            if a.row["status"] is None:
                a.close("candidate_fail", error=f"{type(exc).__name__}: {exc}"[:4000],
                        traceback=traceback.format_exc()[-6000:])
            raise
        else:
            if a.row["status"] is None:
                a.close(a.row.pop("verdict", "pass"))


def write_json(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, default=str))
    tmp.replace(path)
    return path
