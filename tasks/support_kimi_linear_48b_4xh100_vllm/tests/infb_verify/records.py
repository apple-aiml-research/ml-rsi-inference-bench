#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Attempt records and outcome attribution for the private checks.

Every check writes its record here before its verdict is raised, and again when it ends: identities,
every reading it made, and one status -- `pass`, `candidate_fail`, `reference_fault`,
`environment_fault`, `verifier_fault` or `blocked_by:<check>`. Missing required evidence is never a
pass. A trusted observation that cannot establish its evidence raises `ObservationFault` with one of
its reason codes: a verifier fault, an explicit non-result for the affected node. A check that finds the served model gone after an earlier check saw it go is blocked by that
check, not a second independent candidate failure. An exception that escapes the verifier's own analysis of
recorded measurements (`analysis`) is a verifier fault whose record carries `checker_exception`, never the
candidate's.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
import traceback
from pathlib import Path

ROOT = Path(os.environ.get("INFB_RECORDS_DIR", "/logs/verifier/records"))
#: The verifier's own files: a checker exception's frame is the innermost one under this directory.
VERIFIER = Path(__file__).resolve().parents[1]
#: The trusted modules whose explicit `raise ValueError(...)` statements validate candidate evidence (every TP rank
#: exactly once, complete and finite values, the same work timed on both sides): the supplied assessment's trusted
#: copy and the private verdicts.
VALIDATORS = ("eval/kimi_linear/kda/assessment.py", "infb_kda_private/verdicts.py")
NOT_THE_CANDIDATES = ("reference_fault", "environment_fault", "verifier_fault")
_unavailable: dict = {}


class ReferenceFault(Exception):
    """The recorded side of a comparison is unusable: never the candidate's failure."""


class ObservationFault(Exception):
    """A trusted verifier observation could not establish what it must: a verifier fault, never
    the candidate's failure. Raised only by the verifier's own checks, with a reason code and the evidence path; a
    category a candidate reports is never trusted. A broken candidate server, a native execution error or an
    exhausted published operation limit keeps its candidate attribution and is never raised as this."""

    REASONS = ("coverage_unreached", "rank_collection", "observer_incompatible", "observer_overflow",
               "observer_interference")

    def __init__(self, reason: str, message: str, evidence=None):
        if reason not in self.REASONS:
            raise ValueError(f"unknown observation fault reason {reason!r}")
        super().__init__(f"{reason}: {message}")
        self.reason, self.evidence = reason, None if evidence is None else str(evidence)


class VerifierFault(Exception):
    """A trusted verifier procedure could not run as specified (its own material or route, not an observation of the
    candidate): a verifier fault, never the candidate's failure."""


class Blocked(Exception):
    """A check that cannot be decided because a step it depends on ended at its bound or was censored: the status
    names that step (`blocked_by:<step>`), never the candidate's failure."""

    def __init__(self, cause: str, message: str):
        super().__init__(f"blocked by {cause}: {message}")
        self.status = f"blocked_by:{cause}"


def checker_exception(check: str, exc: BaseException) -> dict:
    """An exception that escaped the verifier's own analysis code, as a record: its type and message, the
    innermost frame in the verifier's own files (`file:line`, relative to the tests directory) and the check."""
    frames = traceback.extract_tb(exc.__traceback__)
    ours = [f for f in frames if Path(f.filename).resolve().is_relative_to(VERIFIER)] or frames
    frame = ours[-1] if ours else None
    where = None
    if frame is not None:
        path = Path(frame.filename).resolve()
        where = f"{path.relative_to(VERIFIER) if path.is_relative_to(VERIFIER) else path}:{frame.lineno}"
    return {"check": check, "type": type(exc).__name__, "message": str(exc)[:2000], "frame": where}


def stated_validation(exc: BaseException) -> bool:
    """Whether `exc` is a trusted module's stated validation of candidate evidence -- a ValueError raised by
    an explicit `raise ValueError(...)` in one of `VALIDATORS`. It is judged as the candidate's, so the guard
    never turns a candidate's malformed evidence into a void."""
    if type(exc) is not ValueError:
        return False
    frames = traceback.extract_tb(exc.__traceback__)
    if not frames:
        return False
    path = Path(frames[-1].filename).resolve()
    return (path.is_relative_to(VERIFIER) and str(path.relative_to(VERIFIER)) in VALIDATORS
            and (frames[-1].line or "").startswith("raise ValueError("))


@contextlib.contextmanager
def analysis(check: str):
    """The verifier's own analysis of measurements and responses already recorded. An exception that escapes it
    is the verifier's (`VerifierFault`, carrying `checker_exception`: the node is unmeasured), never the candidate's.
    The verdict exceptions and the trusted modules' stated validations of candidate evidence (`stated_validation`)
    pass through unchanged: a malformed candidate output is a judged problem, never a void."""
    try:
        yield
    except (ReferenceFault, ObservationFault, VerifierFault, Blocked):
        raise
    except Exception as exc:  # noqa: BLE001 - the point: no checker exception is ever the candidate's
        if stated_validation(exc):
            raise
        fault = VerifierFault(f"the {check} analysis raised {type(exc).__name__}: {exc}"[:2000])
        fault.checker_exception = checker_exception(check, exc)
        raise fault from exc


def write(check: str, record: dict) -> Path:
    """Persist one check's record atomically; later writes replace earlier ones."""
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / f"{check}.json"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps({"check": check, "written_unix": time.time(), **record}, indent=1,
                                    default=str) + "\n")
    temporary.replace(path)
    return path


def server_unavailable(check: str) -> str:
    """The first check that finds the served model unreachable owns that failure; later checks are blocked by it."""
    first = _unavailable.setdefault("check", check)
    return "candidate_fail" if first == check else f"blocked_by:{first}"


def classify(check: str, exc: BaseException) -> str:
    """The status of a check that ended in `exc`.

    A reference defect is the reference's; an expired node bound while every operation kept within its own
    published limit is the verifier's (censored); an unreachable server after another check saw it go is blocked.
    Everything else that ends a check at the candidate's surface is the candidate's: a malformed, non-finite or
    failing observation, an operation that exhausted its published limit (`subprocess.TimeoutExpired`, the
    fidelity check's `OperationLimitExceeded`), and an RPC, client or server failure while serving it. A status
    that is not the candidate's needs a positive cause, so no failure of the candidate's can void its own trial."""
    try:
        from infb_efficiency.deadline import DeadlineExceeded  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        DeadlineExceeded = ()  # noqa: N806
    name = type(exc).__name__
    if isinstance(exc, ReferenceFault) or name == "ReferenceFault":
        return "reference_fault"
    if isinstance(exc, (ObservationFault, VerifierFault)):
        return "verifier_fault"
    if isinstance(exc, Blocked):
        return exc.status
    if DeadlineExceeded and isinstance(exc, DeadlineExceeded):
        return "verifier_fault"
    try:
        from requests.exceptions import ConnectionError as Unreachable  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        Unreachable = ()  # noqa: N806
    if Unreachable and isinstance(exc, Unreachable):
        return server_unavailable(check)
    return "candidate_fail"


def run(check: str, meta: dict, body):
    """Run `body(record)` under a record: written before it starts, updated with its end and status.

    `body` fills the record with its readings as it goes and returns (passed, result)."""
    record = {"status": "running", "started_unix": time.time(), **meta}
    write(check, record)
    try:
        passed, result = body(record)
    except BaseException as exc:
        record.update(status=classify(check, exc), error=f"{type(exc).__name__}: {exc}"[:4000],
                      ended_unix=time.time())
        if getattr(exc, "bands", None) is not None and "bands" not in record:
            record["bands"] = exc.bands
        if isinstance(exc, ObservationFault):
            record["observation_fault"] = {"reason": exc.reason, "evidence": exc.evidence}
        if getattr(exc, "checker_exception", None) is not None:
            record["checker_exception"] = exc.checker_exception
        write(check, record)
        raise
    record.update(status="pass" if passed else "candidate_fail", ended_unix=time.time())
    write(check, record)
    return result
