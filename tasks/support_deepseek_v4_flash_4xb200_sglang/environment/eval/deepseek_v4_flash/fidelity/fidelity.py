#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The log-probability fidelity measurement for DeepSeek-V4-Flash on SGLang: the request, the parsing and the
statistic the grader uses, shared byte for byte with it."""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import requests

LIMITS = json.loads((Path(__file__).resolve().parent.parent / "limits.json").read_text())
LONG_INPUT_TOKENS = int(LIMITS["long_input_tokens"])
OPERATION_LIMIT_S = int(LIMITS["operations"]["http_request"]["limit_s"])
LONG_OPERATION_LIMIT_S = int(LIMITS["operations"]["http_request_long"]["limit_s"])
ACTION_LIMIT_S = int(LIMITS["operations"]["action"]["limit_s"])


class StructuralFailure(AssertionError):
    """A served answer that is not the complete, aligned, finite series the request asked for. Terminal for its
    check: no statistic is formed from it and no numerical re-draw replaces it."""

    status = "candidate_fail"


class ReferenceFault(RuntimeError):
    """A malformed reference record: an authoring or verification fault, never a finding about the server."""

    status = "reference_fault"


class ServedRequestFailed(AssertionError):
    """The server refused the request, answered something other than JSON, or took longer than the operation's
    limit."""

    status = "candidate_fail"


def operation_limit_s(n_input_tokens: int) -> int:
    """The wall-clock limit of one request carrying `n_input_tokens` input tokens."""
    return LONG_OPERATION_LIMIT_S if n_input_tokens >= LONG_INPUT_TOKENS else OPERATION_LIMIT_S


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_token_id(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def ids_digest(input_ids: list[int]) -> str:
    """sha256 of the ids' JSON, identifying a request's input in records."""
    return hashlib.sha256(json.dumps(input_ids, separators=(",", ":")).encode()).hexdigest()


def validate_reference(record, where: str = "") -> None:
    """Raise `ReferenceFault` unless `record` is a complete comparison record."""
    label = where or (record.get("id") if isinstance(record, dict) else None) or "?"
    if not isinstance(record, dict):
        raise ReferenceFault(f"reference {label}: not a JSON object")
    rid, ids, ref = record.get("id"), record.get("input_ids"), record.get("ref_logprobs")
    if not isinstance(rid, str) or not rid:
        raise ReferenceFault(f"reference {label}: no id")
    if not isinstance(ids, list) or len(ids) < 2 or not all(_is_token_id(t) for t in ids):
        raise ReferenceFault(f"reference {rid}: input_ids is not a list of at least two token ids")
    if not isinstance(ref, list) or len(ref) != len(ids) - 1:
        raise ReferenceFault(f"reference {rid}: {len(ref) if isinstance(ref, list) else 'no'} comparison values for "
                             f"{len(ids)} tokens; the span is one value per position 1 to {len(ids) - 1}")
    bad = next((i for i, v in enumerate(ref) if not _is_number(v) or not math.isfinite(v)), None)
    if bad is not None:
        raise ReferenceFault(f"reference {rid}: comparison value {ref[bad]!r} for position {bad + 1} is not a finite "
                             f"number")
    if "n_tokens" in record and record["n_tokens"] != len(ids):
        raise ReferenceFault(f"reference {rid}: n_tokens {record['n_tokens']!r} but {len(ids)} input ids")


def load_reference(hidden_dir: Path) -> list[dict]:
    """Every recorded sequence, sorted by file name so the median is over a fixed set in a fixed order, each checked
    by `validate_reference` and the ids checked unique, before anything is sent."""
    out, seen = [], {}
    for path in sorted(Path(hidden_dir).glob("*.json")):
        if path.name == "provenance.json":
            continue
        try:
            record = json.loads(path.read_text())
        except ValueError as exc:
            raise ReferenceFault(f"reference {path.name}: not JSON ({exc})") from exc
        validate_reference(record, path.name)
        if record["id"] in seen:
            raise ReferenceFault(f"reference id {record['id']} appears in {seen[record['id']]} and {path.name}")
        seen[record["id"]] = path.name
        out.append(record)
    if not out:
        raise ReferenceFault(f"no reference records in {hidden_dir}")
    return out


def parse_input_logprobs(input_ids: list[int], start: int, body, sequence_id: str) -> dict:
    """The served log-probabilities of positions `start + 1` to `len(input_ids) - 1`, from the answer `body` of a
    request for `input_ids` with `logprob_start_len` = `start`, after checking the whole answer against the pinned
    wire format. Raises `StructuralFailure` on the first departure, naming the position."""
    n = len(input_ids)
    if not (0 <= start <= n - 2):
        raise ValueError(f"{sequence_id}: logprob_start_len {start} leaves no scored position in {n} tokens")
    where = f"sequence {sequence_id} ({n} tokens, logprob_start_len {start})"
    meta = body.get("meta_info") if isinstance(body, dict) else None
    entries = meta.get("input_token_logprobs") if isinstance(meta, dict) else None
    if not isinstance(entries, list):
        raise StructuralFailure(f"{where}: the answer has no meta_info.input_token_logprobs list")
    if len(entries) != n - start:
        raise StructuralFailure(f"{where}: {len(entries)} input_token_logprobs entries, expected {n - start} (one per "
                                f"position {start} to {n - 1})")
    values = []
    for j, entry in enumerate(entries):
        pos = start + j
        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
            raise StructuralFailure(f"{where}: the entry for position {pos} is not [logprob, token_id, text]: "
                                    f"{entry!r:.120}")
        value, token = entry[0], entry[1]
        if not _is_token_id(token) or token != input_ids[pos]:
            raise StructuralFailure(f"{where}: the entry for position {pos} carries token id {token!r}, the input has "
                                    f"{input_ids[pos]} there")
        if j == 0:
            if value is not None:
                raise StructuralFailure(f"{where}: the start position {pos} carries {value!r}; the engine scores no "
                                        f"token there")
            continue
        if not _is_number(value) or not math.isfinite(value):
            raise StructuralFailure(f"{where}: the log-probability at position {pos} is {value!r}, not a finite "
                                    f"number")
        values.append(float(value))
    return {"values": values, "positions": len(values), "first_position": start + 1, "last_position": n - 1,
            "entries": len(entries)}


def request_logprobs(base_url: str, input_ids: list[int], start: int, sequence_id: str) -> dict:
    """One teacher-forced request (`kl_test_utils._get_input_logprobs`' call, by hand so nothing depends on the engine
    tree): the raw answer, its HTTP status and elapsed seconds. A refused request, an answer that is not JSON, or one
    that took longer than the operation's limit raises `ServedRequestFailed`; the raw answer rides on the exception."""
    limit = operation_limit_s(len(input_ids))
    t0 = time.monotonic()
    try:
        response = requests.post(
            f"{base_url}/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {"max_new_tokens": 0, "temperature": 1, "ignore_eos": True},
                "return_logprob": True,
                "logprob_start_len": start,
                "return_text_in_logprobs": False,
            },
            timeout=(limit, limit),
        )
    except requests.Timeout as exc:
        raise ServedRequestFailed(f"sequence {sequence_id or '?'} ({len(input_ids)} tokens): no complete answer "
                                  f"within its {limit} s limit") from exc
    elapsed = time.monotonic() - t0
    out = {"http_status": response.status_code, "elapsed_s": round(elapsed, 3), "limit_s": limit}
    if response.status_code != 200:
        exc = ServedRequestFailed(f"sequence {sequence_id or '?'} ({len(input_ids)} tokens) was not served: HTTP "
                                  f"{response.status_code}: {response.text[:300]}")
        exc.observation = {**out, "text": response.text[:2000]}
        raise exc
    try:
        out["body"] = response.json()
    except ValueError:
        exc = ServedRequestFailed(f"sequence {sequence_id or '?'}: the answer is not JSON: {response.text[:300]!r}")
        exc.observation = {**out, "text": response.text[:2000]}
        raise exc
    if elapsed > limit:
        exc = ServedRequestFailed(f"sequence {sequence_id or '?'} ({len(input_ids)} tokens): the request took "
                                  f"{elapsed:.1f} s, over its {limit} s limit")
        exc.observation = out
        raise exc
    return out


def served_logprobs(base_url: str, input_ids: list[int], sequence_id: str = "", start: int = 0) -> dict:
    """The request and its parse: `parse_input_logprobs`' result plus `elapsed_s`, `http_status`, `limit_s` and the
    raw entries (`raw`). A structural failure carries the raw answer on the exception (`observation`)."""
    got = request_logprobs(base_url, input_ids, start, sequence_id)
    raw = ((got["body"].get("meta_info") or {}).get("input_token_logprobs")
           if isinstance(got["body"], dict) and isinstance(got["body"].get("meta_info"), dict) else None)
    try:
        parsed = parse_input_logprobs(input_ids, start, got["body"], sequence_id)
    except StructuralFailure as exc:
        exc.observation = {k: v for k, v in got.items() if k != "body"} | {"raw": raw}
        raise
    return {**parsed, **{k: v for k, v in got.items() if k != "body"}, "raw": raw}


def _terms(ref: list[float], served: list[float]) -> list[float]:
    if len(ref) != len(served) or not ref:
        raise ValueError(f"k3 needs two series of the same nonzero length, got {len(ref)} and {len(served)}")
    return [math.inf if not math.isfinite(r := a - b) or r > 700.0 else math.exp(r) - 1.0 - r
            for a, b in zip(ref, served)]


def k3(ref: list[float], served: list[float]) -> float:
    """Mean over positions of exp(r) - 1 - r, r = ref - served. Non-negative by construction."""
    if len(ref) != len(served) or not ref:
        raise ValueError(f"k3 needs two series of the same nonzero length, got {len(ref)} and {len(served)}")
    total = 0.0
    for a, b in zip(ref, served):
        r = a - b
        if not math.isfinite(r) or r > 700.0:
            return math.inf
        total += math.exp(r) - 1.0 - r
    return total / len(ref)


def k3_leaving_out(ref: list[float], served: list[float], leave_out: int) -> float:
    """k3 over a sequence's positions with its `leave_out` largest terms left out."""
    terms = _terms(ref, served)
    n = len(terms)
    if n <= leave_out:
        raise ValueError(f"{n} positions, not more than the {leave_out} left out")
    kept = sorted(terms)[: n - leave_out]
    return math.inf if math.isinf(kept[-1]) else sum(kept) / (n - leave_out)


def median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        raise ValueError("no sequences")
    mid = n // 2
    return ordered[mid] if n % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])


def rescore(base_url: str, record: dict, leave_out: int | None = None, *, draw: int = 1, on_attempt=None,
            check: str = "fidelity") -> dict:
    """One sequence sent and scored: the attempt record (`k3`, and `k3_leaving_out` when `leave_out` is given).
    `on_attempt(attempt)` sees every attempt before this returns or raises, a failed one included."""
    validate_reference(record)
    attempt = {"check": check, "id": record["id"], "draw": draw, "reference": record.get("reference"),
               "tokens": len(record["input_ids"]), "input_digest": ids_digest(record["input_ids"]),
               "logprob_start_len": 0, "expected_positions": len(record["ref_logprobs"]),
               "expected_span": [1, len(record["input_ids"]) - 1]}
    try:
        served = served_logprobs(base_url, record["input_ids"], sequence_id=record["id"], start=0)
    except (StructuralFailure, ServedRequestFailed) as exc:
        attempt.update({"status": exc.status, "structural": "invalid", "error": str(exc)[:1000],
                        **{k: v for k, v in getattr(exc, "observation", {}).items()}})
        if on_attempt:
            on_attempt(attempt)
        raise
    except requests.RequestException as exc:
        attempt.update({"status": "unreachable", "structural": "none", "error": repr(exc)[:1000]})
        if on_attempt:
            on_attempt(attempt)
        raise
    values = served.pop("values")
    attempt.update({**served, "structural": "valid", "status": "measured", "k3": k3(record["ref_logprobs"], values)})
    if leave_out is not None:
        attempt["k3_leaving_out"] = k3_leaving_out(record["ref_logprobs"], values, leave_out)
        attempt["leave_out"] = leave_out
    if on_attempt:
        on_attempt(attempt)
    return attempt


def rescore_leaving_out(base_url: str, record: dict, leave_out: int, **kw) -> float:
    """One sequence sent and scored by `k3_leaving_out`."""
    return rescore(base_url, record, leave_out, **kw)["k3_leaving_out"]


def measure(base_url: str, hidden_dir: Path, *, on_attempt=None, check: str = "fidelity") -> tuple[float, dict[str, float], int]:
    """(median k3 over sequences, per-sequence values, number of validated positions compared), each sequence against
    the comparison values recorded at exactly its own length. Every record is checked before the first request."""
    records = load_reference(hidden_dir)
    per_sequence: dict[str, float] = {}
    positions = 0
    for record in records:
        attempt = rescore(base_url, record, on_attempt=on_attempt, check=check)
        per_sequence[record["id"]] = attempt["k3"]
        positions += attempt["positions"]
    return median(list(per_sequence.values())), per_sequence, positions


def measure_leaving_out(base_url: str, hidden_dir: Path, leave_out: int, *, on_attempt=None,
                        check: str = "fidelity") -> tuple[float, dict[str, float], dict[str, float], int]:
    """`measure`, returning beside each sequence's k3 its `k3_leaving_out` from the same served answer."""
    records = load_reference(hidden_dir)
    per_sequence: dict[str, float] = {}
    trimmed: dict[str, float] = {}
    positions = 0
    for record in records:
        attempt = rescore(base_url, record, leave_out, on_attempt=on_attempt, check=check)
        per_sequence[record["id"]] = attempt["k3"]
        trimmed[record["id"]] = attempt["k3_leaving_out"]
        positions += attempt["positions"]
    return median(list(per_sequence.values())), per_sequence, trimmed, positions
