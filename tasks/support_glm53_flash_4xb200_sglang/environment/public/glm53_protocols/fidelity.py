#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Per-token log-probabilities of given continuations, checked for structure, and their k3 against reference values."""
from __future__ import annotations

import math
from typing import Any

import requests

from .limits import ACTION_S, HTTP_S

LOGPROB_START_LEN = 0
MAX_PROBLEMS = 8


class MalformedAnswer(AssertionError):
    """The server's answer does not have the structure the request asks for."""


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def reference_problems(seq: dict) -> list[str]:
    """What is wrong with one reference record's own structure (empty when it is well formed)."""
    out: list[str] = []
    ids, plen, clen = seq.get("input_ids"), seq.get("prompt_len"), seq.get("cont_len")
    if not isinstance(ids, list) or not ids or not all(isinstance(t, int) and not isinstance(t, bool) and t >= 0
                                                          for t in ids):
        out.append("input_ids is not a non-empty list of token ids")
    if not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in (plen, clen)):
        out.append(f"prompt_len {plen!r} and cont_len {clen!r} must be positive integers")
    elif isinstance(ids, list) and plen + clen != len(ids):
        out.append(f"prompt_len {plen} + cont_len {clen} != {len(ids)} input ids")
    if "ref_logprobs" in seq:
        ref = seq["ref_logprobs"]
        if not isinstance(ref, list) or len(ref) != clen:
            out.append(f"ref_logprobs holds {len(ref) if isinstance(ref, list) else type(ref).__name__}, "
                       f"not cont_len {clen} values")
        elif not all(_is_number(v) and math.isfinite(v) for v in ref):
            out.append("ref_logprobs holds a value that is not a finite number")
    return out


def request_logprobs(base_url: str, seqs: list[dict], routed_dp_rank: int | None = None,
                     timeout: float = HTTP_S) -> Any:
    """The raw answer to one batched teacher-forced request for `seqs`, after a prefix-cache flush. The flush is an
    action (`limits.ACTION_S`) and the request one HTTP workload (`limits.HTTP_S`)."""
    requests.post(base_url + "/flush_cache", params={"timeout": 30}, timeout=ACTION_S).raise_for_status()
    payload = {
        "input_ids": [s["input_ids"] for s in seqs],
        "sampling_params": {"temperature": 0.0, "max_new_tokens": 0, "ignore_eos": True},
        "return_logprob": True,
        "return_text_in_logprobs": False,
        "logprob_start_len": LOGPROB_START_LEN,
    }
    if routed_dp_rank is not None:
        payload["routed_dp_rank"] = routed_dp_rank
    response = requests.post(base_url + "/generate", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def parse_result(seq: dict, result: Any, start: int = LOGPROB_START_LEN) -> dict:
    """One sequence's reading of its result: problems, spans and, when well formed, the continuation's values."""
    ids, plen, clen = seq["input_ids"], seq["prompt_len"], seq["cont_len"]
    reading: dict = {"id": seq.get("id"), "expected_span": [plen, plen + clen], "expected_entries": len(ids) - start,
                     "observed_entries": None, "problems": [], "values": None}
    problems = reading["problems"]
    meta = result.get("meta_info") if isinstance(result, dict) else None
    entries = meta.get("input_token_logprobs") if isinstance(meta, dict) else None
    if not isinstance(entries, list):
        problems.append(f"no meta_info.input_token_logprobs list (got {type(entries).__name__})")
        return reading
    reading["observed_entries"] = len(entries)
    if len(entries) != len(ids) - start:
        problems.append(f"{len(entries)} entries for {len(ids) - start} requested positions "
                        f"({len(ids)} input ids, logprob_start_len {start})")
        return reading
    values: list[float] = []
    for j, entry in enumerate(entries):
        pos = start + j
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            problems.append(f"position {pos}: entry {entry!r} is not [log-probability, token id, ...]")
        else:
            token = entry[1]
            if not (isinstance(token, int) and not isinstance(token, bool)) or token != ids[pos]:
                problems.append(f"position {pos}: token id {token!r}, requested {ids[pos]}")
            elif pos >= plen:
                v = entry[0]
                if not _is_number(v):
                    problems.append(f"continuation position {pos - plen}: log-probability {v!r} is not a number")
                elif not math.isfinite(v):
                    problems.append(f"continuation position {pos - plen}: log-probability {v!r} is not finite")
                else:
                    values.append(float(v))
        if len(problems) >= MAX_PROBLEMS:
            problems.append("(further departures not listed)")
            break
    if not problems:
        if len(values) != clen:
            problems.append(f"{len(values)} continuation values for cont_len {clen}")
        else:
            reading["values"] = values
    return reading


def parse_answer(seqs: list[dict], answer: Any, start: int = LOGPROB_START_LEN) -> list[dict]:
    """One reading per requested sequence, in request order; a missing or extra result is a problem of every one."""
    if not isinstance(answer, list) or len(answer) != len(seqs):
        got = f"{len(answer)} results" if isinstance(answer, list) else f"a {type(answer).__name__}"
        return [{"id": s.get("id"), "expected_span": [s["prompt_len"], s["prompt_len"] + s["cont_len"]],
                 "expected_entries": len(s["input_ids"]) - start, "observed_entries": None, "values": None,
                 "problems": [f"the answer is {got} for {len(seqs)} requested sequences"]} for s in seqs]
    return [parse_result(s, r, start) for s, r in zip(seqs, answer)]


def served_continuation_logprobs(base_url: str, seqs: list[dict], routed_dp_rank: int | None = None) -> list[list[float]]:
    """The continuation values of every sequence, or `MalformedAnswer` naming the first departure."""
    readings = parse_answer(seqs, request_logprobs(base_url, seqs, routed_dp_rank))
    for r in readings:
        if r["problems"]:
            raise MalformedAnswer(f"sequence {r['id']}: {r['problems'][0]}")
    return [r["values"] for r in readings]


def _k3_term(d: float) -> float:
    """`exp(d) - 1 - d`, scored +inf when `d` is not finite or `exp` overflows."""
    if not math.isfinite(d):
        return math.inf
    try:
        return math.exp(d) - 1.0 - d
    except OverflowError:
        return math.inf


def k3_per_sequence(seqs: list[dict], served: list[list[float]]) -> list[float]:
    """k3 per sequence of well-formed values; a length that does not match its reference is refused, never zipped."""
    if len(seqs) != len(served):
        raise MalformedAnswer(f"{len(served)} value lists for {len(seqs)} sequences")
    vals = []
    for s, v in zip(seqs, served):
        ref = s["ref_logprobs"]
        if len(v) != len(ref):
            raise MalformedAnswer(f"sequence {s.get('id')}: {len(v)} values for {len(ref)} reference values")
        vals.append(sum(_k3_term(r - x) for r, x in zip(ref, v)) / len(ref))
    return vals
