#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Numerical fidelity: the served model's per-token log-probabilities against the model author's implementation.

    python check.py --base-url http://127.0.0.1:8000 [--examples examples.json] [--record FILE]

Each example is a recorded token sequence with the model-author implementation's log-probability of every next
token (teacher-forced; `examples.json` says how they were produced). For each sequence the script sends the token
ids as the prompt with `prompt_logprobs=0` and `max_tokens=1`, and compares, at every position after the first:

    lr = logp_author - logp_served         at the sequence's own next token
    k3 = exp(lr) - 1 - lr                  a low-variance estimator of KL; +inf where exp(lr) overflows

Per sequence it reports the mean of k3 over the positions; per length band, the median over that band's
sequences.

Structure comes first, and one parser decides it (`parse_prompt_logprobs`). The recorded data is validated before
any request. A reading is structurally valid only when the response is a JSON object with exactly one choice whose
`prompt_logprobs` list has exactly one entry per prompt token: the documented null at the first position, then at
every later position an entry for the sequence's own token (the key is that token id; no other entry stands in
for it) whose `logprob` is a finite number (a boolean is not a number). A longer or shorter list is a structural
failure before any entry is read. A structurally invalid reading is never a small discrepancy: it is reported as
such, it is never re-sent, averaged or outvoted by another reading, and no statistic is taken over a set that
contains one. A finite log-probability whose difference overflows `exp` is a numerical result (+inf), not a
structural one.

Every attempted reading is kept (`attempts`), with its ordinal, structural status, errors, positions, token
alignment and k3. The command reports the measurements and the structural status (exit 2 on any structurally
invalid reading, otherwise 0); it states no pass threshold.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
MODEL = "/model/deepseek-ai/DeepSeek-V4.1-Flash"
#: The public operation limit for one HTTP correctness request whose input is below 64,000 tokens (README).
REQUEST_TIMEOUT_S = 900

#: A reply as the transport delivered it: (HTTP status or None, the parsed JSON body or None, a transport error or None).
Reply = tuple[int | None, Any, str | None]


class ReferenceFault(ValueError):
    """The recorded reference data is malformed: a defect of the benchmark's data, never a reading of a server."""


def is_number(x: Any) -> bool:
    """An int or a float, and not a bool (`True` is an int in Python, and no log-probability)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate_reference(seq: Any) -> list[str]:
    """Every defect of one recorded sequence: its identity fields, token ids (non-negative ints, at least two) and
    reference log-probabilities (finite numbers, exactly one per position after the first)."""
    if not isinstance(seq, dict):
        return ["the recorded sequence is not an object"]
    errors = [f"no {k!r}" for k in ("id", "band", "family") if k not in seq]
    ids, ref = seq.get("token_ids"), seq.get("reference_logprobs")
    ids_ok = isinstance(ids, list) and len(ids) >= 2 and all(
        isinstance(t, int) and not isinstance(t, bool) and t >= 0 for t in ids)
    if not ids_ok:
        errors.append("token_ids is not a list of at least two non-negative integers")
    if not isinstance(ref, list) or not all(is_number(v) and math.isfinite(v) for v in ref):
        errors.append("reference_logprobs is not a list of finite numbers")
    elif ids_ok and len(ref) != len(ids) - 1:
        errors.append(f"reference_logprobs has {len(ref)} values for {len(ids)} tokens (one per position after the first)")
    return errors


def check_reference(seqs: list[Any]) -> None:
    """Validate every recorded sequence before any request; raise ReferenceFault naming each defect."""
    bad = {str((s or {}).get("id", i)) if isinstance(s, dict) else str(i): e
           for i, s in enumerate(seqs) if (e := validate_reference(s))}
    ids = [s.get("id") for s in seqs if isinstance(s, dict)]
    if len(set(ids)) != len(ids):
        bad["<ids>"] = ["the sequence ids are not unique"]
    if not seqs:
        bad["<set>"] = ["no recorded sequences"]
    if bad:
        raise ReferenceFault("malformed reference data: " + "; ".join(f"{k}: {', '.join(v)}" for k, v in bad.items()))


def request_prompt_logprobs(completions_url: str, model: str, token_ids: list[int],
                            timeout: float = REQUEST_TIMEOUT_S) -> Reply:
    """POST one teacher-forced request and return the reply as delivered; never raises for what the server returned.

    Posted as raw JSON rather than through the OpenAI client: `prompt_logprobs` is a vLLM extension, and reading it
    off a typed client model depends on that client's extra-field policy.
    """
    import requests  # noqa: PLC0415

    try:
        r = requests.post(
            completions_url,
            json={"model": model, "prompt": token_ids, "max_tokens": 1, "temperature": 0.0, "seed": 0,
                  "prompt_logprobs": 0},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        return None, None, f"{type(exc).__name__}: {exc}"[:600]
    try:
        body = r.json()
    except ValueError:
        return r.status_code, None, f"the body is not JSON ({r.text[:200]!r})"
    return r.status_code, body, None


def parse_prompt_logprobs(body: Any, token_ids: list[int]) -> tuple[list[float] | None, list[str], dict]:
    """The one typed parser of a teacher-forced reply.

    Returns (the served log-probability at every position after the first, or None when the reply is structurally
    invalid; the structural errors; the token alignment). Cardinality is checked before any entry is read.
    """
    n = len(token_ids)
    alignment: dict = {"status": "unread", "expected_positions": n - 1}
    if not isinstance(body, dict):
        return None, ["the reply is not a JSON object"], alignment
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        count = len(choices) if isinstance(choices, list) else type(choices).__name__
        return None, [f"the reply has no single choice object (choices: {count})"], alignment
    entries = choices[0].get("prompt_logprobs")
    if not isinstance(entries, list):
        return None, [f"the choice has no prompt_logprobs list (keys {sorted(choices[0])})"], alignment
    alignment["served_entries"] = len(entries)
    if len(entries) != n:
        kind = "oversized" if len(entries) > n else "short"
        return None, [f"prompt_logprobs is {kind}: {len(entries)} entries for {n} prompt tokens"], alignment
    errors: list[str] = []
    if entries[0] is not None:
        errors.append("the first position's entry is not the documented null")
    values: list[float] = []
    missing: list[int] = []
    not_number: list[int] = []
    non_finite: list[int] = []
    for pos in range(1, n):
        entry = entries[pos]
        item = entry.get(str(token_ids[pos])) if isinstance(entry, dict) else None
        if not isinstance(item, dict):
            missing.append(pos)  # no entry for this position's own token; nothing else in the entry stands in for it
            continue
        lp = item.get("logprob")
        if not is_number(lp):
            not_number.append(pos)
            continue
        if not math.isfinite(lp):
            non_finite.append(pos)
            continue
        values.append(float(lp))
    for what, where in (("no log-probability for the sequence's own token", missing),
                        ("a log-probability that is not a number", not_number),
                        ("a non-finite log-probability", non_finite)):
        if where:
            errors.append(f"{what} at {len(where)} position(s), first {where[:5]}")
    alignment.update(status="aligned" if not (missing or entries[0] is not None) else "misaligned",
                     missing_positions=len(missing), first_missing=missing[:5])
    return (None if errors else values), errors, alignment


def k3(lr: float) -> float:
    """exp(lr) - 1 - lr, and +inf where exp(lr) overflows or the term is not finite (never a capped value)."""
    try:
        v = math.exp(lr) - 1 - lr
    except OverflowError:
        return math.inf
    return v if math.isfinite(v) else math.inf


def score(reference: list[float], served: list[float]) -> dict:
    """k3 at every position of a structurally valid reading: its count, mean and maximum."""
    ks = [k3(r - s) for r, s in zip(reference, served, strict=True)]
    return {"n": len(ks), "mean": statistics.fmean(ks), "max": max(ks)}


def observe(fetch: Callable[[list[int]], Reply], seq: dict, ordinal: int) -> dict:
    """One attempted reading of one sequence, kept whole: ordinal, structural status and errors, positions, token
    alignment, k3 and the time it took."""
    t0 = time.monotonic()
    status, body, transport = fetch(seq["token_ids"])
    errors: list[str] = []
    values, alignment = None, {"status": "unread", "expected_positions": len(seq["token_ids"]) - 1}
    if transport is not None:
        errors.append(f"the request failed: {transport}")
    elif status != 200:
        detail = json.dumps(body)[:300] if body is not None else ""
        errors.append(f"HTTP {status} {detail}".strip())
    else:
        values, parse_errors, alignment = parse_prompt_logprobs(body, seq["token_ids"])
        errors += parse_errors
    return {
        "ordinal": ordinal,
        "structural_status": "valid" if not errors else "invalid",
        "errors": errors,
        "positions": {"expected": len(seq["token_ids"]) - 1, "served": alignment.get("served_entries"),
                      "compared": len(values) if values is not None else 0},
        "token_alignment": alignment,
        "k3": score(seq["reference_logprobs"], values) if values is not None else None,
        "seconds": round(time.monotonic() - t0, 3),
        "transport_failed": transport is not None,
    }


def read_sequence(fetch: Callable[[list[int]], Reply], seq: dict, retry_over: float | None = None) -> dict:
    """A sequence's reading under the bounded retry policy.

    The first attempt always runs. A structurally invalid attempt fails the sequence at once (no retry). With
    `retry_over` set (the private trigger), a structurally valid first attempt whose mean k3 is not at most that value
    is sent exactly once more; if the second is invalid the sequence fails, otherwise the lower mean stands. The public
    command passes no trigger and reads each sequence once.
    """
    attempts = [observe(fetch, seq, 1)]
    first = attempts[0]
    if first["structural_status"] == "valid" and retry_over is not None and not first["k3"]["mean"] <= retry_over:
        attempts.append(observe(fetch, seq, 2))
    valid = all(a["structural_status"] == "valid" for a in attempts)
    selected = min(attempts, key=lambda a: (a["k3"]["mean"], a["ordinal"])) if valid else None
    return {
        "id": seq["id"], "band": seq["band"], "family": seq["family"],
        "structural_status": "valid" if valid else "invalid",
        "attempts": attempts,
        "selected_ordinal": selected["ordinal"] if selected else None,
        "k3_mean": selected["k3"]["mean"] if selected else None,
        "k3_max": selected["k3"]["max"] if selected else None,
    }


def band_median(rows: list[dict]) -> float | None:
    """The median of the selected mean k3 over a band's sequences, or None unless every attempted observation of
    every sequence is structurally valid."""
    if not rows or any(a["structural_status"] != "valid" for r in rows for a in r["attempts"]):
        return None
    return statistics.median(r["k3_mean"] for r in rows)


def read_band(fetch: Callable[[list[int]], Reply], seqs: list[dict], retry_over: float | None) -> tuple[list[dict], float | None]:
    """One draw of a band: every sequence read under `read_sequence`; the median only over a structurally valid draw."""
    rows = [read_sequence(fetch, s, retry_over) for s in seqs]
    return rows, band_median(rows)


def structural_failures(rows: list[dict]) -> list[str]:
    """One line per structurally invalid attempted reading, naming the sequence, the attempt and its errors."""
    return [f"{r['id']} attempt {a['ordinal']}: {'; '.join(a['errors'])}"
            for r in rows for a in r["attempts"] if a["structural_status"] != "valid"]


def assert_structural(rows: list[dict]) -> None:
    """Structure first: any structurally invalid attempted reading fails, whatever any other reading says."""
    bad = structural_failures(rows)
    assert not bad, "structurally invalid readings (never read as a discrepancy): " + " | ".join(bad)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--examples", default=str(HERE / "examples.json"))
    ap.add_argument("--record", help="write every attempted reading here as JSON")
    a = ap.parse_args(argv)
    data = json.loads(Path(a.examples).read_text())
    examples = data.get("examples") if isinstance(data, dict) else None
    try:
        check_reference(examples if isinstance(examples, list) else [])
    except ReferenceFault as exc:
        print(f"REFERENCE DATA FAULT: {exc}")
        return 3
    url = a.base_url.rstrip("/") + "/v1/completions"
    fetch = lambda ids: request_prompt_logprobs(url, a.model, ids)  # noqa: E731
    rows = []
    for seq in examples:
        row = read_sequence(fetch, seq)
        rows.append(row)
        att = row["attempts"][-1]
        if row["structural_status"] == "valid":
            print(f"{row['id']} ({att['positions']['expected']} positions, band {row['band']}): mean k3 "
                  f"{row['k3_mean']:.6g}, max {row['k3_max']:.6g}", flush=True)
        else:
            print(f"{row['id']} (band {row['band']}): STRUCTURALLY INVALID: {'; '.join(att['errors'])}", flush=True)
    if a.record:
        Path(a.record).write_text(json.dumps({"examples": a.examples, "sequences": rows}, indent=1))
    bad = structural_failures(rows)
    if bad:
        print("STRUCTURAL FAILURE: " + " | ".join(bad))
        return 2
    bands = sorted({r["band"] for r in rows})
    medians = {b: band_median([r for r in rows if r["band"] == b]) for b in bands}
    print("per-band median of the per-sequence mean k3: " + json.dumps(medians))
    return 0


if __name__ == "__main__":
    sys.exit(main())
