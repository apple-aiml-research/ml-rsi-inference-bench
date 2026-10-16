#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Teacher-forced fidelity protocol used by the verifier and public examples.

Every returned token ID and count must match the requested sequence. Statistics
exclude the declared prefix and compare identical positions. Evaluation samples
and calibrated thresholds are supplied separately.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import statistics
import requests

SKIP_POSITIONS = 32
LONG_WINDOW = 1024

def aligned_values(entries, token_ids, *, first_null=True, seq_id="?"):
    """Reject missing, reordered, duplicate-position, or nonfinite observations."""
    assert isinstance(entries, list) and len(entries) == len(token_ids), (
        f"{seq_id}: expected {len(token_ids)} token observations, got "
        f"{len(entries) if isinstance(entries, list) else type(entries).__name__}")
    result = []
    for index, (entry, token) in enumerate(zip(entries, token_ids)):
        assert isinstance(entry, (list, tuple)) and len(entry) >= 2, f"{seq_id}: malformed entry {index}"
        value, observed = entry[:2]
        assert type(observed) is int and observed == token, f"{seq_id}: token ID mismatch at {index}"
        if index == 0 and first_null:
            assert value is None, f"{seq_id}: first token must have no conditional score"
            result.append(None)
        else:
            assert type(value) in (int, float) and math.isfinite(value), f"{seq_id}: nonfinite or missing score at {index}"
            result.append(float(value))
    return result

def candidate_logprobs(base_url: str, token_ids: list[int], timeout: float = 900.0,
                       seq_id: str = "?") -> list[float]:
    """Teacher-forced log-probabilities from the served model, positions 1..T-1.

    `POST /generate` with `input_ids` and `return_logprob` returns `meta_info.input_token_logprobs` as
    [logprob, token_id, text] per INPUT token, with the first entry null -- position 0 has no
    conditional log-probability, which is exactly the reference's convention too. `max_new_tokens` is 0
    if the engine accepts it and 1 otherwise; a single generated token changes no input log-probability.
    """
    last_error = None
    for max_new in (0, 1):
        payload = {
            "input_ids": token_ids,
            "sampling_params": {"max_new_tokens": max_new, "temperature": 0.0},
            "return_logprob": True,
            "logprob_start_len": 0,
        }
        try:
            r = requests.post(f"{base_url}/generate", json=payload, timeout=timeout)
        except requests.RequestException as exc:
            last_error = exc
            continue
        if r.status_code != 200:
            last_error = RuntimeError(f"/generate returned {r.status_code}: {r.text[:200]}")
            continue
        meta = r.json()["meta_info"]
        entries = meta["input_token_logprobs"]
        values = aligned_values(entries, token_ids, seq_id=seq_id)[1:]
        if len(values) != len(token_ids) - 1:
            last_error = RuntimeError(
                f"expected {len(token_ids) - 1} input log-probabilities, got {len(values)}")
            continue
        out = [float(v) for v in values]
        # A NON-FINITE served value fails the node here, loudly and by name, rather than travelling
        # into the statistic. An unwritten output buffer yields exactly NaN, and NaN does not compare:
        # `statistics.median` over per-sequence values with NaN among them has no defined order, so a
        # candidate with NaN on a MINORITY of sequences could otherwise carry a finite, passing median.
        bad = [(i, v) for i, v in enumerate(out) if not math.isfinite(v)]
        assert not bad, (
            f"{seq_id}: the served model returned {len(bad)} non-finite input log-probabilities, first at "
            f"position {bad[0][0]} with value {bad[0][1]!r}. A model that cannot score a token has not "
            f"implemented it.")
        return out
    raise AssertionError(f"could not retrieve input log-probabilities: {last_error}")


def candidate_logprobs_windowed(base_url: str, token_ids: list[int], first_index: int,
                                window: int = LONG_WINDOW, timeout: float = 900.0, seq_id: str = "?") -> list:
    """Teacher-forced log-probabilities for list indexes `first_index` onward (the reference's convention: index i is
    the log-probability of token i + 1), fetched a window at a time so that no request asks the engine for more than
    `window` positions' logits. Each request prefills the prefix up to its window's end with `logprob_start_len` one
    position before the window, because the engine returns no value for a range's first position; so every wanted
    position gets a value and none is skipped. Indexes before `first_index` are None and are never scored."""
    n = len(token_ids)
    out: list = [None] * (n - 1)
    a = first_index + 1
    while a < n:
        b = min(a + window, n)
        last_error = None
        for max_new in (0, 1):
            payload = {
                "input_ids": token_ids[:b],
                "sampling_params": {"max_new_tokens": max_new, "temperature": 0.0},
                "return_logprob": True,
                "logprob_start_len": a - 1,
            }
            try:
                r = requests.post(f"{base_url}/generate", json=payload, timeout=timeout)
            except requests.RequestException as exc:
                last_error = exc
                continue
            if r.status_code != 200:
                last_error = RuntimeError(f"/generate returned {r.status_code}: {r.text[:200]}")
                continue
            values = aligned_values(r.json()["meta_info"]["input_token_logprobs"],
                                    token_ids[a - 1:b], seq_id=seq_id)
            if len(values) != b - (a - 1):
                last_error = RuntimeError(f"expected {b - (a - 1)} input log-probabilities from position {a - 1}, "
                                          f"got {len(values)}")
                continue
            got = [float(v) for v in values[1:]]
            bad = [(a + i, v) for i, v in enumerate(got) if not math.isfinite(v)]
            assert not bad, (
                f"{seq_id}: the served model returned {len(bad)} non-finite input log-probabilities, first at "
                f"position {bad[0][0]} with value {bad[0][1]!r}. A model that cannot score a token has not "
                f"implemented it.")
            out[a - 1:b - 1] = got
            break
        else:
            raise AssertionError(f"could not retrieve input log-probabilities for {seq_id} positions {a}-{b - 1}: "
                                 f"{last_error}")
        a = b
    return out


def abs_r_median(candidate: list[float], reference: list[float], skip: int = SKIP_POSITIONS) -> float:
    """median_t |r| over positions `skip` onward, r = candidate - reference, clamped to +-30 as in `k3`; a
    non-finite r counts as +inf, which is ordered and fails any bar."""
    assert len(candidate) == len(reference), (len(candidate), len(reference))
    vals = []
    for lc, lr in zip(candidate[skip:], reference[skip:]):
        r = lc - lr
        vals.append(abs(max(min(r, 30.0), -30.0)) if math.isfinite(r) else math.inf)
    assert vals, "no positions left after the skip"
    return statistics.median(vals)


def k3(candidate: list[float], reference: list[float], skip: int = SKIP_POSITIONS) -> float:
    """mean_t[(exp(r) - 1) - r] over positions `skip` onward, r = candidate - reference.

    Two properties this has to hold, so that the median over sequences is a well-defined number and the
    assertion is in the passing direction:

      * `r` is clamped to ±30, so `math.exp` cannot raise `OverflowError` on a hopeless candidate --
        exp(30) is already 1e13, far past any threshold, and a crash would fail the node while losing
        the statistic;
      * a term that still comes out non-finite -- which after the clamp can only mean a non-finite
        INPUT -- is scored as **+inf** rather than NaN. Non-finite served values are already refused by
        name in `candidate_logprobs`; this is the second line of defence, and it is what keeps
        `statistics.fmean` and `statistics.median` ordered. `inf` fails `median < T`; NaN does not
        compare at all.
    """
    assert len(candidate) == len(reference), (len(candidate), len(reference))
    terms = []
    for lc, lr in zip(candidate[skip:], reference[skip:]):
        r = lc - lr
        if not math.isfinite(r):
            terms.append(math.inf)
            continue
        r = max(min(r, 30.0), -30.0)
        term = (math.exp(r) - 1.0) - r
        terms.append(term if math.isfinite(term) else math.inf)
    assert terms, "no positions left after the skip"
    return statistics.fmean(terms)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--examples", type=Path, default=Path(__file__).with_name("fidelity_examples"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.examples.glob("*.json")):
        example = json.loads(path.read_text())
        if "reference" not in example:
            continue
        skip = example.get("skip", SKIP_POSITIONS)
        got = candidate_logprobs(args.base_url, example["token_ids"], seq_id=example["id"])
        expected = example["reference"]["logprobs"]
        rows.append({"id": example["id"], "count": len(got), "skip": skip,
                     "k3": k3(got, expected, skip), "abs_r_median": abs_r_median(got, expected, skip)})
    assert rows, "no examples with reference values"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    print(json.dumps(rows, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

