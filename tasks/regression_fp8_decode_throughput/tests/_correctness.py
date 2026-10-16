#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Output-correctness probe for regression_fp8_decode_throughput."""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from pathlib import Path

_GREEDY = {"temperature": 0.0, "top_p": 1.0, "seed": 12345}

DISTINCT_K = 10
DISTINCT_K_FLOOR: float | None = None
LOGPROB_BAND: float | None = 0.07
MIN_DECODE_POSITIONS = 500


def load_prompts(tests_dir: Path) -> list[str]:
    """The canonical prompt bank, of which this probe scores the first N."""
    prompts = json.loads((tests_dir / "correctness_prompts.json").read_text(encoding="utf-8"))
    n = int(os.environ.get("INFB_CORRECTNESS_PROMPTS", "16"))
    return prompts[:n] if n > 0 else prompts


def _post(port: int, payload: dict, timeout: int) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _grams(text: str, k: int) -> list[str]:
    words = text.split()
    if len(words) < k:
        return []
    return [" ".join(words[i:i + k]) for i in range(len(words) - k + 1)]


def measure_text(text: str, k: int = DISTINCT_K) -> dict:
    """Degeneration measures for one completion. `None` where the text is too short to support them,
    which is itself reported rather than silently defaulting to a passing value."""
    grams = _grams(text, k)
    words = text.split()
    return {
        "chars": len(text),
        "words": len(words),
        "distinct_k": round(len(set(grams)) / len(grams), 6) if grams else None,
        "grams": len(grams),
        "repetition_rate": round(1 - len(set(words)) / len(words), 6) if words else None,
    }


def collect_generation(port: int, model: str, prompts: list[str], max_tokens: int,
                       timeout: int = 900) -> list[dict]:
    """Run by BOTH arms, identically. Greedy-generates and keeps the per-step chosen-token logprobs."""
    out: list[dict] = []
    for idx, prompt in enumerate(prompts):
        try:
            doc = _post(port, {"model": model, "prompt": prompt, "max_tokens": max_tokens,
                               "logprobs": 1, **_GREEDY}, timeout)
            choice = (doc.get("choices") or [{}])[0]
            lp = choice.get("logprobs") or {}
            generated = choice.get("text", "")
            out.append({
                "index": idx,
                "prompt": prompt,
                "generated": generated,
                "tokens": lp.get("tokens") or [],
                "token_logprobs": lp.get("token_logprobs") or [],
                "finish_reason": choice.get("finish_reason"),
                "measures": measure_text(generated),
                "error": None,
            })
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError,
                KeyError) as exc:
            out.append({"index": idx, "prompt": prompt, "generated": "", "tokens": [],
                        "token_logprobs": [], "finish_reason": None,
                        "measures": measure_text(""),
                        "error": f"{type(exc).__name__}: {exc}"})
    return out


def _own_logprobs(record: dict) -> tuple[list[float], int]:
    """Decode-step logprobs for one record, plus the NaN count. Position 0 is prefill-derived and is
    excluded so the statistic describes only the phase the defect lives in."""
    vals, nans = [], 0
    for pos, v in enumerate(record.get("token_logprobs") or []):
        if pos == 0:
            continue
        if v is None or (isinstance(v, float) and math.isnan(v)):
            nans += 1
            continue
        vals.append(float(v))
    return vals, nans


def _arm_summary(records: list[dict]) -> dict:
    """Aggregate one arm. Every count that could hide a failure is reported separately."""
    dk = [r["measures"]["distinct_k"] for r in records
          if r.get("measures", {}).get("distinct_k") is not None]
    rep = [r["measures"]["repetition_rate"] for r in records
           if r.get("measures", {}).get("repetition_rate") is not None]
    lps, nans, empty = [], 0, 0
    for r in records:
        vals, n = _own_logprobs(r)
        nans += n
        lps.extend(vals)
        if not (r.get("generated") or "").strip():
            empty += 1
    mean = lambda a: round(sum(a) / len(a), 6) if a else None  # noqa: E731
    return {
        "prompts": len(records),
        "errored": sum(1 for r in records if r.get("error")),
        "empty_generations": empty,
        "distinct_k": mean(dk),
        "distinct_k_min": round(min(dk), 6) if dk else None,
        "distinct_k_measurable": len(dk),
        "repetition_rate": mean(rep),
        "mean_own_logprob": mean(lps),
        "decode_positions": len(lps),
        "positions_nan": nans,
    }


def compare(reference: list[dict], candidate: list[dict], *,
            distinct_k_floor: float | None = DISTINCT_K_FLOOR,
            logprob_band: float | None = LOGPROB_BAND,
            min_decode_positions: int = MIN_DECODE_POSITIONS) -> dict:
    """Compare the two arms on aggregates measured in decode. No token alignment is attempted."""
    base = _arm_summary(reference)
    cand = _arm_summary(candidate)
    summary = {
        "comparison": "decode_aggregates",
        "distinct_k": DISTINCT_K,
        "distinct_k_floor": distinct_k_floor,
        "logprob_band": logprob_band,
        "min_decode_positions": min_decode_positions,
        "baseline": base,
        "candidate": cand,
        "failures": [],
    }

    def fail(kind: str, **kw) -> None:
        summary["failures"].append({"kind": kind, **kw})

    for who, arm in (("baseline", base), ("candidate", cand)):
        if arm["prompts"] == 0:
            fail("no_records", arm=who)
        if arm["errored"]:
            fail("request_errors", arm=who, count=arm["errored"])
        if arm["empty_generations"]:
            fail("empty_generations", arm=who, count=arm["empty_generations"])
        if arm["positions_nan"]:
            fail("nan_logprobs", arm=who, count=arm["positions_nan"])
        if arm["decode_positions"] < min_decode_positions:
            fail("too_few_decode_positions", arm=who, positions=arm["decode_positions"],
                 minimum=min_decode_positions)

    if distinct_k_floor is not None:
        for who, arm in (("baseline", base), ("candidate", cand)):
            if arm["distinct_k"] is not None and arm["distinct_k"] < distinct_k_floor:
                fail("degenerate_output", arm=who, distinct_k=arm["distinct_k"],
                     floor=distinct_k_floor, worst_prompt=arm["distinct_k_min"])

    if (logprob_band is not None and base["mean_own_logprob"] is not None
            and cand["mean_own_logprob"] is not None):
        delta = cand["mean_own_logprob"] - base["mean_own_logprob"]
        summary["mean_own_logprob_delta"] = round(delta, 6)
        if abs(delta) > logprob_band:
            fail("own_logprob_shift", delta=round(delta, 6), band=logprob_band,
                 baseline=base["mean_own_logprob"], candidate=cand["mean_own_logprob"])
    elif base["mean_own_logprob"] is not None and cand["mean_own_logprob"] is not None:
        summary["mean_own_logprob_delta"] = round(
            cand["mean_own_logprob"] - base["mean_own_logprob"], 6)
        summary["logprob_gating"] = "records only -- band not measured yet"
    elif logprob_band is not None:
        fail("logprob_unmeasurable", baseline=base["mean_own_logprob"],
             candidate=cand["mean_own_logprob"])

    summary["first_failure"] = summary["failures"][0] if summary["failures"] else None
    summary["passed"] = (
        not summary["failures"] and base["prompts"] > 0 and cand["prompts"] > 0
    )
    return summary
