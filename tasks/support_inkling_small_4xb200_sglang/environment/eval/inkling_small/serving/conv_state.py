#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Conv-state restore: does generation continue identically after a prefix-cache restore?"""
from __future__ import annotations

from .server import flush_cache, generate, input_series, output_series


def _first_divergence(a: list[int], b: list[int]) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def _pair(base_url: str, prompt: list[int], new_tokens: int, timeout: float) -> tuple[dict, dict]:
    flush_cache(base_url, timeout)
    cold = generate(base_url, prompt, max_new_tokens=new_tokens, timeout=timeout)
    restored = generate(base_url, prompt, max_new_tokens=new_tokens, timeout=timeout)
    return cold, restored


def _prompt_mode(base_url: str, ids: list[int], start: int, timeout: float) -> dict:
    """The prompt log-probability predicted from position `start`, restored against cold."""
    prompt = ids[: start + 2]
    flush_cache(base_url, timeout)
    cold = generate(base_url, prompt, max_new_tokens=0, timeout=timeout, logprob_start_len=start)
    generate(base_url, prompt, max_new_tokens=0, timeout=timeout, logprob_start_len=start)
    second = generate(base_url, prompt, max_new_tokens=0, timeout=timeout, logprob_start_len=start)
    c_ids, c_lps = input_series(cold)
    s_ids, s_lps = input_series(second)
    rec = {"prompt_mode_cached_tokens": second["meta_info"].get("cached_tokens")}
    if len(c_ids) >= 2 and c_ids == s_ids and c_lps[1] is not None and s_lps[1] is not None:
        rec["prompt_mode_d"] = round(s_lps[1] - c_lps[1], 6)
    else:
        rec["prompt_mode_d"] = None
    return rec


def probe(base_url: str, ids: list[int], L0: int, *, new_tokens: int = 16, timeout: float = 900.0,
          prompt_mode: bool = True) -> dict:
    """One window: the flip at the restore and its diagnostics. `ids` holds at least L0 + 2 tokens."""
    if len(ids) < L0 + 2:
        raise ValueError(f"a window needs L0 + 2 = {L0 + 2} tokens, got {len(ids)}")
    cold, restored = _pair(base_url, ids[: L0 + 1], new_tokens, timeout)
    c = restored["meta_info"].get("cached_tokens") or 0
    rec: dict = {"L0": L0, "cached_tokens": c, "cold_cached_tokens": cold["meta_info"].get("cached_tokens")}
    at = L0
    if 0 < c < L0:
        at = c
        cold, restored = _pair(base_url, ids[: c + 1], new_tokens, timeout)
        rec.update({"reissued_prompt_len": c + 1, "reissue_cached_tokens": restored["meta_info"].get("cached_tokens")})
    cold_ids, cold_lps = output_series(cold)
    rest_ids, rest_lps = output_series(restored)
    problems = [f"the {run} run returned {len(got)} generated tokens and {len(lps)} log-probabilities for {new_tokens}"
                for run, got, lps in (("cold", cold_ids, cold_lps), ("restored", rest_ids, rest_lps))
                if not len(got) == len(lps) == new_tokens]
    flipped = cold_ids[:1] != rest_ids[:1]
    rec.update({
        "structural_problems": problems,
        "prompt_len": at + 1,
        "flipped": flipped,
        "first_divergence": _first_divergence(cold_ids, rest_ids),
        "first_token_d": None if flipped or not cold_lps or not rest_lps else round(rest_lps[0] - cold_lps[0], 6),
        "cold_ids": cold_ids,
        "restored_ids": rest_ids,
    })
    if prompt_mode:
        rec.update(_prompt_mode(base_url, ids, at, timeout))
    return rec


def summarize(rows: list[dict]) -> dict:
    """F (windows whose first generated token flips), the windows below the reuse floor or structurally invalid, the
    diagnostics' spread."""
    ds = sorted(abs(r["first_token_d"]) for r in rows if r.get("first_token_d") is not None)
    pm = sorted(abs(r["prompt_mode_d"]) for r in rows if r.get("prompt_mode_d") is not None)

    def med(v: list[float]):
        return v[len(v) // 2] if v else None

    return {"windows": len(rows), "flips": sum(1 for r in rows if r["flipped"]),
            "below_reuse_floor": sum(1 for r in rows if not r["cached_tokens"] >= 0.5 * r["L0"]),
            "structurally_invalid": sum(1 for r in rows if r["structural_problems"]),
            "reissued": sum(1 for r in rows if "reissued_prompt_len" in r),
            "first_token_abs_d_median": med(ds), "first_token_abs_d_max": ds[-1] if ds else None,
            "prompt_mode_abs_d_median": med(pm)}


def run(base_url: str, windows: list[dict], *, new_tokens: int = 16, timeout: float = 900.0,
        prompt_mode: bool = True) -> dict:
    """windows: [{"id", "L0", "token_ids"}]; returns every window's reading and the summary."""
    rows = []
    for w in windows:
        rec = probe(base_url, w["token_ids"], int(w["L0"]), new_tokens=new_tokens, timeout=timeout,
                    prompt_mode=prompt_mode)
        rec["id"] = w["id"]
        rows.append(rec)
    return {"windows": rows, "summary": summarize(rows)}



DECOY_STEP = 17  # coprime to the window count, so a window's decoys come from different windows


def _history_decoy(base_url: str, windows: list[dict], wi: int, j: int, prompt_len: int,
                   new_tokens: int, timeout: float) -> dict:
    """Decoy j of window i, after `bump` replacements: window (i + DECOY_STEP (j + bump)) mod N's tokens, cut to
    window i's prompt length. Deterministic in (i, j, bump). A decoy that comes back cached is replaced by the next
    index, which is what `bump` counts; the last reading is returned if every index is cached."""
    n = len(windows)
    rec = {}
    for bump in range(n):
        src = (wi + DECOY_STEP * (j + bump)) % n
        k = 0
        while src == wi:
            k += 1
            src = (wi + DECOY_STEP * (j + bump) + k) % n
        ids = windows[src]["token_ids"][:prompt_len]
        r = generate(base_url, ids, max_new_tokens=new_tokens, timeout=timeout, return_logprob=False)
        rec = {"j": j, "source": windows[src].get("id"), "len": len(ids), "bump": bump,
               "cached_tokens": r["meta_info"].get("cached_tokens") or 0}
        if not rec["cached_tokens"]:
            break
    return rec


def _identical(a: dict, b: dict) -> tuple[bool, int | None, float | None, str]:
    """Exact comparison of two runs: the token ids first, then the output log-probabilities. Returns
    (identical, the first differing position, |difference| there where both are numbers, what differed)."""
    a_ids, a_lps = output_series(a)
    b_ids, b_lps = output_series(b)
    if len(a_ids) != len(b_ids) or len(a_lps) != len(b_lps):
        return False, min(len(a_ids), len(b_ids)), None, "length"
    for i, (x, y) in enumerate(zip(a_ids, b_ids)):
        if x != y:
            d = abs(b_lps[i] - a_lps[i]) if a_lps[i] is not None and b_lps[i] is not None else None
            return False, i, d, "token"
    for i, (x, y) in enumerate(zip(a_lps, b_lps)):
        if x != y:
            return False, i, abs(y - x) if x is not None and y is not None else None, "logprob"
    return True, None, None, ""


def _history_attempt(base_url: str, windows: list[dict], wi: int, attempt: int, new_tokens: int,
                     timeout: float) -> dict:
    """One pass of the per-window sequence: flush, cold A, restored A, decoy 1, restored B, flush, decoy 2, cold B."""
    w = windows[wi]
    L0 = int(w["L0"])
    prompt = w["token_ids"][: L0 + 1]

    flush_cache(base_url, timeout)
    cold_a = generate(base_url, prompt, max_new_tokens=new_tokens, timeout=timeout)
    restored_a = generate(base_url, prompt, max_new_tokens=new_tokens, timeout=timeout)
    c_a = restored_a["meta_info"].get("cached_tokens") or 0
    d1 = _history_decoy(base_url, windows, wi, 1 + attempt, len(prompt), new_tokens, timeout)
    restored_b = generate(base_url, prompt, max_new_tokens=new_tokens, timeout=timeout)
    c_b = restored_b["meta_info"].get("cached_tokens") or 0
    flush_cache(base_url, timeout)
    d2 = _history_decoy(base_url, windows, wi, 2 + attempt, len(prompt), new_tokens, timeout)
    cold_b = generate(base_url, prompt, max_new_tokens=new_tokens, timeout=timeout)

    problems = []
    for name, r in (("cold A", cold_a), ("restored A", restored_a), ("restored B", restored_b), ("cold B", cold_b)):
        ids, lps = output_series(r)
        if not len(ids) == len(lps) == new_tokens:
            problems.append(f"the {name} run returned {len(ids)} generated tokens and {len(lps)} log-probabilities "
                            f"for {new_tokens}")
    if c_a != c_b:
        problems.append(f"the two restores reused different prefixes: c_A {c_a} != c_B {c_b}")
    if not c_a >= 0.5 * L0:
        problems.append(f"cached_tokens {c_a} < 0.5 x L0 = {0.5 * L0:g}: the recipe's prefix caching is not active")
    for d in (d1, d2):
        if d["cached_tokens"]:
            problems.append(f"decoy {d['j']} reused {d['cached_tokens']} cached tokens")

    r_same, r_at, r_d, r_why = _identical(restored_a, restored_b)
    c_same, c_at, c_d, c_why = _identical(cold_a, cold_b)
    cold_ids, cold_lps = output_series(cold_a)
    rest_ids, rest_lps = output_series(restored_a)
    flipped = cold_ids[:1] != rest_ids[:1]
    return {
        "L0": L0, "prompt_len": len(prompt), "attempt": attempt,
        "cached_tokens_A": c_a, "cached_tokens_B": c_b,
        "cold_A_cached_tokens": cold_a["meta_info"].get("cached_tokens"),
        "cold_B_cached_tokens": cold_b["meta_info"].get("cached_tokens"),
        "decoys": [d1, d2],
        "structural_problems": problems,
        "restored_pair_identical": r_same, "restored_first_diff": r_at, "restored_abs_d": r_d,
        "restored_diff_kind": r_why,
        "cold_pair_identical": c_same, "cold_first_diff": c_at, "cold_abs_d": c_d, "cold_diff_kind": c_why,
        "flipped_cold_vs_restored": flipped,
        "first_token_d_cold_vs_restored": (None if flipped or not cold_lps or not rest_lps
                                           else round(rest_lps[0] - cold_lps[0], 6)),
    }


def history_probe(base_url: str, windows: list[dict], wi: int, *, new_tokens: int = 16,
                  timeout: float = 900.0) -> dict:
    """One window of the history-independence reading. A structurally invalid window is re-issued once from the
    first step; a second invalid reading is returned as it stands and names the window."""
    rec = _history_attempt(base_url, windows, wi, 0, new_tokens, timeout)
    if rec["structural_problems"]:
        first = rec["structural_problems"]
        rec = _history_attempt(base_url, windows, wi, 1, new_tokens, timeout)
        rec["reissued_after"] = first
    rec["id"] = windows[wi].get("id")
    return rec


def history_summarize(rows: list[dict]) -> dict:
    """The graded counts and the recorded diagnostics."""
    return {
        "windows": len(rows),
        "restored_pair_differing": sum(1 for r in rows if not r["restored_pair_identical"]),
        "cold_pair_differing": sum(1 for r in rows if not r["cold_pair_identical"]),
        "structurally_invalid": sum(1 for r in rows if r["structural_problems"]),
        "reissued": sum(1 for r in rows if "reissued_after" in r),
        "c_A_equals_c_B": sum(1 for r in rows if r["cached_tokens_A"] == r["cached_tokens_B"]),
        "decoys_cached": sum(1 for r in rows for d in r["decoys"] if d["cached_tokens"]),
        "flips_cold_vs_restored": sum(1 for r in rows if r["flipped_cold_vs_restored"]),
        "differing_restored_ids": [r["id"] for r in rows if not r["restored_pair_identical"]],
        "differing_cold_ids": [r["id"] for r in rows if not r["cold_pair_identical"]],
    }


def history_run(base_url: str, windows: list[dict], *, new_tokens: int = 16, timeout: float = 900.0) -> dict:
    """windows: [{"id", "L0", "token_ids"}]; every window's history-independence reading and the summary."""
    rows = [history_probe(base_url, windows, wi, new_tokens=new_tokens, timeout=timeout)
            for wi in range(len(windows))]
    return {"windows": rows, "summary": history_summarize(rows)}
