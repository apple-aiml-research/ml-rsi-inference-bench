#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
'SD3 t2 gates utilities.'
from __future__ import annotations

import math

POLICY = {

    "ISO_PSNR_DB": 45.0,

    "BATCH_PSNR_DB": 40.0,

    "EFFECT_MAX_PSNR_DB": 40.0,

    "FAULT_MARKER": "INFB injected fault",


    "CANCEL_MAX_CALLS_AFTER_ABORT": 3,

    "DEADLINE_MAX_OVERRUN_S": 3.0,
}


def psnr(a, b) -> float | None:
    """The verifier's image psnr (uint8 HWC, mse over every element, 100 for identical); None if either is missing or
    the shapes differ. numpy is imported only here, so the judges stay stdlib."""
    if a is None or b is None:
        return None
    import numpy as np

    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return None
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 100.0 if mse <= 1e-9 else 10.0 * math.log10(255.0 ** 2 / mse)


def _result(verdict: str, reasons: list, details: dict) -> dict:
    return {"verdict": verdict, "reasons": reasons, "details": details}


def request_state(reading: dict | None) -> dict:
    """What one request came back as, recorded as its own reading: `image`; `refused` -- generate() raised, and the
    engine's own reason is kept (the request was not served, whatever the verifier's pair then lacks); `no_output` --
    it returned None or no image WITHOUT raising (a silently missing image); `not_run` -- there is no reading."""
    if not reading:
        return {"state": "not_run"}
    if reading.get("raised"):
        return {"state": "refused", "type": reading.get("type"), "error": str(reading.get("error") or "")[:400]}
    if reading.get("ok"):
        return {"state": "image"}
    return {"state": "no_output", "returned": reading.get("returned")}


def _missing(pair: str, requests: dict | None) -> str:
    """Why a pair has no psnr: the request the engine refused, with its reason, or the one that came back empty;
    else a plain missing image."""
    for key, st in sorted(((requests or {}).get(pair) or {}).items()):
        if st.get("state") == "refused":
            return f"the engine refused request {key} ({st.get('type')}): {st.get('error')}"
        if st.get("state") == "no_output":
            return f"request {key} returned no output without raising (returned {st.get('returned')})"
    return "an image of the pair is missing"


def equal_pair(p: float | None, bar: float) -> tuple[bool | None, str]:
    """(held, why) for a pair that must be the same image: None when a side is missing."""
    if p is None:
        return None, "an image of the pair is missing"
    return p >= bar, f"psnr {p:.2f} dB {'>=' if p >= bar else '<'} {bar:.1f}"


def request_isolation_abab(pairs: dict, policy: dict = POLICY) -> dict:
    """`request_isolation_abab`: A, B, A, B on one engine, B differing from A in prompt, seed, steps, cfg and size.
    pairs = {"A": psnr(r0, r2), "B": psnr(r1, r3)}. PASS iff both pairs are equal images; a missing image FAILS (the
    request itself failed); nothing measured is not_evaluated."""
    if not pairs:
        return _result("not_evaluated", ["no ABAB pair was measured"], {})
    reasons, ok = [], True
    for name, p in sorted(pairs.items()):
        held, why = equal_pair(p, policy["ISO_PSNR_DB"])
        if held is not True:
            ok = False
            reasons.append(f"{name} then {name} again: {why}")
    return _result("pass" if ok else "fail", reasons, {"pairs": pairs, "bar_db": policy["ISO_PSNR_DB"]})


def error_recovery(validation: dict | None, fault: dict | None, recovery_psnr: float | None,
                   policy: dict = POLICY) -> dict:
    """`error_recovery`: (1) an invalid request (a negative height) raises from generate(), before anything runs;
    (2) a request whose DiT forward fails (the verifier injects the fault) raises an error that NAMES the failure --
    B0 returns None, a silently missing image; (3) the next request is served, and equals the same request before
    the fault. `fault["fired"]` is the recorder's proof that the injection ran: without it the gate is not_evaluated,
    never a pass or a fail."""
    if not validation or not fault:
        return _result("not_evaluated", ["the error sequence did not run"], {})
    if not fault.get("fired"):
        return _result("not_evaluated", ["the injected fault was not observed; the request may not have reached "
                       "the instrumented forward, so candidate versus instrument attribution is unresolved"],
                       {"fault": fault, "attribution": "unresolved", "probe_reached": False})
    reasons = []
    if not validation.get("raised"):
        reasons.append(f"an invalid request did not raise (it returned {validation.get('returned')!r})")
    if not fault.get("raised"):
        what = "None" if fault.get("none") else repr(fault.get("returned"))
        reasons.append(f"a request whose DiT forward failed did not raise: generate() returned {what}, so the "
                       "failure reaches the caller as a missing image")
    elif policy["FAULT_MARKER"] not in str(fault.get("error") or ""):
        reasons.append(f"the error raised does not carry the failure's own message: {str(fault.get('error'))[:200]}")
    held, why = equal_pair(recovery_psnr, policy["ISO_PSNR_DB"])
    if held is not True:
        reasons.append(f"after the failure the engine did not serve the same request as before: {why}")
    return _result("fail" if reasons else "pass", reasons,
                   {"validation": validation, "fault": fault, "recovery_psnr": recovery_psnr})


def in_request_batching(list_psnr: list, n_list: int | None, npp_psnr: list, n_npp: int | None,
                        policy: dict = POLICY) -> dict:
    """`in_request_batching`, by B0's convention: result i of a list of prompts equals prompt i asked alone, and
    num_outputs_per_prompt = n gives output i the seed seed + i, so it equals a single request with that seed. Both
    are judged at the batch bar: B0 serves a list one prompt at a time (bit-identical), but an engine that batches
    the list is legal and its GEMM shapes may move the last bits. n_* are how many results came back (2 each)."""
    reasons = []
    if n_list is None and n_npp is None:
        return _result("not_evaluated", ["neither batched request ran"], {})
    if n_list != 2:
        reasons.append(f"a two-prompt request returned {n_list} result(s)")
    for i, p in enumerate(list_psnr):
        held, why = equal_pair(p, policy["BATCH_PSNR_DB"])
        if held is not True:
            reasons.append(f"prompt {i} of the list against the same prompt alone: {why}")
    if n_npp != 2:
        reasons.append(f"num_outputs_per_prompt=2 returned {n_npp} image(s)")
    for i, p in enumerate(npp_psnr):
        held, why = equal_pair(p, policy["BATCH_PSNR_DB"])
        if held is not True:
            reasons.append(f"output {i} against a single request with seed + {i}: {why}")
    return _result("fail" if reasons else "pass", reasons,
                   {"list_psnr": list_psnr, "n_list": n_list, "npp_psnr": npp_psnr, "n_npp": n_npp,
                    "bar_db": policy["BATCH_PSNR_DB"]})


def cfg1_single_row(calls_cfg1: int | None, steps_cfg1: int, calls_cfg7: int | None, steps_cfg7: int) -> dict:
    """`cfg1_single_row` (the pass count; the image is T1's astronaut_cfg1 bar): at cfg = 1 the DiT runs ONE pass per
    step -- no negative branch; at cfg > 1 it runs the two branches, as two passes or as one batched pass per step."""
    if calls_cfg1 is None or calls_cfg7 is None:
        return _result("not_evaluated", ["the DiT forwards of the two requests were not counted"], {})
    reasons = []
    if calls_cfg1 != steps_cfg1:
        reasons.append(f"cfg 1: {calls_cfg1} DiT forwards for {steps_cfg1} steps (expected one per step)")
    if calls_cfg7 not in (steps_cfg7, 2 * steps_cfg7):
        reasons.append(f"cfg > 1: {calls_cfg7} DiT forwards for {steps_cfg7} steps (expected 1 or 2 per step)")
    return _result("fail" if reasons else "pass", reasons,
                   {"cfg1": {"calls": calls_cfg1, "steps": steps_cfg1}, "cfg7": {"calls": calls_cfg7,
                                                                                 "steps": steps_cfg7}})


def prompt_semantics(neg_effect_psnr: float | None, trunc_psnr: float | None, default_neg_psnr: float | None,
                     policy: dict = POLICY, requests: dict | None = None) -> dict:
    '`prompt_semantics`, black-box through the public API:'


    if neg_effect_psnr is None and trunc_psnr is None and default_neg_psnr is None:
        return _result("not_evaluated", ["no prompt-semantics pair was measured"], {})
    reasons = []
    if neg_effect_psnr is None:
        reasons.append(f"the negative-prompt pair: {_missing('negative_effect', requests)}")
    elif neg_effect_psnr >= policy["EFFECT_MAX_PSNR_DB"]:
        reasons.append(f"an explicit negative prompt did not change the cfg > 1 image (psnr {neg_effect_psnr:.2f} dB "
                       f">= {policy['EFFECT_MAX_PSNR_DB']:.1f})")
    held, why = equal_pair(trunc_psnr, policy["ISO_PSNR_DB"])
    if held is False:
        reasons.append(f"words past the 256th T5 token changed the image: {why}")
    elif held is None:
        reasons.append(f"the truncation pair: {_missing('truncation', requests)}")
    held, why = equal_pair(default_neg_psnr, policy["ISO_PSNR_DB"])
    if held is False:
        reasons.append(f"the default negative prompt is not the empty one: {why}")
    elif held is None:
        reasons.append(f'negative_prompt="" (a legal request, the reference\'s own default): '
                       f"{_missing('default_negative', requests)}")
    return _result("fail" if reasons else "pass", reasons,
                   {"neg_effect_psnr": neg_effect_psnr, "trunc_psnr": trunc_psnr, "default_neg_psnr": default_neg_psnr,
                    "bars_db": {"effect_max": policy["EFFECT_MAX_PSNR_DB"], "equal": policy["ISO_PSNR_DB"]},
                    "requests": requests or {}})


def request_cancellation(c: dict | None, after_psnr: float | None, policy: dict = POLICY) -> dict:
    '`request_cancellation`.'


    if not c:
        return _result("not_evaluated", ["the cancellation sequence did not run"], {})
    if not c.get("slow_fired"):
        return _result("not_evaluated", ["the SLOW arm was not observed; the request may not have reached "
                       "the instrumented forward, so candidate versus instrument attribution is unresolved"],
                       {"cancel": c, "attribution": "unresolved", "probe_reached": False})
    reasons, not_eval, steps, bar = [], [], c.get("steps"), policy["CANCEL_MAX_CALLS_AFTER_ABORT"]
    if not c.get("abort_api"):
        reasons.append("the engine offers no abort: DiffGenerator.abort(request_id) is missing")
    run = c.get("running") or {}
    if run.get("state") == "image":
        reasons.append(f"the aborted running request came back with its image after {run.get('calls_total')} of "
                       f"{steps} DiT forwards")
    elif run.get("state") == "not_run":
        reasons.append("the running request was not read")
    elif run.get("calls_after_abort") is None or run["calls_after_abort"] > bar:
        reasons.append(f"the aborted running request ran {run.get('calls_after_abort')} DiT forwards after the abort "
                       f"was sent (bar {bar})")
    dl, t_out = c.get("deadline") or {}, c.get("timeout_s")
    if dl.get("state") == "image":
        reasons.append(f"a request with timeout_s={t_out} ran {dl.get('calls_total')} of {steps} DiT forwards and came "
                       f"back with its image after {dl.get('s')} s")
    elif not dl.get("calls_total"):
        if dl.get("type") == "TypeError" or dl.get("s") is None or float(dl["s"]) < float(t_out or 0):
            reasons.append(f"the timeout_s request never ran ({dl.get('state')}): "
                           f"{dl.get('error') or dl.get('returned')}")
        else:
            not_eval.append(f"the timeout_s={t_out} request ended after {dl.get('s')} s without a DiT forward: its "
                            f"deadline passed before step 0, so the stop in mid-denoising was not exercised")
    elif dl["calls_total"] >= (steps or 0):
        reasons.append(f"the timeout_s request ran all {steps} DiT forwards")
    elif dl.get("s") is None or dl["s"] > float(t_out or 0) + policy["DEADLINE_MAX_OVERRUN_S"]:
        reasons.append(f"the timeout_s={t_out} request ended after {dl.get('s')} s (bar timeout_s + "
                       f"{policy['DEADLINE_MAX_OVERRUN_S']:.1f} s)")
    q = c.get("queued") or {}
    ql, qq = q.get("long") or {}, q.get("queued") or {}
    for name, reading in (("running", run), ("deadline", dl), ("queued", qq)):
        if reading.get("state") == "refused":
            text = (str(reading.get("type") or "") + " " + str(reading.get("error") or "")).lower()
            allowed = ("abort", "cancel", "499") + (("timeout", "deadline") if name == "deadline" else ())
            if not any(marker in text for marker in allowed):
                reasons.append(f"{name} cancellation returned an unrelated exception: {text[:250]}")
    if ql.get("state") != "image":
        reasons.append(f"the request running next to the queued abort did not return its image ({ql.get('state')}): "
                       f"{ql.get('error') or ''}".rstrip(": "))
    answer = (q.get("abort") or {}).get("answer")
    reached = (q.get("wire_receipt") or {}).get("ok") is True
    public = q.get('public_outcome')
    if public is not None:


        reached = public.get('verdict') == 'pass'
        if public.get('verdict') == 'fail':
            reasons.append(public.get('reason') or 'queued cancellation control failed')
        elif not reached:
            not_eval.append(public.get('reason') or 'queued cancellation control unmeasured')
    if q.get("peer_row_work") is None:
        not_eval.append("queued cancellation peer row work unmeasured")
    elif q.get("peer_row_work") != q.get("peer_expected_steps"):
        reasons.append("queued member performed extra native row work while its peer was running")
    if qq.get("state") == "image" or (q.get("calls_after_long") or 0) > 0:
        reasons.append(f"the aborted queued request ran: {q.get('calls_after_long')} DiT forwards after the running "
                       f"request ended, and it came back as {qq.get('state')}")
    elif q.get("calls_after_long") is None:
        not_eval.append("the queued request's DiT forwards after the running one were not counted")
    elif not reached:
        not_eval.append(f"the queued request did not run, but nothing shows the abort reached it (the abort answered "
                        f"{answer if answer is not None else (q.get('abort') or {}).get('error')}; the request came "
                        f"back as {qq.get('state')} after {qq.get('s')} s)")
    held, why = equal_pair(after_psnr, policy["ISO_PSNR_DB"])
    if held is not True:
        reasons.append(f"after the cancellations the engine did not serve the same request as before: {why}")
    verdict = "fail" if reasons else ("not_evaluated" if not_eval else "pass")
    return _result(verdict, reasons + not_eval,
                   {"cancel": c, "after_psnr": after_psnr, "not_evaluated": not_eval,
                    "bars": {"calls_after_abort": bar, "deadline_overrun_s": policy["DEADLINE_MAX_OVERRUN_S"],
                             "equal_db": policy["ISO_PSNR_DB"]}})


def count_calls_between(events: list, t0: float, t1: float) -> int:
    """The recorder's `dit_call` events stamped inside [t0, t1] (one request's window; requests run one at a time)."""
    return sum(1 for e in events if e.get("event") == "dit_call" and t0 <= float(e.get("t") or 0.0) <= t1)
