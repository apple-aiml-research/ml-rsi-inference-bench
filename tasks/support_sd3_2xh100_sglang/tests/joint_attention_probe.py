#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
'SD3 joint attention probe utilities.'


from __future__ import annotations

import math

CAPABILITY_JOINT = "native_joint_attention"
CAPABILITY_BACKEND = "attention_backend_selection"
LAYER = "T3"
CTX_TOKENS = 333
PATCH_PIXELS = 16
MIN_EFFECT = 1e-4
NOISE_FACTOR = 10.0
MATH_TOL = 2e-2
DELTA = 0.5
CAPTURE_CALLS = (0, 11, 23)
FA_IMPL = "FlashAttentionImpl"
SDPA_IMPL = "SDPAImpl"


def joint_len(h: int, w: int) -> int:
    """Image patches + context tokens for an h x w request (legal sizes are multiples of 16)."""
    return (int(h) // PATCH_PIXELS) * (int(w) // PATCH_PIXELS) + CTX_TOKENS


def forwards_between(summary, t0, t1) -> "list | None":
    """Pure: the `attn` summaries of the served DiT's forwards the worker recorded inside [t0, t1], or None when the
    record carries no attention accounting at all (a recorder without it, or no pre-hook), so "not recorded" never
    reads as "zero calls"."""
    s = summary if isinstance(summary, dict) else {}
    att = s.get("attention") or {}
    if (not att.get("window_installed") or not att.get("impl_wrapped_pids") or not isinstance(t0, (int, float))
            or not isinstance(t1, (int, float))):
        return None
    workers = set((s.get("recorder") or {}).get("worker_pids") or [])
    return [a for pid, t, a in (att.get("dit_attn_calls") or [])
            if (not workers or pid in workers) and isinstance(t, (int, float)) and t0 <= t <= t1
            and isinstance(a, dict)]


def _joint_calls(attn: dict, s_joint: int) -> int:
    """How many of one forward's calls had keys spanning the joint sequence."""
    return sum(int(c) for s, c in (attn.get("k_seq") or {}).items() if str(s).isdigit() and int(s) >= s_joint)


def request_joint_evidence(forwards, s_joint: int, n_blocks: int) -> dict:
    """Pure: one request's DiT forwards judged on the `joint` rule."""
    if forwards is None:
        return {"ok": False, "not_evaluated": True, "forwards": 0,
                "reasons": ["the worker recorded no engine-attention accounting for this request"]}
    if not forwards:
        return {"ok": False, "not_evaluated": False, "forwards": 0,
                "reasons": ["no DiT forward was recorded inside this request's window"]}
    joint = [_joint_calls(a, s_joint) for a in forwards]
    calls = [int(a.get("n") or 0) for a in forwards]
    impls: dict = {}
    k_seq: dict = {}
    for a in forwards:
        for k, v in (a.get("impls") or {}).items():
            impls[k] = impls.get(k, 0) + int(v)
        for k, v in (a.get("k_seq") or {}).items():
            k_seq[k] = k_seq.get(k, 0) + int(v)
    reasons = []
    short = [i for i, j in enumerate(joint) if j < n_blocks]
    if short:
        reasons.append(f"{len(short)} of {len(forwards)} DiT forwards made fewer than {n_blocks} engine-attention "
                       f"calls over the joint sequence (>= {s_joint} keys); min {min(joint)}, "
                       f"calls per forward {min(calls)}..{max(calls)}, key lengths seen {sorted(k_seq)}")
    return {"ok": not reasons, "not_evaluated": False, "forwards": len(forwards), "joint_calls_min": min(joint),
            "calls_min": min(calls), "calls_max": max(calls), "impls": impls, "k_seq": k_seq,
            "s_joint": s_joint, "reasons": reasons}


def attention_rows_reference(q_rows, k, v, scale: float):
    """fp64 softmax(q k^T * scale) v for the captured rows: q_rows [B, R, H, D], k/v [B, S, H, D] -> [B, R, H, D]."""
    import numpy as np

    q = np.asarray(q_rows, dtype=np.float64)
    kk = np.asarray(k, dtype=np.float64)
    vv = np.asarray(v, dtype=np.float64)
    s = np.einsum("brhd,bshd->bhrs", q, kk) * float(scale)
    s -= s.max(axis=-1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(axis=-1, keepdims=True)
    return np.einsum("bhrs,bshd->brhd", p, vv)


def rel_l2(a, b) -> float:
    "||a - b|| / ||b||. Where the reference's RMS is ~0 (< 1e-8) the relative error is undefined (pixel decision"


    import numpy as np

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    size = max(b.size, 1) ** 0.5
    nb, nd = float(np.linalg.norm(b)), float(np.linalg.norm(a - b))
    if nb / size < 1e-8:
        return 0.0 if nd / size < 1e-8 else float("nan")
    return nd / nb


def capture_math(cap: dict, head_dim: int) -> dict:
    """Pure: one captured call against the fp64 recompute at SD3's scale 1/sqrt(head_dim)."""
    try:
        ref = attention_rows_reference(cap["q_rows"], cap["k"], cap["v"], 1.0 / math.sqrt(head_dim))
        r = rel_l2(cap["out_rows"], ref)
        return {"rel_l2": r, "finite": math.isfinite(r), "q_shape": list(map(int, cap.get("q_shape", []))),
                "k_shape": list(map(int, cap.get("k_shape", []))), "rows": int(len(cap.get("rows", [])))}
    except Exception as exc:  # noqa: BLE001 - an unreadable capture is a failed check with its reason
        return {"rel_l2": None, "finite": False, "error": f"{type(exc).__name__}: {exc}"[:300]}


def probe_evidence(probe: "dict | None", math_results: dict, n_blocks: int, s_joint: int,
                   min_effect: float = MIN_EFFECT, noise_factor: float = NOISE_FACTOR,
                   math_tol: float = MATH_TOL) -> dict:
    """Pure: the armed forward judged on the `causal` and `math` rules."""
    if not isinstance(probe, dict):
        return {"ok": False, "not_evaluated": True, "reasons": ["the worker recorded no joint-attention probe"]}
    reasons = []
    if probe.get("error"):
        reasons.append(f"the probe failed in the worker: {probe['error']}")
    n = int(probe.get("n") or 0)
    per = [x for x in (probe.get("per_call_rel") or []) if isinstance(x, (int, float))]
    rep = probe.get("repeat_rel")
    rep_ok = isinstance(rep, (int, float)) and math.isfinite(rep)
    floor = max(min_effect, noise_factor * rep) if rep_ok else None
    if n < n_blocks:
        reasons.append(f"the armed forward made {n} engine-attention calls, fewer than the {n_blocks} blocks")
    if not rep_ok:
        reasons.append("no unchanged re-run was measured, so there is no noise floor")
    if len(per) < n:
        reasons.append(f"only {len(per)} of {n} calls were perturbed")
    dead = [j for j, x in enumerate(per) if floor is not None and not (x >= floor)]
    if dead:
        reasons.append(f"{len(dead)} of {len(per)} attention calls do not decide the DiT output (scaling their "
                       f"output by {1 + float(probe.get('delta') or 0):.2f} moved it by < {floor:.3g}): calls "
                       f"{dead[:12]}")
    bad_math = {j: m for j, m in (math_results or {}).items()
                if not (m.get("finite") and m.get("rel_l2") is not None and m["rel_l2"] <= math_tol)}
    if not math_results:
        reasons.append("no captured attention call to recompute")
    for j, m in sorted(bad_math.items()):
        reasons.append(f"call {j}: output vs fp64 softmax(q k^T / sqrt(d)) v rel_l2 "
                       f"{m.get('rel_l2')} > {math_tol}" + (f" ({m['error']})" if m.get("error") else ""))
    short_keys = {j: m for j, m in (math_results or {}).items()
                  if len(m.get("k_shape") or []) >= 2 and int(m["k_shape"][1]) < s_joint}
    for j, m in sorted(short_keys.items()):
        reasons.append(f"call {j}: keys cover {m['k_shape'][1]} tokens, not the joint {s_joint}")
    return {"ok": not reasons, "not_evaluated": False, "n": n, "repeat_rel": rep, "effect_floor": floor,
            "per_call_rel_min": min(per) if per else None, "per_call_rel_max": max(per) if per else None,
            "dead_calls": dead, "math": math_results, "reasons": reasons}


def joint_attention_verdict(requests: dict, probe_ev: dict) -> dict:
    """Pure: the gate's verdict from every scored request's `joint` evidence and the probe's."""
    reasons = []
    not_eval = False
    for rid, ev in (requests or {}).items():
        not_eval |= bool(ev.get("not_evaluated"))
        reasons += [f"{rid}: {r}" for r in ev.get("reasons") or []]
    not_eval |= bool(probe_ev.get("not_evaluated"))
    reasons += [f"probe: {r}" for r in probe_ev.get("reasons") or []]
    if not requests:
        reasons.append("no scored request to judge")
    return {"passed": not reasons, "not_evaluated": bool(not_eval and reasons), "reasons": reasons}


def _impl_kinds(forwards) -> dict:
    out: dict = {}
    for a in forwards or []:
        for k, v in (a.get("impls") or {}).items():
            out[k.rsplit(".", 1)[-1]] = out.get(k.rsplit(".", 1)[-1], 0) + int(v)
    return out


def backend_evidence(default_forwards, sdpa_forwards, sdpa_image: dict, n_blocks: int) -> dict:
    """Pure: the default engine ran FlashAttention, the torch_sdpa engine ran SDPA, the same number of calls per
    forward, and the SDPA engine's image cleared the fixture's bars."""
    reasons = []
    if default_forwards is None or sdpa_forwards is None:
        return {"passed": False, "not_evaluated": True,
                "reasons": ["an engine's attention accounting was not recorded"
                            + ("" if default_forwards is not None else " (default engine)")
                            + ("" if sdpa_forwards is not None else " (torch_sdpa engine)")]}
    d_kinds, s_kinds = _impl_kinds(default_forwards), _impl_kinds(sdpa_forwards)
    if not default_forwards:
        reasons.append("the default engine recorded no DiT forward")
    elif set(d_kinds) != {FA_IMPL}:
        reasons.append(f"with no --attention-backend the DiT's attention ran on {sorted(d_kinds) or 'nothing'}, "
                       f"not {FA_IMPL} (the platform default)")
    if not sdpa_forwards:
        reasons.append("the torch_sdpa engine recorded no DiT forward")
    elif set(s_kinds) != {SDPA_IMPL}:
        reasons.append(f"with --attention-backend torch_sdpa the DiT's attention ran on {sorted(s_kinds) or 'nothing'}"
                       f", not {SDPA_IMPL}")
    d_n = sorted({int(a.get("n") or 0) for a in default_forwards or []})
    s_n = sorted({int(a.get("n") or 0) for a in sdpa_forwards or []})
    if d_n and s_n and d_n != s_n:
        reasons.append(f"calls per forward differ between backends: default {d_n}, torch_sdpa {s_n}")
    if (d_n and min(d_n) < n_blocks) or (s_n and min(s_n) < n_blocks):
        reasons.append(f"fewer than {n_blocks} attention calls per forward (default {d_n}, torch_sdpa {s_n})")
    img = sdpa_image if isinstance(sdpa_image, dict) else {}
    if not img.get("match"):
        reasons.append("the torch_sdpa engine's image did not clear the fixture's bars: "
                       + (img.get("note") or ", ".join(f"{k}={img.get(k)}" for k in ("psnr", "ssim", "clip_img",
                                                                                      "clip_txt"))))
    return {"passed": not reasons, "not_evaluated": False, "reasons": reasons, "default_impls": d_kinds,
            "sdpa_impls": s_kinds, "calls_per_forward": {"default": d_n, "torch_sdpa": s_n}}
