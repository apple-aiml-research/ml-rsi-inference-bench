#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The autoregressive smoke session: test_ar_model_serves (P2P), on the ar_smoke recipe
(Qwen2.5-0.5B-Instruct).

Eight prompts of 1,024 tokens from prose windows (the prose pool decoded and re-tokenized with the
AR model's tokenizer). For each: prompt_logprobs 0 (teacher-forced) and a 64-token greedy
continuation (temperature 0, ignore_eos).

It guards the engine paths the task touches for other models: the FlashAttention version selection,
Triton and the metadata builders.
"""
from __future__ import annotations

import math

from eval.diffusion_gemma.serving.recipes import AR_MODEL
from eval.diffusion_gemma.serving import reads as R

from infb_verify import common
from infb_verify.common import SESSIONS

N_PROMPTS, PROMPT_LEN, CONTINUATION = 8, 1024, 64
OFFSET, STRIDE = 1_200_000, 5_000


def prompts() -> list[list[int]]:
    path = common.HIDDEN / "ar_smoke.json"
    if path.exists():
        return [p["ids"] for p in common.hidden("ar_smoke.json")["prompts"]]
    from transformers import AutoTokenizer

    from infb_verify.serving_nodes import prose, tok

    ar = AutoTokenizer.from_pretrained(AR_MODEL)
    out = []
    for k in range(N_PROMPTS):
        text = tok().decode(prose(1400, OFFSET + STRIDE * k))
        out.append(ar.encode(text, add_special_tokens=False)[:PROMPT_LEN])
    return out


def _prompt_lp(res: dict, ids: list[int]) -> list[float | None]:
    plp = ((res.get("choices") or [{}])[0].get("prompt_logprobs")) or []
    out = []
    for i, entry in enumerate(plp[1:], start=1):
        e = (entry or {}).get(str(ids[i])) if isinstance(entry, dict) else None
        out.append(e.get("logprob") if isinstance(e, dict) else None)
    return out


def test_ar_model_serves():
    srv = SESSIONS.get("ar_smoke")
    rec_path = common.HIDDEN / "ar_smoke.json"
    recorded = common.hidden("ar_smoke.json") if rec_path.exists() else None
    T = None if recorded is None or common.CALIBRATION else recorded.get("T_ar")
    rows, ok = [], True
    for k, ids in enumerate(prompts()):
        r = srv.post("/v1/completions", {"model": AR_MODEL, "prompt": ids, "max_tokens": CONTINUATION, "temperature": 0.0,
                                         "ignore_eos": True, "prompt_logprobs": 0}, timeout=600)
        n = (r.get("usage") or {}).get("completion_tokens")
        lp = _prompt_lp(r, ids)
        row = {"k": k, "status": r.get("_status"), "completion_tokens": n, "prompt_logprobs": lp}
        good = r.get("_status") == 200 and n == CONTINUATION and lp and all(v is not None and math.isfinite(v) for v in lp)
        if T is not None:
            ref = recorded["prompts"][k]["reference_logprobs"]
            row["k3"] = R.k3(ref, lp)
            good = good and row["k3"] <= T
        row["pass"] = bool(good)
        rows.append(row)
        ok &= bool(good)
    health = srv.get("/health")[0]
    rec = {"T_ar": T, "health": health, "rows": rows, "pass": bool(ok and health == 200)}
    common.record("test_ar_model_serves", rec)
    assert rec["pass"], {k: v for k, v in rec.items() if k != "rows"} | {"failing": [r["k"] for r in rows if not r["pass"]]}


def teardown_module(module):
    SESSIONS.stop()
