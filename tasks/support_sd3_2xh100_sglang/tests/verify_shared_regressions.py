#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Regression checks for shared weight selection and Flux text encoding."""


from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import GateResult, safe_run  # noqa: E402

SELECTION_GATE = "shared_loader_selection"
FLUX_GATE = "flux_text_encoding"
SELECTION_MODULE = "sglang.multimodal_gen.runtime.loader.transformer_load_utils"
SELECTION_FN = "_filter_duplicate_precision_variant_safetensors"


SELECTION_CASES = (
    ("canonical + fp16", ["d/model.safetensors", "d/model.fp16.safetensors"], ["d/model.safetensors"]),
    ("sharded canonical + fp16",
     ["d/diffusion_pytorch_model-00001-of-00002.safetensors",
      "d/diffusion_pytorch_model-00002-of-00002.safetensors",
      "d/diffusion_pytorch_model.fp16-00001-of-00002.safetensors",
      "d/diffusion_pytorch_model.fp16-00002-of-00002.safetensors"],
     ["d/diffusion_pytorch_model-00001-of-00002.safetensors",
      "d/diffusion_pytorch_model-00002-of-00002.safetensors"]),
    ("fp16 only", ["d/model.fp16.safetensors"], ["d/model.fp16.safetensors"]),
    ("two precision-only variants", ["d/model.fp16.safetensors", "d/model.bf16.safetensors"],
     ["d/model.fp16.safetensors", "d/model.bf16.safetensors"]),
)


def selection_verdict(select) -> dict:
    """Pure: run every layout through `select` and compare with the baseline's selection."""
    per = []
    for case, given, want in SELECTION_CASES:
        rec = {"case": case, "given": list(given), "want": list(want)}
        try:
            got = list(select(list(given)))
            rec["got"] = got
            rec["ok"] = sorted(got) == sorted(want) and len(got) == len(set(got))
        except Exception as exc:  # noqa: BLE001 - a raising selector is a failing selector
            rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
            rec["ok"] = False
        per.append(rec)
    return {"passed": bool(per) and all(p["ok"] for p in per), "per": per}


def run_selection() -> GateResult:
    try:
        import importlib

        select = getattr(importlib.import_module(SELECTION_MODULE), SELECTION_FN)
    except Exception as exc:  # noqa: BLE001
        return GateResult(gate=SELECTION_GATE, passed=False, score=0.0,
                          error=(f"the shared selector {SELECTION_MODULE}.{SELECTION_FN} could not be imported: "
                                 f"{type(exc).__name__}: {exc} -- fail-closed")[:500],
                          details={"not_evaluated": True})
    v = selection_verdict(select)
    bad = [p["case"] for p in v["per"] if not p["ok"]]
    return GateResult(gate=SELECTION_GATE, passed=v["passed"], score=1.0 if v["passed"] else 0.0,
                      error=None if v["passed"] else f"the shared loader selects differently from baseline: {bad}",
                      details={"per": v["per"], "selector": f"{SELECTION_MODULE}.{SELECTION_FN}"})


def flux_verdict(observed: dict) -> dict:
    """Pure: what Flux v1's text encoding must look like, from the harness's plain-data observations.

    observed = {"mask_passed": [bool per encoder], "pooled_sources": [names], "embed_sources": [names]}.
    Names are the fake tensors the values were identified as ("clip.pooler_output", ...), or "unknown".
    """
    reasons = []
    masks = list(observed.get("mask_passed") or [])
    if len(masks) != 2:
        reasons.append(f"expected two encoder calls (CLIP, T5), saw {len(masks)}")
    elif any(masks):
        reasons.append(f"Flux v1 encoders must receive no attention mask; passed per encoder: {masks}")
    pooled = list(observed.get("pooled_sources") or [])
    if pooled != ["clip.pooler_output"]:
        reasons.append(f"Flux v1 pools CLIP's pooler_output and nothing else; got {pooled}")
    embeds = list(observed.get("embed_sources") or [])
    if embeds != ["clip.pooler_output", "t5.last_hidden_state"]:
        reasons.append(f"embeddings must be what Flux's own postprocess functions return; got {embeds}")
    return {"passed": not reasons, "reasons": reasons}


def _flux_observe() -> dict:
    """Drive the candidate's TextEncodingStage.encode_text with Flux v1's real config and fakes (needs torch)."""
    import types

    import torch
    from sglang.multimodal_gen.configs.pipeline_configs.flux import FluxPipelineConfig
    from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import TextEncodingStage

    torch.manual_seed(0)
    received: list = []

    class _Enc(dict):
        def to(self, *a, **k):
            return self

    class _Tok:
        def __call__(self, prompts, **kw):
            n = len(prompts)
            return _Enc(input_ids=torch.ones((n, 8), dtype=torch.long),
                        attention_mask=torch.ones((n, 8), dtype=torch.long))

    class _Out:
        def __init__(self, last, pooled):
            self.last_hidden_state, self.pooler_output = last, pooled
            self.hidden_states, self.attentions = (last,), None

    class _FakeEncoder:
        uses_sglang_forward_context = False

        def __init__(self, name, pooled):
            self.name = name
            self.out = _Out(torch.randn(1, 8, 4), torch.randn(1, 4) if pooled else None)

        def forward(self, input_ids=None, output_hidden_states=None, attention_mask=None, **kw):
            received.append({"encoder": self.name, "has_mask": attention_mask is not None})
            return self.out

        def __call__(self, **kw):
            return self.forward(**kw)

    clip, t5 = _FakeEncoder("clip", True), _FakeEncoder("t5", False)
    server_args = types.SimpleNamespace(pipeline_config=FluxPipelineConfig(), comfyui_mode=False)


    from sglang.multimodal_gen.runtime import server_args as _sa

    previous = getattr(_sa, "_global_server_args", None)
    _sa.set_global_server_args(server_args)
    try:
        stage = TextEncodingStage(text_encoders=[clip, t5], tokenizers=[_Tok(), _Tok()])
        result = stage.encode_text(["a red cube on a white table"], server_args, encoder_index=[0, 1],
                                   return_type="list", device="cpu")
    finally:
        _sa._global_server_args = previous
    embeds, pooled = result[0], result[-1]
    named = {"clip.pooler_output": clip.out.pooler_output, "clip.last_hidden_state": clip.out.last_hidden_state,
             "t5.last_hidden_state": t5.out.last_hidden_state}

    def _name(t):
        for k, v in named.items():
            if isinstance(t, torch.Tensor) and t.shape == v.shape and torch.equal(t, v):
                return k
        return "unknown"

    return {"mask_passed": [r["has_mask"] for r in received], "encoders_called": [r["encoder"] for r in received],
            "pooled_sources": [_name(t) for t in pooled], "embed_sources": [_name(t) for t in embeds]}


def run_flux() -> GateResult:
    try:
        observed = _flux_observe()
    except Exception as exc:  # noqa: BLE001
        return GateResult(gate=FLUX_GATE, passed=False, score=0.0,
                          error=(f"the Flux text-encoding harness could not run: {type(exc).__name__}: {exc} "
                                 "-- fail-closed")[:500],
                          details={"not_evaluated": True, "trace": traceback.format_exc()[-1500:]})
    v = flux_verdict(observed)
    return GateResult(gate=FLUX_GATE, passed=v["passed"], score=1.0 if v["passed"] else 0.0,
                      error=None if v["passed"] else "; ".join(v["reasons"]),
                      details={"observed": observed, "config": "FluxPipelineConfig (Flux v1)"})


if __name__ == "__main__":
    rc = safe_run(run_selection, SELECTION_GATE, fail_exit_code=0)
    rc |= safe_run(run_flux, FLUX_GATE, fail_exit_code=0)
    sys.exit(rc)
