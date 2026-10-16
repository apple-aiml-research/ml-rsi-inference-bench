#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"The three precision profiles' gate ids."


from __future__ import annotations

from dataclasses import asdict

from _common import GateResult

PROFILES = ("bf16", "fp32", "mixed")


DEFAULT_PROFILE = "mixed"

DEFAULT_PROFILE_IDS = ("native_joint_attention", "attention_backend_selection")

PROFILE_GATE_IDS = (
    ("sd3_registered_bf16", "bf16", "sd3_registered"),
    ("sd3_dit_importable_bf16", "bf16", "sd3_dit_importable"),
    ("components_loaded_bf16", "bf16", "components_loaded"),
    ("native_text_encoders_bf16", "bf16", "native_text_encoders"),
    ("native_vae_bf16", "bf16", "native_vae"),
    ("native_transformer_bf16", "bf16", "native_transformer"),
    ("pipeline_runs_bf16", "bf16", "pipeline_runs"),
    ("image_similarity_smoke_bf16", "bf16", "image_similarity_smoke"),
    ("image_similarity_full_bf16", "bf16", "image_similarity_full"),
    ("prompt_conditioning_bf16", "bf16", "prompt_conditioning"),
    ("native_dit_forward_bf16", "bf16", "native_dit_forward"),
    ("runs_through_sglang_engine_bf16", "bf16", "runs_through_sglang_engine"),
    ("stage_step_count_bf16", "bf16", "stage_step_count"),
    ("precision_recorded_bf16", "bf16", "precision_recorded"),
    ("component_alignment_bf16", "bf16", "component_alignment"),
    ("sd3_registered_fp32", "fp32", "sd3_registered"),
    ("sd3_dit_importable_fp32", "fp32", "sd3_dit_importable"),
    ("components_loaded_fp32", "fp32", "components_loaded"),
    ("native_text_encoders_fp32", "fp32", "native_text_encoders"),
    ("native_vae_fp32", "fp32", "native_vae"),
    ("native_transformer_fp32", "fp32", "native_transformer"),
    ("pipeline_runs_fp32", "fp32", "pipeline_runs"),
    ("image_similarity_smoke_fp32", "fp32", "image_similarity_smoke"),
    ("image_similarity_full_fp32", "fp32", "image_similarity_full"),
    ("prompt_conditioning_fp32", "fp32", "prompt_conditioning"),
    ("native_dit_forward_fp32", "fp32", "native_dit_forward"),
    ("runs_through_sglang_engine_fp32", "fp32", "runs_through_sglang_engine"),
    ("stage_step_count_fp32", "fp32", "stage_step_count"),
    ("precision_recorded_fp32", "fp32", "precision_recorded"),
    ("component_alignment_fp32", "fp32", "component_alignment"),
    ("sd3_registered_mixed", "mixed", "sd3_registered"),
    ("sd3_dit_importable_mixed", "mixed", "sd3_dit_importable"),
    ("components_loaded_mixed", "mixed", "components_loaded"),
    ("native_text_encoders_mixed", "mixed", "native_text_encoders"),
    ("native_vae_mixed", "mixed", "native_vae"),
    ("native_transformer_mixed", "mixed", "native_transformer"),
    ("pipeline_runs_mixed", "mixed", "pipeline_runs"),
    ("image_similarity_smoke_mixed", "mixed", "image_similarity_smoke"),
    ("image_similarity_full_mixed", "mixed", "image_similarity_full"),
    ("prompt_conditioning_mixed", "mixed", "prompt_conditioning"),
    ("native_dit_forward_mixed", "mixed", "native_dit_forward"),
    ("runs_through_sglang_engine_mixed", "mixed", "runs_through_sglang_engine"),
    ("stage_step_count_mixed", "mixed", "stage_step_count"),
    ("precision_recorded_mixed", "mixed", "precision_recorded"),
    ("component_alignment_mixed", "mixed", "component_alignment"),
)


BASE_IDS = tuple(dict.fromkeys(b for _gate, _prof, b in PROFILE_GATE_IDS))


def profile_gate_id(base: str, profile: str) -> str:
    """The id `base` is written under for `profile`. KeyError for a gate the table does not name."""
    for gate, prof, b in PROFILE_GATE_IDS:
        if prof == profile and b == base:
            return gate
    raise KeyError(f"no profile id for gate {base!r} under profile {profile!r} (tests/profile_gates.py)")


def write_profile_gate(r: GateResult, profile: str, write) -> str:
    """Write `r` under its profile id through `write` (_common.write_gate) and return that id. The details keep the
    profile and the base gate, so a reader of one gate file knows which profile process wrote it. A gate in
    DEFAULT_PROFILE_IDS is written under its own name, and only by DEFAULT_PROFILE's process; anything else with no
    row raises KeyError -- never an unsuffixed write that another profile's process would overwrite."""
    if profile not in PROFILES:
        raise KeyError(f"{profile!r} is not a precision profile ({PROFILES})")
    fields = {k: v for k, v in asdict(r).items() if k not in ("gate", "details")}
    details = dict(r.details or {})
    details.setdefault("profile", profile)
    if r.gate in DEFAULT_PROFILE_IDS:
        if profile != DEFAULT_PROFILE:
            raise KeyError(f"{r.gate!r} is measured on the {DEFAULT_PROFILE} profile only, not on {profile!r}")
        write(r)
        return r.gate
    for gate, prof, base in PROFILE_GATE_IDS:
        if prof == profile and base == r.gate:
            details["profile_base_gate"] = base
            write(GateResult(gate=gate, details=details, **fields))
            return gate
    raise KeyError(f"no profile id for gate {r.gate!r} under profile {profile!r} (tests/profile_gates.py)")
