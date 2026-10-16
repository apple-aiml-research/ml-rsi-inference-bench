#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
'SD3 day0 policy utilities.'


from __future__ import annotations


MARLIN_POLICY = "M1"
KERNEL_NAME_POLICY = "K2"
MARLIN_POLICIES = ("M1", "M2")
KERNEL_NAME_POLICIES = ("K1", "K2")
FLASHINFER_BOUNDARY_POLICY = "F1"
FLASHINFER_BOUNDARY_POLICIES = ("F1", "F2")


RESULT_CONSUMPTION_POLICY = "R1"
RESULT_CONSUMPTION_POLICIES = ("R0", "R1")


FLASHINFER_PINNED_ENTRIES = {"0.6.7.post3": ("flashinfer.gemm.gemm_base.fp8_blockscale_gemm_sm90",)}
ACTIVATION_QUANT = {"scheme": "dynamic", "granularity": "per-token", "group_size": 128, "dtype": "float8_e4m3fn"}
WEIGHT_QUANT = {"dtype": "float8_e4m3fn", "block": [128, 128], "scale": "weight_scale_inv, float32, a dequant multiplier"}

POSITIVE_EVIDENCE_SOURCES = ("dispatch_fp8_operand_gemm", "triton_fp8_operand_launch", "ncu_fp8_tensor_pipe")


BETA_W8A8_MIN = 0.90
BETA_W8A16_MAX = 0.20
BETA_MIN_LAYERS = 16


Q_MODULES = 285
Q_FP8_WEIGHT_BYTES = 1_337_720_832


LORA_RECIPE = "lora_recipe_v4.json"
LORA_SEQUENCE = "request_sequence_v4.json"
LORA_MULTI_GATE = "lora_multi_adapter_composition"


LORA_PUBLIC_SUPPLEMENT = "linoyts/Yarn_art_SD3_LoRA@127ead02fd14754d490977db6dfe163765eed09f"


LORA_COEF = 2.0


LORA_FLOOR_MIN = 1e-4


LORA_DISCRIMINATION_MIN = 2.0 * LORA_COEF
LORA_SCORED_POINTS = (("astronaut", 0), ("astronaut", 13), ("astronaut_cfg1", 13))
LORA_DIAGNOSTIC_POINTS = (("astronaut", 27), ("astronaut_cfg1", 0))

LORA_CERTIFICATION_SCOPE = ("certifies the DiT at the scored reading points only (astronaut@0, astronaut@13, "
                            "astronaut_cfg1@13); step 27 is diagnostic, so this is NOT a claim that every denoising "
                            "stage is certified")


LORA_REPEAT_COEF = 2.0


LORA_ISO_PSNR_MARGIN_DB = 3.0
LORA_ISO_SSIM_MARGIN = 0.01


LORA_MARGIN_MAX = 0.414


LORA_FLOOR_RATIO_MAX = 1.689


LORA_POWER_MARGIN = LORA_MARGIN_MAX / LORA_FLOOR_RATIO_MAX


LORA_TUNING_SET = ("calibration_case_1", "calibration_case_2", "calibration_case_3",
                   "calibration_case_4", "calibration_case_5", "calibration_case_6")
LORA_HELD_OUT_SET = ("held_out_case_1", "held_out_case_2")


def fp8_policy() -> dict:
    """The FP8 switches as recorded in every FP8 gate file."""
    return {"marlin": MARLIN_POLICY, "kernel_names": KERNEL_NAME_POLICY, "activation_quant": ACTIVATION_QUANT,
            "weight_quant": WEIGHT_QUANT, "positive_evidence_sources": list(POSITIVE_EVIDENCE_SOURCES),
            "beta_w8a8_min": BETA_W8A8_MIN, "beta_w8a16_max": BETA_W8A16_MAX, "beta_min_layers": BETA_MIN_LAYERS,
            "q_modules": Q_MODULES, "flashinfer_boundary": FLASHINFER_BOUNDARY_POLICY,
            "result_consumption": RESULT_CONSUMPTION_POLICY,
            "flashinfer_pinned_entries": {k: list(v) for k, v in FLASHINFER_PINNED_ENTRIES.items()},
            "calibration_status": "Experimental FP8 policy outside core49 scoring",
            "calibrated": False}


def lora_policy() -> dict:
    """The LoRA switches as recorded in every LoRA gate file."""
    return {"recipe": LORA_RECIPE, "sequence": LORA_SEQUENCE, "statistic": "v2", "margin_max": LORA_MARGIN_MAX,
            "floor_ratio_max": LORA_FLOOR_RATIO_MAX, "power_margin": LORA_POWER_MARGIN,
            "multi_adapter_gate": {"gate": LORA_MULTI_GATE, "requests": "r12-r13 (v4)", "statistic": "v2, same bars",
                                   "grid": ["base", "state", "first_only", "last_only", "first_at_last_scale"],
                                   "pair": "r12/r13 at the repeat pair's level", "calibrated": False},
            "frozen_sets": "Requests, comparison grids and repeat pairs are fixed by the "
                           "sequence; an unread one is not_evaluated, a request the engine did not serve fails",
            "tuning_set": list(LORA_TUNING_SET), "held_out_set": list(LORA_HELD_OUT_SET),
            "v1_recorded": {"coef": LORA_COEF, "floor_min": LORA_FLOOR_MIN,
                            "discrimination_min": LORA_DISCRIMINATION_MIN},
            "scored_points": [list(p) for p in LORA_SCORED_POINTS],
            "diagnostic_points": [list(p) for p in LORA_DIAGNOSTIC_POINTS], "repeat_coef": LORA_REPEAT_COEF,
            "isolation_image_layer": {"metric": "psnr + ssim (verify_callsite_main's definitions)",
                                      "psnr_margin_db": LORA_ISO_PSNR_MARGIN_DB, "ssim_margin": LORA_ISO_SSIM_MARGIN,
                                      "calibrated": False, "rel_l2": "recorded as a diagnostic, not judged"},
            "certification_scope": LORA_CERTIFICATION_SCOPE, "public_supplement": LORA_PUBLIC_SUPPLEMENT,
            "calibration_status": "Experimental LoRA policy outside core49 scoring",
            "calibrated": False}
