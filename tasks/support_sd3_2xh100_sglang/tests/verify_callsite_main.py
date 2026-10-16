#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
'SD3 native execution and independent per-precision image correctness.'
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (GateResult, HIDDEN_DIR, NoImageProduced, RESULTS_DIR, WORK_DIR,  # noqa: E402
                     save_stderr_log, to_image_array)
from _common import write_gate as _write_gate_file  # noqa: E402
import native_execution
import image_thresholds as _IMAGE_BARS
import profile_gates  # noqa: E402

_PG = profile_gates


def write_gate(r: GateResult) -> None:
    """Write a gate under its profile-specific id when SD3_PRECISION_PROFILE is set."""


    if PRECISION_PROFILE:
        _PG.write_profile_gate(r, PRECISION_PROFILE, _write_gate_file)
    else:
        _write_gate_file(r)


def _gate_file(gate: str) -> Path:
    """Where `gate`'s file is: under its profile id in profile mode (a gate this process wrote), else its own name."""
    if PRECISION_PROFILE and gate in _PG.BASE_IDS:
        return RESULTS_DIR / f"{_PG.profile_gate_id(gate, PRECISION_PROFILE)}.gate.json"
    return RESULTS_DIR / f"{gate}.gate.json"


_cprobe = None
_CPROBE_IMPORT_ERROR = None
try:
    import component_probe as _cprobe  # noqa: E402
except Exception as _cprobe_exc:  # noqa: BLE001
    _CPROBE_IMPORT_ERROR = f"{type(_cprobe_exc).__name__}: {_cprobe_exc}"

_jattn = None
_JATTN_IMPORT_ERROR = None
try:
    import joint_attention_probe as _jattn  # noqa: E402
except Exception as _jattn_exc:  # noqa: BLE001
    _JATTN_IMPORT_ERROR = f"{type(_jattn_exc).__name__}: {_jattn_exc}"

def _resolve_model() -> str:
    'Resolve the explicitly selected or offline-staged checkpoint. The workspace model symlink also supports public Docker mounts; no runtime download is prepared here.'

    explicit = os.environ.get("SD3_MODEL")
    if explicit:
        return explicit
    for name in ("INFB_MODEL_PATH",):
        staged = os.environ.get(name)
        if staged and Path(staged).is_dir():
            return staged
    link = Path(os.environ.get("INFB_WORKSPACE", "/app")) / "model"
    if link.is_dir():
        return str(link.resolve())
    return "stabilityai/stable-diffusion-3-medium-diffusers"


MODEL = _resolve_model()


def _golden_dir():


    inrun = os.environ.get("SD3_GOLDEN_DIR")
    if inrun and (Path(inrun) / "sd3_golden.json").exists():
        return Path(inrun)
    try:
        rec = json.loads((WORK_DIR / "effective_target_env.json").read_text())
        gd = rec.get("golden_dir")
        if gd and (Path(gd) / "sd3_golden.json").exists():
            return Path(gd)
    except Exception:
        pass
    for ev in ("INFB_VLLM_TARGET_ENV_ARTIFACT", "INFB_SGLANG_TARGET_ENV_ARTIFACT"):
        art = os.environ.get(ev)
        if art and (Path(art) / "golden" / "sd3_golden.json").exists():
            return Path(art) / "golden"
    return HIDDEN_DIR


GOLDEN_DIR = _golden_dir()
GOLDEN = GOLDEN_DIR / "sd3_golden.json"

PSNR_MIN = float(os.environ.get("SD3_SIM_PSNR_MIN", "16.5"))
SSIM_MIN = float(os.environ.get("SD3_SIM_SSIM_MIN", "0.70"))
CLIP_MIN = float(os.environ.get("SD3_SIM_CLIP_MIN", "0.95"))
CLIP_TXT_MIN = float(os.environ.get("SD3_CLIP_TXT_MIN", "0.25"))
SHARED_BARS = {"psnr": PSNR_MIN, "ssim": SSIM_MIN, "clip_img": CLIP_MIN, "clip_txt": CLIP_TXT_MIN}
SIM_THRESHOLD_REL_L2 = float(os.environ.get("SD3_SIM_THRESHOLD_REL_L2", "0.05"))


FULL_RATE_MIN = float(os.environ.get("SD3_SIM_FULL_RATE_MIN", "0.95"))


PROMPT_COND_FACTOR = float(os.environ.get("SD3_PROMPT_COND_FACTOR", "2"))


PC_SAME_BARS = {"psnr": float(os.environ.get("SD3_PC_SAME_PSNR", "16.5")),
                "ssim": float(os.environ.get("SD3_PC_SAME_SSIM", "0.70")),
                "clip_img": float(os.environ.get("SD3_PC_SAME_CLIP_IMG", "0.95"))}


STEP_COUNT_TOLERANCE = int(os.environ.get("SD3_STEP_COUNT_TOLERANCE", "0"))


CFG1_PSNR_ORACLE = float(os.environ.get("SD3_SIM_CFG1_PSNR_ORACLE", "37.631"))
CFG1_PSNR_COEF = float(os.environ.get("SD3_SIM_CFG1_PSNR_COEF", "0.80"))


SG_PSNR_LEGAL = float(os.environ.get("SD3_SIM_STAINEDGLASS_PSNR_LEGAL", "15.988563284254717"))
SG_PSNR_COEF = float(os.environ.get("SD3_SIM_STAINEDGLASS_PSNR_COEF", "0.95"))
SG_SSIM_LEGAL = float(os.environ.get("SD3_SIM_STAINEDGLASS_SSIM_LEGAL", "0.6740803839646289"))
SG_SSIM_COEF = float(os.environ.get("SD3_SIM_STAINEDGLASS_SSIM_COEF", "0.95"))
SG_CLIP_LEGAL = float(os.environ.get("SD3_SIM_STAINEDGLASS_CLIP_LEGAL", "0.948726236820221"))
SG_CLIP_COEF = float(os.environ.get("SD3_SIM_STAINEDGLASS_CLIP_COEF", "0.98"))


FIXTURE_BARS = {"stainedglass": {"psnr": SG_PSNR_LEGAL * SG_PSNR_COEF, "ssim": SG_SSIM_LEGAL * SG_SSIM_COEF,
                                 "clip_img": SG_CLIP_LEGAL * SG_CLIP_COEF}}


FX_SSIM_LEGAL = json.loads(os.environ.get("SD3_SIM_FIXTURE_SSIM_LEGAL", '{"redlacquer": 0.8619625577950888, "ramen": 0.8776179586988145, "origami": 0.9110763138120704, "candles": 0.8953151194284058, "astronaut_cfg1": 0.9925021891036724, "stainedglass_cfg1": 0.9621614427532686}'))
FX_SSIM_COEF = float(os.environ.get("SD3_SIM_FIXTURE_SSIM_COEF", "0.95"))
FX_CLIP_LEGAL = json.loads(os.environ.get("SD3_SIM_FIXTURE_CLIP_LEGAL", '{"redlacquer": 0.9895836710929871, "ramen": 0.9860738515853882, "astronaut_cfg1": 0.9970365762710571}'))
FX_CLIP_COEF = float(os.environ.get("SD3_SIM_FIXTURE_CLIP_COEF", "0.98"))
for _fid, _anchor in FX_SSIM_LEGAL.items():
    FIXTURE_BARS.setdefault(_fid, {})["ssim"] = float(_anchor) * FX_SSIM_COEF
for _fid, _anchor in FX_CLIP_LEGAL.items():
    FIXTURE_BARS.setdefault(_fid, {})["clip_img"] = float(_anchor) * FX_CLIP_COEF


MOUNTAIN_CLIP_TXT_MIN = float(os.environ.get("SD3_SIM_MOUNTAIN_CLIP_TXT_MIN", "0.2430"))
FIXTURE_BARS.setdefault("mountain", {})["clip_txt"] = MOUNTAIN_CLIP_TXT_MIN


REF_SETS = ("bf16", "fp32")
REF_TIE = "bf16"
REF_DIRS = {"bf16": GOLDEN_DIR, "fp32": GOLDEN_DIR / "fp32", "mixed": GOLDEN_DIR / "mixed"}


PRECISION_PROFILE = os.environ.get("SD3_PRECISION_PROFILE") or None


FP32_FIXTURE_BARS = {_fid: dict(_b) for _fid, _b in FIXTURE_BARS.items()}
BF16_FIXTURE_LEGAL = json.loads(os.environ.get("SD3_SIM_BF16_FIXTURE_LEGAL", '{"redlacquer": {"ssim": 0.8771486951771649}, "ramen": {"ssim": 0.8820650672602586}, "astronaut_cfg1": {"ssim": 0.994639555089177, "clip_img": 0.9971283078193665}}'))
FP32_FIXTURE_LEGAL = json.loads(os.environ.get("SD3_SIM_FP32_FIXTURE_LEGAL", '{"redlacquer": {"ssim": 0.8783076883214225}, "candles": {"ssim": 0.9188329426979082}, "ramen": {"clip_img": 0.9887076020240784}}'))
for _table, _own_bars in ((FIXTURE_BARS, BF16_FIXTURE_LEGAL), (FP32_FIXTURE_BARS, FP32_FIXTURE_LEGAL)):
    for _fid, _own in _own_bars.items():
        for _m, _anchor in _own.items():
            _bar = float(_anchor) * {"ssim": FX_SSIM_COEF, "clip_img": FX_CLIP_COEF}[_m]

            _table.setdefault(_fid, {})[_m] = max(_bar, _table.get(_fid, {}).get(_m, float("-inf")))
BARS_BY_SET = {"bf16": FIXTURE_BARS, "fp32": FP32_FIXTURE_BARS}

CFG1_ANCHOR_BY_SET = {"bf16": CFG1_PSNR_ORACLE, "fp32": CFG1_PSNR_ORACLE}
CFG1_COEF_BY_SET = {"bf16": CFG1_PSNR_COEF, "fp32": CFG1_PSNR_COEF}

BARS_BY_SET = {p: dict(_IMAGE_BARS.TABLE["profiles"][p]["fixtures"]) for p in _IMAGE_BARS.PROFILES}
CFG1_ANCHOR_BY_SET = {p: _IMAGE_BARS.TABLE["profiles"][p]["cfg1_psnr"] for p in _IMAGE_BARS.PROFILES}
CFG1_COEF_BY_SET = {p: 1.0 for p in _IMAGE_BARS.PROFILES}
if PRECISION_PROFILE:
    import sd3_precision_profiles as _PREC

    if PRECISION_PROFILE not in _PREC.PROFILES:
        raise SystemExit(f"[callsite] SD3_PRECISION_PROFILE={PRECISION_PROFILE!r} is not a profile "
                         f"({sorted(_PREC.PROFILES)})")


    REF_SETS = (PRECISION_PROFILE,)
    REF_TIE = PRECISION_PROFILE


PROFILE_BARS_STATE = {p: _IMAGE_BARS.STATE for p in _IMAGE_BARS.PROFILES}


def _profile_env() -> dict:
    """Build the precision environment and execution witness settings inherited by engine workers."""

    if not PRECISION_PROFILE:
        return {}
    env = dict(_PREC.PROFILES[PRECISION_PROFILE]["env"])
    if PRECISION_PROFILE == "fp32" and os.environ.get("SD3_PRECISION_TF32") == "on":
        env = {"TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1"}
    env["INFB_SD3_PRECISION_WITNESS"] = "1"
    return env


def missing_reference_sets(fixtures, ref_dirs: dict) -> dict:
    """{set: [ids]} for every reference set that cannot score every fixture -- its directory, its spec, or a fixture's
    `.npy` absent. {} means all required sets are complete. Pure over paths; nothing is loaded."""
    out = {}
    for name, d in ref_dirs.items():
        d = Path(d)
        if not (d / "sd3_golden.json").is_file():
            out[name] = [str(fx.get("id")) for fx in fixtures]
            continue
        gone = [str(fx.get("id")) for fx in fixtures if not fx.get("golden_npy") or not (d / fx["golden_npy"]).is_file()]
        if gone:
            out[name] = gone
    return out


PROXIMITY_METRIC = "mean psnr over the scored fixtures, higher is closer"
_PROXIMITY_TIE_DB = 1e-9


def reference_proximity(per_by_set: dict, scored_ids) -> dict:
    """{set: mean psnr over the scored fixtures every set has a reading for}. A fixture with no reading in some set
    (no image) enters no set's mean, so the sets are compared on the same fixtures; none at all -> 0.0 for every set
    (a tie). Pure."""
    ids = [i for i in scored_ids if all(
        next((r.get("psnr") for r in per_by_set[s] if r.get("id") == i), None) is not None for s in per_by_set)]
    out = {}
    for s, recs in per_by_set.items():
        by_id = {r.get("id"): r.get("psnr") for r in recs}
        out[s] = sum(float(by_id[i]) for i in ids) / len(ids) if ids else 0.0
    return out


def choose_reference_set(proximity: dict, tie: str = REF_TIE) -> str:
    """The ONE set a candidate is judged on: the NEAREST, i.e. the highest proximity (reference_proximity); `tie` wins
    any tie it is part of (within 1e-9 dB), otherwise the first nearest in REF_SETS order. Pure."""
    best = max(proximity.values())
    if abs(proximity.get(tie, float("-inf")) - best) <= _PROXIMITY_TIE_DB:
        return tie
    return next(name for name in REF_SETS if abs(proximity.get(name, float("-inf")) - best) <= _PROXIMITY_TIE_DB)


def psnr_bar_for(fx, psnr_min: float, cfg1_oracle: float, cfg1_coef: float) -> float:
    """The psnr bar that decides ONE fixture: a cfg=1.0 fixture's own bar (oracle x coefficient), every other
    fixture the shared one. Never mixed -- neither bar enters the other's fixtures. Pure."""
    try:
        cfg = float((fx or {}).get("cfg"))
    except (TypeError, ValueError):
        return psnr_min
    return cfg1_oracle * cfg1_coef if abs(cfg - 1.0) < 1e-9 else psnr_min


def profile_bars_for(fx, profile: str) -> dict:
    """The task-frozen image policy used by solo and serving-reference gates."""
    return _IMAGE_BARS.bars_for(fx, profile)


def bars_for(fx, shared: dict, cfg1_oracle: float, cfg1_coef: float, fixture_bars: dict) -> dict:
    """The four bars that decide ONE fixture: {"psnr", "ssim", "clip_img", "clip_txt"}. Pure.

    `shared` decides every metric of every fixture unless (a) the fixture is cfg=1.0, whose psnr bar is its own
    (psnr_bar_for), or (b) `fixture_bars` names the fixture, whose named metrics override the shared bars."""
    out = {m: shared[m] for m in ("psnr", "ssim", "clip_img", "clip_txt")}
    out["psnr"] = psnr_bar_for(fx, shared["psnr"], cfg1_oracle, cfg1_coef)
    own = (fixture_bars or {}).get((fx or {}).get("id")) or {}
    out.update({m: float(v) for m, v in own.items() if m in out})
    return out


LOAD_BOUND = ("sd3_registered", "sd3_dit_importable", "components_loaded",
              "pipeline_runs", "native_dit_forward", "stage_step_count",
              "image_similarity_smoke", "image_similarity_full",
              "prompt_conditioning", "runs_through_sglang_engine")


def rel_l2(a, b) -> float:
    a = a.astype(np.float64); b = b.astype(np.float64)
    return float(np.sqrt(np.sum((a - b) ** 2)) / (np.sqrt(np.sum(a ** 2)) + 1e-9))


def prompt_images_differ(m: dict, bars: dict) -> dict:
    """Pure: prompt_conditioning's verdict from the two prompts' images compared with each other (`m`: psnr, ssim,
    clip_img of image a against image b). They are the SAME image iff every metric reaches its bar -- the rule
    image_similarity_full applies to a candidate and its golden -- and they DIFFER iff at least one falls below. A
    metric that is missing cannot show a difference: with none below and one missing the verdict is undetermined, and
    the gate fails closed."""
    below = sorted(k for k in bars if isinstance(m.get(k), (int, float)) and m[k] < bars[k])
    missing = sorted(k for k in bars if not isinstance(m.get(k), (int, float)))
    if below:
        return {"differ": True, "below": below, "missing": missing, "reason": None}
    if missing:
        return {"differ": False, "below": [], "missing": missing,
                "reason": f"cannot establish a difference: {', '.join(missing)} unavailable, and every other metric "
                          f"is at or above its same-image bar"}
    parts = ", ".join(f"{k} {m[k]:.4g} >= {bars[k]:.4g}" for k in bars)
    return {"differ": False, "below": [], "missing": [],
            "reason": f"{parts}: two different prompts at one seed gave the same image by the task's own image rule, "
                      f"so the output does not follow the prompt"}


_DEV = None
_MM = {}


_CLIP_DEV = os.environ.get("SD3_CLIP_DEVICE", "cpu")


def _dev() -> str:
    global _DEV
    if _DEV is None:
        try:
            import torch
            _DEV = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            _DEV = "cpu"
    return _DEV


def _fmt(v, p="%.4g"):
    return (p % v) if isinstance(v, (int, float)) else "na"


def psnr(a_img, b_img) -> float:
    """Pixel PSNR (dB) between two uint8 HWC images; higher = more similar (inf->100)."""
    a = a_img.astype(np.float64); b = b_img.astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-9:
        return 100.0
    return float(20.0 * np.log10(255.0) - 10.0 * np.log10(mse))


def ssim(a_img, b_img) -> float:
    """SSIM (per-channel, RGB-averaged) between two uint8 HWC images; 1.0 = identical."""
    from skimage.metrics import structural_similarity
    return float(structural_similarity(a_img, b_img, channel_axis=-1, data_range=255))


def _clip_id():
    """Resolve a VALID local CLIP dir (has config.json + preprocessor_config.json), else
    glob the staged HF hub cache, else fall back to the repo-id. Validating the dir avoids
    the HFValidationError when an explicit SD3_CLIP_MODEL dir exists but is empty (e.g. the
    build host couldn't reach HF and left an empty clip_model)."""
    import glob
    cid = os.environ.get("SD3_CLIP_MODEL", "openai/clip-vit-large-patch14")

    def _ok(d):
        return (os.path.isfile(os.path.join(d, "config.json"))
                and os.path.isfile(os.path.join(d, "preprocessor_config.json")))

    if os.path.isdir(cid) and _ok(cid):
        return cid
    roots = [os.environ.get("HF_HOME", ""), os.path.expanduser("~/.cache/huggingface")]
    for r in roots:
        if not r:
            continue
        for base in (os.path.join(r, "hub"), r):
            for snap in sorted(glob.glob(os.path.join(
                    base, "models--openai--clip-vit-large-patch14", "snapshots", "*"))):
                if _ok(snap):
                    return snap
    return "openai/clip-vit-large-patch14"


def _clip_model():
    if "clipm" not in _MM:
        from transformers import CLIPModel, CLIPProcessor
        cid = _clip_id()
        print(f"[callsite] loading CLIP from: {cid}", flush=True)
        _MM["clipm"] = (CLIPProcessor.from_pretrained(cid),
                        CLIPModel.from_pretrained(cid).eval().to(_CLIP_DEV))
    return _MM["clipm"]


def _clip_embeds(images, prompts):
    """Normalized CLIP (image_embeds, text_embeds) via the full-model forward — version-
    robust: CLIPModel.get_image_features returned a BaseModelOutputWithPooling under the
    env's transformers (5.3.0), but CLIPOutput.image_embeds/.text_embeds are stable."""
    import torch
    from PIL import Image
    proc, m = _clip_model()
    with torch.no_grad():
        inputs = proc(text=list(prompts), images=[Image.fromarray(x) for x in images],
                      return_tensors="pt", padding=True, truncation=True)
        inputs = {k: (v.to(_CLIP_DEV) if hasattr(v, "to") else v) for k, v in inputs.items()}
        out = m(**inputs)
        ie = torch.nn.functional.normalize(out.image_embeds, dim=-1)
        te = torch.nn.functional.normalize(out.text_embeds, dim=-1)
        return ie, te


def clip_text_scores(images, prompts):
    'Normalized image-prompt CLIP cosine matrix, used for prompt adherence and retrieval diagnostics.'
    ie, te = _clip_embeds(images, prompts)
    return (ie @ te.T).float().cpu().numpy()


def clip_cos(a_img, b_img) -> float:
    """Compute normalized image-image CLIP cosine against the reference image."""
    ie, _ = _clip_embeds([a_img, b_img], [""])
    return float((ie[0] * ie[1]).sum())


def pixel_diag(img, gold):
    """Compute image-reference metrics; callers apply the scoring policy."""
    m = {}
    try:
        if img.shape == gold.shape:
            m["psnr"] = psnr(img, gold)
            m["rel_l2"] = rel_l2(img, gold)
            m["clip_img"] = clip_cos(img, gold)
        else:
            m["shape_mismatch"] = f"{tuple(img.shape)} != {tuple(gold.shape)}"
    except Exception:  # noqa: BLE001
        pass
    try:
        if img.shape == gold.shape:
            m["ssim"] = ssim(img, gold)
    except Exception:  # noqa: BLE001
        pass
    return m


_LOADED_RE = re.compile(r"Loaded (\w+): (\S+) \(([^)]+) version\)")


def _loaded_impls(captured: str) -> dict:
    """{component: {"cls": ..., "impl": "sgl-diffusion"|"native"}} from the loader's own lines.

    Captures EVERY component the loader announces (text_encoder*, vae, transformer, scheduler,
    tokenizer*), not just the text encoders, so a gate can be added over any of them without new
    instrumentation. The lines are emitted by sglang's own ComponentLoader.
    """
    out: dict = {}
    for comp, cls, ver in _LOADED_RE.findall(captured or ""):
        out[comp] = {"cls": cls, "impl": ver.strip()}
    return out


_SAVE_DIR = Path(os.environ.get("INFB_ARTIFACTS_DIR", "/logs/artifacts")) / "sd3_images"


def save_image(arr, name) -> None:
    """Persist a generated/golden image so we can eyeball the port (collected artifact)."""
    try:
        from PIL import Image
        _SAVE_DIR.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(_SAVE_DIR / name)
    except Exception as e:  # noqa: BLE001
        print(f"[callsite] save_image {name} failed: {type(e).__name__}: {e}", flush=True)


def _step_count(r0) -> int:
    """How many denoising steps ONE request's result reports. Never raises; 0 means "none reported".

    sglang STACKS the per-step trajectory before returning (denoising.py `_post_denoising_loop`):
      trajectory_timesteps -> tensor (steps,)                [stack dim=0]
      trajectory_latents   -> tensor (batch, steps, C, H, W)  [stack dim=1]
    Thus the step count is timesteps.shape[0] or latents.shape[1]; len(latents) is the batch size.
    """
    ts = getattr(r0, "trajectory_timesteps", None)
    if ts is not None:
        try:
            return int(ts.shape[0])
        except Exception:
            try:
                return int(len(ts))
            except Exception:
                pass
    lat = getattr(r0, "trajectory_latents", None)
    if lat is not None:
        try:
            if hasattr(lat, "shape") and len(lat.shape) >= 2:
                return int(lat.shape[1])
        except Exception:
            pass
        try:
            return int(len(lat))
        except Exception:
            pass
    return 0


def _request_record(out) -> dict:
    """What one generate() call reports besides its image. Never raises: an unreadable field is absent."""
    r0 = out[0] if isinstance(out, (list, tuple)) and out else out
    rec = {"trajectory_len": 0, "latents_shape": None, "timesteps_shape": None,
           "metrics_keys": [], "peak_memory_mb": None}
    if r0 is None:
        return rec
    rec["trajectory_len"] = _step_count(r0)
    for key, attr in (("latents_shape", "trajectory_latents"), ("timesteps_shape", "trajectory_timesteps")):
        try:
            v = getattr(r0, attr, None)
            rec[key] = list(v.shape) if hasattr(v, "shape") else (None if v is None else type(v).__name__)
        except Exception:  # noqa: BLE001
            pass
    try:
        metrics = getattr(r0, "metrics", {}) or {}
        rec["metrics_keys"] = list(metrics.keys()) if isinstance(metrics, dict) else []
        peak = (metrics.get("peak_memory_mb") if isinstance(metrics, dict) else None) \
            or getattr(r0, "peak_memory_mb", None)


        rec["peak_memory_mb"] = float(peak) if isinstance(peak, (int, float)) and peak > 0 else None
    except Exception:  # noqa: BLE001
        pass
    return rec


_REQUEST_LEDGER: list = []


def _gen(gen, prompt, fx):
    'ONE request -> (uint8 image or None, its record). The only call shape this verifier issues.'
    entry = {"label": fx.get("id"), "prompt": prompt, "cfg": fx["cfg"], "seed": fx["seed"], "steps": fx["steps"],
             "t0": time.time(), "t1": None, "ok": False}
    _REQUEST_LEDGER.append(entry)
    try:
        out = gen.generate(sampling_params_kwargs=dict(
            prompt=prompt, seed=fx["seed"], num_inference_steps=fx["steps"],
            guidance_scale=fx["cfg"], num_frames=1, height=fx["h"], width=fx["w"],
            return_frames=True, save_output=False))
        entry["ok"] = True
    finally:
        entry["t1"] = time.time()
    rec = dict(_request_record(out), cfg=fx["cfg"], num_outputs=1)
    try:
        img = to_image_array(out)
    except NoImageProduced as e:
        print(f"[callsite] {fx['id']}: {e}", flush=True)
        img = None
    del out
    return img, rec


def image_gate_verdicts(per: list, diagnostic_ids, smoke_ids, rate_min: float) -> dict:
    'The two scored image gates over the NON-diagnostic fixtures. Pure; no numpy, no GPU.'


    diag_set = set(diagnostic_ids or [])
    smoke_set = set(smoke_ids or [])
    scored = [p for p in per if p["id"] not in diag_set]
    diag = [p for p in per if p["id"] in diag_set]
    smoke = [p for p in scored if p["id"] in smoke_set] or scored[:1]
    smoke_ok = all(p["match"] for p in smoke) if smoke else False
    full_rate = sum(1 for p in scored if p["match"]) / len(scored) if scored else 0.0
    return {"smoke_ok": bool(smoke_ok), "full_rate": full_rate, "full_ok": bool(full_rate >= rate_min),
            "scored": scored, "diagnostic": diag, "smoke": smoke}


_BAR_METRICS = ("clip_txt", "psnr", "ssim", "clip_img")


def image_gate_failure(scored: list, rate: float, rate_min: float, chosen, proximity: dict | None = None,
                       max_items: int = 6) -> str | None:
    """The concrete reason a pixel verdict failed, or None when it passed. Pure; it reads only the records given.

    It names the rate against rate_min, the ONE set the candidate was judged on (with the proximity that chose it),
    and every unmatched scored fixture with each metric under its bar.
    A record whose metrics are missing is named by its note (generation failed, the CLIP error, a missing set).
    """
    if chosen is None or rate >= rate_min:
        return None
    items = []
    for p in scored:
        if p.get("match"):
            continue
        bars = p.get("bars") or {}
        bad = []
        for m in _BAR_METRICS:
            v, b = p.get(m), bars.get(m)
            if not isinstance(v, (int, float)):
                bad.append(f"{m} missing")
            elif isinstance(b, (int, float)) and not v >= b:
                bad.append(f"{m} {v:.2f} < {b:.2f}" if m == "psnr" else f"{m} {v:.4f} < {b:.4f}")
        if p.get("note") and all(x.endswith(" missing") for x in bad):
            bad = [str(p["note"])]
        items.append(f"{p.get('id')}[{p.get('reference_set') or chosen}] " + (", ".join(bad) or "not matched"))
    if len(items) > max_items:
        items = items[:max_items] + [f"+{len(items) - max_items} more"]
    near = ""
    if proximity:
        near = " (mean psnr " + ", ".join(f"{k} {v:.3f}" for k, v in proximity.items()
                                         if isinstance(v, (int, float))) + ")"
    return f"rate {round(rate, 4):g} < {rate_min:g} on the {chosen} set{near}: " + "; ".join(items)


def trajectory_gate_verdicts(records: list, tol: int) -> dict:
    """The two scored trajectory gates, judged on the SAME requests that produced the scored images. Pure.

    `records` holds one entry per scored fixture: {"id", "steps", "image": bool, "trajectory_len": int},
    where `image` says whether THAT request returned a scorable image. Per request:
      native_dit_forward  image AND trajectory_len >= max(1, steps - tol)
      stage_step_count    image AND steps - tol <= trajectory_len <= steps + tol
    Each gate passes iff every scored request passes it, and there is at least one. A request that returned
    a trajectory but no image fails both: it is not evidence that a scored image was denoised natively,
    which is what these gates establish.
    """
    tol = int(tol)
    per = []
    for r in records:
        n = int(r.get("steps") or 0)
        t = int(r.get("trajectory_len") or 0)
        img = bool(r.get("image"))
        per.append({"id": r.get("id"), "steps": n, "trajectory_len": t, "image": img,
                    "native_ok": bool(img and t >= max(1, n - tol)),
                    "stage_ok": bool(img and (n - tol) <= t <= (n + tol))})
    return {"native_dit_forward": bool(per) and all(p["native_ok"] for p in per),
            "stage_step_count": bool(per) and all(p["stage_ok"] for p in per),
            "per": per}


def trajectory_failures(per: list, tol: int, ok_key: str) -> list:
    """One message per request that fails `ok_key` ("stage_ok" / "native_ok").

    Each message uses that request's own step count, including when requests have different lengths.
    """
    out = []
    for p in per or []:
        if p.get(ok_key):
            continue
        line = f"{p.get('id')}: trajectory {p.get('trajectory_len')} for {p.get('steps')} requested steps (tol {tol})"
        out.append(line if p.get("image") else line + ", no image")
    return out


_CONCURRENCY_N = 2


_CONCURRENCY_TIMEOUT_S = 900.0
_CONCURRENCY_OUT = Path(os.environ.get("INFB_ARTIFACTS_DIR", "/logs/artifacts")) / "sd3_concurrency.json"

_REFUSAL_MARKERS = ("ZMQError", "Operation cannot be accomplished in current state")


class _RefusalCapture(logging.Handler):
    """The engine client's log records while the concurrent requests run, kept only if they carry a refusal.

    DiffGenerator.generate CATCHES the REQ socket's ZMQError, logs it (utils/logging_utils.py "Failed to generate
    output for prompt %d: %s", entrypoints/diffusion_generator.py "Generation failed for prompt %d/%d: %s") and
    returns no image -- so the thread sees no exception, only a missing image, and the refusal exists only in the
    log. Attached to the root logger and
    every `sglang*` logger (theirs may not propagate); a record seen twice is kept once."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.lines: list = []
        self._seen: set = set()
        self._loggers: list = []

    def emit(self, record) -> None:
        try:
            if id(record) in self._seen:
                return
            self._seen.add(id(record))
            text = record.getMessage()
            exc = record.exc_info[1] if record.exc_info else None
            if exc is not None:
                text += f" [{type(exc).__name__}: {exc}]"
            if any(m in text for m in _REFUSAL_MARKERS):
                self.lines.append(text[:300])
        except Exception:  # noqa: BLE001 - a capture must never break the engine's own logging
            pass

    def attach(self) -> "_RefusalCapture":
        names = [n for n in list(logging.Logger.manager.loggerDict) if n.startswith("sglang")]
        self._loggers = [logging.getLogger()] + [logging.getLogger(n) for n in names]
        for lg in self._loggers:
            lg.addHandler(self)
        return self

    def detach(self) -> None:
        for lg in self._loggers:
            try:
                lg.removeHandler(self)
            except Exception:  # noqa: BLE001
                pass


def concurrency_refusals(results: dict, refusal_lines: list) -> list:
    """Pure: the concurrent requests the client refused. By the request's own exception text; or, as the pinned
    DiffGenerator does it, by a caught-and-logged refusal -- then the requests that came back without an image are
    the refused ones."""
    results = results if isinstance(results, dict) else {}
    by_exc = sorted(fid for fid, r in results.items()
                    if any(m in str((r or {}).get("error") or "") for m in _REFUSAL_MARKERS))
    if by_exc:
        return by_exc
    if refusal_lines:
        return sorted(fid for fid, r in results.items() if (r or {}).get("img") is None) or ["<logged, unattributed>"]
    return []


def _sync_client_socket() -> "str | None":
    """The live sync client's socket type ("REQ" at the pinned baseline: one request in flight). None if unreadable."""
    try:
        sc = sys.modules.get("sglang.multimodal_gen.runtime.scheduler_client")
        sock = getattr(getattr(sc, "sync_scheduler_client", None), "scheduler_socket", None)
        if sock is None:
            return None
        import zmq

        names = {getattr(zmq, n): n for n in ("REQ", "REP", "DEALER", "ROUTER", "PAIR", "PUSH", "PULL")}
        return names.get(sock.type, str(sock.type))
    except Exception:  # noqa: BLE001
        return None


def _probe_concurrency(gen, fixtures, smoke_ids, per, imgs, ref_dir=None) -> None:
    'UNSCORED. Regenerate two smoke fixtures AT THE SAME TIME and record whether they survive it.'


    import threading
    import time


    def _ge(v, t):
        return v is not None and v >= t


    out = {"n": _CONCURRENCY_N, "measured": False, "label": "concurrency (unscored)",
           "client": "DiffGenerator sync client: one ZMQ REQ socket; one request in flight",
           "bars": {"psnr": PSNR_MIN, "ssim": SSIM_MIN, "clip_img": CLIP_MIN}}
    try:
        by_id = {fx["id"]: i for i, fx in enumerate(fixtures)}


        ordered = ([fx for fx in fixtures if fx["id"] in smoke_ids]
                   + [fx for fx in fixtures if fx["id"] not in smoke_ids])
        chosen = [fx for fx in ordered if imgs[by_id[fx["id"]]] is not None][:_CONCURRENCY_N]
        if len(chosen) < 2:
            out["reason"] = ("fewer than two fixtures produced a serial image, so concurrency cannot "
                             "be exercised. This is a fixture/port-availability fact, not a result.")
            _CONCURRENCY_OUT.parent.mkdir(parents=True, exist_ok=True)
            _CONCURRENCY_OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
            print(f"[concurrency] SKIP: {out['reason']}", flush=True)
            return
        out["fixture_ids"] = [fx["id"] for fx in chosen]

        results: dict = {}

        def _one(fx):
            try:
                t = time.perf_counter()
                img, _ = _gen(gen, fx["prompt"], fx)
                results[fx["id"]] = {"img": img, "secs": time.perf_counter() - t}
            except Exception as exc:  # noqa: BLE001
                results[fx["id"]] = {"img": None, "error": f"{type(exc).__name__}: {exc}"}

        out["client_socket"] = _sync_client_socket()
        capture = _RefusalCapture().attach()
        try:
            t0 = time.perf_counter()
            threads = [threading.Thread(target=_one, args=(fx,), daemon=True) for fx in chosen]
            for th in threads:
                th.start()
            deadline = t0 + _CONCURRENCY_TIMEOUT_S
            for th in threads:
                th.join(max(0.0, deadline - time.perf_counter()))
            out["wall_secs_concurrent"] = round(time.perf_counter() - t0, 2)
        finally:
            capture.detach()
        refused = concurrency_refusals(results, capture.lines)
        out["refusal_log_lines"] = capture.lines[:5]
        out["concurrency_reachable"] = not refused
        if refused:
            out["label"] = ("single-request: the client's REQ socket refused the concurrent send(s) "
                            f"{refused}; no two requests were ever in flight, so nothing about concurrency "
                            "was measured")
        alive = [th.name for th in threads if th.is_alive()]
        if alive:
            out["reason"] = (f"{len(alive)} of {len(threads)} concurrent request(s) had not returned "
                            f"after {_CONCURRENCY_TIMEOUT_S:.0f}s. Recorded as unmeasured rather than "
                            "as a port defect -- a deadlock here is equally consistent with the probe "
                            "being wrong about thread-safety expectations.")
            _CONCURRENCY_OUT.parent.mkdir(parents=True, exist_ok=True)
            _CONCURRENCY_OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
            print(f"[concurrency] TIMEOUT: {out['reason']}", flush=True)
            return


        out["per"] = []
        for fx in chosen:
            i = by_id[fx["id"]]
            r = results.get(fx["id"]) or {}
            rec = {"id": fx["id"], "secs": round(r.get("secs", -1.0), 2)}
            cimg = r.get("img")
            if r.get("error"):
                rec["error"] = r["error"]
            if cimg is None:
                rec["no_image"] = True
            else:
                gold = to_image_array(np.load(Path(ref_dir or GOLDEN_DIR) / fx["golden_npy"]))
                vs_gold = pixel_diag(cimg, gold)
                rec["vs_golden"] = {k: vs_gold.get(k) for k in ("psnr", "ssim", "clip_img", "rel_l2")}


                rec["clears_serial_bars"] = bool(
                    _ge(vs_gold.get("psnr"), PSNR_MIN) and _ge(vs_gold.get("ssim"), SSIM_MIN)
                    and _ge(vs_gold.get("clip_img"), CLIP_MIN))
                vs_serial = pixel_diag(cimg, imgs[i])
                rec["vs_serial"] = {k: vs_serial.get(k) for k in ("psnr", "ssim", "clip_img", "rel_l2")}
                rec["identical_to_serial"] = bool(np.array_equal(cimg, imgs[i]))
                rec["serial_match"] = bool((per[i] if i < len(per) else {}).get("match"))
                save_image(cimg, f"{fx['id']}_concurrent.png")
            out["per"].append(rec)
        out["measured"] = True
        out["all_clear_serial_bars"] = all(
            p.get("clears_serial_bars") for p in out["per"]) if out["per"] else False
        out["all_identical_to_serial"] = all(
            p.get("identical_to_serial") for p in out["per"]) if out["per"] else False
        _CONCURRENCY_OUT.parent.mkdir(parents=True, exist_ok=True)
        _CONCURRENCY_OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
        for p in out["per"]:
            g = p.get("vs_golden") or {}
            s = p.get("vs_serial") or {}
            print(f"[concurrency] {p['id']}: clears_serial_bars={p.get('clears_serial_bars')} "
                  f"identical_to_serial={p.get('identical_to_serial')} "
                  f"vs_golden(psnr={_fmt(g.get('psnr'), '%.1f')} ssim={_fmt(g.get('ssim'))} "
                  f"clip_img={_fmt(g.get('clip_img'))}) "
                  f"vs_serial(psnr={_fmt(s.get('psnr'), '%.1f')} rel_l2={_fmt(s.get('rel_l2'))}) "
                  f"{p['secs']}s", flush=True)
        print(f"[concurrency] UNSCORED probe: all_clear_serial_bars="
              f"{out['all_clear_serial_bars']} all_identical_to_serial="
              f"{out['all_identical_to_serial']} wall={out.get('wall_secs_concurrent')}s "
              f"-> {_CONCURRENCY_OUT}", flush=True)
    except Exception as exc:  # noqa: BLE001

        save_stderr_log("callsite_concurrency", traceback.format_exc())
        print(f"[concurrency] probe failed (UNSCORED, no gate affected): "
              f"{type(exc).__name__}: {exc}", flush=True)


_CPROBE_RECORD = Path(os.environ.get("INFB_ARTIFACTS_DIR", "/logs/artifacts")) / "sd3_component_record.jsonl"
_CPROBE_OUT = Path(os.environ.get("INFB_ARTIFACTS_DIR", "/logs/artifacts")) / "sd3_component_probe.json"


def _cprobe_arm():
    """Arm the component recorder for the engine about to be built. Never raises."""
    if _cprobe is None:
        return {"armed": False, "error": f"component_probe not importable: {_CPROBE_IMPORT_ERROR}"}
    try:
        return _cprobe.arm(os.environ, _CPROBE_RECORD)
    except Exception as exc:  # noqa: BLE001
        return {"armed": False, "error": f"{type(exc).__name__}: {exc}"[:300]}


def _cprobe_disarm(token) -> None:
    try:
        if _cprobe is not None:
            _cprobe.disarm(os.environ, token)
    except Exception:  # noqa: BLE001
        pass


NATIVE_RECORD_GATES = ("native_text_encoders", "native_vae", "native_transformer")


def _cprobe_summary(token, loader_lines) -> dict:
    """What the three worker-record gates are judged on: the worker's record, read now. Never raises."""
    if _cprobe is None:
        why = f"component_probe not importable ({_CPROBE_IMPORT_ERROR}) -- fail-closed"
        return {"error": why, "gate_verdicts": {g: {"verdict": "FAIL", "not_evaluated": True, "reasons": [why]}
                                                for g in NATIVE_RECORD_GATES}}
    try:
        return _cprobe.read_summary(token, loader_lines, parent_pid=os.getpid())
    except Exception as exc:  # noqa: BLE001
        why = f"the component record could not be summarized: {type(exc).__name__}: {exc} -- fail-closed"
        return {"error": why, "gate_verdicts": {g: {"verdict": "FAIL", "not_evaluated": True, "reasons": [why]}
                                                for g in NATIVE_RECORD_GATES}}


def _native_gate_result(gate, comps, summary, impls, loader_captured) -> GateResult:
    """native_text_encoders / native_vae from the worker's record, with the loader's own line as a diagnostic."""
    summary = summary if isinstance(summary, dict) else {}
    v = (summary.get("gate_verdicts") or {}).get(gate)
    if not isinstance(v, dict):
        v = {"verdict": "FAIL", "not_evaluated": True,
             "reasons": [f"no verdict for {gate} in the component summary -- fail-closed"]}
    ok = v.get("verdict") == "PASS"
    comp = summary.get("components") or {}
    rec = summary.get("recorder") or {}
    details = {
        "evidence": "the classes the spawned worker holds (component_recorder)",
        "not_evaluated": bool(v.get("not_evaluated")),
        "reasons": list(v.get("reasons") or []),
        "gated_components": list(comps),
        "components": {c: {k: (comp.get(c) or {}).get(k) for k in
                           ("kind", "impl", "source", "served", "loaded", "swapped_after_load",
                            "foreign_built_in_window", "foreign_method_owner")
                           + (("whole_model",) if gate == "native_transformer" else ())} for c in comps},
        "recorder": {k: rec.get(k) for k in ("ran_in_worker", "evidence_complete", "missing", "worker_pids",
                                             "armed_error")},


        "loader_line": {c: (impls or {}).get(c) for c in comps},
        "loader_line_captured": bool(loader_captured),
        "disagrees_with_loader_line": [c for c in comps
                                       if (comp.get(c) or {}).get("agrees_with_loader_line") is False],
    }
    if summary.get("error"):
        details["summary_error"] = summary["error"]
    if not ok:
        details["hard_gate_reason"] = (
            "a DiT that is, wraps or subclasses a whole transformers/diffusers model runs that model, not a port"
            if gate == "native_transformer" else
            "delegating a text encoder or the VAE implements less of SGLang while scoring better against a "
            "diffusers reference")
    return GateResult(gate=gate, passed=ok, is_hard_gate=True, score=1.0 if ok else 0.0,
                      error=None if ok else "; ".join(v.get("reasons") or ["no reason recorded"]),
                      details=details)


def _probe_components(token, loader_lines, load_error=None) -> None:
    'Write an auxiliary end-of-run component summary. Scored native gates use the trusted worker record separately; this report never writes gate verdicts.'
    try:
        if _cprobe is None:
            print(f"[component-probe] UNSCORED: not run -- {_CPROBE_IMPORT_ERROR}", flush=True)
            return
        summary = _cprobe.probe(token, loader_lines, _CPROBE_OUT, load_error=load_error, parent_pid=os.getpid())
        print(_cprobe.one_line(summary) + f" -> {_CPROBE_OUT}", flush=True)
    except Exception as exc:  # noqa: BLE001 - an unscored probe must never change a verdict, including by raising
        print(f"[component-probe] probe failed (UNSCORED, no gate affected): {type(exc).__name__}: {exc}",
              flush=True)


STEPS_PROBE = 12
_STEPS_PROBE_OUT = Path(os.environ.get("INFB_ARTIFACTS_DIR", "/logs/artifacts")) / "sd3_steps_probe.json"


_TAP_OUT = Path(os.environ.get("INFB_ARTIFACTS_DIR", "/logs/artifacts")) / "sd3_component_tap.json"


def _steps_request(gen, fixtures) -> dict:
    'SD3 verification helper.'


    rec = {"id": f"steps_probe_{STEPS_PROBE}", "steps": STEPS_PROBE, "image": False, "trajectory_len": 0,
           "sd9": True}
    try:
        fx = next(f for f in fixtures if int(f.get("steps") or 0) != STEPS_PROBE)
        rec["fixture"] = fx["id"]
        rec["t0"] = time.time()
        img, req = _gen(gen, fx["prompt"], {**fx, "steps": STEPS_PROBE})
        rec["t1"] = time.time()
        rec.update(req)
        rec["image"] = img is not None
    except Exception as exc:  # noqa: BLE001 - a failed request is a failed record, never a crash
        rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
    try:
        _STEPS_PROBE_OUT.parent.mkdir(parents=True, exist_ok=True)
        _STEPS_PROBE_OUT.write_text(json.dumps({**rec, "scored": True, "judged_by": "stage_step_count"}, indent=2,
                                               default=str), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    print(f"[steps] requested {STEPS_PROBE} -> trajectory_len={rec.get('trajectory_len')} "
          f"image={rec.get('image')} {rec.get('error') or ''}", flush=True)
    return rec


def native_request_evidence(records: list, tol: int, summary=None) -> dict:
    per = [{"id": r.get("id"), "image": bool(r.get("image")),
            **native_execution.observed_request(summary, r, tol)} for r in records or []]
    return {"ok": bool(per) and all(p["ok"] for p in per), "per": per,
            "not_evaluated": not per or all(p["ok"] or p["not_evaluated"] for p in per)}


def generation_evidence(summary, records: list, tol: int) -> dict:

    summary = summary if isinstance(summary, dict) else {}
    try:
        pipe = _cprobe.native_pipeline_evidence(summary) if _cprobe is not None else {
            "ok": False, "reasons": ["component recorder not importable"]}
    except Exception as exc:
        pipe = {"ok": False, "reasons": [f"recorder could not be judged: {exc}"]}
    req = native_request_evidence(records, tol, summary)
    consumed = summary.get("native_consumption")
    if not isinstance(consumed, dict) or type(consumed.get("ok")) is not bool:
        consumed = {"ok": False, "not_evaluated": True,
                    "reason": "output consumption absent or malformed"}
    ok = bool(pipe.get("ok") and req["ok"] and consumed.get("ok"))
    reasons = list(pipe.get("reasons") or [])
    reasons += [str(p["id"]) + ": " + p["reason"] for p in req["per"] if not p["ok"]]
    if not consumed.get("ok"):
        reasons.append(consumed.get("reason", "output consumption unmeasured"))
    return {"ok": ok, "reason": None if ok else "; ".join(reasons)[:800],
            "pipeline": pipe, "requests": req["per"], "consumption": consumed,
            "not_evaluated": not ok and all(x.get("ok") or x.get("not_evaluated") for x in (pipe, req, consumed))}


def _native_consumption_request(gen, fixtures, token, scored_img):
    """One paired repeat/intervention. Neither image enters the quality score."""
    if not fixtures or not (token or {}).get("armed") or not (token or {}).get("tap_dir"):
        return {"ok": False, "not_evaluated": True, "reason": "native recorder not armed"}
    fx, record = fixtures[0], {}
    tap = Path(token["tap_dir"])
    arm = tap / "NATIVE_DIT_ARM"
    try:
        if not (token or {}).get("armed") or not tap.is_dir():
            raise RuntimeError("native recorder not armed")
        readings, images = {}, {}
        for name in ("repeat", "armed"):
            if name == "armed":
                arm.write_text(json.dumps({"delta": .25}))
            rec = {"id": "native_consumption_" + name, "steps": fx["steps"], "cfg": fx["cfg"], "t0": time.time()}
            try:
                img, req = _gen(gen, fx["prompt"], fx)
                rec.update(req, image=img is not None)
                images[name] = img
            except Exception as exc:
                rec.update(image=False, error=f"{type(exc).__name__}: {exc}"[:500])
                images[name] = None
            finally:
                rec["t1"] = time.time()
                arm.unlink(missing_ok=True)
            readings[name] = rec
        summary = _cprobe_summary(token, None)
        observed = {k: native_execution.observed_request(summary, v, STEP_COUNT_TOLERANCE) for k, v in readings.items()}
        import t2_gates
        repeat_db = t2_gates.psnr(images["repeat"], scored_img)
        armed_db = t2_gates.psnr(images["armed"], images["repeat"])
        repeat_equal = repeat_db >= t2_gates.POLICY["ISO_PSNR_DB"] if repeat_db is not None else None
        armed_equal = armed_db > t2_gates.POLICY["EFFECT_MAX_PSNR_DB"] if armed_db is not None else None
        record = native_execution.consumption_verdict(observed["repeat"], observed["armed"], repeat_equal, armed_equal)
        record.update(requests=readings, repeat_psnr_db=repeat_db, armed_psnr_db=armed_db,
                      unchanged_bar_db=t2_gates.POLICY["ISO_PSNR_DB"],
                      effect_bar_db=t2_gates.POLICY["EFFECT_MAX_PSNR_DB"])
        for name, image in images.items():
            if image is not None:
                save_image(image, "native_consumption_" + name + ".png")
    except Exception as exc:
        record = {"ok": False, "not_evaluated": True, "reason": f"consumption instrument/request failed: {exc}"[:500]}
    finally:
        arm.unlink(missing_ok=True)
    _write_json(Path(os.environ.get("INFB_ARTIFACTS_DIR", "/logs/artifacts"))/("sd3_native_consumption_" + (PRECISION_PROFILE or "default") + ".json"), record)
    return record


VACUOUS_ON_FALLBACK = ("sd3_registered", "sd3_dit_importable", "components_loaded", "pipeline_runs",
                       "runs_through_sglang_engine")
VACUOUS_LABEL = "vacuous (diffusers fallback)"


def _label_vacuous_on_fallback(gen_ev) -> list:
    """When the worker record shows no native pipeline, mark each of those diagnostics that PASSED with VACUOUS_LABEL
    and the evidence's reasons. Their `passed` and `score` are left as their own checks set them and none is scored,
    so no reward and no partial can move. Never raises; returns the gates it labelled."""
    labelled = []
    try:
        pipe = (gen_ev or {}).get("pipeline") or {}
        if pipe.get("ok"):
            return labelled
        for gate in VACUOUS_ON_FALLBACK:
            f = _gate_file(gate)
            try:
                g = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(g, dict) or not g.get("passed"):
                continue
            det = g.get("details") if isinstance(g.get("details"), dict) else {}
            g["details"] = {**det, "label": VACUOUS_LABEL, "vacuous_because": list(pipe.get("reasons") or [])[:5]}
            f.write_text(json.dumps(g, default=str, indent=2), encoding="utf-8")
            labelled.append(gate)
        print(f"[callsite] {VACUOUS_LABEL}: {labelled}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[callsite] could not label the vacuous diagnostics: {type(exc).__name__}: {exc}", flush=True)
    return labelled


def _probe_component_tap(gen, fixtures, token, scored_img) -> None:
    """Arm the worker's taps for ONE request -- the first fixture's own parameters -- and
    record what they captured, for tests/verify_component_alignment.py to replay against diffusers.

    Runs after every gate file is on disk; writes no gate. The request repeats a scored one exactly, so its image
    must equal that fixture's scored image: a free check that the taps change no pixel (`identical_to_scored`).
    """
    out = {"scored": False, "measured": False}
    try:
        tap = Path((token or {}).get("tap_dir") or "")
        if not (token or {}).get("armed") or not tap.is_dir():
            out["reason"] = "the recorder was not armed, so the worker has no taps"
        else:
            fx = fixtures[0]
            arm = tap / "ARM"
            arm.write_text("armed by verify_callsite_main._probe_component_tap\n", encoding="utf-8")
            try:
                img, req = _gen(gen, fx["prompt"], fx)
            finally:
                try:
                    arm.unlink()
                except FileNotFoundError:
                    pass
            files = sorted(p.name for p in tap.glob("*.pt"))
            out.update({"measured": bool(files), "tap_dir": str(tap), "files": files,
                        "fixture": {k: fx.get(k) for k in ("id", "prompt", "seed", "steps", "cfg", "h", "w")},
                        "image": img is not None, "trajectory_len": req.get("trajectory_len"),
                        "identical_to_scored": (bool(np.array_equal(img, scored_img))
                                                if img is not None and scored_img is not None else None)})
    except Exception as exc:  # noqa: BLE001 - an unscored probe must never change a verdict
        out["error"] = f"{type(exc).__name__}: {exc}"[:300]
    try:
        _TAP_OUT.parent.mkdir(parents=True, exist_ok=True)
        _TAP_OUT.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    print(f"[component-tap] UNSCORED: measured={out.get('measured')} files={out.get('files')} "
          f"identical_to_scored={out.get('identical_to_scored')} {out.get('reason') or out.get('error') or ''}",
          flush=True)


_JATTN_OUT = Path(os.environ.get("INFB_ARTIFACTS_DIR", "/logs/artifacts")) / "sd3_joint_attention_probe.json"
_BACKEND_OUT = Path(os.environ.get("INFB_ARTIFACTS_DIR", "/logs/artifacts")) / "sd3_attention_backend_probe.json"
_BACKEND_RECORD = (Path(os.environ.get("INFB_ARTIFACTS_DIR", "/logs/artifacts")) / "sd3_backend_probe"
                   / "sd3_component_record.jsonl")


def _dit_arch(model) -> dict:
    """num_layers / attention_head_dim from the checkpoint's own transformer config (SD3-medium's when unreadable)."""
    try:
        d = json.loads((Path(model) / "transformer" / "config.json").read_text(encoding="utf-8"))
        return {"num_layers": int(d["num_layers"]), "head_dim": int(d["attention_head_dim"]), "source": "checkpoint"}
    except Exception as exc:  # noqa: BLE001
        return {"num_layers": 24, "head_dim": 64, "source": f"SD3-medium default ({type(exc).__name__})"}


def _write_json(path: Path, obj) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def _diag_gate(gate: str, capability: str, verdict: dict, details: dict) -> None:
    ' diag gate.'
    ok = bool(verdict.get("passed"))
    reasons = list(verdict.get("reasons") or [])
    write_gate(GateResult(gate=gate, passed=ok, score=1.0 if ok else 0.0, is_hard_gate=False,
                          error=None if ok else ("; ".join(reasons) or "no reason recorded")[:800],
                          details={"capability_id": capability, "layer": "T3", "scored": True,
                                   "diagnostic": False, "not_evaluated": bool(verdict.get("not_evaluated")),
                                   "reasons": reasons[:40], **details}))


def _probe_joint_attention(gen, fixtures, token, traj, scored_img) -> None:
    ' probe joint attention.'
    arch = _dit_arch(MODEL)
    n_blocks, head_dim = arch["num_layers"], arch["head_dim"]
    out: dict = {"scored": True, "gate": "native_joint_attention", "arch": arch}
    verdict: dict = {"passed": False, "not_evaluated": True, "reasons": []}
    try:
        if _jattn is None:
            raise RuntimeError(f"joint_attention_probe not importable: {_JATTN_IMPORT_ERROR}")
        tap = Path((token or {}).get("tap_dir") or "")
        if not (token or {}).get("armed") or not tap.is_dir():
            raise RuntimeError("the recorder was not armed, so the worker has no attention accounting")
        fx = fixtures[0]
        plan = {"capture": sorted({0, n_blocks // 2 - 1, n_blocks - 1}), "delta": _jattn.DELTA, "rows": 64}
        arm = tap / "ATTN_ARM"
        arm.write_text(json.dumps(plan), encoding="utf-8")
        t0 = time.time()
        try:
            img, _req = _gen(gen, fx["prompt"], fx)
        finally:
            try:
                arm.unlink()
            except FileNotFoundError:
                pass
        t1 = time.time()
        out["armed_request"] = {"fixture": fx["id"], "t0": t0, "t1": t1, "image": img is not None,
                                "identical_to_scored": (bool(np.array_equal(img, scored_img))
                                                        if img is not None and scored_img is not None else None),
                                "plan": plan}
        summary = _cprobe_summary(token, None)
        att = summary.get("attention") or {}
        out["recorder"] = {k: att.get(k) for k in ("window_installed", "impl_wrapped_pids")}
        out["recorder"]["errors"] = att.get("errors")
        probes = [p for p in att.get("probes") or [] if t0 <= (p.get("t") or 0) <= t1 + 1.0]
        caps = [c for c in att.get("captures") or [] if t0 <= (c.get("t") or 0) <= t1 + 1.0]
        math_res: dict = {}
        for c in caps:
            try:
                with np.load(c["file"]) as z:
                    cap = {k: z[k] for k in z.files}
                math_res[int(c["call"])] = {**_jattn.capture_math(cap, head_dim), "impl": c.get("impl"),
                                            "impl_softmax_scale": c.get("softmax_scale"),
                                            "impl_causal": c.get("causal")}
            except Exception as exc:  # noqa: BLE001
                math_res[int(c.get("call", -1))] = {"rel_l2": None, "finite": False,
                                                     "error": f"{type(exc).__name__}: {exc}"[:300]}
        probe_ev = _jattn.probe_evidence(probes[0] if probes else None, math_res, n_blocks,
                                         _jattn.joint_len(fx["h"], fx["w"]))
        by_id = {f["id"]: f for f in fixtures}
        requests = {}
        for r in traj or []:
            f = by_id.get(r.get("fixture") or r.get("id"))
            if f is None:
                continue
            fw = _jattn.forwards_between(summary, r.get("t0"), r.get("t1"))
            requests[r["id"]] = _jattn.request_joint_evidence(fw, _jattn.joint_len(f["h"], f["w"]), n_blocks)
        verdict = _jattn.joint_attention_verdict(requests, probe_ev)
        out.update({"probe": probe_ev, "requests": requests})
    except Exception as exc:  # noqa: BLE001 - a probe that could not run is a not-evaluated FAIL, never a crash
        verdict = {"passed": False, "not_evaluated": True, "reasons": [f"{type(exc).__name__}: {exc}"[:300]]}
        out["error"] = verdict["reasons"][0]
    out["verdict"] = verdict
    _write_json(_JATTN_OUT, out)
    try:
        req = out.get("requests") or {}
        _diag_gate("native_joint_attention", "native_joint_attention", verdict, {
            "evidence": "the worker's engine-attention accounting per DiT forward, a perturbation probe on every "
                        "attention call, and an fp64 recompute of captured calls (component_recorder, "
                        "joint_attention_probe.py)",

            "arch": arch,
            "armed_request": {k: (out.get("armed_request") or {}).get(k)
                              for k in ("fixture", "image", "identical_to_scored")},
            "recorder": {"window_installed": (out.get("recorder") or {}).get("window_installed"),
                         "impl_wrapped": bool((out.get("recorder") or {}).get("impl_wrapped_pids")),
                         "errors": len((out.get("recorder") or {}).get("errors") or [])},
            "probe": {k: (out.get("probe") or {}).get(k) for k in ("n", "repeat_rel", "effect_floor",
                                                                    "per_call_rel_min", "per_call_rel_max",
                                                                    "dead_calls", "math")},
            "requests": {rid: {k: ev.get(k) for k in ("ok", "forwards", "joint_calls_min", "calls_min",
                                                      "calls_max", "impls", "k_seq", "s_joint")}
                         for rid, ev in req.items()},
            "thresholds": ({"min_effect": _jattn.MIN_EFFECT, "noise_factor": _jattn.NOISE_FACTOR,
                            "math_tol": _jattn.MATH_TOL, "delta": _jattn.DELTA} if _jattn else None)})
    except Exception as exc:  # noqa: BLE001
        print(f"[joint-attention] could not write its gate: {type(exc).__name__}: {exc}", flush=True)
    print(f"[joint-attention] native_joint_attention passed={verdict.get('passed')} "
          f"not_evaluated={verdict.get('not_evaluated')} {'; '.join(verdict.get('reasons') or [])[:400]}",
          flush=True)


def _bounded(fn, timeout_s: float) -> "str | None":
    """Run fn() in a daemon thread for at most timeout_s; the error, 'timed out', or None."""
    import threading

    box: dict = {}

    def run():
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            box["error"] = f"{type(exc).__name__}: {exc}"[:300]

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        return f"timed out after {timeout_s:.0f} s"
    return box.get("error")


def _fixture_match(img, fx, ref_set: str) -> dict:
    """One image on ONE fixture's own bars, against the chosen reference set (the pixel gates' rule, reused)."""
    if img is None:
        return {"match": False, "note": "no image produced"}
    rec = dict(pixel_diag(img, to_image_array(np.load(REF_DIRS[ref_set] / fx["golden_npy"]))))
    try:
        rec["clip_txt"] = float(clip_text_scores([img], [fx["prompt"]])[0, 0])
    except Exception as exc:  # noqa: BLE001
        rec["note"] = f"CLIP unavailable: {type(exc).__name__}: {exc}"[:200]
    bars = profile_bars_for(fx, ref_set)
    rec["bars"] = bars
    rec["reference_set"] = ref_set
    rec["threshold_policy"] = _IMAGE_BARS.receipt()
    rec["match"] = _IMAGE_BARS.matches(rec, bars)
    return rec


_PREC_PROBE: dict = {}


def _probe_precision_consumption(gen, fixtures, token, scored_img) -> None:
    """Profile mode, after every other gate: ONE armed request (`<tap>/PREC_ARM`, fixture 0's own parameters) on which
    the worker's precision witness re-runs the first call of each component once unchanged and once per sampled main op
    with that op's result perturbed (component_recorder `_pw_maybe_probe`). The pipeline gets the original outputs, so
    the armed request's image must equal fixture 0's scored one (`identical_to_scored`). Never raises: a request that
    fails leaves no probe line, and precision_recorded then reads consumption as unmeasured."""
    import secrets

    _PREC_PROBE.clear()
    try:
        tap = Path((token or {}).get("tap_dir") or "")
        if not (token or {}).get("armed") or not tap.is_dir():
            raise RuntimeError("the recorder was not armed, so the worker has no precision witness")
        fx = fixtures[0]
        plan = dict(_PREC.PROBE_PLAN, seed=secrets.randbelow(2 ** 31))
        arm = tap / "PREC_ARM"
        arm.write_text(json.dumps(plan), encoding="utf-8")
        t0 = time.time()
        try:
            img, _req = _gen(gen, fx["prompt"], fx)
        finally:
            try:
                arm.unlink()
            except FileNotFoundError:
                pass
        _PREC_PROBE.update({"fixture": fx["id"], "plan": plan, "t0": t0, "t1": time.time(), "image": img is not None,
                            "identical_to_scored": (bool(np.array_equal(img, scored_img))
                                                    if img is not None and scored_img is not None else None)})
    except Exception as exc:  # noqa: BLE001 - a probe that could not run leaves consumption unmeasured, never a crash
        _PREC_PROBE["error"] = f"{type(exc).__name__}: {exc}"[:300]
    print(f"[callsite] precision consumption probe: {_PREC_PROBE}", flush=True)


def _t5_tokenizer():
    """The verifier's own T5 tokenizer (the model's tokenizer_3): prompt -> token ids, for the ledger's prompt
    identities. None when it cannot be loaded -- request coverage is then unmeasured, never assumed."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL, subfolder="tokenizer_3")
        return lambda text: tok(text, max_length=256, truncation=True)["input_ids"]
    except Exception as exc:  # noqa: BLE001
        print(f"[callsite] T5 tokenizer unavailable for the precision ledger: {type(exc).__name__}: {exc}"[:300],
              flush=True)
        return None


def _write_precision_gate() -> None:
    """`precision_recorded` (profile mode only, scored): the worker's record judged against the profile's definition
    (tests/sd3_precision_profiles.precision_verdict): compute from the operand witness of every served call, the
    consumption probe of the armed request, the weights census, the initial noise, TF32. Unmeasured is never a pass;
    the measurement state and the attribution are recorded separately."""
    import hashlib
    events, raw = [], b""
    try:
        raw = _CPROBE_RECORD.read_bytes()
        for ln in raw.decode("utf-8", errors="replace").splitlines():
            if ln.startswith("{"):
                try:
                    events.append(json.loads(ln))
                except ValueError:
                    pass
    except OSError:
        pass
    ledger = [dict(q) for q in _REQUEST_LEDGER]
    tok = _t5_tokenizer()
    for q in ledger:
        try:
            q["ids_sha"], q["ntok"] = _PREC.canonical_ids_sha(tok(q["prompt"])) if tok else (None, None)
        except Exception as exc:  # noqa: BLE001
            q["ids_sha"], q["tokenize_error"] = None, f"{type(exc).__name__}: {exc}"[:200]
    v = _PREC.precision_verdict(PRECISION_PROFILE, events, require_consumption=True, ledger=ledger)
    jc = v["observed"].get("joint_conditioning") or {}
    bad = v["mismatches"] + v["unmeasured"]
    write_gate(GateResult(gate="precision_recorded", passed=not bad, score=0.0 if bad else 1.0, is_hard_gate=False,
                          error="; ".join(bad)[:500] or None,
                          details={"profile": PRECISION_PROFILE, "definition": _PREC.PROFILES[PRECISION_PROFILE],
                                   "instrument": "v2 (operand witness + consumption probe)",
                                   "mismatches": v["mismatches"], "unmeasured": v["unmeasured"],
                                   "measured": v["measured"], "attribution": v["attribution"],
                                   "not_evaluated": bool(v["unmeasured"]) and not v["mismatches"],
                                   "observed": v["observed"], "consumption_request": dict(_PREC_PROBE),
                                   "ledger": ledger, "tokenizer": "tokenizer_3" if tok else None,
                                   "record": {"path": str(_CPROBE_RECORD), "bytes": len(raw),
                                              "sha256": hashlib.sha256(raw).hexdigest(), "lines": len(events)},
                                   "joint_coverage": {"requests": jc.get("requests"), "gaps": jc.get("gaps"),
                                                      "rows": jc.get("rows")},
                                   "tf32_arm": os.environ.get("SD3_PRECISION_TF32"), "scored": True, "diagnostic": False}))
    print(f"[callsite] precision_recorded ({PRECISION_PROFILE}): {'honoured' if not bad else bad}", flush=True)


def _probe_attention_backend(gen, fixtures, token, traj, ref_set: str) -> None:
    ' probe attention backend.'
    arch = _dit_arch(MODEL)
    out: dict = {"scored": True, "gate": "attention_backend_selection", "arch": arch}
    ev: dict = {"passed": False, "not_evaluated": True, "reasons": []}
    try:
        if _jattn is None or _cprobe is None:
            raise RuntimeError(f"judges not importable: {_JATTN_IMPORT_ERROR or _CPROBE_IMPORT_ERROR}")
        summary = _cprobe_summary(token, None)
        default_fw: "list | None" = []
        for r in traj or []:
            fw = _jattn.forwards_between(summary, r.get("t0"), r.get("t1"))
            if fw is None:
                default_fw = None
                break
            default_fw += fw
        out["shutdown_default"] = _bounded(gen.shutdown, 90)
        _write_json(_BACKEND_OUT, out)
        if out['shutdown_default'] is not None:
            raise RuntimeError('default engine shutdown did not complete cleanly: ' + str(out['shutdown_default']))
        tok2 = _cprobe.arm(os.environ, _BACKEND_RECORD)
        gen2, t0, t1, img2 = None, None, None, None
        try:
            from sglang.multimodal_gen import DiffGenerator
            gen2 = DiffGenerator.from_pretrained(model_path=MODEL, num_gpus=1, attention_backend="torch_sdpa",
                                                 **(_PREC.engine_kwargs(PRECISION_PROFILE) if PRECISION_PROFILE else {}),
                                                 **_engine_ports())
        finally:
            _cprobe.disarm(os.environ, tok2)
        try:
            fx = fixtures[0]
            t0 = time.time()
            img2, _req2 = _gen(gen2, fx["prompt"], fx)
            t1 = time.time()
        finally:
            out["shutdown_sdpa"] = _bounded(gen2.shutdown, 90)
        if out['shutdown_sdpa'] is not None:
            raise RuntimeError('SDPA engine shutdown did not complete cleanly: ' + str(out['shutdown_sdpa']))
        sum2 = _cprobe.read_summary(tok2, None, parent_pid=os.getpid())
        sdpa_fw = _jattn.forwards_between(sum2, t0, t1)
        img_ev = _fixture_match(img2, fixtures[0], ref_set)
        try:
            save_image(img2, f"{fixtures[0]['id']}_candidate_torch_sdpa.png")
        except Exception:  # noqa: BLE001
            pass
        ev = _jattn.backend_evidence(default_fw, sdpa_fw, img_ev, arch["num_layers"])
        out.update({"sdpa_request": {"t0": t0, "t1": t1, "image": img2 is not None}, "sdpa_image": img_ev,
                    "evidence": ev})
    except Exception as exc:  # noqa: BLE001 - never a crash; a probe that could not run says so
        ev = {"passed": False, "not_evaluated": True, "reasons": [f"{type(exc).__name__}: {exc}"[:300]]}
        out["error"] = ev["reasons"][0]
    _write_json(_BACKEND_OUT, out)
    try:
        _diag_gate("attention_backend_selection", "attention_backend_selection", ev, {
            "evidence": "impl classes of the DiT's attention calls under the default engine and a torch_sdpa engine, "
                        "and the torch_sdpa engine's fixture-0 image on that fixture's bars",
            "arch": arch, "calls_per_forward": ev.get("calls_per_forward"), "default_impls": ev.get("default_impls"),
            "sdpa_impls": ev.get("sdpa_impls"),
            "sdpa_image": {k: (out.get("sdpa_image") or {}).get(k)
                           for k in ("match", "psnr", "ssim", "clip_img", "clip_txt", "reference_set", "note")},
            "shutdowns": {k: out.get(k) for k in ("shutdown_default", "shutdown_sdpa")}})
    except Exception as exc:  # noqa: BLE001
        print(f"[attention-backend] could not write its gate: {type(exc).__name__}: {exc}", flush=True)
    print(f"[attention-backend] attention_backend_selection passed={ev.get('passed')} "
          f"not_evaluated={ev.get('not_evaluated')} {'; '.join(ev.get('reasons') or [])[:400]}", flush=True)


def main() -> None:

    if not GOLDEN.exists():

        for g in LOAD_BOUND:
            write_gate(GateResult(gate=g, passed=False, score=0.0,
                                  error="Independent reference fixtures are missing or invalid"))
        print("[callsite] missing reference fixtures; emitted 0-score gates", flush=True)
        return

    spec = json.loads(GOLDEN.read_text())
    fixtures = spec["fixtures"]
    smoke_ids = set(spec.get("smoke_ids", []))


    diagnostic_ids = set(spec.get("diagnostic_ids", []))

    def _probe_sd3_progress():


        p = {}
        try:
            import sglang.multimodal_gen  # noqa: F401  (registry import side effects)
            from sglang.multimodal_gen.configs.pipeline_configs.stablediffusion3 import (  # noqa: F401
                StableDiffusion3PipelineConfig,
            )
            p["registered"] = True
        except Exception as e:  # noqa: BLE001
            p["registered"] = False
            p["reg_err"] = f"{type(e).__name__}: {e}"[:140]
        try:
            from sglang.multimodal_gen.runtime.models.dits.stablediffusion3 import (  # noqa: F401
                SD3Transformer2DModel,
            )
            p["dit_importable"] = True
        except Exception as e:  # noqa: BLE001
            p["dit_importable"] = False
            p["dit_err"] = f"{type(e).__name__}: {e}"[:140]
        return p

    _cp_token = None
    try:
        from sglang.multimodal_gen import DiffGenerator
        _cap_path = None
        import tempfile as _tf
        _capf = _tf.NamedTemporaryFile("w+", suffix=".loaderlog", delete=False)
        _cap_path = _capf.name
        _capf.close()
        _saved1, _saved2 = os.dup(1), os.dup(2)
        _tee = subprocess.Popen(["tee", "-a", _cap_path], stdin=subprocess.PIPE,
                                stdout=_saved1, stderr=subprocess.DEVNULL)
        os.dup2(_tee.stdin.fileno(), 1)
        os.dup2(_tee.stdin.fileno(), 2)


        _cp_token = _cprobe_arm()
        _penv = dict(_profile_env(), INFB_SD3_BATCH_OBSERVE="1")
        _penv_saved = {k: os.environ.get(k) for k in list(_penv) + ["NVIDIA_TF32_OVERRIDE", "INFB_SD3_PRECISION_WITNESS"]}
        if PRECISION_PROFILE == "fp32" and os.environ.get("SD3_PRECISION_TF32") == "on":
            os.environ.pop("NVIDIA_TF32_OVERRIDE", None)
        os.environ.update(_penv)
        try:
            _pkw = _PREC.engine_kwargs(PRECISION_PROFILE) if PRECISION_PROFILE else {}
            if PRECISION_PROFILE:
                print(f"[callsite] precision profile {PRECISION_PROFILE}: switches {_pkw}, env {_penv}", flush=True)
            gen = DiffGenerator.from_pretrained(model_path=MODEL, num_gpus=1, **_pkw, **_engine_ports())
        finally:
            for _k, _v in _penv_saved.items():
                if _v is None:
                    os.environ.pop(_k, None)
                else:
                    os.environ[_k] = _v
            _cprobe_disarm(_cp_token)
            sys.stdout.flush(); sys.stderr.flush()
            os.dup2(_saved1, 1); os.dup2(_saved2, 2)
            os.close(_saved1); os.close(_saved2)
            try:
                _tee.stdin.close(); _tee.wait(timeout=20)
            except Exception:  # noqa: BLE001
                pass
        try:
            _LOADER_TEXT = Path(_cap_path).read_text(errors="replace")
        except Exception:  # noqa: BLE001
            _LOADER_TEXT = ""
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
        save_stderr_log("callsite_load", traceback.format_exc())


        prog = _probe_sd3_progress()
        write_gate(GateResult(gate="sd3_registered", passed=prog.get("registered", False),
                              score=1.0 if prog.get("registered") else 0.0,
                              error=None if prog.get("registered") else f"load failed: {err}",
                              details=prog))
        write_gate(GateResult(gate="sd3_dit_importable", passed=prog.get("dit_importable", False),
                              score=1.0 if prog.get("dit_importable") else 0.0, details=prog))
        for g in ("components_loaded", "pipeline_runs", "native_dit_forward", "stage_step_count",
                  "image_similarity_smoke", "image_similarity_full", "prompt_conditioning",
                  "runs_through_sglang_engine", "native_text_encoders", "native_vae", "native_transformer"):
            write_gate(GateResult(gate=g, passed=False, score=0.0, error=f"load failed: {err}"))
        print(f"[callsite] load failed; static progress: registered={prog.get('registered')} "
              f"dit_importable={prog.get('dit_importable')} ({err})", flush=True)

        _probe_components(_cp_token, None, load_error=err)
        return


    write_gate(GateResult(gate="sd3_registered", passed=True, score=1.0))
    write_gate(GateResult(gate="sd3_dit_importable", passed=True, score=1.0))
    write_gate(GateResult(gate="components_loaded", passed=True, score=1.0,
                          details={"note": "DiffGenerator.from_pretrained succeeded"}))


    _impls = _loaded_impls(_LOADER_TEXT)
    _cp_summary = _cprobe_summary(_cp_token, _impls)
    _enc_comps = ["text_encoder", "text_encoder_2", "text_encoder_3"]
    for _native_gate, _native_comps in (("native_text_encoders", _enc_comps), ("native_vae", ["vae"])):
        write_gate(_native_gate_result(_native_gate, _native_comps, _cp_summary, _impls,
                                       bool(_LOADER_TEXT.strip())))
    print(f"[callsite] loader lines (diagnostic): {_impls}", flush=True)
    print("[callsite] native gates judged on the worker record: "
          + " ".join(f"{g}={(v or {}).get('verdict')}"
                     for g, v in ((_cp_summary or {}).get("gate_verdicts") or {}).items()), flush=True)


    prompts = [fx["prompt"] for fx in fixtures]
    imgs, gen_ok, ran, traj = [], [], False, []
    for fx in fixtures:
        tr = {"id": fx["id"], "steps": int(fx["steps"]), "image": False, "trajectory_len": 0}
        try:
            tr["t0"] = time.time()
            img, req = _gen(gen, fx["prompt"], fx); ran = True
            tr["t1"] = time.time()
            tr.update(req)
            tr["image"] = img is not None
            print(f"[callsite] {fx['id']}: trajectory_len={tr['trajectory_len']} "
                  f"(latents {tr.get('latents_shape')}, timesteps {tr.get('timesteps_shape')})", flush=True)
            if img is None:


                print(f"[callsite] {fx['id']}: no image produced", flush=True)
                imgs.append(None); gen_ok.append(False)
            else:
                save_image(img, f"{fx['id']}_candidate.png")
                imgs.append(img); gen_ok.append(True)
        except Exception as exc:  # noqa: BLE001
            print(f"[callsite] {fx['id']} gen error: {type(exc).__name__}: {exc}", flush=True)
            tr["error"] = f"{type(exc).__name__}: {exc}"[:300]
            imgs.append(None); gen_ok.append(False)
        traj.append(tr)

    traj.append(_steps_request(gen, fixtures))


    _native_consumption = _native_consumption_request(gen, fixtures, _cp_token, imgs[0] if imgs else None)
    _cp_after = _cprobe_summary(_cp_token, _impls)
    _cp_after["native_consumption"] = _native_consumption
    _gen_ev = generation_evidence(_cp_after, traj[:len(fixtures)], STEP_COUNT_TOLERANCE)
    print(f"[callsite] native generation evidence: ok={_gen_ev['ok']} {_gen_ev['reason'] or ''}", flush=True)


    write_gate(_native_gate_result("native_transformer", ["transformer"], _cp_after, _impls,
                                   bool(_LOADER_TEXT.strip())))
    print("[callsite] native_transformer judged on the post-generation worker record: "
          + str(((_cp_after or {}).get("gate_verdicts") or {}).get("native_transformer", {}).get("verdict")),
          flush=True)
    for fx in fixtures:
        for _set, _suffix in (("bf16", ""), ("fp32", "_fp32"), ("mixed", "_mixed")):
            if _set not in REF_SETS:
                continue
            try:
                save_image(to_image_array(np.load(REF_DIRS[_set] / fx["golden_npy"])), f"{fx['id']}_golden{_suffix}.png")
            except Exception:  # noqa: BLE001
                pass

    S, clip_err = None, None


    _clip_imgs = [im if im is not None
                  else np.zeros((int(fixtures[j]["h"]), int(fixtures[j]["w"]), 3), np.uint8)
                  for j, im in enumerate(imgs)]
    try:
        S = clip_text_scores(_clip_imgs, prompts)
    except Exception as exc:  # noqa: BLE001
        clip_err = f"{type(exc).__name__}: {exc}"
        save_stderr_log("callsite_clip", traceback.format_exc())
        print(f"[callsite] CLIP prompt-adherence UNAVAILABLE: {clip_err}", flush=True)


        print("[callsite]   *** this is an ENVIRONMENT failure, not a candidate one: the CLIP "
              "scorer is missing. ***", flush=True)
        print(f"[callsite]   resolved id: {_clip_id()}   SD3_CLIP_MODEL="
              f"{os.environ.get('SD3_CLIP_MODEL', '<unset>')}   HF_HOME="
              f"{os.environ.get('HF_HOME', '<unset>')}", flush=True)
        print("[callsite]   Prepare the pinned SD3 and CLIP assets on the host before starting "
              "the offline trial. From the repository root, run:", flush=True)
        print("[callsite]     uv run --with huggingface-hub==1.28.0 python "
              "tasks/support_sd3_2xh100_sglang/environment/scripts/model_assets.py fetch "
              "--root /absolute/path/sd3-assets", flush=True)
        print("[callsite]   See this task's README for Hugging Face access/login and the "
              "read-only /opt/infb/models mount. The asset helper verifies the pinned "
              "revisions and file hashes.", flush=True)

    def _ge(v, t):
        return v is not None and v >= t


    _missing_sets = missing_reference_sets(fixtures, {k: REF_DIRS[k] for k in REF_SETS})
    per_by_set = {name: [] for name in REF_SETS}
    for i, fx in enumerate(fixtures):
        rec = {"id": fx["id"]}
        _pix = {}
        if imgs[i] is None:


            rec["no_image"] = True
        else:
            for _set in REF_SETS:
                if _set in _missing_sets:
                    continue
                try:
                    _pix[_set] = pixel_diag(imgs[i], to_image_array(np.load(REF_DIRS[_set] / fx["golden_npy"])))
                except Exception:  # noqa: BLE001
                    _pix[_set] = {}
        if S is not None and gen_ok[i]:
            adh = float(S[i, i]); best = int(S[i].argmax())
            rec["clip_txt"] = adh


            rec["retrieval_ok"] = (prompts[best] == prompts[i])
            rec["clip_txt_best_prompt"] = prompts[best][:36]


        for _set in REF_SETS:
            r = {**rec, **_pix.get(_set, {}), "reference_set": _set}
            r["bars"] = profile_bars_for(fx, _set)
            r["psnr_bar"] = r["bars"]["psnr"]
            r["match"] = bool(gen_ok[i] and _set not in _missing_sets
                              and _IMAGE_BARS.matches(r, r["bars"]))
            if not r["match"] and (clip_err or not gen_ok[i] or _set in _missing_sets):
                r["note"] = (f"reference set {_set} missing" if _set in _missing_sets
                             else clip_err or "generation failed")
            per_by_set[_set].append(r)
            print(f"[callsite] {fx['id']} [{_set}]: match={r['match']} | "
                  f"clip_txt={_fmt(r.get('clip_txt'))}>={_fmt(r['bars']['clip_txt'])} "
                  f"psnr={_fmt(r.get('psnr'), '%.1f')}>={_fmt(r['bars']['psnr'], '%.2f')} "
                  f"ssim={_fmt(r.get('ssim'))}>={_fmt(r['bars']['ssim'])} "
                  f"clip_img={_fmt(r.get('clip_img'))}>={_fmt(r['bars']['clip_img'])} "
                  f"(retrieval_ok={r.get('retrieval_ok')} rel_l2={_fmt(r.get('rel_l2'))})", flush=True)

    write_gate(GateResult(gate="pipeline_runs", passed=ran, score=1.0 if ran else 0.0))


    for _set in REF_SETS:
        for p in per_by_set[_set]:
            p["scored"] = p["id"] not in diagnostic_ids
    iv_by_set = {_set: image_gate_verdicts(per_by_set[_set], diagnostic_ids, smoke_ids, FULL_RATE_MIN)
                 for _set in REF_SETS}
    rates = {_set: iv_by_set[_set]["full_rate"] for _set in REF_SETS}


    proximity = reference_proximity(per_by_set, [p["id"] for p in iv_by_set[REF_TIE]["scored"]])
    chosen = None if _missing_sets else choose_reference_set(proximity, REF_TIE)
    if PRECISION_PROFILE:
        chosen = None if PRECISION_PROFILE in _missing_sets else PRECISION_PROFILE
        _missing_sets = {k: v for k, v in _missing_sets.items() if k == PRECISION_PROFILE}
    per = per_by_set[chosen or REF_TIE]
    iv = iv_by_set[chosen or REF_TIE]
    scored, diag = iv["scored"], iv["diagnostic"]
    smoke_ok, full_rate = iv["smoke_ok"] and chosen is not None, iv["full_rate"] if chosen else 0.0
    _ref_detail = {"profile": PRECISION_PROFILE, "threshold_policy": _IMAGE_BARS.receipt(),
                   "bars_state": PROFILE_BARS_STATE.get(PRECISION_PROFILE) if PRECISION_PROFILE else None,
                   "chosen_set": chosen, "tie_set": REF_TIE, "missing_sets": _missing_sets,
                   "proximity_metric": PROXIMITY_METRIC, "proximity_by_set": proximity, "image_score_by_set": rates,
                   "per_by_set": {_set: iv_by_set[_set]["scored"] for _set in REF_SETS}}
    _ref_error = (None if chosen else "reference set(s) missing, so the candidate cannot be judged (fail closed): "
                  + "; ".join(f"{k}: {len(v)} fixture(s)" for k, v in sorted(_missing_sets.items())))
    print(f"[callsite] reference sets: proximity ({PROXIMITY_METRIC}) {proximity}; image score {rates}; "
          f"judged on {chosen} (tie -> {REF_TIE})"
          + (f"; MISSING {sorted(_missing_sets)}" if _missing_sets else ""), flush=True)
    _gate_desc = (f"All four finite image metrics must meet the explicit fixture bars in "
                  f"{_IMAGE_BARS.REVISION}; precision={PRECISION_PROFILE or chosen}; "
                  f"table_sha256={_IMAGE_BARS.SHA256}. {_IMAGE_BARS.STATE}")

    _full_pixels_ok = bool(iv["full_ok"] and chosen is not None)
    _smoke_ok, _full_ok = smoke_ok and _gen_ev["ok"], _full_pixels_ok and _gen_ev["ok"]
    _img_error = _ref_error or (None if _gen_ev["ok"] else _gen_ev["reason"])


    _full_error = "; ".join(x for x in (_img_error, image_gate_failure(scored, full_rate, FULL_RATE_MIN, chosen,
                                                                       proximity)) if x) or None
    write_gate(GateResult(gate="image_similarity_smoke", passed=_smoke_ok,
                          score=1.0 if _smoke_ok else 0.0, error=_img_error,
                          details={"metric": _gate_desc, "per": iv["smoke"], "clip_err": clip_err,
                                   "pixels_ok": smoke_ok, "native_generation": _gen_ev,
                                   "not_evaluated": smoke_ok and _gen_ev.get("not_evaluated", False),
                                   "chosen_set": chosen, "proximity_by_set": proximity,
                                   "image_score_by_set": rates}))
    write_gate(GateResult(gate="image_similarity_full", passed=_full_ok,
                          score=full_rate if _gen_ev["ok"] else 0.0, error=_full_error,
                          details={"metric": _gate_desc, "rate": full_rate, "rate_min": FULL_RATE_MIN,
                                   "scored_ids": [p["id"] for p in scored],
                                   "diagnostic_ids": sorted(diagnostic_ids),
                                   "per": scored, "diagnostic_per": diag, "clip_err": clip_err,
                                   "pixels_ok": _full_pixels_ok, "native_generation": _gen_ev,
                                   "not_evaluated": _full_pixels_ok and _gen_ev.get("not_evaluated", False),
                                   **_ref_detail}))
    if diag:
        print("[callsite] DIAGNOSTIC fixtures (generated + scored for metrics, NOT counted in the gate): "
              + ", ".join(f"{p['id']} match={p.get('match')}" for p in diag), flush=True)


    try:
        f0 = fixtures[0]
        _ra = {"id": "prompt_conditioning_a", "steps": int(f0["steps"]), "t0": time.time()}
        a, _qa = _gen(gen, "a red cube on a white table", f0)
        _ra.update({"t1": time.time(), **_qa, "image": a is not None})
        _rb = {"id": "prompt_conditioning_b", "steps": int(f0["steps"]), "t0": time.time()}
        b, _qb = _gen(gen, "a snowy mountain at sunset", f0)
        _rb.update({"t1": time.time(), **_qb, "image": b is not None})
        _pc_summary = _cprobe_summary(_cp_token, _impls)
        _pc_summary["native_consumption"] = _native_consumption
        _pc_ev = generation_evidence(_pc_summary, [_ra, _rb], STEP_COUNT_TOLERANCE)
        if a is None or b is None:
            write_gate(GateResult(gate="prompt_conditioning", passed=False, score=0.0,
                                  error="no image produced; conditioning cannot be assessed",
                                  details={"a_produced": a is not None,
                                           "b_produced": b is not None}))
        else:
            save_image(a, "prompt_conditioning_a.png")
            save_image(b, "prompt_conditioning_b.png")
            _pc_m = pixel_diag(a, b)
            _pc_v = prompt_images_differ(_pc_m, PC_SAME_BARS)
            differ = _pc_v["differ"]
            _pc_ok = differ and _pc_ev["ok"]

            _cond_min = SIM_THRESHOLD_REL_L2 * PROMPT_COND_FACTOR
            _pc_rel = _pc_m.get("rel_l2")

            _pc_error = "; ".join(x for x in (
                None if _pc_ev["ok"] else _pc_ev["reason"],
                None if differ else _pc_v["reason"]) if x) or None
            write_gate(GateResult(gate="prompt_conditioning", passed=_pc_ok,
                                  score=1.0 if _pc_ok else 0.0, error=_pc_error,
                                  details={"differ": differ, "psnr": _pc_m.get("psnr"), "ssim": _pc_m.get("ssim"),
                                           "clip_img": _pc_m.get("clip_img"), "same_image_bars": PC_SAME_BARS,
                                           "below": _pc_v["below"], "missing": _pc_v["missing"],
                                           "rel_l2": _pc_rel, "rel_l2_min": _cond_min,
                                           "legacy_rel_l2_differ": (_pc_rel > _cond_min) if _pc_rel is not None else None,
                                           "legacy_rule": "rel_l2 is diagnostic; image metrics determine the verdict",
                                           "native_generation": _pc_ev,
                                           "not_evaluated": differ and _pc_ev.get("not_evaluated", False)}))
    except Exception as exc:  # noqa: BLE001
        write_gate(GateResult(gate="prompt_conditioning", passed=False, score=0.0,
                              error=f"{type(exc).__name__}: {exc}"))


    try:
        _step_readings = native_request_evidence(traj, STEP_COUNT_TOLERANCE, _cp_after)
        _step_per = [{**p, "steps": p["expected_steps"], "trajectory_len": p["observed_steps"],
                      "native_ok": p["ok"], "stage_ok": p["ok"] and _native_consumption.get("ok", False)}
                     for p in _step_readings["per"]]
        tv = {"native_dit_forward": _step_readings["ok"],
              "stage_step_count": _step_readings["ok"] and _native_consumption.get("ok", False), "per": _step_per}
        t0 = traj[0] if traj else {}
        tcount = int(t0.get("trajectory_len") or 0)
        n_steps = int(t0.get("steps") or 0)
        native_ok, stage_ok = tv["native_dit_forward"], tv["stage_step_count"]
        _traj_detail = {"trajectory_len": tcount, "num_inference_steps": n_steps,
                        "tolerance": STEP_COUNT_TOLERANCE, "per_request": tv["per"],
                        "source": "trusted served-DiT timestep/CFG rows and paired output-consumption intervention",
                        "consumption": _native_consumption}


        _native_fail = [str(p["id"]) + ": " + p["reason"] for p in tv["per"] if not p["native_ok"]]
        _stage_fail = list(_native_fail)
        if not _native_consumption.get("ok"):
            _stage_fail.append("output consumption: " + _native_consumption.get("reason", "unmeasured"))
        _stage_ne = not stage_ok and _step_readings["not_evaluated"] and (
            _native_consumption.get("ok") or _native_consumption.get("not_evaluated"))
        write_gate(GateResult(gate="native_dit_forward", passed=native_ok, is_hard_gate=False,
            score=1.0 if native_ok else 0.0,
            error=None if native_ok else ("; ".join(_native_fail) or "no scored request")[:800],
            details={**_traj_detail, "failing_requests": _native_fail,
                     "note": "unscored duplicate evidence; stage_step_count is the scored gate"}))


        metrics_keys = list(t0.get("metrics_keys") or [])
        peak = t0.get("peak_memory_mb")
        engine_ok = bool(tcount > 0 or len(metrics_keys) > 0 or (peak or 0) > 0)
        write_gate(GateResult(gate="runs_through_sglang_engine", passed=engine_ok,
            score=1.0 if engine_ok else 0.0,
            details={"metrics_keys": metrics_keys, "peak_memory_mb": peak, "trajectory_len": tcount}))


        write_gate(GateResult(gate="stage_step_count", passed=stage_ok, is_hard_gate=True,
            score=1.0 if stage_ok else 0.0,
            error=None if stage_ok else ("; ".join(_stage_fail) or "no scored request")[:800],
            details={**_traj_detail, "expected": n_steps, "failing_requests": _stage_fail,
                     "not_evaluated": _stage_ne,
                     "hard_gate_reason": "denoising must run n_steps times through sglang's stages"}))
        print("[callsite] observed/requested steps per request: "
              + ", ".join(f"{p['id']} {p['trajectory_len']}/{p['steps']}" for p in tv["per"])
              + f" tol={STEP_COUNT_TOLERANCE} peak_mb={peak} native_ok={native_ok} engine_ok={engine_ok} "
              f"stage_ok={stage_ok}" + (f" FAILING: {'; '.join(_stage_fail)}" if _stage_fail else ""), flush=True)
    except Exception as exc:  # noqa: BLE001
        save_stderr_log("callsite_steps", traceback.format_exc())
        for g in ("native_dit_forward", "runs_through_sglang_engine", "stage_step_count"):
            write_gate(GateResult(gate=g, passed=False, score=0.0,
                                  is_hard_gate=g == "stage_step_count",
                                  error=f"{type(exc).__name__}: {exc}"))
    print(f"[callsite] done. full_rate={full_rate:.3f}", flush=True)


    _label_vacuous_on_fallback(_gen_ev)


    _probe_components(_cp_token, _impls)


    _probe_component_tap(gen, fixtures, _cp_token, imgs[0] if imgs else None)
    if PRECISION_PROFILE:

        _probe_precision_consumption(gen, fixtures, _cp_token, imgs[0] if imgs else None)
        _write_precision_gate()
        if PRECISION_PROFILE != _PG.DEFAULT_PROFILE:


            print(f"[callsite] profile {PRECISION_PROFILE}: joint-attention / concurrency / backend probes are "
                  f"measured on the {_PG.DEFAULT_PROFILE} profile's process", flush=True)
            return

    _probe_joint_attention(gen, fixtures, _cp_token, traj, imgs[0] if imgs else None)


    _probe_concurrency(gen, fixtures, smoke_ids, per, imgs, ref_dir=REF_DIRS[chosen or REF_TIE])


    _probe_attention_backend(gen, fixtures, _cp_token, traj, chosen or REF_TIE)


def _engine_ports() -> dict:
    """Return this verifier queue's assigned engine port block."""
    from _common import engine_port_kwargs

    return engine_port_kwargs()

if __name__ == "__main__":
    main()
