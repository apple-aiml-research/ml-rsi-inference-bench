#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
'SD3 verify fp8 image reference utilities.'


from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
DAY0 = HERE / "day0"
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(DAY0))

ARTIFACTS = Path(os.environ.get("INFB_ARTIFACTS_DIR", "/logs/artifacts"))
OUT = ARTIFACTS / "sd3_day0_fp8_reference.json"
IMG_DIR = ARTIFACTS / "sd3_day0_fp8_reference"
SCHEMA = "sd3-day0-fp8-image-reference/v1"
RECORDED_ONLY = "record only: no gate and no bar; bars require independent legal-kernel calibration"
CHILD_TIMEOUT_S = 600


CONTROL_MIN_PSNR_DB = 40.0


WEIGHT_MAP_MAX_REL_L2 = 0.25
NEGATIVE_PROMPT = " "
GOLDEN_DIR = HERE
CHILD_DEVICE = "cuda"


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")


def image_sha256(img) -> str:
    """The FP8 verifier's image identity: sha256 of the contiguous uint8 (H, W, 3) bytes."""
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest()


def psnr_db(a, b) -> float:
    """PSNR on uint8 RGB, data range 255; 100.0 for identical images (the in-run golden check's convention)."""
    import numpy as np
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    mse = float((d * d).mean())
    return 100.0 if mse <= 1e-9 else float(10.0 * np.log10(255.0 ** 2 / mse))


def ssim(a, b) -> float | None:
    """The task's pixel SSIM (verify_callsite_main.ssim); None where scikit-image is not importable."""
    try:
        from skimage.metrics import structural_similarity
    except Exception:  # noqa: BLE001
        return None
    return float(structural_similarity(a, b, channel_axis=-1, data_range=255))


def image_metrics(a, b) -> dict:
    import numpy as np
    if np.asarray(a).shape != np.asarray(b).shape:
        return {"shape_mismatch": [list(np.asarray(a).shape), list(np.asarray(b).shape)]}
    d = np.abs(np.asarray(a, np.int16) - np.asarray(b, np.int16))
    return {"psnr_db": psnr_db(a, b), "ssim": ssim(a, b), "bit_identical": bool(d.max() == 0),
            "mean_abs": float(d.mean()), "max_abs": int(d.max())}


def _rel_l2(a, b) -> float:
    import torch
    a = a.detach().double().flatten()
    b = b.detach().double().flatten()
    return float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b))


def _emulated_linear_class():
    import torch

    import fp8_torch_ref as T

    class EmulatedW8A8Linear(torch.nn.Module):
        """The W8A8 emulation of one quantised linear from the asset's bytes; counts its calls (the swap's proof)."""

        def __init__(self, name: str, lin, wq, scale, device):
            super().__init__()
            self.name = name
            self.in_features, self.out_features = lin.in_features, lin.out_features
            self.w_deq = T.dequant_block(wq.to(device), scale.to(device))
            self.b32 = None if lin.bias is None else lin.bias.detach().to(device=device, dtype=torch.float32)
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            return T.linear_w8a8(x, None, None, self.b32, w_deq=self.w_deq).to(x.dtype)

    return EmulatedW8A8Linear


def install_w8a8_emulation(transformer, asset: Path, modules: list, device) -> tuple[list, dict]:
    """Replace each listed linear of a diffusers SD3Transformer2DModel by its W8A8 emulation from the asset's bytes.
    Returns (the installed modules, a summary). A module that is missing, is not a Linear, has another shape, or whose
    dequantised bytes are not near its own weight is NOT installed and is named in the summary's `problems`."""
    import torch
    from safetensors import safe_open
    cls = _emulated_linear_class()
    installed, problems, rel = [], [], {}
    with safe_open(str(asset / "diffusion_pytorch_model.safetensors"), "pt", device="cpu") as f:
        keys = set(f.keys())
        for name in modules:
            parent_name, _, leaf = name.rpartition(".")
            try:
                parent = transformer.get_submodule(parent_name)
                lin = getattr(parent, leaf)
            except (AttributeError, KeyError) as exc:
                problems.append(f"{name}: not a module of the pipeline's transformer ({type(exc).__name__})")
                continue
            if not isinstance(lin, torch.nn.Linear):
                problems.append(f"{name}: a {type(lin).__name__}, not a Linear")
                continue
            if name + ".weight" not in keys or name + ".weight_scale_inv" not in keys:
                problems.append(f"{name}: the asset holds no weight / weight_scale_inv for it")
                continue
            wq, sc = f.get_tensor(name + ".weight"), f.get_tensor(name + ".weight_scale_inv")
            if wq.dtype != torch.float8_e4m3fn or tuple(wq.shape) != tuple(lin.weight.shape):
                problems.append(f"{name}: asset {wq.dtype} {tuple(wq.shape)} vs the layer's {tuple(lin.weight.shape)}")
                continue
            emu = cls(name, lin, wq, sc, device)
            rel[name] = _rel_l2(emu.w_deq, lin.weight.to(device=device, dtype=torch.float32))
            if not rel[name] <= WEIGHT_MAP_MAX_REL_L2:
                problems.append(f"{name}: dequantised asset bytes {rel[name]:.3f} (rel L2) from the layer's own weight")
                continue
            setattr(parent, leaf, emu)
            installed.append(emu)
    vals = sorted(rel.values())
    worst = max(rel, key=rel.get) if rel else None
    return installed, {"expected": len(modules), "installed": len(installed), "problems": problems[:20],
                       "n_problems": len(problems),
                       "weight_rel_l2": {"min": vals[0], "median": vals[len(vals) // 2], "max": vals[-1],
                                         "worst_module": worst} if vals else None}


def load_pipeline(model: str, device: str):
    """diffusers' own SD3 pipeline, every component bf16: the committed golden's recipe (generate_goldens_inrun.py)."""
    import torch
    from diffusers import StableDiffusion3Pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(model, torch_dtype=torch.bfloat16).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate(pipe, fx: dict, device):
    """One image under the committed golden's recipe (generate_goldens_inrun.py), uint8 (H, W, 3)."""
    import numpy as np
    import torch
    g = torch.Generator(device=device).manual_seed(int(fx["seed"]))
    o = pipe(prompt=fx["prompt"], negative_prompt=NEGATIVE_PROMPT, num_inference_steps=int(fx["steps"]),
             guidance_scale=float(fx["cfg"]), height=int(fx["h"]), width=int(fx["w"]), generator=g,
             output_type="pil")
    return np.asarray(o.images[0])


def _inside(path: str, root: str) -> bool:
    p, r = Path(path).resolve(), Path(root).resolve()
    return p == r or r in p.parents


def child(model: str, asset: Path, fx: dict, golden: Path, out: Path, device: str = "cuda") -> dict:
    """Isolated: diffusers + torch + safetensors + tests/day0 only. Writes control.npy / reference.npy into `out`."""
    import numpy as np
    import torch

    import diffusers
    out.mkdir(parents=True, exist_ok=True)
    rec = {"fixture": fx.get("id"), "device": device, "timing_s": {}}
    missing = [what for what, ok in (("the model is not a local checkpoint directory", Path(model).is_dir()),
                                     ("the asset has no fp8_manifest.json", (asset / "fp8_manifest.json").is_file()),
                                     ("the committed golden is missing", golden.is_file())) if not ok]
    if missing:
        return dict(rec, status="not_evaluated", reason="; ".join(missing))
    loc = os.path.dirname(os.path.abspath(diffusers.__file__))
    rec["diffusers"] = {"version": diffusers.__version__, "file": loc}
    if _inside(loc, os.environ.get("INFB_WORKSPACE", "/app")):
        return dict(rec, status="not_evaluated", reason=f"diffusers resolved inside the workspace ({loc}): the "
                                                        f"candidate could supply the reference")
    if device.startswith("cuda") and not torch.cuda.is_available():
        return dict(rec, status="not_evaluated", reason="no CUDA device visible to the reference child")
    torch.backends.cuda.matmul.allow_tf32 = False
    t = time.time()
    pipe = load_pipeline(model, device)
    rec["timing_s"]["pipeline_load"] = round(time.time() - t, 1)
    comps = {}
    for name in ("text_encoder", "text_encoder_2", "text_encoder_3", "transformer", "vae"):
        m = getattr(pipe, name, None)
        comps[name] = sorted({str(q.dtype) for q in m.parameters()}) if m is not None else ["absent"]
    rec["precision"] = {"components": comps, "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32)}
    if device.startswith("cuda"):
        rec["gpu"] = torch.cuda.get_device_name(0)
        torch.cuda.reset_peak_memory_stats()

    t = time.time()
    control = generate(pipe, fx, device)
    rec["timing_s"]["control"] = round(time.time() - t, 1)
    np.save(out / "control.npy", control)
    gold = np.load(golden)
    rec["control"] = {"sha256": image_sha256(control), "vs_committed_golden": image_metrics(control, gold)}
    p = rec["control"]["vs_committed_golden"].get("psnr_db")
    if p is None or not p >= CONTROL_MIN_PSNR_DB:
        return dict(rec, status="not_evaluated",
                    reason=f"the unmodified pipeline does not reproduce the committed golden ({p} dB < "
                           f"{CONTROL_MIN_PSNR_DB}): this environment is not the golden's, so the reference would "
                           f"differ in more than the quantised linears")
    man = json.loads((asset / "fp8_manifest.json").read_text())
    mods = sorted(man.get("quantized_modules") or [])
    t = time.time()
    installed, summary = install_w8a8_emulation(pipe.transformer, asset, mods, device)
    rec["timing_s"]["install"] = round(time.time() - t, 1)
    rec["swap"] = dict(summary, asset_tensor_set_sha256=man.get("tensor_set_sha256"),
                       n_quantized_modules_declared=man.get("n_quantized_modules"))
    if not mods or summary["n_problems"] or len(installed) != len(mods):
        return dict(rec, status="not_evaluated", reason=f"the emulation was not installed on every quantised "
                                                        f"linear ({len(installed)}/{len(mods)}): "
                                                        f"{'; '.join(summary['problems'][:3])}")
    t = time.time()
    ref = generate(pipe, fx, device)
    rec["timing_s"]["reference"] = round(time.time() - t, 1)
    calls = [m.calls for m in installed]
    rec["swap"].update(calls_min=min(calls), calls_max=max(calls), calls_expected=int(fx["steps"]))
    if device.startswith("cuda"):
        rec["peak_memory_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    if not (min(calls) == max(calls) == int(fx["steps"])):
        return dict(rec, status="not_evaluated", reason=f"the emulated linears were called {min(calls)}..{max(calls)}"
                                                        f" times, not once per step ({fx['steps']})")
    np.save(out / "reference.npy", ref)
    rec["reference"] = {"sha256": image_sha256(ref), "vs_committed_golden": image_metrics(ref, gold),
                        "vs_control": image_metrics(ref, control)}
    return dict(rec, status="measured")


def read_candidates(profiles: dict, artifacts: Path) -> dict:
    """{capability: record}. Each FP8 profile's image comes from the lossless PNG the FP8 verifier saved, and counts
    only if it decodes to the sha256 that verifier recorded for its request (the bytes it judged)."""
    import numpy as np
    out = {}
    for _tp, (gate, cap, _layer, artifact) in sorted(profiles.items()):
        c = {"gate": gate, "record": str(artifacts / artifact)}
        out[cap] = c
        p = artifacts / artifact
        if not p.is_file():
            c.update(status="not_evaluated", reason="the FP8 verifier wrote no record for this profile")
            continue
        try:
            r = json.loads(p.read_text())
        except Exception as exc:  # noqa: BLE001
            c.update(status="not_evaluated", reason=f"the FP8 record does not parse: {type(exc).__name__}")
            continue
        run = r.get("run") or {}
        c.update(gate_verdict=(r.get("verdict") or {}).get("verdict"),
                 asset_tensor_set_sha256=(r.get("asset") or {}).get("tensor_set_sha256"),
                 image_sha256=run.get("image_sha256"), image_png=run.get("image_png"))
        if not run.get("image_sha256") or not run.get("image_png"):
            c.update(status="not_evaluated", reason="the profile's request left no image (or no lossless PNG) "
                                                    "to read against the reference")
            continue
        try:
            from PIL import Image
            png = artifacts / Path(artifact).stem / Path(run["image_png"]).name
            img = np.asarray(Image.open(png if png.is_file() else run["image_png"]).convert("RGB"))
        except Exception as exc:  # noqa: BLE001
            c.update(status="not_evaluated", reason=f"the PNG does not load: {type(exc).__name__}: {exc}"[:300])
            continue
        if image_sha256(img) != run["image_sha256"]:
            c.update(status="not_evaluated", reason="the PNG is not the image the FP8 verifier recorded "
                                                    "(sha256 differs)")
            continue
        c["image"] = img
    return out


def _save_png(img, name: str) -> str | None:
    try:
        from PIL import Image
        IMG_DIR.mkdir(parents=True, exist_ok=True)
        p = IMG_DIR / name
        Image.fromarray(img).save(p)
        return str(p)
    except Exception:  # noqa: BLE001
        return None


def child_env() -> dict:
    'child env.'
    env = dict(os.environ)
    for k in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE", "PYTHONSTARTUP",
              "INFB_SD3_DAY0_TAP_DIR", "INFB_SD3_DAY0_LIB", "INFB_SD3_DAY0_FI_MANIFEST", "INFB_SD3_DAY0_FI_T0"):
        env.pop(k, None)
    env["HF_HUB_OFFLINE"] = "1"
    return env


def run_child(model: str, asset_dir: Path, fx: dict, golden: Path, cdir: Path) -> tuple[int, str]:
    """The isolated child (this file, --child): (returncode, stderr tail). Raises subprocess.TimeoutExpired."""

    model, asset_dir, golden, cdir = (Path(p).resolve() for p in (model, asset_dir, golden, cdir))
    cdir.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--child", str(model),
                        str(asset_dir), json.dumps(fx), str(golden), str(cdir), CHILD_DEVICE],
                       cwd=str(cdir), env=child_env(), capture_output=True, text=True, timeout=CHILD_TIMEOUT_S)
    return r.returncode, r.stderr[-2000:]


def main() -> dict:
    import numpy as np

    import verify_fp8_execution as V
    rec = {"schema": SCHEMA, "recorded_only": RECORDED_ONLY, "fixture": V.FIXTURE, "status": "not_evaluated",
           "timing_s": {}}
    t_start = time.time()
    deadline = os.environ.get("SD3_DAY0_DEADLINE_EPOCH")
    if deadline and time.time() > float(deadline):
        return dict(rec, reason="the verifier's time budget for day-0 diagnostics is spent")
    cands = read_candidates(V.PROFILES, ARTIFACTS)
    if not any("image" in c for c in cands.values()):
        return dict(rec, candidates=cands, reason="no FP8 profile served an image to read against a reference")
    fx = {f["id"]: f for f in json.loads(V.GOLDEN_SPEC.read_text())["fixtures"]}[V.FIXTURE]
    golden = GOLDEN_DIR / fx["golden_npy"]
    model = V._resolve_model()
    rec["model"] = model
    work = Path(tempfile.mkdtemp(prefix="sd3_day0_fp8ref_"))
    try:
        t = time.time()
        try:
            asset = V.build_asset(model, work / "fp8_transformer")
        except Exception as exc:  # noqa: BLE001 - ours, not the candidate's
            return dict(rec, candidates=cands, reason=f"verifier_defect: the asset did not build: "
                                                      f"{type(exc).__name__}: {exc}"[:500])
        rec["timing_s"]["asset_build"] = round(time.time() - t, 1)
        rec["asset"] = {"tensor_set_sha256": asset.get("tensor_set_sha256")}
        cdir = work / "child"
        t = time.time()
        try:
            rc, err = run_child(model, work / "fp8_transformer", fx, golden, cdir)
        except subprocess.TimeoutExpired:
            return dict(rec, candidates=cands, reason=f"the reference child did not finish in {CHILD_TIMEOUT_S} s")
        rec["timing_s"]["child"] = round(time.time() - t, 1)
        cp = cdir / "child.json"
        if rc != 0 or not cp.is_file():
            return dict(rec, candidates=cands, reason=f"the reference child exited {rc}", child_stderr=err)
        ch = json.loads(cp.read_text())
        rec["child"] = ch
        if ch.get("status") != "measured":
            if (cdir / "control.npy").is_file():
                rec["control_png"] = _save_png(np.load(cdir / "control.npy"), f"control.{V.FIXTURE}.png")
            return dict(rec, candidates=cands, reason=f"reference not_evaluated: {ch.get('reason')}")
        ref = np.load(cdir / "reference.npy")
        gold = np.load(golden)
        rec["reference_png"] = _save_png(ref, f"reference.{V.FIXTURE}.png")
        rec["control_png"] = _save_png(np.load(cdir / "control.npy"), f"control.{V.FIXTURE}.png")
        for c in cands.values():
            img = c.pop("image", None)
            if img is None:
                continue
            c.update(status="measured", vs_fp8_reference=image_metrics(img, ref), vs_bf16_golden=image_metrics(img, gold),
                     same_asset=c.get("asset_tensor_set_sha256") == asset.get("tensor_set_sha256"))
            a, b = c["vs_fp8_reference"].get("psnr_db"), c["vs_bf16_golden"].get("psnr_db")
            if a is not None and b is not None:
                c["psnr_gain_over_bf16_golden_db"] = a - b
        rec.update(status="measured", candidates=cands, reason=None)
        return rec
    finally:
        rec["timing_s"]["total"] = round(time.time() - t_start, 1)
        for c in cands.values():
            c.pop("image", None)
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--child"]:
        _model, _asset, _fx, _golden, _out, _dev = sys.argv[2:8]
        try:
            res = child(_model, Path(_asset), json.loads(_fx), Path(_golden), Path(_out), _dev)
        except Exception as exc:  # noqa: BLE001
            res = {"status": "not_evaluated", "reason": f"{type(exc).__name__}: {exc}"[:400],
                   "trace": traceback.format_exc()[-2000:]}
        _write_json(Path(_out) / "child.json", res)
        sys.exit(0)
    try:
        res = main()
    except Exception as exc:  # noqa: BLE001
        res = {"schema": SCHEMA, "status": "not_evaluated", "recorded_only": RECORDED_ONLY,
               "reason": f"{type(exc).__name__}: {exc}"[:400], "trace": traceback.format_exc()[-2000:]}
    try:
        _write_json(OUT, res)
    except Exception:  # noqa: BLE001
        pass
    line = {k: (v.get("vs_fp8_reference") or {}).get("psnr_db") for k, v in (res.get("candidates") or {}).items()}
    print(f"[day0-fp8-ref] {res.get('status')}: {res.get('reason') or line}", flush=True)
    sys.exit(0)
