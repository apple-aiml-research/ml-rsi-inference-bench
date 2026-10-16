#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Replay captured component calls through diffusers and score component_alignment."""


from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
ARTIFACTS = Path(os.environ.get("INFB_ARTIFACTS_DIR", "/logs/artifacts"))
TAP_MANIFEST = ARTIFACTS / "sd3_component_tap.json"
OUT = ARTIFACTS / "sd3_component_alignment.json"


SEQ_SEGMENTS = {"clip_l": ((0, 77), (0, 768)), "clip_g": ((0, 77), (768, 2048)),
                "clip_zero_pad": ((0, 77), (2048, 4096)), "t5": ((77, None), (0, None))}
POOLED_SEGMENTS = {"pooled_clip_l": (0, 768), "pooled_clip_g": (768, 2048)}

ALIGN_TIMEOUT_S = 1800

GATE = "component_alignment"

ALIGN_COEF = float(os.environ.get("SD3_ALIGN_REL_L2_COEF", "1.5"))
LEGAL_CLIP_L_HIDDEN = float(os.environ.get("SD3_ALIGN_LEGAL_CLIP_L_HIDDEN", "0.002815689575877673"))
LEGAL_CLIP_L_POOLED = float(os.environ.get("SD3_ALIGN_LEGAL_CLIP_L_POOLED", "0.008176863242576545"))
LEGAL_CLIP_G_HIDDEN = float(os.environ.get("SD3_ALIGN_LEGAL_CLIP_G_HIDDEN", "0.01757848938436147"))
LEGAL_CLIP_G_POOLED = float(os.environ.get("SD3_ALIGN_LEGAL_CLIP_G_POOLED", "0.009419541681727173"))
LEGAL_T5_HIDDEN = float(os.environ.get("SD3_ALIGN_LEGAL_T5_HIDDEN", "0.02554605933650533"))
ORACLE_DIT = float(os.environ.get("SD3_ALIGN_ORACLE_DIT", "0.011771794107186083"))
LEGAL_VAE_DECODE = float(os.environ.get("SD3_ALIGN_LEGAL_VAE_DECODE", "0.0061937210184702645"))

ALIGN_TENSORS = (
    ("clip_l.penultimate_hidden", ("text_encoder", "penultimate_hidden"), LEGAL_CLIP_L_HIDDEN, "legal bf16"),
    ("clip_l.pooled_projected", ("text_encoder", "pooled_projected"), LEGAL_CLIP_L_POOLED, "legal bf16"),
    ("clip_g.penultimate_hidden", ("text_encoder_2", "penultimate_hidden"), LEGAL_CLIP_G_HIDDEN, "legal bf16"),
    ("clip_g.pooled_projected", ("text_encoder_2", "pooled_projected"), LEGAL_CLIP_G_POOLED, "legal bf16"),
    ("t5.last_hidden", ("text_encoder_3", "last_hidden"), LEGAL_T5_HIDDEN, "legal bf16"),
    ("dit.step1_output", ("transformer", "step1_output"), ORACLE_DIT, "oracle"),
    ("vae.decode", ("vae", "decode"), LEGAL_VAE_DECODE, "legal bf16 VAE"),
)


def _at(tree, path):
    cur = tree
    for k in path:
        cur = cur.get(k) if isinstance(cur, dict) else None
    return cur


def alignment_verdict(alignment, tensors=ALIGN_TENSORS, coef: float = ALIGN_COEF) -> dict:
    """Pure: the component_alignment verdict from the replay's JSON. Fail-closed on anything unmeasured."""
    a = alignment if isinstance(alignment, dict) else {}
    comps = a.get("components") or {}
    if not a.get("measured") or not comps:
        why = a.get("reason") or "the replay measured no component"
        return {"passed": False, "not_evaluated": True, "per": [], "failed": [],
                "reasons": [f"{why} -- fail-closed"]}
    per, failed = [], []
    for name, path, anchor, source in tensors:
        m = _at(comps, path)
        rec = {"tensor": name, "anchor": anchor, "anchor_source": source, "bar": anchor * coef}
        if isinstance(m, dict) and m.get("near_zero_reference"):
            ok = isinstance(m.get("abs_rms"), (int, float)) and m["abs_rms"] < NEAR_ZERO_RMS
            rec.update({"ok": ok, "abs_rms": m.get("abs_rms"), "max_abs": m.get("max_abs")})
            if not ok:
                rec["why"] = (f"the reference is ~0 (RMS < {NEAR_ZERO_RMS:g}), so the relative error is undefined, and "
                              f"the candidate is not: absolute RMS {m.get('abs_rms')}, max_abs {m.get('max_abs')}")
        elif not isinstance(m, dict) or not m.get("comparable") or not isinstance(m.get("rel_l2"), (int, float)):
            rec.update({"ok": False, "why": "not measured / not comparable"})
        else:
            rec.update({"rel_l2": m["rel_l2"], "ok": m["rel_l2"] <= anchor * coef,
                        "x_anchor": (m["rel_l2"] / anchor) if anchor else None})
        per.append(rec)
        if not rec["ok"]:
            failed.append(name)

    diag = {"vae_decode_psnr_db": (comps.get("vae") or {}).get("decode_psnr_db")}
    cond = comps.get("conditioning") or {}
    dup = {}
    for seg, enc in (("clip_l", ("text_encoder", "penultimate_hidden")), ("clip_g", ("text_encoder_2", "penultimate_hidden")),
                     ("t5", ("text_encoder_3", "last_hidden")), ("pooled_clip_l", ("text_encoder", "pooled_projected")),
                     ("pooled_clip_g", ("text_encoder_2", "pooled_projected"))):
        c, e = (cond.get(seg) or {}).get("rel_l2"), (_at(comps, enc) or {}).get("rel_l2")
        dup[seg] = (c == e) if (c is not None and e is not None) else None
    reasons = [f"{r['tensor']}: rel_l2 {r.get('rel_l2')} > bar {r['bar']:.6g} ({r['anchor_source']} "
               f"{r['anchor']:.6g} x {coef})" if "rel_l2" in r else f"{r['tensor']}: {r.get('why')}"
               for r in per if not r["ok"]]
    return {"passed": not failed, "not_evaluated": False, "per": per, "failed": failed, "reasons": reasons,
            "vae": diag, "conditioning_duplicates_encoder_taps": dup, "binds": "fixture 0's request only (n=1)"}


def _write_gate(alignment) -> None:
    'The scored gate, from whatever the replay wrote (or did not). Never raises.'


    try:
        from _common import GateResult, write_gate
        import profile_gates
        _PG = profile_gates

        profile = os.environ.get("SD3_PRECISION_PROFILE") or None
        v = alignment_verdict(alignment)
        r = GateResult(gate=GATE, passed=v["passed"], is_hard_gate=False, score=1.0 if v["passed"] else 0.0,
                       error=None if v["passed"] else "; ".join(v["reasons"])[:800],
                       details={**v, "coef": ALIGN_COEF, "fixture": (alignment or {}).get("fixture"),
                                "profile": profile,
                                "bars_state": ("the default configuration's bars; uncalibrated per profile"
                                               if profile else None)})
        written = _PG.write_profile_gate(r, profile, write_gate) if profile else (write_gate(r) or GATE)
        print(f"[component-alignment] {written}: passed={v['passed']} failed={v['failed']} "
              f"{'; '.join(v['reasons'])[:300]}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[component-alignment] could not write {GATE}: {type(exc).__name__}: {exc}", flush=True)


NEAR_ZERO_RMS = 1e-8


def align_metrics(a, b) -> dict:
    """Candidate `a` vs reference `b` (array-likes). Pure numpy. Shapes must match after dropping size-1 dims."""
    import numpy as np

    x = np.squeeze(np.asarray(a, dtype=np.float64))
    y = np.squeeze(np.asarray(b, dtype=np.float64))
    if x.shape != y.shape:
        return {"comparable": False, "shape_candidate": list(x.shape), "shape_reference": list(y.shape)}
    d = x - y
    nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
    if nx == 0.0 and ny == 0.0:
        cos = 1.0
    elif nx == 0.0 or ny == 0.0:
        cos = 0.0
    else:
        cos = float(np.dot(x.ravel(), y.ravel()) / (nx * ny))
    max_abs = float(np.max(np.abs(d))) if d.size else 0.0
    out = {"comparable": True, "cosine": cos, "max_abs": max_abs, "norm_candidate": nx, "norm_reference": ny,
           "shape": list(x.shape)}


    size = max(d.size, 1)
    if ny / size ** 0.5 < NEAR_ZERO_RMS:
        out.update(rel_l2=None, near_zero_reference=True, abs_rms=float(np.linalg.norm(d)) / size ** 0.5)
    else:
        out["rel_l2"] = float(np.linalg.norm(d) / ny)
    return out


def image_psnr(a, b) -> float:
    """PSNR (dB) of two decoded images in [-1, 1], compared as the uint8-range images they become."""
    import numpy as np

    def to255(v):
        return np.clip(np.asarray(v, dtype=np.float64) / 2.0 + 0.5, 0.0, 1.0) * 255.0

    x, y = np.squeeze(to255(a)), np.squeeze(to255(b))
    if x.shape != y.shape:
        return float("nan")
    mse = float(np.mean((x - y) ** 2))
    return 100.0 if mse <= 1e-12 else float(20.0 * np.log10(255.0) - 10.0 * np.log10(mse))


def segment_metrics(candidate_seq, reference_seq, candidate_pooled, reference_pooled) -> dict:
    """Split SD3's joint conditioning into its encoder segments and compare each. Pure numpy."""
    import numpy as np

    cs, rs = np.asarray(candidate_seq, dtype=np.float64), np.asarray(reference_seq, dtype=np.float64)
    cp, rp = np.asarray(candidate_pooled, dtype=np.float64), np.asarray(reference_pooled, dtype=np.float64)
    out = {"seq_shape_candidate": list(cs.shape), "seq_shape_reference": list(rs.shape),
           "pooled_shape_candidate": list(cp.shape), "pooled_shape_reference": list(rp.shape)}
    if cs.ndim != 3 or rs.ndim != 3 or cs.shape[1:] != rs.shape[1:]:
        out["comparable"] = False
        return out
    for name, ((t0, t1), (c0, c1)) in SEQ_SEGMENTS.items():
        out[name] = align_metrics(cs[:1, t0:t1, c0:c1], rs[:1, t0:t1, c0:c1])
    if cp.shape[-1] == rp.shape[-1] == 2048:
        for name, (c0, c1) in POOLED_SEGMENTS.items():
            out[name] = align_metrics(cp[:1, c0:c1], rp[:1, c0:c1])
    out["comparable"] = True
    return out


def _resolve_model() -> str:
    """The same weight routes as verify_callsite_main._resolve_model."""
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


def _write(out: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")


def dit_replay_tensors(capture, reference_config, fixture, vae_scale_factor):
    """Map checkpoint patch tokens to spatial tensors for independent replay.

    The grid comes from the verifier request and independent VAE config, never
    a candidate's latent_hw. SD3 patch coordinates are (ph, pw, channel), the
    same order used by Diffusers' output projection. This is an exact layout
    permutation; it neither rescales values nor chooses a layout by score.
    """
    import torch

    p = int(reference_config.patch_size)
    channels = int(reference_config.in_channels)
    out_channels = int(reference_config.out_channels or channels)
    scale = vae_scale_factor
    h, w = fixture.get("h"), fixture.get("w")
    if (not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in (h, w, scale))
            or p < 1 or h % scale or w % scale or (h // scale) % p or (w // scale) % p):
        raise ValueError("DiT replay: incompatible verifier fixture / reference patch grid")
    height, width = h // scale, w // scale

    def spatial(value, count, label):
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise ValueError(f"DiT replay {label}: missing floating tensor")
        shape = tuple(value.shape)
        if value.ndim == 3:
            if shape[0] < 1 or shape[1:] != ((height // p) * (width // p), p * p * count):
                raise ValueError(f"DiT replay {label}: incompatible packed shape {shape}")
            value = value.reshape(shape[0], height // p, width // p, p, p, count)
            value = value.permute(0, 5, 1, 3, 2, 4).reshape(shape[0], count, height, width)
            return value, "packed"
        layout = "spatial"
        if value.ndim == 5 and shape[2] == 1:
            value, layout = value[:, :, 0], "single_frame"
        if value.ndim != 4 or value.shape[0] < 1 or tuple(value.shape[1:]) != (count, height, width):
            raise ValueError(f"DiT replay {label}: incompatible spatial shape {shape}")
        return value, layout

    x, x_layout = spatial(capture.get("hidden_states"), channels, "input")
    output, y_layout = spatial(capture.get("output"), out_channels, "output")
    if x.shape[0] != output.shape[0]:
        raise ValueError("DiT replay: input/output batch mismatch")
    inputs = {name: capture[name] for name in ("encoder_hidden_states", "pooled_projections", "timestep")}
    for name in ("encoder_hidden_states", "pooled_projections"):
        value = inputs[name]
        if not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[0] != x.shape[0]:
            raise ValueError(f"DiT replay: incompatible {name} batch")
    step = inputs["timestep"]
    if not isinstance(step, torch.Tensor) or step.ndim > 1 or step.numel() not in (1, x.shape[0]):
        raise ValueError("DiT replay: incompatible timestep batch")
    inputs["timestep"] = step.reshape(-1)
    inputs["hidden_states"] = x
    return inputs, output, {"input_layout": x_layout, "output_layout": y_layout,
        "input_shape": list(capture["hidden_states"].shape), "output_shape": list(capture["output"].shape),
        "spatial_shape": list(x.shape), "patch_size": p, "patch_order": "ph_pw_channel",
        "grid_source": "verifier_fixture_and_reference_vae"}


def _child() -> int:
    """Runs in the isolated process: diffusers + torch only, never the candidate's tree."""
    import numpy as np
    import torch
    from projected_pooling import projected_pooled

    import diffusers

    ws = os.path.abspath(os.environ.get("INFB_WORKSPACE", "/app"))
    loc = os.path.dirname(os.path.abspath(diffusers.__file__))
    if loc.startswith(ws):
        _write({"measured": False, "scored": True, "reason": f"diffusers resolved INSIDE the workspace ({loc})"})
        return 0
    from diffusers import AutoencoderKL, SD3Transformer2DModel, StableDiffusion3Pipeline

    manifest = json.loads(TAP_MANIFEST.read_text(encoding="utf-8"))
    tap, model = Path(manifest["tap_dir"]), os.environ["SD3_MODEL"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float32
    res = {"measured": False, "scored": True, "diffusers": diffusers.__version__, "diffusers_file": loc,
           "reference_dtype": "float32", "device": dev, "fixture": manifest.get("fixture"), "components": {},
           "missing": [], "errors": {}}

    def load(name):
        p = tap / f"{name}.pt"
        return torch.load(p, map_location="cpu", weights_only=True) if p.is_file() else None

    def npy(t):
        return t.detach().to("cpu", torch.float64).numpy()

    def frame(t):
        return t[:, :, 0] if t.dim() == 5 and t.shape[2] == 1 else t

    with torch.no_grad():
        try:
            pipe = StableDiffusion3Pipeline.from_pretrained(model, transformer=None, vae=None, torch_dtype=dt).to(dev)
            encoders = {"text_encoder": pipe.text_encoder, "text_encoder_2": pipe.text_encoder_2,
                        "text_encoder_3": pipe.text_encoder_3}
            for name, enc in encoders.items():
                cap = load(name)
                if cap is None or cap.get("input_ids") is None:
                    res["missing"].append(name)
                    continue
                o = enc(cap["input_ids"].to(dev), output_hidden_states=True)
                m = {"input_ids_shape": list(cap["input_ids"].shape), "captured": sorted(k for k in cap)}
                if name == "text_encoder_3":
                    if "last_hidden_state" in cap:
                        m["last_hidden"] = align_metrics(npy(cap["last_hidden_state"]), npy(o.last_hidden_state))
                else:
                    if "hidden_states_m2" in cap:
                        m["penultimate_hidden"] = align_metrics(npy(cap["hidden_states_m2"]), npy(o.hidden_states[-2]))
                    ref_pooled = getattr(o, "text_embeds", None)
                    captured_pooled = projected_pooled(cap)
                    if captured_pooled is not None and ref_pooled is not None:
                        m["pooled_projected"] = align_metrics(npy(captured_pooled), npy(ref_pooled))
                res["components"][name] = m
            dit_cap = load("transformer")
            prompt = (manifest.get("fixture") or {}).get("prompt")
            if dit_cap is not None and prompt:
                pe, _, pooled, _ = pipe.encode_prompt(prompt=prompt, prompt_2=None, prompt_3=None, device=dev,
                                                      num_images_per_prompt=1, do_classifier_free_guidance=False,
                                                      max_sequence_length=256)
                ehs, pp = dit_cap.get("encoder_hidden_states"), dit_cap.get("pooled_projections")
                if hasattr(ehs, "shape") and hasattr(pp, "shape"):
                    res["components"]["conditioning"] = segment_metrics(npy(ehs), npy(pe), npy(pp), npy(pooled))
                else:
                    res["missing"].append("conditioning (DiT inputs encoder_hidden_states / pooled_projections)")
            del pipe
            if dev == "cuda":
                torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            res["errors"]["encoders"] = f"{type(exc).__name__}: {exc}"[:400]
        try:
            dit_cap = load("transformer")
            if dit_cap is None:
                res["missing"].append("transformer")
            else:
                need = ("hidden_states", "encoder_hidden_states", "pooled_projections", "timestep", "output")
                absent = [k for k in need if not hasattr(dit_cap.get(k), "shape")]
                if absent:
                    res["missing"].append(f"transformer inputs {absent}")
                else:
                    dit = SD3Transformer2DModel.from_pretrained(model, subfolder="transformer", torch_dtype=dt).to(dev)
                    vae_config = AutoencoderKL.load_config(model, subfolder="vae")
                    scale = 2 ** (len(vae_config["block_out_channels"]) - 1)
                    inputs, observed, layout = dit_replay_tensors(dit_cap, dit.config, manifest["fixture"], scale)
                    inputs = {name: value.to(dev) if name == "timestep" else value.to(dev, dt)
                              for name, value in inputs.items()}
                    ref = dit(**inputs, return_dict=False)[0]
                    res["components"]["transformer"] = {
                        "step1_output": align_metrics(npy(observed), npy(ref)), "layout": layout,
                        "timestep": npy(inputs["timestep"]).ravel().tolist()[:4]}
                    del dit
                    if dev == "cuda":
                        torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            res["errors"]["transformer"] = f"{type(exc).__name__}: {exc}"[:400]
        try:
            vae_cap = load("vae_decode")
            if vae_cap is None or not hasattr(vae_cap.get("z"), "shape"):
                res["missing"].append("vae_decode")
            else:
                vae = AutoencoderKL.from_pretrained(model, subfolder="vae", torch_dtype=dt).to(dev)
                ref = vae.decode(frame(vae_cap["z"]).to(dev, dt), return_dict=False)[0]
                cand = frame(vae_cap["sample"])
                res["components"]["vae"] = {"decode": align_metrics(npy(cand), npy(ref)),
                                            "decode_psnr_db": image_psnr(npy(cand), npy(ref))}
        except Exception as exc:  # noqa: BLE001
            res["errors"]["vae"] = f"{type(exc).__name__}: {exc}"[:400]
    res["measured"] = bool(res["components"])
    _write(res)
    print("[component-alignment] replay: measured=%s components=%s missing=%s errors=%s"
          % (res["measured"], sorted(res["components"]), res["missing"], sorted(res["errors"])), flush=True)
    return 0


def main() -> int:
    if "--child" in sys.argv[1:]:
        return _child()
    if not TAP_MANIFEST.is_file():
        _write({"measured": False, "scored": True, "reason": f"no tap manifest at {TAP_MANIFEST}"})
        print("[component-alignment] no tap manifest -- nothing to replay", flush=True)
        return 0
    try:
        manifest = json.loads(TAP_MANIFEST.read_text(encoding="utf-8"))
    except ValueError as exc:
        _write({"measured": False, "scored": True, "reason": f"unreadable tap manifest: {exc}"})
        return 0
    if not manifest.get("measured"):
        _write({"measured": False, "scored": True,
                "reason": f"the tap request captured nothing ({manifest.get('reason') or manifest.get('error')})"})
        print("[component-alignment] the tap request captured nothing", flush=True)
        return 0
    repo = str(Path(os.environ.get("INFB_WORKSPACE", "/app")).resolve() / "repo")
    env = dict(os.environ)
    env["PYTHONPATH"] = ":".join(q for q in (env.get("PYTHONPATH") or "").split(":") if q and not q.startswith(repo))
    env["SD3_MODEL"] = _resolve_model()
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "--child"], env=env, capture_output=True, text=True,
                       timeout=ALIGN_TIMEOUT_S)
    sys.stdout.write(r.stdout[-4000:])
    if r.returncode != 0 or not OUT.is_file():
        _write({"measured": False, "scored": True, "reason": f"replay child exited {r.returncode}",
                "stderr_tail": r.stderr[-3000:]})
    return 0


def _read_out():
    try:
        return json.loads(OUT.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


if __name__ == "__main__":
    if "--child" in sys.argv[1:]:
        try:
            main()
        except (Exception, SystemExit) as exc:  # noqa: BLE001 - the parent reads whatever the child left
            _write({"measured": False, "scored": True, "reason": f"{type(exc).__name__}: {exc}"[:400]})
        sys.exit(0)
    try:
        main()
    except (Exception, SystemExit) as exc:  # noqa: BLE001 - the gate below still gets written, fail-closed
        try:
            _write({"measured": False, "scored": True, "reason": f"{type(exc).__name__}: {exc}"[:400]})
        except Exception:  # noqa: BLE001
            pass
    _write_gate(_read_out())
    sys.exit(0)
