#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
'SD3 verify vae tiled decode utilities.'


from __future__ import annotations

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
RECORDER = HERE / "component_recorder"
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(DAY0))

ARTIFACTS = Path(os.environ.get("INFB_ARTIFACTS_DIR", "/logs/artifacts"))
OUT = ARTIFACTS / "sd3_day0_vae_tiled.json"
IMG_DIR = ARTIFACTS / "sd3_day0_vae_tiled"
LAYER = "T2"
GATE = "vae_tiled_decode_is_real"
CAPABILITY = "vae_tiled_decode"
GOLDEN_SPEC = Path(os.environ.get("SD3_DAY0_GOLDEN_SPEC", HERE / "sd3_golden.json"))
SIZE = int(os.environ.get("SD3_VAE_TILED_SIZE", "1152"))
STEPS = int(os.environ.get("SD3_VAE_TILED_STEPS", "12"))
CFG = 7.0
FIXTURE = os.environ.get("SD3_VAE_TILED_FIXTURE", "mountain")
REPLAY_TIMEOUT_S = 600.0
UNTILED_KWARGS = {"vae_tiling": False, "vae_sp": False}


def _request() -> dict:
    fx = {f["id"]: f for f in json.loads(GOLDEN_SPEC.read_text())["fixtures"]}
    f = fx.get(FIXTURE) or next(iter(fx.values()))
    return dict(prompt=f["prompt"], seed=int(f["seed"]), num_inference_steps=STEPS, guidance_scale=CFG,
                height=SIZE, width=SIZE)


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


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")


def _events(record: Path) -> list:
    out = []
    if record.is_file():
        for ln in record.read_text(errors="replace").splitlines():
            try:
                rec = json.loads(ln)
            except ValueError:
                continue
            if isinstance(rec, dict):
                out.append(rec)
    return out


def run_engine(work: Path, tag: str, req: dict, keys: list, engine_kwargs: dict) -> dict:
    """Build one engine with the recorder (its VAE tap armed for the whole run), serve `req` once per key, return
    every reading (images kept in memory, PNGs saved for eyeballing)."""
    import numpy as np
    from _common import NoImageProduced, to_image_array

    record, tap = work / f"record_{tag}.jsonl", work / f"tap_{tag}"
    tap.mkdir(parents=True, exist_ok=True)
    (tap / "VAE_ARM").write_text(f"armed by verify_vae_tiled_decode ({tag})\n", encoding="utf-8")
    saved = {k: os.environ.get(k) for k in ("PYTHONPATH", "INFB_SD3_COMPONENT_RECORD_FILE",
                                             "INFB_SD3_COMPONENT_TAP_DIR")}
    os.environ["PYTHONPATH"] = ":".join([str(RECORDER)] + [q for q in (saved["PYTHONPATH"] or "").split(":") if q])
    os.environ["INFB_SD3_COMPONENT_RECORD_FILE"] = str(record)
    os.environ["INFB_SD3_COMPONENT_TAP_DIR"] = str(tap)
    run = {"tag": tag, "engine_kwargs": engine_kwargs, "model": _resolve_model(), "readings": {}, "timing_s": {},
           "record": str(record), "tap_dir": str(tap), "built": False}
    t0 = time.time()
    try:
        from sglang.multimodal_gen import DiffGenerator
        gen = DiffGenerator.from_pretrained(model_path=run["model"], num_gpus=1, **engine_kwargs, **_engine_ports())
        run["built"] = True
    except Exception as exc:  # noqa: BLE001 - an engine that does not build is a reading of this run
        run["build_error"] = f"{type(exc).__name__}: {exc}"[:600]
        run["trace"] = traceback.format_exc()[-2000:]
        return run
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        run["timing_s"]["engine_build"] = round(time.time() - t0, 1)
    images: dict = {}

    def ask(key: str, **kw) -> dict:
        t = time.time()
        r = {"key": key, "t0": t, "raised": False, "none": False, "ok": False, "n": 0}
        try:
            out = gen.generate(sampling_params_kwargs=dict(num_frames=1, return_frames=True, save_output=False, **kw))
            items = out if isinstance(out, list) else [out]
            imgs = []
            for o in items:
                try:
                    imgs.append(to_image_array(o))
                except NoImageProduced:
                    imgs.append(None)
            r.update(n=len(items), ok=all(i is not None for i in imgs) and bool(imgs), returned=type(out).__name__)
            r["none"] = out is None or not any(i is not None for i in imgs)
            images[key] = imgs
            for j, img in enumerate(imgs):
                if img is not None:
                    r["shape"] = list(np.asarray(img).shape)
                    try:
                        from PIL import Image
                        IMG_DIR.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(img).save(IMG_DIR / f"{key}.{j}.png")
                    except Exception:  # noqa: BLE001 - eyeballing only
                        pass
        except Exception as exc:  # noqa: BLE001 - a failed request is a reading
            r.update(raised=True, type=type(exc).__name__, error=f"{type(exc).__name__}: {exc}"[:600])
            images[key] = []
        r["t1"] = time.time()
        r["s"] = round(r["t1"] - t, 2)
        run["readings"][key] = r
        return r

    try:
        for key in keys:
            ask(key, **req)
    finally:
        try:
            gen.shutdown()
        except Exception:  # noqa: BLE001
            pass
    run["timing_s"]["requests"] = round(time.time() - t0 - run["timing_s"]["engine_build"], 1)
    run["images"] = images
    run["events"] = _events(record)
    run["vae_lines"] = [e for e in run["events"] if e.get("event") == "vae_decode"]
    run["installed"] = [e for e in run["events"] if e.get("event") == "vae_tiling_installed"]
    run["taps"] = sorted(p.name for p in tap.glob("vae_decode_*.pt"))
    return run


def _line_for(run: dict, key: str) -> dict | None:
    """The `vae_decode` line stamped inside the request's window (the first, if a request decoded more than once)."""
    import vae_tiled_gates as G

    r = (run.get("readings") or {}).get(key) or {}
    if not r.get("t0") or not r.get("t1"):
        return None
    lines = G.lines_in_window(run.get("events") or [], r["t0"], r["t1"])
    return lines[0] if lines else None


def _tap_for(run: dict, key: str) -> str | None:
    """The tap file of the request's first decode: the recorder numbers taps by decode (n), in order."""
    line = _line_for(run, key)
    if not line or line.get("n") is None:
        return None
    p = Path(run["tap_dir"]) / f"vae_decode_{int(line['n'])}.pt"
    return str(p) if p.is_file() else None


def _child() -> int:
    """Runs in the isolated process: diffusers + torch only, never the candidate's tree. Reads the manifest
    (INFB_VAE_TILED_MANIFEST), writes the readings JSON next to it."""
    import hashlib

    import numpy as np
    import torch

    man_path = Path(os.environ["INFB_VAE_TILED_MANIFEST"])
    man = json.loads(man_path.read_text())
    out_path = Path(man["out"])
    res: dict = {"measured": False, "device": None, "psnr": {}, "seam_jump": {}, "shapes": {}, "errors": {}}
    try:
        import diffusers
        from diffusers import AutoencoderKL

        from _common import to_image_array
        import vae_tiled_gates as G

        ws = os.path.abspath(os.environ.get("INFB_WORKSPACE", "/app"))
        loc = os.path.dirname(os.path.abspath(diffusers.__file__))
        res.update(diffusers=diffusers.__version__, diffusers_file=loc)
        if loc.startswith(ws + os.sep):
            res["errors"]["isolation"] = f"diffusers resolved INSIDE the workspace ({loc})"
            _write_json(out_path, res)
            return 0
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        res["device"] = dev

        def load(p):
            if not p:
                return None
            d = torch.load(p, map_location="cpu", weights_only=True)
            return d if isinstance(d, dict) else None

        def frame(t):
            return t[:, :, 0] if t is not None and t.dim() == 5 and t.shape[2] == 1 else t

        def u8(t):
            """A raw VAE sample (about [-1, 1], unclamped) -> uint8 HWC exactly as the engine's image reaches the
            verifier: the decoding stage's `(x / 2 + 0.5).clamp(0, 1)`, then the verifier's single coercion
            (_common.to_image_array, which scales [0, 1] by 255 and truncates). The raw sample must NOT go into
            to_image_array directly: a max above 1 + 1e-3 makes it skip the x255 and every image reads near-constant."""
            if t is None:
                return None
            x = (frame(t).detach().to("cpu", torch.float32) / 2 + 0.5).clamp(0, 1)
            return to_image_array(x)

        def seam_jump(img, line, axis):
            if img is None:
                return None
            a = np.asarray(img).astype(np.float64)
            if axis == 0:
                return float(np.abs(a[line] - a[line - 1]).mean()) if 0 < line < a.shape[0] else None
            return float(np.abs(a[:, line] - a[:, line - 1]).mean()) if 0 < line < a.shape[1] else None

        A1, A2, B1 = load(man.get("tiled")), load(man.get("tiled_repeat")), load(man.get("untiled"))
        if A1 is None or not hasattr(A1.get("z"), "shape"):
            res["errors"]["tiled"] = "no captured latent of the tiled request"
            _write_json(out_path, res)
            return 0
        z = frame(A1["z"]).to(torch.float32)
        res["shapes"]["z"] = list(z.shape)


        res["z_sha256"] = hashlib.sha256(z.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        e_t = u8(A1.get("sample"))
        e_t2 = u8(A2.get("sample")) if A2 else None
        same_latent = None
        e_u = None
        if B1 is not None and hasattr(B1.get("z"), "shape"):
            zb = frame(B1["z"]).to(torch.float32)
            same_latent = bool(zb.shape == z.shape and torch.equal(zb, z))
            res["untiled_latent_max_abs_diff"] = float((zb - z).abs().max()) if zb.shape == z.shape else None
            e_u = u8(B1.get("sample"))
        res["same_latent"] = same_latent
        rec_t = man.get("recorded_tiling") or {}
        with torch.no_grad():
            from vae_replay_contract import decode_reference
            raw_u, raw_default, raw_recorded, replay_meta = decode_reference(
                man["model"], dev, frame(A1["z"]), rec_t, AutoencoderKL.from_pretrained,
                {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16})
            defaults = replay_meta["reference_tiles"]
            at = replay_meta["recorded_tiling_replayed"]
            res["reference_tiles"] = defaults
            res["observed_config_checked"] = replay_meta["observed_config_checked"]
            d_u, d_t, d_ts = u8(raw_u), u8(raw_default), u8(raw_recorded)
        res["shapes"].update({k: (None if v is None else list(v.shape)) for k, v in
                              {"engine_tiled": e_t, "engine_tiled_repeat": e_t2, "engine_untiled": e_u,
                               "ref_untiled": d_u, "ref_tiled": d_t, "ref_tiled_at_recorded": d_ts}.items()})
        p = res["psnr"]
        p["replay_dtype"] = replay_meta["replay_dtype"]
        if same_latent and e_u is not None:
            u, p["untiled_source"] = e_u, "engine"
        else:
            u, p["untiled_source"] = d_u, "reference"
            p["untiled_missing"] = ("the untiled engine's latent differs from the tiled engine's (bitwise)"
                                    if same_latent is False else "no untiled engine capture")
        p["tiled_vs_untiled"] = G.psnr(e_t, u)
        p["seam_excess"] = G.seam_excess(e_t, u)
        p["ref_tiled_vs_ref_untiled"] = G.psnr(d_ts, d_u)
        p["ref_default_tiled_vs_ref_untiled"] = G.psnr(d_t, d_u)
        p["tiled_vs_ref_tiled_at_recorded"] = G.psnr(e_t, d_ts) if d_ts is not None else None
        p["recorded_tiling_replayed"] = at

        p["tiled_vs_ref_tiled_default"] = G.psnr(e_t, d_t)
        p["tiled_vs_ref_untiled"] = G.psnr(e_t, d_u)
        p["untiled_vs_ref_untiled"] = G.psnr(e_u, d_u) if e_u is not None else None
        p["engine_tiled_vs_engine_untiled_any_latent"] = G.psnr(e_t, e_u) if e_u is not None else None
        p["tiled_repeat_vs_tiled"] = G.psnr(e_t2, e_t)
        p["seam_excess_ref_tiled_vs_ref_untiled"] = G.seam_excess(d_ts, d_u)
        tl, ts = at["tile_latent"], at["tile_sample"]
        ov = at["overlap"]
        lines = sorted({int(ts - int(ts * ov)), int(ts)})
        for tag, img in (("engine_tiled", e_t), ("engine_untiled", e_u), ("ref_tiled", d_t), ("ref_untiled", d_u)):
            if img is not None:
                res["seam_jump"][tag] = {f"{ax}{ln}": seam_jump(img, ln, 0 if ax == "row" else 1)
                                         for ax in ("row", "col") for ln in lines}
        res["measured"] = p.get("ref_tiled_vs_ref_untiled") is not None
    except Exception as exc:  # noqa: BLE001
        res["errors"]["replay"] = f"{type(exc).__name__}: {exc}"[:400]
        res["trace"] = traceback.format_exc()[-2000:]
    _write_json(out_path, res)
    return 0


def replay(work: Path, runs: dict) -> dict:
    """Spawn the isolated child on the captured taps; return its readings (or why there are none)."""
    a, b = runs.get("A") or {}, runs.get("B") or {}
    t1 = _line_for(a, "t1") or {}
    man = {"model": _resolve_model(), "tiled": _tap_for(a, "t1"), "tiled_repeat": _tap_for(a, "t2"),
           "untiled": _tap_for(b, "u1"), "out": str(work / "replay.json"),
           "recorded_tiling": {k: t1.get(k) for k in ("overlap", "tile_latent", "tile_sample", "in_dtype", "vae_config")}}
    man_path = work / "replay_manifest.json"
    _write_json(man_path, man)
    if not man["tiled"]:
        return {"measured": False, "manifest": man, "reason": "the tiled engine's decode was not captured "
                                                               "(no vae_decode_<n>.pt for request t1)"}
    repo = str(Path(os.environ.get("INFB_WORKSPACE", "/app")).resolve() / "repo")
    env = dict(os.environ)
    env["PYTHONPATH"] = ":".join(q for q in (env.get("PYTHONPATH") or "").split(":") if q and not q.startswith(repo))
    env["INFB_VAE_TILED_MANIFEST"] = str(man_path)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    t0 = time.time()
    try:
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child"], env=env, cwd=str(work),
                              capture_output=True, text=True, timeout=REPLAY_TIMEOUT_S)
        tail = {"rc": proc.returncode, "stderr": proc.stderr[-1500:], "stdout": proc.stdout[-500:]}
    except subprocess.TimeoutExpired:
        return {"measured": False, "manifest": man, "reason": f"the replay exceeded {REPLAY_TIMEOUT_S:.0f} s"}
    out = Path(man["out"])
    if not out.is_file():
        return {"measured": False, "manifest": man, "reason": "the replay wrote no readings", "proc": tail}
    res = json.loads(out.read_text())
    res.update(manifest=man, proc=tail, s=round(time.time() - t0, 1))
    return res


def judge(runs: dict, rep: dict) -> dict:
    import vae_tiled_gates as G

    a, b = runs.get("A") or {}, runs.get("B") or {}
    t1, t2, u1 = _line_for(a, "t1"), _line_for(a, "t2"), _line_for(b, "u1")
    p = dict((rep or {}).get("psnr") or {})
    if not (rep or {}).get("measured"):
        p.setdefault("reference_missing", (rep or {}).get("reason") or "; ".join(
            f"{k}: {v}" for k, v in ((rep or {}).get("errors") or {}).items()) or "the replay did not run")
    extra = {"untiled_engine_line": u1, "repeat_line": t2,
             "repeat_recorded_same": (bool(t2) and bool(t1) and t2.get("tiled") == t1.get("tiled")
                                      and t2.get("tiles") == t1.get("tiles")) if t1 and t2 else None,
             "peak_mem_bytes": {"tiled": (t1 or {}).get("peak_mem_bytes"), "untiled": (u1 or {}).get("peak_mem_bytes"),
                                "ratio_tiled_over_untiled": G.peak_ratio(t1, u1),
                                "delta_tiled": (t1 or {}).get("peak_delta_bytes"),
                                "delta_untiled": (u1 or {}).get("peak_delta_bytes"),
                                "delta_ratio_tiled_over_untiled": G.peak_delta_ratio(t1, u1),
                                "notes": {"tiled": (t1 or {}).get("peak_note"), "untiled": (u1 or {}).get("peak_note")}},
             "probe_passes": (t1 or {}).get("probe_passes"), "z_sha256": (rep or {}).get("z_sha256"),
             "same_latent": (rep or {}).get("same_latent"), "seam_jump": (rep or {}).get("seam_jump"),
             "reference": {k: (rep or {}).get(k) for k in ("diffusers", "device", "reference_tiles")},
             "requests": {tag: {k: {kk: vv for kk, vv in v.items() if kk != "key"}
                                for k, v in (r.get("readings") or {}).items()}
                          for tag, r in runs.items()},
             "engine_errors": {tag: r.get("build_error") for tag, r in runs.items() if r.get("build_error")}}
    return G.vae_tiled_decode_is_real(t1, p, extra=extra)


def write_gate(result: dict) -> None:
    import day0_gates as DG
    import vae_tiled_gates as G
    from _common import GateResult, write_gate as _wg

    v = result.get("verdict") or {"verdict": "not_evaluated", "reasons": [result.get("reason", "not run")]}
    ok = v.get("verdict") == "pass"
    _wg(GateResult(gate=GATE, passed=ok, score=1.0 if ok else 0.0, is_hard_gate=False, level=LAYER,
                   error=None if ok else "; ".join(v.get("reasons", []))[:500] or v.get("verdict"),
                   details=DG.gate_details(CAPABILITY, LAYER, v, dict(G.POLICY), artifact=str(OUT),
                                           policy_state=G.POLICY_STATE, policy_basis=dict(G.POLICY_BASIS),
                                           request={"size": SIZE, "steps": STEPS, "cfg": CFG, "fixture": FIXTURE},
                                           untiled_engine_kwargs=UNTILED_KWARGS)))


def main() -> dict:


    req = _request()
    work = Path(tempfile.mkdtemp(prefix="sd3_day0_vae_tiled_"))
    result = {"measured": False, "request": req, "work": str(work)}
    try:
        runs = {}
        for tag, keys, kw in (("A", ["t1", "t2"], {}), ("B", ["u1"], dict(UNTILED_KWARGS))):
            try:
                runs[tag] = run_engine(work, tag, req, keys, kw)
            except Exception as exc:  # noqa: BLE001 - one engine's failure is a reading of this run
                from _common import save_stderr_log
                save_stderr_log(f"day0_vae_tiled_engine_{tag}", traceback.format_exc())
                runs[tag] = {"tag": tag, "built": False, "build_error": f"{type(exc).__name__}: {exc}"[:400]}
        rep = replay(work, runs)
        result.update(measured=True, model=_resolve_model(), replay=rep,
                      runs={tag: {k: v for k, v in r.items() if k not in ("images", "events")}
                            for tag, r in runs.items()},
                      vae_lines={tag: r.get("vae_lines") for tag, r in runs.items()})
        result["verdict"] = judge(runs, rep)
        if not runs["A"].get("built"):
            result["verdict"] = {"verdict": "fail", "reasons": ["the engine did not build or serve: "
                                                                + str(runs["A"].get("build_error"))],
                                 "details": result["verdict"].get("details", {})}
        return result
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _engine_ports() -> dict:
    """Return this verifier queue's assigned engine port block."""
    from _common import engine_port_kwargs

    return engine_port_kwargs()

if __name__ == "__main__":
    if "--child" in sys.argv[1:]:
        sys.exit(_child())
    try:
        res = main()
    except Exception as exc:  # noqa: BLE001 - the gate below is still written
        res = {"measured": False, "reason": f"not_evaluated: {type(exc).__name__}: {exc}"[:400],
               "trace": traceback.format_exc()[-2000:]}
    try:
        _write_json(OUT, res)
    except Exception:  # noqa: BLE001
        pass
    write_gate(res)
    v = res.get("verdict") or {}
    print(f"[day0-vae-tiled] {GATE}: {v.get('verdict', res.get('reason'))}; "
          + "; ".join(v.get("reasons") or [])[:600], flush=True)
    sys.exit(0)
