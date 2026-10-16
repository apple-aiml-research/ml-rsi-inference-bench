#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Required SD3 CFG parallel verifier. Both candidate engines explicitly use
the bf16 profile. Guided/unguided parallel and serial images each use the same
request's independent, same-trial Diffusers reference and frozen four-metric
bars. Recorder branch/collective evidence and pairwise consistency remain
mandatory. Missing reference or instrument evidence never passes."""
from __future__ import annotations

import json
import os
import shutil
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
OUT = ARTIFACTS / "sd3_day0_cfg_parallel.json"
IMG_DIR = ARTIFACTS / "sd3_day0_cfg_parallel"
LAYER = "T3"
GATE = "cfg_parallel_is_real"
CAPABILITY = "cfg_parallel"
GOLDEN_SPEC = Path(os.environ.get("SD3_DAY0_GOLDEN_SPEC", HERE / "sd3_golden.json"))
FIXTURES = {"7": "astronaut", "1": "astronaut_cfg1"}
from sd3_precision_profiles import engine_kwargs

REFERENCE_PROFILE = "bf16"
CFGP_KWARGS = {"num_gpus": 2, "enable_cfg_parallel": True, **engine_kwargs(REFERENCE_PROFILE)}
SERIAL_KWARGS = {"num_gpus": 1, **engine_kwargs(REFERENCE_PROFILE)}


def _fixtures() -> dict:
    fx = {f["id"]: f for f in json.loads(GOLDEN_SPEC.read_text())["fixtures"]}
    steps = os.environ.get("SD3_CFGP_STEPS")
    size = os.environ.get("SD3_CFGP_SIZE")
    out = {}
    for g, fid in FIXTURES.items():
        f = fx[fid]
        out[g] = {"fixture": f, "req": dict(prompt=f["prompt"], seed=int(f["seed"]),
                                            num_inference_steps=int(steps or f["steps"]), guidance_scale=float(f["cfg"]),
                                            height=int(size or f["h"]), width=int(size or f["w"]))}
    return out


def _resolve_model() -> str:
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


def run_engine(work: Path, tag: str, reqs: dict, engine_kwargs: dict) -> dict:
    import numpy as np
    from _common import NoImageProduced, to_image_array

    record = work / f"record_{tag}.jsonl"
    saved = {k: os.environ.get(k) for k in ("PYTHONPATH", "INFB_SD3_COMPONENT_RECORD_FILE", "INFB_SD3_CFG_OBSERVE")}
    os.environ["PYTHONPATH"] = ":".join([str(RECORDER)] + [q for q in (saved["PYTHONPATH"] or "").split(":") if q])
    os.environ["INFB_SD3_COMPONENT_RECORD_FILE"] = str(record)
    os.environ["INFB_SD3_CFG_OBSERVE"] = "1"
    run = {"tag": tag, "engine_kwargs": engine_kwargs, "model": _resolve_model(), "readings": {}, "timing_s": {},
           "built": False}
    t0 = time.time()
    try:
        from sglang.multimodal_gen import DiffGenerator
        gen = DiffGenerator.from_pretrained(model_path=run["model"], **engine_kwargs, **_engine_ports())
        run["built"] = True
    except Exception as exc:  # noqa: BLE001
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
    try:
        for key, req in reqs.items():
            t = time.time()
            r = {"key": key, "t0": t, "raised": False, "ok": False, "guidance_scale": req["guidance_scale"],
                 "cfg_parallel_degree": 2 if engine_kwargs.get("enable_cfg_parallel") else 1}
            try:
                out = gen.generate(sampling_params_kwargs=dict(num_frames=1, return_frames=True, save_output=False,
                                                               **req))
                items = out if isinstance(out, list) else [out]
                img = None
                try:
                    img = to_image_array(items[0]) if items else None
                except NoImageProduced:
                    img = None
                r["ok"] = img is not None
                images[key] = img
                if img is not None:
                    r["shape"] = list(np.asarray(img).shape)
                    try:
                        from PIL import Image
                        IMG_DIR.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(img).save(IMG_DIR / f"{key}.png")
                    except Exception:  # noqa: BLE001
                        pass
            except Exception as exc:  # noqa: BLE001 - a failed request is a reading
                r.update(raised=True, error=f"{type(exc).__name__}: {exc}"[:600])
                images[key] = None
            r["t1"] = time.time()
            r["s"] = round(r["t1"] - t, 2)
            run["readings"][key] = r
    finally:
        try:
            gen.shutdown()
        except Exception:  # noqa: BLE001
            pass
    run["timing_s"]["requests"] = round(time.time() - t0 - run["timing_s"]["engine_build"], 1)
    run["images"] = images
    run["events"] = _events(record)
    return run


def _window(run: dict, key: str) -> list:
    import cfg_gates as G

    r = (run.get("readings") or {}).get(key) or {}
    if not r.get("t0") or not r.get("t1"):
        return []
    return G.lines_in_window(run.get("events") or [], r["t0"], r["t1"])


def _reference(fx: dict, req: dict):
    if req["height"] != int(fx["h"]) or req["num_inference_steps"] != int(fx["steps"]):
        return None, "the request is not the fixture's settings (local override)"
    import numpy as np

    for d in (os.environ.get("SD3_GOLDEN_DIR"), str(HERE)):
        if d and (Path(d) / fx["golden_npy"]).is_file():
            return np.load(Path(d) / fx["golden_npy"]), str(Path(d) / fx["golden_npy"])
    return None, "no reference file"


def judge(runs: dict, fxs: dict, references: dict | None = None) -> dict:
    import cfg_gates as G
    import reference_contract as RC

    p, s = runs.get("P") or {}, runs.get("S") or {}
    images = {"p7": (p.get("images") or {}).get("p7"), "p1": (p.get("images") or {}).get("p1"),
              "s7": (s.get("images") or {}).get("s7"), "s1": (s.get("images") or {}).get("s1")}
    checks = {}
    for role, image in images.items():
        rows = (references or {}).get("images", {}).get(role) or []
        checks[role] = RC.image_check(image, rows[0] if rows else None, fxs[role[1:]]["req"],
                                      profile=REFERENCE_PROFILE)
    extra = {"reference_checks": checks, "reference_profile": REFERENCE_PROFILE,
             "guidance_scale": {"p7": 7.0, "p1": 1.0}, "cfg_parallel_degree": 2,
             "requests": {tag: {k: {kk: vv for kk, vv in v.items() if kk != "key"}
                                for k, v in (r.get("readings") or {}).items()} for tag, r in runs.items()},
             "timing_s": {tag: r.get("timing_s") for tag, r in runs.items()},
             "engine_errors": {tag: r.get("build_error") for tag, r in runs.items() if r.get("build_error")}}
    return G.cfg_parallel_is_real(_window(p, "p7"), _window(s, "s7"), images,
                                  steps=int(fxs["7"]["req"]["num_inference_steps"]), extra=extra)


def write_gate(result: dict) -> None:
    import cfg_gates as G
    import day0_gates as DG
    from _common import GateResult, write_gate as _wg

    v = result.get("verdict") or {"verdict": "not_evaluated", "reasons": [result.get("reason", "not run")]}
    ok = v.get("verdict") == "pass"
    _wg(GateResult(gate=GATE, passed=ok, score=1.0 if ok else 0.0, is_hard_gate=False, level=LAYER,
                   error=None if ok else "; ".join(v.get("reasons", []))[:500] or v.get("verdict"),
                   details=DG.gate_details(CAPABILITY, LAYER, v, dict(G.POLICY), artifact=str(OUT),
                                           policy_state=G.POLICY_STATE, policy_basis=dict(G.POLICY_BASIS),
                                           requests=result.get("requests"), cfgp_engine_kwargs=CFGP_KWARGS,
                                           serial_engine_kwargs=SERIAL_KWARGS)))


def _gpus() -> int:
    try:
        import torch
        return int(torch.cuda.device_count())
    except Exception:  # noqa: BLE001
        return 0


def main() -> dict:


    if _gpus() < 2:
        return {"measured": False, "reason": f"not_evaluated: CFG parallel needs 2 GPUs, {_gpus()} visible"}
    fxs = _fixtures()
    work = Path(tempfile.mkdtemp(prefix="sd3_day0_cfgp_"))
    result = {"measured": False, "requests": {g: v["req"] for g, v in fxs.items()}, "work": str(work)}
    try:
        import reference_contract as RC
        references = RC.load_reference(work,
            {k: fxs[k[1:]]["req"] for k in ("p7", "p1", "s7", "s1")},
            _resolve_model(), ARTIFACTS, profile=REFERENCE_PROFILE)
        result["independent_reference"] = {k:v for k,v in references.items() if k != "images"}
        runs = {}
        for tag, kw, keys in (("P", dict(CFGP_KWARGS), ("p7", "p1")), ("S", dict(SERIAL_KWARGS), ("s7", "s1"))):
            reqs = {k: fxs[k[1:]]["req"] for k in keys}
            try:
                runs[tag] = run_engine(work, tag, reqs, kw)
            except Exception as exc:  # noqa: BLE001
                runs[tag] = {"tag": tag, "built": False, "build_error": f"{type(exc).__name__}: {exc}"[:400]}
        result.update(measured=True, model=_resolve_model(),
                      runs={tag: {k: v for k, v in r.items() if k not in ("images", "events")} for tag, r in runs.items()},
                      cfg_lines={tag: [e for e in (r.get("events") or []) if e.get("event") == "dit_cfg"][:200]
                                 for tag, r in runs.items()})
        result["verdict"] = judge(runs, fxs, references)
        if not runs["P"].get("built"):
            result["verdict"] = {"verdict": "fail", "reasons": ["the CFG-parallel engine did not build or serve: "
                                                                + str(runs["P"].get("build_error"))],
                                 "details": result["verdict"].get("details", {})}
        return result
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _engine_ports() -> dict:
    """Return this verifier queue's assigned engine port block."""
    from _common import engine_port_kwargs

    return engine_port_kwargs()

if __name__ == "__main__":
    try:
        res = main()
    except Exception as exc:  # noqa: BLE001
        res = {"measured": False, "reason": f"not_evaluated: {type(exc).__name__}: {exc}"[:400],
               "trace": traceback.format_exc()[-2000:]}
    try:
        _write_json(OUT, res)
    except Exception:  # noqa: BLE001
        pass
    write_gate(res)
    v = res.get("verdict") or {}
    print(f"[day0-cfg-parallel] {GATE}: {v.get('verdict', res.get('reason'))}; "
          + "; ".join(v.get("reasons") or [])[:600], flush=True)
    sys.exit(0)
