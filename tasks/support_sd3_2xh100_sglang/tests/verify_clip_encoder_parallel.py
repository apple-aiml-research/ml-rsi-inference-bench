#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
'SD3 verify clip encoder parallel utilities.'
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
OUT = ARTIFACTS / "sd3_day0_clip_parallel.json"
LAYER = "T3"
GATE = "clip_encoder_parallel_is_real"
CAPABILITY = "clip_encoder_parallel"
GOLDEN_SPEC = Path(os.environ.get("SD3_DAY0_GOLDEN_SPEC", HERE / "sd3_golden.json"))
FIXTURE = "astronaut"
SP_KWARGS = {"num_gpus": 2, "sp_degree": 2, "ulysses_degree": 2, "tp_size": 1, "backend": "sglang", "dist_timeout": 300}
ENCODERS = ("text_encoder", "text_encoder_2")
CHILD_TIMEOUT_S = 600
CHILD_DEVICE = os.environ.get("SD3_CLIPP_CHILD_DEVICE", "auto")


def _request() -> dict:
    fx = {f["id"]: f for f in json.loads(GOLDEN_SPEC.read_text())["fixtures"]}[FIXTURE]
    size = int(os.environ.get("SD3_CLIPP_SIZE") or 512)
    return dict(prompt=fx["prompt"], seed=int(fx["seed"]), num_inference_steps=int(os.environ.get("SD3_CLIPP_STEPS") or 2),
                guidance_scale=float(fx["cfg"]), height=size, width=size)


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


def run_engine(work: Path, req: dict, engine_kwargs: dict) -> dict:
    record, cap = work / "record_sp2.jsonl", work / "cap"
    cap.mkdir(parents=True, exist_ok=True)
    keys = ("PYTHONPATH", "INFB_SD3_COMPONENT_RECORD_FILE", "INFB_SD3_CLIP_OBSERVE", "INFB_SD3_CLIP_OBSERVE_DIR")
    saved = {k: os.environ.get(k) for k in keys}
    os.environ["PYTHONPATH"] = ":".join([str(RECORDER)] + [q for q in (saved["PYTHONPATH"] or "").split(":") if q])
    os.environ["INFB_SD3_COMPONENT_RECORD_FILE"] = str(record)
    os.environ["INFB_SD3_CLIP_OBSERVE"] = "1"
    os.environ["INFB_SD3_CLIP_OBSERVE_DIR"] = str(cap)
    run = {"engine_kwargs": engine_kwargs, "model": _resolve_model(), "request": req, "timing_s": {}, "built": False}
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
    try:
        t = time.time()
        try:
            out = gen.generate(sampling_params_kwargs=dict(num_frames=1, return_frames=True, save_output=False, **req))
            run["served"] = bool(out if not isinstance(out, list) else len(out))
        except Exception as exc:  # noqa: BLE001 - a failed request is a reading
            run["request_error"] = f"{type(exc).__name__}: {exc}"[:600]
        run["timing_s"]["request"] = round(time.time() - t, 1)
    finally:
        try:
            gen.shutdown()
        except Exception:  # noqa: BLE001
            pass
    run["events"] = _events(record)
    run["cap_dir"] = str(cap)
    return run


def child_env() -> dict:
    """Isolate the child from Python startup overrides and recorders; keep the hub offline."""
    env = dict(os.environ)
    for k in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE", "PYTHONSTARTUP", "INFB_SD3_COMPONENT_RECORD_FILE",
              "INFB_SD3_COMPONENT_TAP_DIR", "INFB_SD3_CLIP_OBSERVE", "INFB_SD3_CLIP_OBSERVE_DIR"):
        env.pop(k, None)
    env["HF_HUB_OFFLINE"] = "1"
    return env


def run_reference(model: str, cap: Path, out: Path) -> dict:
    """The isolated child (this file, --child): its JSON, or the reason it has none."""
    model_p, cap, out = (Path(p).resolve() for p in (model, cap, out))
    cdir = out.parent / "child"
    cdir.mkdir(parents=True, exist_ok=True)
    try:
        r = subprocess.run([sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--child", str(model_p),
                            str(cap), str(out), CHILD_DEVICE, os.path.abspath(os.environ.get("INFB_WORKSPACE", "/app"))],
                           cwd=str(cdir), env=child_env(), capture_output=True, text=True, timeout=CHILD_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return {"error": f"the reference child ran past {CHILD_TIMEOUT_S} s"}
    if r.returncode != 0 or not out.is_file():
        return {"error": f"the reference child exited {r.returncode}", "stderr_tail": r.stderr[-2000:]}
    try:
        return json.loads(out.read_text())
    except ValueError as exc:
        return {"error": f"unreadable reference: {exc}"}


def _child(model: str, cap: str, out: str, device: str, workspace: str) -> int:
    """Runs in the isolated process: torch + transformers from the verifier's own environment, never the candidate."""
    import torch
    from projected_pooling import projected_pooled
    import transformers

    res: dict = {"transformers": transformers.__version__, "transformers_file": transformers.__file__,
                 "dtype": "float32"}
    loc = os.path.dirname(os.path.abspath(transformers.__file__))
    if workspace and loc.startswith(workspace.rstrip("/") + "/"):
        res["error"] = f"transformers resolved INSIDE the workspace ({loc}); refusing to use it as the reference"
        Path(out).write_text(json.dumps(res))
        return 0
    from transformers import CLIPTextModelWithProjection

    dev = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    res["device"] = dev

    def rel(a, b):
        a, b = a.detach().double().cpu(), b.detach().double().cpu()
        if a.shape != b.shape:
            return {"rel_l2": None, "why": f"shape {list(a.shape)} vs reference {list(b.shape)}"}
        out = {"cand_finite": bool(torch.isfinite(a).all()), "ref_finite": bool(torch.isfinite(b).all())}
        n = float(torch.linalg.norm(b))
        out.update(rel_l2=float(torch.linalg.norm(a - b)) / max(n, 1e-12), max_abs=float((a - b).abs().max()))
        return out

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for enc in ENCODERS:
        er: dict = {"ranks": {}}
        res[enc] = er
        files = sorted(Path(cap).glob(f"clip_{enc}_rank*.pt"))
        if not files:
            er["error"] = "no capture"
            continue
        try:
            m = CLIPTextModelWithProjection.from_pretrained(os.path.join(model, enc), torch_dtype=torch.float32).to(dev).eval()
            calls = []

            def hook(mod, args, output):
                x = args[0] if args else None
                y = output[0] if isinstance(output, (tuple, list)) else output
                if hasattr(x, "shape") and hasattr(y, "shape") and x.shape[-1] > 0:
                    calls.append(x.numel() // x.shape[-1] * x.shape[-1] * y.shape[-1])

            hs = [sub.register_forward_hook(hook) for sub in m.modules() if isinstance(sub, torch.nn.Linear)]
            caps = {}
            for f in files:
                rk = f.stem.rsplit("rank", 1)[1]
                caps[rk] = torch.load(f, map_location="cpu", weights_only=True)
            first = None
            with torch.no_grad():
                for rk, c in sorted(caps.items()):
                    ids = c.get("input_ids")
                    if ids is None:
                        er["ranks"][rk] = {"why": "no input ids captured"}
                        continue
                    calls.clear()
                    o = m(ids.to(dev).long(), output_hidden_states=True)
                    if "dense_macs" not in er:
                        er["dense_macs"] = int(sum(calls))
                        er["dense_calls"] = len(calls)
                    got = {}
                    if c.get("hidden_states_m2") is not None:
                        got["hidden_m2"] = rel(c["hidden_states_m2"], o.hidden_states[-2])
                    captured_pooled = projected_pooled(c)
                    if captured_pooled is not None:
                        got["pooled"] = rel(captured_pooled, o.text_embeds)
                    got["ids_shape"] = list(ids.shape)
                    er["ranks"][rk] = got
                    first = first or rk
            for h in hs:
                h.remove()
            if len(caps) >= 2:
                a, b = (caps[k] for k in sorted(caps)[:2])
                er["cross_rank"] = {k: {"bit_identical": bool(torch.equal(a[k], b[k])),
                                        "max_abs": float((a[k].double() - b[k].double()).abs().max())}
                                    for k in ("hidden_states_m2", "pooler_output")
                                    if a.get(k) is not None and b.get(k) is not None and a[k].shape == b[k].shape}
            del m
        except Exception as exc:  # noqa: BLE001
            er["error"] = f"{type(exc).__name__}: {exc}"[:400]
    Path(out).write_text(json.dumps(res, default=str))
    return 0


def judge(run: dict, reference: dict) -> dict:
    import clip_gates as G

    extra = {"request": run.get("request"), "engine_kwargs": run.get("engine_kwargs"), "timing_s": run.get("timing_s"),
             "reference_meta": {k: reference.get(k) for k in ("transformers", "device", "dtype", "error")}}
    return G.clip_encoder_parallel_is_real(run.get("events") or [], reference, extra=extra)


def write_gate(result: dict) -> None:
    import clip_gates as G
    import day0_gates as DG
    from _common import GateResult, write_gate as _wg

    v = result.get("verdict") or {"verdict": "not_evaluated", "reasons": [result.get("reason", "not run")]}
    ok = v.get("verdict") == "pass"
    _wg(GateResult(gate=GATE, passed=ok, score=1.0 if ok else 0.0, is_hard_gate=False, level=LAYER,
                   error=None if ok else "; ".join(v.get("reasons", []))[:500] or v.get("verdict"),
                   details=DG.gate_details(CAPABILITY, LAYER, v, dict(G.POLICY), artifact=str(OUT),
                                           policy_state=G.POLICY_STATE, policy_basis=dict(G.POLICY_BASIS),
                                           sp_engine_kwargs=SP_KWARGS, request=result.get("request"))))


def _gpus() -> int:
    try:
        import torch
        return int(torch.cuda.device_count())
    except Exception:  # noqa: BLE001
        return 0


def main() -> dict:
    if os.environ.get("SD3_CLIP_REUSE_PARALLEL") == "1":


        bundle_path = ARTIFACTS / "sd3_day0_parallel.json"
        if not bundle_path.is_file():
            return {"reason": "formal SP2 producer left no receipt"}
        bundle = json.loads(bundle_path.read_text())
        run_id = os.environ.get("SD3_VERIFIER_RUN_ID")
        if not run_id or bundle.get("run_id") != run_id:
            return {"reason": "SP2 evidence is not from this verifier invocation"}
        run = (bundle.get("runs") or {}).get("sp2") or {}
        if run.get('execution',{}).get('quiescent') is not True or not run.get('basic_stage'):
            return {"reason": "SP2 basic stage lacks a completed receipt or worker quiescence"}
        import native_sequence as N
        try:
            run = N.load_checkpoint(Path(bundle['work'])/'sp2', 'basic', run_id)
        except (OSError, ValueError, KeyError) as exc:
            return {'reason': f'SP2 basic stage cannot be verified: {exc}'}
        if not run.get("built"):
            return {"verdict": {"verdict": "fail", "reasons": ["the SP2 producer did not build", str(run.get("build_error", ""))]}}
        model = run["model"]
        ref = run_reference(model, Path(run["cap_dir"]), Path(bundle["work"]) / "clip_reference.json")
        return {"measured": True, "reference": ref, "request": bundle["requests"]["a"],
                "producer": str(bundle_path), "verdict": judge(run, ref)}


    if _gpus() < 2:
        return {"measured": False, "reason": f"not_evaluated: CLIP encoder parallel needs 2 GPUs, {_gpus()} visible"}
    req = _request()
    work = Path(tempfile.mkdtemp(prefix="sd3_day0_clipp_"))
    result = {"measured": False, "request": req, "work": str(work)}
    try:
        run = run_engine(work, req, dict(SP_KWARGS))
        if not run.get("built"):
            result.update(measured=True, run={k: v for k, v in run.items() if k != "events"})
            result["verdict"] = {"verdict": "fail", "reasons": ["the SP=2 engine did not build: "
                                                                + str(run.get("build_error"))]}
            return result
        t = time.time()
        reference = run_reference(run["model"], Path(run["cap_dir"]), work / "reference.json")
        run["timing_s"]["reference"] = round(time.time() - t, 1)
        result.update(measured=True, model=run["model"], reference=reference,
                      run={k: v for k, v in run.items() if k != "events"},
                      clip_lines=[e for e in run.get("events") or [] if str(e.get("event", "")).startswith("clip_")][:80])
        result["verdict"] = judge(run, reference)
        if run.get("request_error") and result["verdict"].get("verdict") != "fail":
            result["verdict"] = {"verdict": "fail", "reasons": ["the SP=2 engine did not serve the request: "
                                                                + run["request_error"]],
                                 "details": result["verdict"].get("details", {})}
        return result
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _engine_ports() -> dict:
    """Return this verifier queue's assigned engine port block."""
    from _common import engine_port_kwargs

    return engine_port_kwargs()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        try:
            sys.exit(_child(*sys.argv[2:7]))
        except Exception as exc:  # noqa: BLE001 - the parent reads whatever the child left
            Path(sys.argv[4]).write_text(json.dumps({"error": f"{type(exc).__name__}: {exc}"[:400]}))
            sys.exit(0)
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
    print(f"[day0-clip-parallel] {GATE}: {v.get('verdict', res.get('reason'))}; "
          + "; ".join(v.get("reasons") or [])[:600], flush=True)
    sys.exit(0)
