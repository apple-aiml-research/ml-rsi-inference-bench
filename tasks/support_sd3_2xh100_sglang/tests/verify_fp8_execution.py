#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
'SD3 verify fp8 execution utilities.'


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
TAP = HERE / "day0_tap"
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(DAY0))

ARTIFACTS = Path(os.environ.get("INFB_ARTIFACTS_DIR", "/logs/artifacts"))


TP = int(os.environ.get("SD3_DAY0_FP8_TP_SIZE", "1") or 1)
PARALLEL_DIST_TIMEOUT_S = 300
PROFILES = {
    1: ("fp8_dit_execution", "fp8_dit_execution", "T3", "sd3_day0_fp8.json"),
    2: ("fp8_dit_execution_tp2", "tp2_x_fp8", "T4", "sd3_day0_fp8_tp2.json"),
}
if TP not in PROFILES:
    raise SystemExit(f"SD3_DAY0_FP8_TP_SIZE={TP}: the FP8 verifier's profiles are {sorted(PROFILES)}")
GATE, CAPABILITY, LAYER, _ARTIFACT = PROFILES[TP]


GATE_IDS = ("fp8_dit_execution", "fp8_dit_execution_tp2")
OUT = ARTIFACTS / _ARTIFACT
IMG_DIR = ARTIFACTS / Path(_ARTIFACT).stem

DG_PRECOMPILE_ENV = "SGLANG_JIT_DEEPGEMM_PRECOMPILE"
FIXTURE = os.environ.get("SD3_DAY0_FP8_FIXTURE", "astronaut_cfg1")
LABEL = f"fp8.{FIXTURE}"
REPLAY_TIMEOUT_S = 1200
GOLDEN_SPEC = Path(os.environ.get("SD3_DAY0_GOLDEN_SPEC", HERE / "sd3_golden.json"))
RECIPE = Path(os.environ.get("SD3_DAY0_FP8_RECIPE", DAY0 / "recipe_v2.json"))


FI_MANIFEST = Path(os.environ.get("SD3_DAY0_FI_MANIFEST", DAY0 / "flashinfer_0.6.7.post3_trusted.json"))


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


def write_gate_file(result: dict) -> None:
    import day0_gates as G
    import day0_policy as P
    from _common import GateResult, write_gate
    v = result.get("verdict") or {"verdict": "not_evaluated", "reasons": [result.get("reason", "not run")]}
    ok = v.get("verdict") == "pass"
    for gate in GATE_IDS:
        if gate != GATE:
            continue
        write_gate(GateResult(gate=gate, passed=ok, score=1.0 if ok else 0.0, is_hard_gate=False, level=LAYER,
                              error=None if ok else ("; ".join(v.get("reasons", [])) or v.get("verdict"))[:500],
                              details=G.gate_details(CAPABILITY, LAYER, v, P.fp8_policy(), artifact=str(OUT),
                                                     execution_type=v.get("execution_type"),
                                                     quality=result.get("quality"), attribution=v.get("attribution"),
                                                     engine_env=(result.get("run") or {}).get("engine_env"))))


def engine_kwargs() -> dict:
    """DiffGenerator.from_pretrained arguments for one GPU or the requested TP profile."""
    if TP == 1:
        return {"num_gpus": 1}
    return {"num_gpus": TP, "tp_size": TP, "backend": "sglang", "dist_timeout": PARALLEL_DIST_TIMEOUT_S}


def build_asset(model: str, out: Path) -> dict:
    """The frozen FP8 asset, on CPU, checked against the recipe's frozen content identity."""
    src = Path(model) / "transformer"
    py = sys.executable
    g = subprocess.run([py, str(DAY0 / "sd3_fp8_recipe.py"), "generate", "--src", str(src), "--out", str(out),
                        "--device", "cpu", "--recipe", str(RECIPE)], capture_output=True, text=True, timeout=1800)
    if g.returncode != 0:
        raise RuntimeError(f"recipe generate exited {g.returncode}: {g.stderr[-600:]}")
    v = subprocess.run([py, str(DAY0 / "sd3_fp8_recipe.py"), "verify", "--asset", str(out), "--recipe", str(RECIPE)],
                       capture_output=True, text=True, timeout=1800)
    if v.returncode != 0:
        raise RuntimeError(f"recipe verify exited {v.returncode}: {(v.stdout + v.stderr)[-600:]}")
    man = json.loads((out / "fp8_manifest.json").read_text())
    return {"dir": str(out), "tensor_set_sha256": man.get("tensor_set_sha256"), "generate_stdout": g.stdout[-400:]}


def _save_png(img, label: str) -> str | None:
    """The request's image, lossless, for the gallery and later image-level readings (eyeballing only: best effort)."""
    try:
        from PIL import Image
        IMG_DIR.mkdir(parents=True, exist_ok=True)
        p = IMG_DIR / f"{label}.png"
        Image.fromarray(img).save(p)
        return str(p)
    except Exception:  # noqa: BLE001
        return None


def run_engine(model: str, asset: Path, tap_dir: Path, fx: dict, instrumented: bool = True,
               label: str = LABEL) -> dict:
    """Build the engine on the asset and serve the fixture's request. Raises only when the engine does not BUILD; a
    request that raised or returned nothing comes back without an "image". instrumented=False is the control: the
    same route and request on a fresh engine with no tap on the path and nothing armed."""
    import numpy as np
    from _common import NoImageProduced, to_image_array
    saved = {k: os.environ.get(k) for k in ("PYTHONPATH", "INFB_SD3_DAY0_TAP_DIR", "INFB_SD3_DAY0_LIB",
                                            "INFB_SD3_DAY0_FI_MANIFEST", "FLASHINFER_WORKSPACE_BASE",
                                            "INFB_SD3_DAY0_FI_T0", "TRTLLM_DG_CACHE_DIR", "FLASHINFER_LOGLEVEL",
                                            DG_PRECOMPILE_ENV)}
    path = [q for q in (saved["PYTHONPATH"] or "").split(":") if q and os.path.abspath(q) != str(TAP)]
    if instrumented:
        os.environ["PYTHONPATH"] = ":".join([str(TAP)] + path)
        os.environ["INFB_SD3_DAY0_TAP_DIR"] = str(tap_dir)
        os.environ["INFB_SD3_DAY0_LIB"] = str(DAY0)


        os.environ["INFB_SD3_DAY0_FI_MANIFEST"] = str(FI_MANIFEST)
        os.environ["INFB_SD3_DAY0_FI_T0"] = repr(time.time() - 1.0)
    else:
        os.environ["PYTHONPATH"] = ":".join(path)
        for k in ("INFB_SD3_DAY0_TAP_DIR", "INFB_SD3_DAY0_LIB", "INFB_SD3_DAY0_FI_MANIFEST", "INFB_SD3_DAY0_FI_T0"):
            os.environ.pop(k, None)
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.environ["FLASHINFER_WORKSPACE_BASE"] = str(tap_dir.parent / "fi_workspace")
    os.environ["TRTLLM_DG_CACHE_DIR"] = str(tap_dir.parent / "dg_cache")
    os.environ["FLASHINFER_LOGLEVEL"] = "0"
    if saved[DG_PRECOMPILE_ENV] is None:
        os.environ[DG_PRECOMPILE_ENV] = "0"
    for k in ("FLASHINFER_WORKSPACE_BASE", "TRTLLM_DG_CACHE_DIR"):
        Path(os.environ[k]).mkdir(parents=True, exist_ok=True)
    rec = {"timing_s": {}, "instrumented": instrumented,
           "engine_env": {DG_PRECOMPILE_ENV: os.environ.get(DG_PRECOMPILE_ENV)}}
    t0 = time.time()
    try:
        from sglang.multimodal_gen import DiffGenerator
        gen = DiffGenerator.from_pretrained(model_path=model, transformer_weights_path=str(asset), **engine_kwargs(),
                                            **_engine_ports())
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    rec["timing_s"]["engine_build"] = round(time.time() - t0, 1)
    try:
        if instrumented:
            arm = {"label": LABEL, "save_steps": [], "fp8_step": 0, "fp8_layer_stride": 3, "fp8_rows": 64,
                   "all_ranks": TP > 1}
            (tap_dir / "ARM.tmp").write_text(json.dumps(arm))
            os.replace(tap_dir / "ARM.tmp", tap_dir / "ARM")
        t1 = time.time()
        try:
            out = gen.generate(sampling_params_kwargs=dict(
                prompt=fx["prompt"], seed=fx["seed"], num_inference_steps=fx["steps"], guidance_scale=fx["cfg"],
                num_frames=1, height=fx["h"], width=fx["w"], return_frames=True, save_output=False))
            err = getattr(out, "error", None)
            if err:
                rec["engine_error"] = str(err)[:600]
            img = to_image_array(out)
            rec["image_sha256"] = hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest()
            rec["image"] = img
            rec["image_png"] = _save_png(img, label)
        except NoImageProduced as exc:
            rec["no_image"] = str(exc)
        except Exception as exc:  # noqa: BLE001 - no image; the uninstrumented control supports attribution
            rec["request_error"] = f"{type(exc).__name__}: {exc}"[:600]
        finally:
            try:
                (tap_dir / "ARM").unlink()
            except FileNotFoundError:
                pass
        rec["timing_s"]["request"] = round(time.time() - t1, 1)
    finally:
        try:
            gen.shutdown()
        except Exception:  # noqa: BLE001
            pass
    return rec


def control_run(model: str, work: Path, fx: dict) -> dict:
    """The armed engine did not serve, so the same request runs once more on a fresh engine without the
    tap. {"built", "served", "why", ...the run record}; never raises. It shares the armed run's JIT cache directories
    (their parent is `work`), so it recompiles nothing the armed run compiled."""
    try:
        rec = run_engine(model, work / "fp8_transformer", work / "tap_control", fx, instrumented=False,
                         label=f"control.{LABEL}")
    except Exception as exc:  # noqa: BLE001 - the control engine did not build
        return {"built": False, "served": False, "why": f"the control engine did not build: {type(exc).__name__}: "
                                                        f"{exc}"[:400]}
    img = rec.pop("image", None)
    rec.update(built=True, served=img is not None,
               why=None if img is not None else (rec.get("engine_error") or rec.get("request_error")
                                                 or rec.get("no_image") or "no image"))
    return rec


def _sha_t(t) -> str:
    import torch
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def replay(tap_dir: Path, asset_dir: Path) -> dict:
    """Isolated: torch + safetensors + tests/day0 only. Evidence for day0_gates.fp8_execution_verdict."""
    if TP > 1:
        return replay_tp(tap_dir, asset_dir, TP)
    import torch
    from safetensors import safe_open

    import fp8_evidence as E
    rec_p = tap_dir / f"{LABEL}.fp8.json"
    if not rec_p.is_file():
        started = False
        ev_p = tap_dir / "events.jsonl"
        if ev_p.is_file():
            started = any('"fp8_started"' in line for line in ev_p.read_text(errors="replace").splitlines())
        return {"captured": False, "capture_started_not_finished": started,
                "reason": ("the armed DiT forward started an FP8 capture and never finished it (it raised inside the "
                           "candidate)" if started else "the tap wrote no FP8 record (no armed DiT forward?)")}
    rec = json.loads(rec_p.read_text())
    if rec.get("forward_raised"):

        return {"captured": False, "capture_started_not_finished": True, "forward_raised": True,
                "layers_fired_before_the_raise": len(rec.get("layers_fired") or []),
                "reason": "the armed DiT forward raised inside the candidate"}
    layers_io = torch.load(tap_dir / f"{LABEL}.fp8_layers.pt", map_location="cpu", weights_only=True) \
        if (tap_dir / f"{LABEL}.fp8_layers.pt").is_file() else {}
    man = json.loads((asset_dir / "fp8_manifest.json").read_text())
    q_mods = sorted(man.get("quantized_modules") or [])
    by_sha = {}
    scale_sha = set()
    with safe_open(str(asset_dir / "diffusion_pytorch_model.safetensors"), "pt", device="cpu") as f:
        for m in q_mods:
            w = f.get_tensor(m + ".weight")
            by_sha[hashlib.sha256(w.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()] = m
            scale_sha.add(_sha_t(f.get_tensor(m + ".weight_scale_inv")))

    matched, where = set(), {}
    for t in rec.get("store") or []:
        for key in ("sha256", "sha256_t"):
            m = by_sha.get(t.get(key))
            if m:
                matched.add(m)
                where[t["name"]] = {"asset": [m], "transposed": key == "sha256_t"}
        slabs = [by_sha.get(s) for s in t.get("slabs") or []]
        if slabs and all(slabs):
            matched.update(slabs)
            where[t["name"]] = {"asset": slabs, "transposed": False}

    betas = []
    with safe_open(str(asset_dir / "diffusion_pytorch_model.safetensors"), "pt", device="cpu") as f:
        for L in rec.get("layers_found") or []:
            io = layers_io.get(L["module"])
            src = where.get(f"{L['module']}.{L['weight']}")
            if io is None or src is None:
                continue
            ws = [f.get_tensor(m + ".weight") for m in src["asset"]]
            ss = [f.get_tensor(m + ".weight_scale_inv") for m in src["asset"]]
            w, s = torch.cat(ws, 0), torch.cat(ss, 0)


            names = [io.get("orig_dtype") or L.get("orig_dtype"), io.get("x_dtype"), io.get("y_dtype")]
            cands = []
            for nm in names:
                dt = getattr(torch, nm, None) if isinstance(nm, str) else None
                if isinstance(dt, torch.dtype) and dt.is_floating_point and dt not in cands:
                    cands.append(dt)
            cands.append(None)
            try:
                b = E.layer_beta_on_quantised_operand(io["x"], io["y"], w, s, cands)
                row = {"module": L["module"], "asset": src["asset"], **{k: b[k] for k in b if k in (
                    "beta", "orthogonal_residual_over_d", "d_rel", "rel_l2_vs_w8a8", "rel_l2_vs_w8a16")}}
                row.update(operand=b["input_dtype"], alternatives=b["alternatives"], x_dtype=io.get("x_dtype"),
                           orig_dtype=io.get("orig_dtype") or L.get("orig_dtype"))


                if io.get("x_dtype") == "float32" and row["operand"] != "float32":
                    row["beta_captured_input"] = next((a["beta"] for a in b["alternatives"]
                                                       if a["input_dtype"] == "float32"), None)
                betas.append(row)
            except Exception as exc:  # noqa: BLE001
                betas.append({"module": L["module"], "error": f"{type(exc).__name__}: {exc}"[:200]})
    disp = E.gemm_census(rec.get("dispatch_records") or [], E.q_shapes_for(fused_qkv=True), min_tokens=64)
    fp8_names = ("float8_e4m3fn", "float8_e5m2")
    triton_fp8 = sum(1 for lz in rec.get("triton_launches") or [] if sum(d in fp8_names for d in lz["dtypes"]) >= 2)
    kc = E.kernel_census(rec.get("kernels") or {}) if rec.get("kernels") else None

    import day0_policy as P
    ptr_to_asset = {}
    for t in rec.get("store") or []:
        if t.get("data_ptr") is not None and t.get("name") in where:
            ptr_to_asset[int(t["data_ptr"])] = where[t["name"]]["asset"]
    fb = E.flashinfer_boundary(rec, ptr_to_asset, P.FLASHINFER_PINNED_ENTRIES) \
        if rec.get("flashinfer_calls") or rec.get("flashinfer_patched") else None
    store = {"q_modules_expected": len(q_mods), "q_modules_matched": len(matched),
             "float8_tensors_held": len(rec.get("store") or []), "unmatched_sample":
                 [m for m in q_mods if m not in matched][:8]}
    if "store_scales" in rec:
        store.update(scales_expected=len(q_mods), scales_matched=sum(
            1 for t in rec["store_scales"] if t.get("sha256") in scale_sha or t.get("sha256_t") in scale_sha))
    found = rec.get("layers_found") or []
    return {"captured": True,
            "store": store,
            "dispatch": disp, "triton": {"fp8_operand_launches": triton_fp8,
                                         "launches": len(rec.get("triton_launches") or [])},
            "ncu": None, "flashinfer": fb, "direct": E.direct_channel_evidence(rec, ptr_to_asset),
            "q_layers_found": len(found),
            "q_layers_fired": len(rec.get("layers_fired") or []),
            "beta": [b for b in betas if "beta" in b], "beta_errors": [b for b in betas if "error" in b],
            "beta_expected": _beta_expected(found, lambda L: where.get(f"{L['module']}.{L['weight']}")),
            "kernels": kc, "kernels_empty": rec.get("kernels_empty"), "asset_tensor_set_sha256":
                man.get("tensor_set_sha256")}


def _beta_expected(found: list, holds_asset) -> int | None:
    """Count sampled layers that hold the expected asset; return None if sampling metadata is absent."""


    if not found or any("keep" not in L for L in found):
        return None
    return sum(1 for L in found if L["keep"] and holds_asset(L))


def replay_tp(tap_dir: Path, asset_dir: Path, tp: int) -> dict:
    """The TP profile's replay: one record per rank (`<label>.r<rank>.fp8.json`), each judged against ITS OWN slabs of
    the asset -- a column-parallel shard is rows [r N/tp, (r+1) N/tp) of the asset's weight and the matching 128-block
    rows of its scale, a row-parallel shard the columns [r K/tp, (r+1) K/tp) -- so a rank holding the whole tensor
    (replicated weights) or another rank's slab matches nothing. Per rank: the store census (weights and scales), the
    dispatch census at the shard shapes, beta (column-parallel on the rank's own rows; row-parallel on the ranks'
    captured inputs joined, against the reduced output), and result consumption with row-parallel layers judged across ranks."""
    import torch
    from safetensors import safe_open

    import day0_policy as P
    import fp8_evidence as E
    recs, missing, raised = [], [], False
    for r in range(tp):
        p = tap_dir / f"{LABEL}.r{r}.fp8.json"
        if not p.is_file():
            missing.append(r)
            recs.append(None)
            continue
        rec = json.loads(p.read_text())
        raised = raised or bool(rec.get("forward_raised"))
        recs.append(rec)
    if missing or raised:
        started = False
        ev_p = tap_dir / "events.jsonl"
        if ev_p.is_file():
            started = any('"fp8_started"' in line for line in ev_p.read_text(errors="replace").splitlines())
        return {"captured": False, "tp": tp, "capture_started_not_finished": raised or started,
                "reason": f"rank record(s) missing: {missing}" if missing else "the armed forward raised on a rank"}
    ios = []
    for r in range(tp):
        q = tap_dir / f"{LABEL}.r{r}.fp8_layers.pt"
        ios.append(torch.load(q, map_location="cpu", weights_only=True) if q.is_file() else {})
    man = json.loads((asset_dir / "fp8_manifest.json").read_text())
    q_mods = sorted(man.get("quantized_modules") or [])
    slab = [dict() for _ in range(tp)]
    sslab = [set() for _ in range(tp)]
    whole = set()
    shapes = {}
    with safe_open(str(asset_dir / "diffusion_pytorch_model.safetensors"), "pt", device="cpu") as f:
        for m in q_mods:
            w, sc = f.get_tensor(m + ".weight"), f.get_tensor(m + ".weight_scale_inv")
            n, k = w.shape
            shapes[m] = (n, k)
            whole.add(_sha_t(w))
            for r in range(tp):
                if n % (tp * 128) == 0:
                    rows = slice(r * n // tp, (r + 1) * n // tp)
                    slab[r][_sha_t(w[rows])] = (m, "rows")
                    sslab[r].add(_sha_t(sc[r * n // tp // 128:(r + 1) * n // tp // 128]))
                if k % (tp * 128) == 0:
                    cols = slice(r * k // tp, (r + 1) * k // tp)
                    slab[r][_sha_t(w[:, cols])] = (m, "cols")
                    sslab[r].add(_sha_t(sc[:, r * k // tp // 128:(r + 1) * k // tp // 128]))
        where = []
        for r, rec in enumerate(recs):
            wr = {}
            for t in rec.get("store") or []:
                for key in ("sha256", "sha256_t"):
                    hit = slab[r].get(t.get(key))
                    if hit:
                        wr[t["name"]] = {"asset": hit[0], "kind": hit[1], "data_ptr": t.get("data_ptr")}
            where.append(wr)
        kinds = {}
        for wr in where:
            for name, h in wr.items():
                kinds.setdefault(name.rsplit(".", 1)[0], h["kind"])
        row_parallel = {L for L, kd in kinds.items() if kd == "cols"}
        betas = [[] for _ in range(tp)]
        for L in recs[0].get("layers_found") or []:
            mod, wname = L["module"], L["weight"]
            kind = kinds.get(mod)
            src = where[0].get(f"{mod}.{wname}")
            if kind is None or src is None:
                continue
            m = src["asset"]
            cands = [c for c in (getattr(torch, nm, None) for nm in (L.get("orig_dtype"), "bfloat16"))
                     if isinstance(c, torch.dtype)] + [None]
            try:
                if kind == "rows":
                    n, k = shapes[m]
                    w, sc = f.get_tensor(m + ".weight"), f.get_tensor(m + ".weight_scale_inv")
                    for r in range(tp):
                        io = ios[r].get(mod)
                        if io is None:
                            continue
                        rows = slice(r * n // tp, (r + 1) * n // tp)
                        b = E.layer_beta_on_quantised_operand(io["x"], io["y"], w[rows], sc[r * n // tp // 128:
                                                              (r + 1) * n // tp // 128], cands)
                        betas[r].append({"module": mod, "beta": b["beta"], "operand": b["input_dtype"], "kind": kind})
                else:
                    got = [ios[r].get(mod) for r in range(tp)]
                    if any(g is None for g in got):
                        continue
                    x = torch.cat([g["x"] for g in got], dim=-1)
                    b = E.layer_beta_on_quantised_operand(x, got[0]["y"], f.get_tensor(m + ".weight"),
                                                          f.get_tensor(m + ".weight_scale_inv"), cands)
                    for r in range(tp):
                        betas[r].append({"module": mod, "beta": b["beta"], "operand": b["input_dtype"], "kind": kind,
                                         "joined_ranks": tp})
            except Exception as exc:  # noqa: BLE001
                for r in range(tp):
                    betas[r].append({"module": mod, "error": f"{type(exc).__name__}: {exc}"[:200]})
    ptrs = [{int(h["data_ptr"]): [h["asset"]] for h in wr.values() if h.get("data_ptr") is not None} for wr in where]
    direct = E.tp_direct_evidence(recs, ptrs, row_parallel)


    fi_reads = E.tp_flashinfer_row_sum(recs, row_parallel)
    q_shapes = E.q_shapes_for(tp=tp, fused_qkv=True) | E.q_shapes_for(fused_qkv=True)
    fp8_names = ("float8_e4m3fn", "float8_e5m2")

    beta_exp = _beta_expected(recs[0].get("layers_found") or [], lambda L: kinds.get(L["module"]) is not None
                              and where[0].get(f"{L['module']}.{L['weight']}") is not None)
    ranks = []
    for r, rec in enumerate(recs):
        matched = {h["asset"] for h in where[r].values()}
        held_whole = sum(1 for t in rec.get("store") or [] if t.get("sha256") in whole or t.get("sha256_t") in whole)
        store = {"q_modules_expected": len(q_mods), "q_modules_matched": len(matched),
                 "float8_tensors_held": len(rec.get("store") or []), "whole_tensors_held": held_whole,
                 "unmatched_sample": [m for m in q_mods if m not in matched][:8]}
        if "store_scales" in rec:
            store.update(scales_expected=len(q_mods), scales_matched=sum(
                1 for t in rec["store_scales"] if t.get("sha256") in sslab[r] or t.get("sha256_t") in sslab[r]))
        fb = E.flashinfer_boundary(E.with_tp_flashinfer_provenance(rec, fi_reads), ptrs[r],
                                   P.FLASHINFER_PINNED_ENTRIES) \
            if rec.get("flashinfer_calls") or rec.get("flashinfer_patched") else None
        ranks.append({"captured": True, "rank": r, "store": store,
                      "dispatch": E.gemm_census(rec.get("dispatch_records") or [], q_shapes, min_tokens=64),
                      "triton": {"fp8_operand_launches": sum(1 for lz in rec.get("triton_launches") or []
                                                             if sum(d in fp8_names for d in lz["dtypes"]) >= 2)},
                      "ncu": None, "flashinfer": fb, "direct": direct[r],
                      "q_layers_found": len(rec.get("layers_found") or []),
                      "q_layers_fired": len(rec.get("layers_fired") or []),
                      "beta": [b for b in betas[r] if "beta" in b], "beta_errors": [b for b in betas[r] if "error" in b],
                      "beta_expected": beta_exp})
    fi_tally = {v: list(fi_reads.values()).count(v) for v in sorted(set(fi_reads.values()))}
    return {"captured": True, "tp": tp, "ranks": ranks, "row_parallel_layers": len(row_parallel),
            "flashinfer_tp_row_sum": dict(fi_tally, row_parallel_layers_with_calls=len(fi_reads)),
            "asset_tensor_set_sha256": man.get("tensor_set_sha256")}


def _psnr(a, b) -> float:
    import numpy as np
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    mse = float((d * d).mean())
    return float("inf") if mse == 0 else 10.0 * float(np.log10(255.0 ** 2 / mse))


def main() -> dict:
    import numpy as np

    import day0_gates as G


    model = _resolve_model()
    fx = {f["id"]: f for f in json.loads(GOLDEN_SPEC.read_text())["fixtures"]}[FIXTURE]
    work = Path(tempfile.mkdtemp(prefix="sd3_day0_fp8_"))
    result = {"fixture": FIXTURE, "model": model, "profile": engine_kwargs()}
    try:
        try:
            result["asset"] = build_asset(model, work / "fp8_transformer")
        except Exception as exc:  # noqa: BLE001 - ours, not the candidate's
            return dict(result, reason=f"not_evaluated: verifier_defect: {type(exc).__name__}: {exc}"[:500])
        tap_dir = work / "tap"
        tap_dir.mkdir()
        try:
            run = run_engine(model, work / "fp8_transformer", tap_dir, fx)
        except Exception as exc:  # noqa: BLE001 - the armed engine did not build with the asset
            from _common import save_stderr_log
            save_stderr_log("day0_fp8_engine", traceback.format_exc())
            why = f"the engine did not build with the FP8 asset: {type(exc).__name__}: {exc}"[:400]
            result["control"] = control_run(model, work, fx)
            return dict(result, verdict=G.fp8_e4_verdict(why, result["control"], armed_built=False))
        img = run.pop("image", None)
        result["run"] = run
        if img is None:


            why = run.get("engine_error") or run.get("request_error") or run.get("no_image") or "no image"
            result["control"] = control_run(model, work, fx)
            return dict(result, verdict=G.fp8_e4_verdict(
                f"the engine built with the FP8 asset but did not serve it: {why}"[:500], result["control"],
                armed_built=True))
        if img is not None:
            golden = HERE / fx["golden_npy"]
            if golden.is_file():
                result["quality"] = {"psnr_vs_bf16_golden_db": _psnr(img, np.load(golden)),
                                     "recorded_only": "no bar until independent legal-kernel calibration is complete"}
        repo = str(Path(os.environ.get("INFB_WORKSPACE", "/app")).resolve() / "repo")
        env = dict(os.environ)
        env["PYTHONPATH"] = ":".join(q for q in (env.get("PYTHONPATH") or "").split(":") if q and not q.startswith(repo))
        rp = work / "replay.json"
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--replay", str(tap_dir),
                            str(work / "fp8_transformer"), str(rp)], env=env, capture_output=True, text=True,
                           timeout=REPLAY_TIMEOUT_S)
        if r.returncode != 0 or not rp.is_file():
            return dict(result, reason=f"not_evaluated: replay exited {r.returncode}", replay_stderr=r.stderr[-2000:])
        ev = json.loads(rp.read_text())
        result["evidence"] = ev
        result["verdict"] = G.fp8_tp_verdict(ev) if TP > 1 else G.fp8_execution_verdict(ev)
        return result
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _engine_ports() -> dict:
    """Return this verifier queue's assigned engine port block."""
    from _common import engine_port_kwargs

    return engine_port_kwargs()

if __name__ == "__main__":
    if sys.argv[1:2] == ["--replay"]:
        try:
            res = replay(Path(sys.argv[2]), Path(sys.argv[3]))
        except Exception as exc:  # noqa: BLE001
            res = {"captured": False, "reason": f"{type(exc).__name__}: {exc}"[:400], "trace": traceback.format_exc()[-2000:]}
        _write_json(Path(sys.argv[4]), res)
        sys.exit(0)
    try:
        res = main()
    except Exception as exc:  # noqa: BLE001
        res = {"reason": f"not_evaluated: {type(exc).__name__}: {exc}"[:400], "trace": traceback.format_exc()[-2000:]}
    try:
        _write_json(OUT, res)
    except Exception:  # noqa: BLE001
        pass
    write_gate_file(res)
    print(f"[day0-fp8] {GATE}: {(res.get('verdict') or {}).get('verdict', res.get('reason'))}", flush=True)
    sys.exit(0)
