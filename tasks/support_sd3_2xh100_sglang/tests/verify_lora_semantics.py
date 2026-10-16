#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
'SD3 verify lora semantics utilities.'


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


PROFILES = {
    (1, 1): {"name": "1gpu", "capability": "lora_semantics", "multi_capability": "lora_multi_adapter", "layer": "T3",
             "sequence": "request_sequence_v4.json",
             "gates": {"reference": "lora_reference_agreement", "isolation": "lora_state_isolation",
                       "multi": "lora_multi_adapter_composition"},
             "artifact": "sd3_day0_lora.json", "images": "sd3_day0_lora"},
    (2, 1): {"name": "tp2", "capability": "tp2_x_lora", "multi_capability": "tp2_x_lora", "layer": "T4",
             "sequence": "request_sequence_v4.json",
             "gates": {"reference": "lora_reference_agreement_tp2", "isolation": "lora_state_isolation_tp2",
                       "multi": "lora_multi_adapter_composition_tp2"},
             "artifact": "sd3_day0_lora_tp2.json", "images": "sd3_day0_lora_tp2"},
    (1, 2): {"name": "sp2", "capability": "sp2_x_lora_or_fp8", "multi_capability": "sp2_x_lora_or_fp8", "layer": "T4",
             "sequence": "request_sequence_v4.json",
             "gates": {"reference": "lora_reference_agreement_sp2", "isolation": "lora_state_isolation_sp2",
                       "multi": "lora_multi_adapter_composition_sp2"},
             "artifact": "sd3_day0_lora_sp2.json", "images": "sd3_day0_lora_sp2"},
    (1, 1, "offload"): {"name": "offload", "capability": "offload_x_lora", "multi_capability": "offload_x_lora", "layer": "T4",
             "sequence": "request_sequence_v4.json", "gates": {"reference": "offload_x_lora"},
             "artifact": "sd3_day0_lora_offload.json", "images": "sd3_day0_lora_offload"},
}


def profile_dims(env=None) -> tuple:
    """(tp, sp) as the profile switch spells them, for NAMING the run's gates; never raises (an unparseable value
    names no profile of the matrix). engine_kwargs() alone judges the profile's consistency (num_gpus = tp x sp), and
    an inconsistent one is reported under the gates of its (tp, sp)."""
    env = os.environ if env is None else env

    def _v(name: str) -> int:
        raw = env.get(name)
        if raw is None or str(raw).strip() == "":
            return 1
        try:
            return int(raw)
        except ValueError:
            return -1

    return _v("SD3_DAY0_LORA_TP_SIZE"), _v("SD3_DAY0_LORA_SP_DEGREE")


def profile_key(env=None):
    env = os.environ if env is None else env
    dims = profile_dims(env)
    return (*dims, "offload") if env.get("SD3_DAY0_LORA_OFFLOAD") == "1" else dims


if profile_key() not in PROFILES:
    raise SystemExit(f"SD3_DAY0_LORA_TP_SIZE / SD3_DAY0_LORA_SP_DEGREE = {profile_dims()}: the LoRA verifier's "
                     f"profiles (tp, sp) are {sorted(PROFILES)}")
PROFILE = PROFILES[profile_key()]


GATE_IDS = ("lora_reference_agreement", "lora_state_isolation", "lora_multi_adapter_composition",
            "lora_reference_agreement_tp2", "lora_state_isolation_tp2", "lora_multi_adapter_composition_tp2",
            "lora_reference_agreement_sp2", "lora_state_isolation_sp2", "lora_multi_adapter_composition_sp2", "offload_x_lora")
OUT = ARTIFACTS / PROFILE["artifact"]
IMG_DIR = ARTIFACTS / PROFILE["images"]
GATE_REF = PROFILE["gates"]["reference"]
GATE_ISO = PROFILE["gates"].get("isolation")
GATE_MULTI = PROFILE["gates"].get("multi")
CAPABILITY, LAYER = PROFILE["capability"], PROFILE["layer"]
CAPABILITY_MULTI = PROFILE["multi_capability"]
REPLAY_TIMEOUT_S = 1800


VAE_SCALE = 8

RECIPE = Path(os.environ.get("SD3_DAY0_LORA_RECIPE", DAY0 / "lora_recipe_v4.json"))
SHAPES = Path(os.environ.get("SD3_DAY0_LORA_SHAPES", DAY0 / "sd3m_dit_linear_shapes.json"))
SEQUENCE = Path(os.environ.get("SD3_DAY0_LORA_SEQUENCE", DAY0 / PROFILE["sequence"]))
GOLDEN_SPEC = Path(os.environ.get("SD3_DAY0_GOLDEN_SPEC", HERE / "sd3_golden.json"))
CALIBRATED_PROFILE = {"num_gpus": 1}
PARALLEL_DIST_TIMEOUT_S = 300


def engine_kwargs(env=None) -> dict:
    """DiffGenerator.from_pretrained's profile kwargs: {"num_gpus": 1} unless SD3_DAY0_LORA_TP_SIZE /
    SD3_DAY0_LORA_SP_DEGREE / SD3_DAY0_LORA_NUM_GPUS ask for more, then the spelling of core's F probes (tp2:
    num_gpus 2 + tp_size 2; sp2: num_gpus 2 + sp_degree 2 + ulysses_degree 2 + tp_size 1) plus backend="sglang" and
    dist_timeout. Raises ValueError on a value < 1 or num_gpus != tp x sp."""
    env = os.environ if env is None else env

    def _int(name: str, default: int) -> int:
        raw = env.get(name)
        if raw is None or str(raw).strip() == "":
            return default
        v = int(raw)
        if v < 1:
            raise ValueError(f"{name}={raw!r}: must be >= 1")
        return v

    tp = _int("SD3_DAY0_LORA_TP_SIZE", 1)
    sp = _int("SD3_DAY0_LORA_SP_DEGREE", 1)
    n = _int("SD3_DAY0_LORA_NUM_GPUS", tp * sp)
    if n != tp * sp:
        raise ValueError(f"inconsistent LoRA engine profile: num_gpus {n} != tp {tp} x sp {sp}")
    if n == 1:
        kw = dict(CALIBRATED_PROFILE)
        if env.get("SD3_DAY0_LORA_OFFLOAD") == "1":
            kw["dit_layerwise_offload"] = True
        return kw
    kw = {"num_gpus": n, "tp_size": tp, "backend": "sglang", "dist_timeout": PARALLEL_DIST_TIMEOUT_S}
    if sp > 1:
        kw.update(sp_degree=sp, ulysses_degree=sp)
    return kw


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


def _points(seq: dict) -> tuple[set, set]:
    rp = seq["reading_points"]
    return {tuple(p) for p in rp["scored"]}, {tuple(p) for p in rp["diagnostic"]}


def offload_combination_verdict(result: dict) -> dict:
    import offload_gates as O
    reasons, missing, per_request = [], [], {}
    for key in ("reference_verdict", "isolation_verdict", "multi_verdict"):
        v = result.get(key) or {}
        if v.get("verdict") == "fail":
            reasons.extend([f"{key}: {x}" for x in v.get("reasons", [])] or [key + ": fail"])
        elif v.get("verdict") != "pass":
            missing.append(key + ": not measured")
    run = result.get("run") or {}
    events = run.get("residency_events") or []
    requests = run.get("requests") or []
    if not requests:
        missing.append("no LoRA request sequence")
    for request in requests:
        if not request.get("fixtures"):
            missing.append(f"r{request.get('request')}: no fixture residency windows")
        for fid, r in request.get("fixtures", {}).items():
            label = r.get("label", str(request.get("request")) + "." + fid)
            lines = O.lines_in_window(events, r.get("t0", 0), r.get("t1", 0))
            problem = O._record_problem(lines)
            if problem:
                missing.append(f"{label}: {problem}")
                continue


            fractions = []
            for line in lines:
                full = sum(line.get("self") or [])
                if not full or any(x <= 0 for x in line["self"]):
                    missing.append(f"{label}: missing block weight residency")
                    break
                fractions.append(max(line["total"]) / full)
            per_request[label] = fractions
            if fractions and max(fractions) > O.POLICY["MAX_RESIDENT_FRAC"]:
                reasons.append(f"{label}: DiT stayed resident through the LoRA request")
    verdict = "fail" if reasons else ("not_evaluated" if missing else "pass")
    return {"verdict": verdict, "reasons": reasons + missing, "details": {
        "max_resident_fraction_by_request": per_request, "policy_state": O.POLICY_STATE,
        "constituents": {k: result.get(k) for k in ("reference_verdict", "isolation_verdict", "multi_verdict")}}}


def write_gates(result: dict) -> None:
    """The profile's gates (PROFILES): reference + isolation (+ the multi-adapter gate at one GPU). The policy records
    the sequence this run actually replayed and the profile it ran under."""
    import day0_gates as G
    import day0_policy as P
    from _common import GateResult, write_gate
    pol = dict(P.lora_policy(), sequence=SEQUENCE.name, profile=PROFILE["name"])
    missing = {"verdict": "not_evaluated", "reasons": [result.get("reason", "not run")]}
    by_role = {"reference": (result.get("reference_verdict"), CAPABILITY),
               "isolation": (result.get("isolation_verdict"), CAPABILITY),
               "multi": (result.get("multi_verdict"), CAPABILITY_MULTI)}
    for role, name in PROFILE["gates"].items():
        v, cap = by_role[role]
        v = offload_combination_verdict(result) if PROFILE["name"] == "offload" else (v or missing)
        ok = v.get("verdict") == "pass"
        for gate in GATE_IDS:
            if gate != name:
                continue
            write_gate(GateResult(gate=gate, passed=ok, score=1.0 if ok else 0.0, is_hard_gate=False, level=LAYER,
                                  error=None if ok else "; ".join(v.get("reasons", []))[:500] or v.get("verdict"),
                                  details=G.gate_details(cap, LAYER, v, pol, artifact=str(OUT),
                                                         certification_scope=P.LORA_CERTIFICATION_SCOPE,
                                                         engine_profile=result.get("engine_profile"),
                                                         calibrated_profile=dict(CALIBRATED_PROFILE),
                                                         profile=PROFILE["name"], sequence=SEQUENCE.name,
                                                         profile_reached=result.get("profile_reached"))))


def _op_fixtures(op: dict, seq: dict) -> list:
    """The fixtures generated after an op: its own list (v4's r8-r14: astronaut_cfg1 only), else the sequence's."""
    return list(op.get("fixtures") or seq["fixtures"])


def profile_reached(tap_events: dict | None, world: int) -> dict:
    """Did the requested 2-GPU profile OCCUR -- the condition every TP / SP clause depends on? The tap hooks the DiT in
    every rank's worker (`hooked`), and a rank other than 0 that runs a DiT forward under an armed tap emits one
    `rank_skip` (tests/day0_tap). One GPU has nothing to prove. Not reached means the gates are not_evaluated: an engine that took tp_size 2 and served from one process was never read under TP."""
    te = tap_events or {}
    hooked, skip = int(te.get("hooked") or 0), int(te.get("rank_skip") or 0)
    reasons = []
    if world > 1:
        if hooked < world:
            reasons.append(f"the tap hooked {hooked} DiT(s), expected one per rank ({world})")
        if skip < 1:
            reasons.append("no rank_skip: no second rank ran a DiT forward under the tap")
    return {"reached": not reasons, "world": world, "hooked": hooked, "rank_skip": skip, "reasons": reasons}


def _pair_fixtures(seq: dict, i: int, j: int) -> list:
    """A pair is compared on the fixtures BOTH its requests generate, by design (never on a missing capture)."""
    fi, fj = _op_fixtures(seq["ops"][i], seq), _op_fixtures(seq["ops"][j], seq)
    return [f for f in seq["fixtures"] if f in fi and f in fj]


def grid_states(op: dict, exp: dict) -> dict:
    """The served-strength grid of one request beyond base and its own state: {tag: state}. A single adapter X@s has
    its factor-2 neighbours half (X@s/2) and double (X@2s); every request adds its op's named neighbours (v4:
    rslora_ignored, pattern_ignored; for several adapters first_only / last_only / first_at_last_scale, which have no
    half / double). A base request has no grid (it is judged against the base reference alone)."""
    if exp.get("adapter") is None:
        return {}
    g = {}
    if not isinstance(exp["adapter"], list) and exp.get("scale"):
        s0 = float(exp["scale"])
        g["half"] = {"adapter": exp["adapter"], "scale": s0 / 2}
        g["double"] = {"adapter": exp["adapter"], "scale": s0 * 2}
    for tag, st in sorted((op.get("neighbours") or {}).items()):
        g[tag] = dict(st)
    return g


def frozen_request_set(seq: dict, run: dict | None = None) -> list:
    "The sequence's request set, frozen before the run."


    scored, _ = _points(seq)
    ran = {r["request"]: r for r in (run or {}).get("requests") or []}
    out = []
    for k, (op, exp) in enumerate(zip(seq["ops"], seq["expected_states"])):
        fx = _op_fixtures(op, seq)
        errs = []
        r = ran.get(k)
        if run is not None and r is None:
            errs.append(f"r{k}: the request never ran")
        for fid, v in sorted(((r or {}).get("fixtures") or {}).items()):
            if v.get("error") or v.get("no_image"):
                errs.append(f"r{k} {fid}: {v.get('error') or v.get('no_image')}"[:300])
        grid = [] if exp.get("adapter") is None else ["base", "state"] + list(grid_states(op, exp))
        out.append({"request": k, "expected": exp, "points": sorted([list(p) for p in scored if p[0] in fx]),
                    "grid": grid, "serve_errors": errs})
    return out


def frozen_pair_set(seq: dict, key: str = "equal_pairs") -> dict:
    """{"<fixture>/image" | "<fixture>/dit_step0": [pair keys]}: every pair of seq[key] on every fixture both of its
    requests generate, at both levels -- what day0_gates.lora_isolation_verdict must find measured."""
    req: dict = {}
    for i, j in seq.get(key) or []:
        for fid in _pair_fixtures(seq, i, j):
            for lvl in (f"{fid}/image", f"{fid}/dit_step0"):
                req.setdefault(lvl, []).append(f"{i}-{j}")
    return req


def _adapters(tmp: Path) -> dict:
    """Generate every adapter of the frozen recipe and check each against its `expected` file hash."""
    import sd3_lora_synth as S
    rc = S.main(["generate", "--out", str(tmp), "--recipe", str(RECIPE), "--shapes", str(SHAPES)])
    if rc != 0:
        raise RuntimeError(f"adapter generation exited {rc}")
    recipe = json.loads(RECIPE.read_text())
    exp = recipe["expected"]
    out = {}
    for name in sorted(recipe["adapters"]):
        p = tmp / name / S.WEIGHT_NAME
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != exp[name]["file_sha256"]:
            raise RuntimeError(f"adapter {name} hashes to {got}, the recipe froze {exp[name]['file_sha256']}")
        out[name] = str(p)
    return out


def _events(tap_dir: Path) -> list[dict]:
    p = tap_dir / "events.jsonl"
    if not p.is_file():
        return []
    out = []
    for line in p.read_text(errors="replace").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def _arm(tap_dir: Path, label: str, save_steps) -> None:
    tmp = tap_dir / "ARM.tmp"
    tmp.write_text(json.dumps({"label": label, "save_steps": sorted(save_steps), "fp8_step": None}))
    os.replace(tmp, tap_dir / "ARM")


def _disarm(tap_dir: Path) -> None:
    try:
        (tap_dir / "ARM").unlink()
    except FileNotFoundError:
        pass


def run_engine(tap_dir: Path, adapters: dict, seq: dict, fixtures: dict) -> dict:
    """Build the engine with the tap, replay the ops, generate every fixture after each; returns the run record."""
    import numpy as np
    from _common import NoImageProduced, to_image_array
    scored, diag = _points(seq)
    steps = sorted({s for _, s in scored | diag})
    saved_env = {k: os.environ.get(k) for k in ("PYTHONPATH", "INFB_SD3_DAY0_TAP_DIR", "INFB_SD3_DAY0_LIB",
                                                 "TORCHDYNAMO_DISABLE", "INFB_SD3_COMPONENT_RECORD_FILE",
                                                 "INFB_SD3_OFFLOAD_OBSERVE")}
    os.environ["PYTHONPATH"] = ":".join([str(TAP)] + [q for q in (saved_env["PYTHONPATH"] or "").split(":") if q])
    if PROFILE["name"] == "offload":
        os.environ["PYTHONPATH"] = str(TAP) + ":" + str(HERE / "component_recorder") + ":" + (saved_env["PYTHONPATH"] or "")
        os.environ["INFB_SD3_COMPONENT_RECORD_FILE"] = str(tap_dir / "residency.jsonl")
        os.environ["INFB_SD3_OFFLOAD_OBSERVE"] = "1"
    os.environ["INFB_SD3_DAY0_TAP_DIR"] = str(tap_dir)
    os.environ["INFB_SD3_DAY0_LIB"] = str(DAY0)
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    rec = {"model": _resolve_model(), "tap_dir": str(tap_dir), "requests": [], "images": {}, "api_errors": [],
           "timing_s": {}, "engine_kwargs": engine_kwargs()}
    t0 = time.time()
    try:
        from sglang.multimodal_gen import DiffGenerator
        gen = DiffGenerator.from_pretrained(model_path=rec["model"], **engine_kwargs(), **_engine_ports())
    finally:
        for k, v in saved_env.items():
            if k == "TORCHDYNAMO_DISABLE":
                continue
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    rec["timing_s"]["engine_build"] = round(time.time() - t0, 1)
    active = None
    images = {}
    rec["refusals"] = {}
    try:
        for k, (op, exp) in enumerate(zip(seq["ops"], seq["expected_states"])):
            err = None
            t1 = time.time()
            try:
                if op["op"] == "set":
                    if isinstance(op["adapter"], list):
                        gen.set_lora(list(op["adapter"]), lora_path=[adapters[a] for a in op["adapter"]],
                                     strength=[float(s) for s in op["scale"]])
                    else:
                        gen.set_lora(op["adapter"], lora_path=adapters[op["adapter"]], strength=float(op["scale"]))
                    if not op.get("refused"):
                        active = op["adapter"]
                elif op["op"] == "scale":
                    if active is None or isinstance(active, list):
                        raise RuntimeError("scale op with no single adapter set")
                    gen.set_lora(active, lora_path=adapters[active], strength=float(op["scale"]))
                elif op["op"] == "disable":
                    gen.unmerge_lora_weights()
            except Exception as exc:  # noqa: BLE001 - the candidate's API refused: recorded, the request still runs
                err = f"r{k} {op['op']}: {type(exc).__name__}: {exc}"[:300]
                if not op.get("refused"):
                    rec["api_errors"].append(err)
            if op.get("refused"):

                rec["refusals"][str(k)] = {"refused": err is not None, "error": err, "adapter": op.get("adapter")}
            entry = {"request": k, "op": op, "expected": exp, "api_error": err, "fixtures": {}}
            for fid in _op_fixtures(op, seq):
                fx = fixtures[fid]
                label = f"r{k}.{fid}"
                _arm(tap_dir, label, steps)
                request_start = time.time()
                try:
                    out = gen.generate(sampling_params_kwargs=dict(
                        prompt=fx["prompt"], seed=fx["seed"], num_inference_steps=fx["steps"],
                        guidance_scale=fx["cfg"], num_frames=1, height=fx["h"], width=fx["w"],
                        return_frames=True, save_output=False))
                    img = to_image_array(out)
                    images[label] = img
                    entry["fixtures"][fid] = {"label": label, "image_sha256": hashlib.sha256(
                        np.ascontiguousarray(img).tobytes()).hexdigest(), "shape": list(img.shape)}
                    try:
                        from PIL import Image
                        IMG_DIR.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(img).save(IMG_DIR / f"{label}.png")
                    except Exception:  # noqa: BLE001 - eyeballing only
                        pass
                except NoImageProduced as exc:
                    entry["fixtures"][fid] = {"label": label, "no_image": str(exc)}
                except Exception as exc:  # noqa: BLE001
                    entry["fixtures"][fid] = {"label": label, "error": f"{type(exc).__name__}: {exc}"[:300]}
                finally:
                    entry["fixtures"][fid].update(t0=request_start, t1=time.time())
                    _disarm(tap_dir)
            entry["seconds"] = round(time.time() - t1, 1)
            rec["requests"].append(entry)
    finally:
        try:
            gen.shutdown()
        except Exception:  # noqa: BLE001
            pass
    rec["timing_s"]["requests"] = round(time.time() - t0 - rec["timing_s"]["engine_build"], 1)
    if PROFILE["name"] == "offload":
        from verify_dit_layerwise_offload import _events
        rec["residency_events"] = _events(tap_dir / "residency.jsonl")
    rec["image_pairs"] = image_pairs(images, seq)
    rec["image_pairs_multi"] = image_pairs(images, seq, "multi_equal_pairs")
    return rec


def _rel_l2(a, b) -> float:
    import numpy as np
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def image_psnr(a, b) -> float:
    """The task's pixel PSNR (verify_callsite_main.psnr, the T1 gate's): uint8 HWC RGB, data range 255, 100 dB for
    an identical pair (MSE <= 1e-9)."""
    import numpy as np
    mse = float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))
    return 100.0 if mse <= 1e-9 else float(20.0 * np.log10(255.0) - 10.0 * np.log10(mse))


def image_ssim(a, b) -> float | None:
    """The task's pixel SSIM (verify_callsite_main.ssim): skimage, per channel on uint8 RGB, data range 255, averaged.
    None where scikit-image is not importable (the pair is then not judged at the pixel layer, never passed)."""
    try:
        from skimage.metrics import structural_similarity
    except Exception:  # noqa: BLE001
        return None
    return float(structural_similarity(a, b, channel_axis=-1, data_range=255))


def image_pairs(images: dict, seq: dict, pairs_key: str = "equal_pairs") -> dict:
    """Measure repeat and sequence-pair equality, PSNR and SSIM on final uint8 fixture images."""


    import numpy as np
    out = {}
    ra, rb = seq["repeat_pair"]
    for fid in seq["fixtures"]:
        def m(i, j):
            a, b = images.get(f"r{i}.{fid}"), images.get(f"r{j}.{fid}")
            if a is None or b is None:
                return None
            return {"identical": bool(np.array_equal(a, b)), "psnr": image_psnr(a, b), "ssim": image_ssim(a, b),
                    "rel_l2": _rel_l2(a, b)}
        pairs = {f"{i}-{j}": m(i, j) for i, j in seq.get(pairs_key, []) if fid in _pair_fixtures(seq, i, j)}
        if not pairs:
            continue
        out[f"{fid}/image"] = {"repeat": m(ra, rb), "pairs": pairs,
                               "cross_regime": {f"{i}-{j}": m(i, j) for i, j in seq.get("cross_regime_pairs", [])
                                                if fid in _pair_fixtures(seq, i, j)}}
    return out


def _replay_input_digest(payload: dict) -> str:
    """Exact boundary inputs, including dtype, shape and row order. No rounded fingerprints."""
    import torch

    digest = hashlib.sha256()
    for name, value in sorted(payload.items()):
        value = value.detach().cpu().contiguous()
        meta = json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode()
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(meta).to_bytes(8, 'big'))
        digest.update(meta)
        digest.update(len(raw).to_bytes(8, 'big'))
        digest.update(raw)
    return digest.hexdigest()


def _replay_reference_jobs(todo, payload, forward, set_state, dit16, dtype16, grid,
                           max_cache_bytes=64 * 1024 * 1024):
    """Reuse only identical inputs at one reference state and precision, within this replay.

    Every request and every grid comparison is still returned. The cache is destroyed at
    each weight-state transition, never persisted or shared with another candidate/profile.
    The independent models are in eval mode, with no random sampling in these forwards.
    """
    identities = {key: _replay_input_digest(pl) for key, pl in payload.items()}
    ref, y16 = {}, {}
    stats = dict(requested=0, executed=0, cache_hits=0, peak_cache_bytes=0,
                 max_cache_bytes=max_cache_bytes, scope='exact input bytes, one state and precision, this replay only')
    for sk in sorted(todo):
        st, jobs = todo[sk]
        set_state(st)
        cache, used = {}, 0

        def read(key, kind):
            nonlocal used
            identity = (kind, identities[key])
            stats['requested'] += 1
            if identity in cache:
                stats['cache_hits'] += 1
                return cache[identity]
            value = forward(payload[key]) if kind == 'float32' else forward(payload[key], dit16, dtype16)
            stats['executed'] += 1
            size = value.numel() * value.element_size()
            if used + size <= max_cache_bytes:
                cache[identity] = value
                used += size
                stats['peak_cache_bytes'] = max(stats['peak_cache_bytes'], used)
            return value

        for key, tag in jobs:
            ref[(key, tag)] = read(key, 'float32')
            if tag == 'state' or (tag == 'base' and key[0] not in grid):
                y16[key] = read(key, 'bfloat16')
    return ref, y16, stats


def replay(manifest_path: Path) -> dict:
    """Isolated: diffusers + torch + numpy + tests/day0 only, never the candidate's tree."""
    import numpy as np
    import torch

    import diffusers
    import lora_reference as LR

    ws = os.path.abspath(os.environ.get("INFB_WORKSPACE", "/app"))
    loc = os.path.dirname(os.path.abspath(diffusers.__file__))
    if loc.startswith(ws + os.sep):
        return {"measured": False, "reason": f"diffusers resolved INSIDE the workspace ({loc})"}
    from diffusers import SD3Transformer2DModel

    man = json.loads(manifest_path.read_text())
    seq = man["sequence"]
    tap = Path(man["tap_dir"])
    scored, diag = _points(seq)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ads = {n: LR.load_adapter(p, name=n) for n, p in man["adapters"].items()}


    dit = SD3Transformer2DModel.from_pretrained(man["model"], subfolder="transformer",
                                                torch_dtype=torch.float32).to(dev).eval()
    dit16 = SD3Transformer2DModel.from_pretrained(man["model"], subfolder="transformer",
                                                  torch_dtype=torch.bfloat16).to(dev).eval()
    linears = {n: m for n, m in dit.named_modules() if isinstance(m, torch.nn.Linear)}
    linears16 = {n: m for n, m in dit16.named_modules() if isinstance(m, torch.nn.Linear)}
    touched = sorted(set().union(*(set(a.modules) for a in ads.values())))
    missing = [n for n in touched if n not in linears]
    if missing:
        return {"measured": False, "reason": f"adapter modules absent from the reference DiT: {missing[:4]}"}
    w0 = {n: linears[n].weight.detach().to("cpu", torch.float32).clone() for n in touched}

    def set_state(state: dict) -> None:


        terms = LR.state_terms(state, ads) if state.get("adapter") is not None else {}
        with torch.no_grad():
            for n in touched:
                w = w0[n].to(dev, torch.float64)
                for coef, lo in terms.get(n, []):
                    B = torch.from_numpy(lo.B).to(dev, torch.float64)
                    A = torch.from_numpy(lo.A).to(dev, torch.float64)
                    w = w + float(coef) * (B @ A)
                linears[n].weight.data.copy_(w.to(torch.float32))
                linears16[n].weight.data.copy_(w.to(torch.bfloat16))

    def frame(t):
        return t[:, :, 0] if t.dim() == 5 and t.shape[2] == 1 else t

    calls = []
    for r in man["requests"]:
        for fid in seq["fixtures"]:
            for p in sorted(tap.glob(f"r{r['request']}.{fid}.s*.p*.pt")):
                s, q = p.name[len(f"r{r['request']}.{fid}.s"):-3].split(".p")
                if (fid, int(s)) in scored | diag:
                    calls.append((r["request"], fid, int(s), int(q), p))
    if not calls:
        return {"measured": False, "reason": "the tap saved no DiT forward (was the served DiT hooked?)"}
    need = ("hidden_states", "encoder_hidden_states", "pooled_projections", "timestep", "output")


    whole = {f: tuple(int(x) for x in v) for f, v in (man.get("latent_hw") or {}).items()}
    not_whole = []

    def forward(pl, model=None, dt=torch.float32):
        model = dit if model is None else model
        return model(hidden_states=frame(pl["hidden_states"]).to(dev, dt),
                     encoder_hidden_states=pl["encoder_hidden_states"].to(dev, dt),
                     pooled_projections=pl["pooled_projections"].to(dev, dt),
                     timestep=pl["timestep"].to(dev), return_dict=False)[0].to("cpu", torch.float32)

    def st_key(st: dict) -> str:
        sc = st.get("scale")
        sc = [float(x or 0.0) for x in sc] if isinstance(sc, list) else float(sc or 0.0)
        return json.dumps({"adapter": st["adapter"], "scale": sc, "variant": st.get("variant")}, sort_keys=True)

    base_st = {"adapter": None, "scale": 0.0}
    expected = {r["request"]: r["expected"] for r in man["requests"]}


    grid = {k: g for k, e in expected.items() if (g := grid_states(seq["ops"][k], e)) or e.get("adapter") is not None}
    cand, payload, absent = {}, {}, []
    for (k, fid, s, q, p) in calls:
        pl = torch.load(p, map_location="cpu", weights_only=True)
        if any(not hasattr(pl.get(x), "shape") for x in need):
            absent.append(f"{p.name}: {[x for x in need if not hasattr(pl.get(x), 'shape')]}")
            continue
        hw = tuple(int(x) for x in frame(pl["hidden_states"]).shape[-2:])
        if fid in whole and hw != whole[fid]:
            not_whole.append(f"{p.name}: latent {list(hw)}, the fixture's whole latent is {list(whole[fid])}")
            continue
        cand[(k, fid, s, q)] = frame(pl["output"]).to(torch.float32)
        payload[(k, fid, s, q)] = {x: pl[x] for x in need if x != "output"}
    ref = {}
    y16 = {}
    with torch.no_grad():
        todo = {st_key(base_st): (base_st, [])}
        for key in cand:
            k = key[0]
            todo[st_key(base_st)][1].append((key, "base"))
            e = expected[k]
            if key[0] in grid:
                todo.setdefault(st_key(e), (e, []))[1].append((key, "state"))
                for tag, st in grid[k].items():
                    todo.setdefault(st_key(st), (st, []))[1].append((key, tag))
        ref, y16, reuse = _replay_reference_jobs(todo, payload, forward, set_state, dit16, torch.bfloat16, grid)
        set_state(base_st)

    def rel(a, b, norm):
        return float(torch.linalg.norm(a.double() - b.double()) / norm)

    readings = []
    for key, y in sorted(cand.items()):
        k, fid, s, q = key
        rb = ref[(key, "base")].double()
        rs = ref.get((key, "state"), ref[(key, "base")]).double()
        d = rs - rb
        dd = float((d * d).sum())
        ns = float(torch.linalg.norm(rs))
        rec = {"request": k, "fixture": fid, "step": s, "pass": q, "expected": expected[k],
               "scored": (fid, s) in scored,
               "err_state": float(torch.linalg.norm(y.double() - rs) / ns),
               "err_base": float(torch.linalg.norm(y.double() - rb) / torch.linalg.norm(rb)),
               "effect_ref": float(torch.linalg.norm(d) / torch.linalg.norm(rb)),
               "beta": (float(((y.double() - rb) * d).sum()) / dd) if dd > 0 else None,
               "floor16": rel(y16[key], rs, ns) if key in y16 else None,
               "timestep": None}
        if k in grid:


            refs = {"base": rb, "state": rs, **{t: ref[(key, t)].double() for t in grid[k]}}
            rec["dist"] = {t: rel(y, r_, ns) for t, r_ in refs.items()}
            rec["dist16"] = {t: rel(y16[key], r_, ns) for t, r_ in refs.items()}
            rec["ref_dist"] = {t: rel(refs[t], rs, ns) for t in refs if t != "state"}
        readings.append(rec)


    ra, rbq = seq["repeat_pair"]

    def dit_pairs_of(key_: str) -> dict:
        lv = {}
        for fid in seq["fixtures"]:
            plist = [(i, j) for i, j in seq.get(key_) or [] if fid in _pair_fixtures(seq, i, j)]
            if not plist:
                continue
            passes = sorted({q for (k, f, s, q) in cand if f == fid and s == 0})
            for q in passes:
                def m(i, j):
                    a, b = cand.get((i, fid, 0, q)), cand.get((j, fid, 0, q))
                    if a is None or b is None:
                        return None
                    return {"identical": bool(torch.equal(a, b)),
                            "rel_l2": float(torch.linalg.norm(a - b) / torch.linalg.norm(b))}
                lv[f"{fid}/dit_step0.p{q}"] = {"repeat": m(ra, rbq), "pairs": {f"{i}-{j}": m(i, j) for i, j in plist},
                                               "cross_regime": {f"{i}-{j}": m(i, j)
                                                                for i, j in seq.get("cross_regime_pairs", [])
                                                                if fid in _pair_fixtures(seq, i, j)}}
        return lv

    return {"measured": True, "device": dev, "diffusers": diffusers.__version__, "diffusers_file": loc,
            "reference_dtype": "float32 forward, float64 merge",
            "legal_port_dtype": "bfloat16 forward, float64 merge rounded to bfloat16 once", "n_calls": len(calls),
            "n_forwards": {"float32": len(ref), "bfloat16": len(y16)}, "reference_compute": reuse,
            "absent": absent, "not_whole_latent": not_whole,
            "readings": readings, "dit_pairs": dit_pairs_of("equal_pairs"),
            "dit_pairs_multi": dit_pairs_of("multi_equal_pairs")}


def main() -> dict:
    import day0_gates as G
    seq = json.loads(SEQUENCE.read_text())


    fixtures = {f["id"]: f for f in json.loads(GOLDEN_SPEC.read_text())["fixtures"]}


    if PROFILE["name"] == "offload":
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="sd3_day0_lora_",
                                dir=ARTIFACTS if PROFILE["name"] == "offload" else None))
    result = {"work": str(work), "measured": False, "sequence": seq["sequence_id"], "recipe": RECIPE.name,
              "calibrated_profile": dict(CALIBRATED_PROFILE),
              "profile": {"name": PROFILE["name"], "capability": CAPABILITY, "layer": LAYER,
                          "gates": sorted(PROFILE["gates"].values()), "sequence_file": SEQUENCE.name}}
    try:
        try:
            result["engine_profile"] = engine_kwargs()
        except ValueError as exc:
            return dict(result, reason=f"not_evaluated: verifier_defect: {exc}"[:400])
        try:
            adapters = _adapters(work / "adapters")
        except Exception as exc:  # noqa: BLE001 - ours, not the candidate's
            return dict(result, reason=f"not_evaluated: verifier_defect: {type(exc).__name__}: {exc}"[:400])
        tap_dir = work / "tap"
        tap_dir.mkdir()
        try:
            run = run_engine(tap_dir, adapters, seq, fixtures)
        except Exception as exc:  # noqa: BLE001
            from _common import save_stderr_log
            save_stderr_log("day0_lora_engine", traceback.format_exc())
            return dict(result, reason=f"the engine did not build or serve: {type(exc).__name__}: {exc}"[:400],
                        engine_failed=True)
        result["run"] = run
        ev = _events(tap_dir)
        result["tap_events"] = {e.get("event"): sum(1 for x in ev if x.get("event") == e.get("event")) for e in ev}
        if not any(e.get("event") == "hooked" for e in ev):
            return dict(result, reason="not_evaluated: the day-0 tap never hooked the served DiT")
        world = int((result.get("engine_profile") or {}).get("num_gpus") or 1)
        result["profile_reached"] = profile_reached(result["tap_events"], world)
        if not result["profile_reached"]["reached"]:
            return dict(result, reason=f"not_evaluated: the requested engine profile {PROFILE['name']} did not occur: "
                        + "; ".join(result["profile_reached"]["reasons"]))
        manifest = work / "manifest.json"
        _write_json(manifest, {"tap_dir": str(tap_dir), "model": run["model"], "adapters": adapters,
                               "sequence": seq, "requests": [{"request": r["request"], "expected": r["expected"]}
                                                             for r in run["requests"]],
                               "latent_hw": {f: [fixtures[f]["h"] // VAE_SCALE, fixtures[f]["w"] // VAE_SCALE]
                                             for f in seq["fixtures"]}})
        repo = str(Path(os.environ.get("INFB_WORKSPACE", "/app")).resolve() / "repo")
        env = dict(os.environ)
        env["PYTHONPATH"] = ":".join(q for q in (env.get("PYTHONPATH") or "").split(":") if q and not q.startswith(repo))
        rp = work / "replay.json"
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--replay", str(manifest), str(rp)], env=env,
                           capture_output=True, text=True, timeout=REPLAY_TIMEOUT_S)
        sys.stdout.write(r.stdout[-3000:])
        if r.returncode != 0 or not rp.is_file():
            return dict(result, reason=f"not_evaluated: replay exited {r.returncode}", replay_stderr=r.stderr[-3000:])
        rep = json.loads(rp.read_text())
        result["replay"] = {k: v for k, v in rep.items() if k != "readings"}
        result["readings"] = rep.get("readings", [])
        if not rep.get("measured"):
            return dict(result, reason=f"not_evaluated: {rep.get('reason')}")
        result["measured"] = True


        multi = set(seq.get("multi_adapter_requests") or [])
        frozen = frozen_request_set(seq, run)
        first_base = next((e for e in frozen if e["expected"].get("adapter") is None), None)
        result["frozen_request_set"] = frozen
        refusal_failures = [f"r{k} set {v.get('adapter')}: set_lora accepted a file the contract says is refused "
                            "before any weight changes (a DoRA / LoRA-bias variant)"
                            for k, v in sorted((run.get("refusals") or {}).items(), key=lambda kv: int(kv[0]))
                            if not v.get("refused")]
        single = [x for x in rep["readings"] if int(x["request"]) not in multi]
        result["reference_verdict"] = G.lora_reference_verdict(
            single, run["api_errors"], expected=[e for e in frozen if e["request"] not in multi],
            refusal_failures=refusal_failures)

        result["reference_v1_recorded"] = G.lora_reference_v1_readings(single, run["api_errors"])
        levels = dict(run["image_pairs"])
        levels.update(rep.get("dit_pairs") or {})
        result["cross_regime_pairs"] = {k: v.get("cross_regime") for k, v in levels.items()}
        levels = {k: {kk: vv for kk, vv in v.items() if kk != "cross_regime"} for k, v in levels.items()}
        result["isolation_verdict"] = G.lora_isolation_verdict(levels, required=frozen_pair_set(seq))
        if GATE_MULTI is None and PROFILE["name"] != "offload":
            pass
        elif multi:


            mlev = dict(run.get("image_pairs_multi") or {})
            mlev.update(rep.get("dit_pairs_multi") or {})
            mlev = {k: {kk: vv for kk, vv in v.items() if kk != "cross_regime"} for k, v in mlev.items()}
            mreq = [x for x in rep["readings"] if int(x["request"]) in multi
                    or (first_base is not None and int(x["request"]) == first_base["request"])]
            result["multi_verdict"] = G.lora_multi_adapter_verdict(
                mreq, mlev, [e for e in run["api_errors"] if int(e.split(" ", 1)[0][1:]) in multi],
                expected=[e for e in frozen if e["request"] in multi or e is first_base],
                required=frozen_pair_set(seq, "multi_equal_pairs"))
        else:
            result["multi_verdict"] = {"verdict": "not_evaluated", "reasons": [
                f"the sequence {seq.get('sequence_id')} has no multi-adapter request"]}
        if run["api_errors"] and result["isolation_verdict"]["verdict"] == "pass":
            result["isolation_verdict"] = dict(result["isolation_verdict"], verdict="fail", reasons=[
                "the LoRA API refused an op, so the pairs compare states the sequence never reached: "
                + run["api_errors"][0]])
        serve_errs = [x for e in frozen for x in e["serve_errors"]]
        if serve_errs and result["isolation_verdict"]["verdict"] != "fail":


            result["isolation_verdict"] = dict(result["isolation_verdict"], verdict="fail", reasons=[
                "the engine did not serve a request of the sequence: " + serve_errs[0]]
                + list(result["isolation_verdict"].get("reasons") or []))
        nw = rep.get("not_whole_latent") or []
        if nw:


            why = (f"{len(nw)} DiT capture(s) were not the fixture's whole latent (a port that shards before its DiT; "
                   f"the replay needs the whole image): {nw[0]}")
            for key in ("reference_verdict", "isolation_verdict", "multi_verdict"):
                v = result.get(key)
                if v and v.get("verdict") not in ("fail", "pass"):
                    result[key] = dict(v, reasons=[why] + list(v.get("reasons") or []))
        return result
    finally:
        if PROFILE["name"] != "offload":
            shutil.rmtree(work, ignore_errors=True)


def _engine_ports() -> dict:
    """Return this verifier queue's assigned engine port block."""
    from _common import engine_port_kwargs

    return engine_port_kwargs()

if __name__ == "__main__":
    if sys.argv[1:2] == ["--replay"]:
        try:
            res = replay(Path(sys.argv[2]))
        except Exception as exc:  # noqa: BLE001 - the parent reads whatever the child left
            res = {"measured": False, "reason": f"{type(exc).__name__}: {exc}"[:400], "trace": traceback.format_exc()[-2000:]}
        _write_json(Path(sys.argv[3]), res)
        sys.exit(0)
    try:
        res = main()
    except Exception as exc:  # noqa: BLE001 - the gates below are still written
        res = {"measured": False, "reason": f"not_evaluated: {type(exc).__name__}: {exc}"[:400],
               "trace": traceback.format_exc()[-2000:]}
    try:
        _write_json(OUT, res)
    except Exception:  # noqa: BLE001
        pass
    write_gates(res)
    print(f"[day0-lora:{PROFILE['name']}] {GATE_REF}: "
          f"{(res.get('reference_verdict') or {}).get('verdict', res.get('reason'))}; "
          f"{GATE_ISO}: {(res.get('isolation_verdict') or {}).get('verdict', res.get('reason'))}", flush=True)
    sys.exit(0)
