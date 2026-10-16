#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
'SD3 verify dynamic batching utilities.'
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
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
TP_SIZE = int(os.environ.get("SD3_BATCH_TP_SIZE", "1"))
if TP_SIZE not in (1, 2):
    raise ValueError("SD3_BATCH_TP_SIZE must be 1 or 2")
SUFFIX = "_tp2" if TP_SIZE == 2 else ""
OUT = ARTIFACTS / f"sd3_day0_batching{SUFFIX}.json"
IMG_DIR = ARTIFACTS / f"sd3_day0_batching{SUFFIX}"
GOLDEN_SPEC = HERE / "sd3_golden.json"
LORA_RECIPE = DAY0 / "lora_recipe_v3.json"

ADAPTER_FENCE_REQUIRED = False
LORA_SHAPES = DAY0 / "sd3m_dit_linear_shapes.json"
LAYER = "T4"
PROFILES = {1: ("dynamic_batching_is_real", "batching_x_multioutput_isolation"),
            2: ("dynamic_batching_is_real_tp2", "batching_x_multioutput_isolation_tp2")}
GATE_BATCHING, GATE_ISOLATION = PROFILES[TP_SIZE]
GATE_IDS = ("dynamic_batching_is_real", "batching_x_multioutput_isolation",
            "dynamic_batching_is_real_tp2", "batching_x_multioutput_isolation_tp2")
CAPABILITY = {GATE_BATCHING: "tp2_x_batching" if TP_SIZE == 2 else "dynamic_batching",
              GATE_ISOLATION: "tp2_x_batching" if TP_SIZE == 2 else "batching_x_multioutput_isolation"}


MAX_SIZE = 3
DELAY_MS = 1500.0
STAGGER_S = 0.03
SLOW_S = 0.4
WAIT_S = 120.0
BASE_PROFILE = "mixed"


def _requests() -> dict:
    from reference_contract import request as q

    reqs = {
        "g7_astronaut": q("astronaut"),
        "g7_ramen": q("ramen"),
        "g7_mountain2": q("mountain", num_outputs_per_prompt=2),
        "g1_astronaut": q("astronaut_cfg1"),
        "g1_stainedglass": q("stainedglass_cfg1"),
        "c_candles": q("candles", cfg=1.0),
        "c_origami": q("origami", cfg=1.0),
        "c_teapot": q("teapot", cfg=1.0),
        "blocker": q("redlacquer", cfg=1.0),
        "x1_before": q("teapot", seed=311, cfg=1.0),
        "x2_after": q("teapot", seed=312, cfg=1.0),
    }
    if UNIQUE_BURST:


        reqs["g7_ramen"] = q("ramen", seed=4202)
        reqs["g1_astronaut"] = q("teapot", seed=4301, cfg=1.0)
    return reqs


GROUPS = {"G7": {"members": ["g7_astronaut", "g7_ramen", "g7_mountain2"], "rows": 4, "branches": 2},
          "G1": {"members": ["g1_astronaut", "g1_stainedglass"], "rows": 2, "branches": 1}}
ROLES = {"multioutput": "g7_mountain2", "same_seed": ["g7_astronaut", "g7_ramen"]}


MULTISTEP = os.environ.get("SD3_QFE_CAPTURE") == "multistep"
UNIQUE_BURST = MULTISTEP and os.environ.get("SD3_QFE_UNIQUE_BURST") == "1"
MULTISTEP_KEYS = frozenset(GROUPS["G7"]["members"] + GROUPS["G1"]["members"])


def mark(tap, plan):
    """The capture marker, replaced atomically between requests (never while a forward reads it)."""
    tmp = tap / "BATCH_CAPTURE.tmp"
    tmp.write_text(json.dumps(plan), encoding="utf-8")
    os.replace(tmp, tap / "BATCH_CAPTURE")


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


def _forwards(record: Path, t0: float, t1: float | None = None, event="batch_dit", rank=0) -> list:
    return [e for e in _events(record) if e.get("event") == event and int((e.get("rank") or {}).get("dist_rank", 0)) == rank
            and float(e.get("t") or 0) >= t0
            and (t1 is None or float(e.get("t") or 0) <= t1)]


def _engine_kwargs() -> dict:
    import sd3_precision_profiles as P

    return dict(num_gpus=TP_SIZE, tp_size=TP_SIZE, sp_degree=1, dist_timeout=30, backend="sglang", batching_max_size=MAX_SIZE, batching_delay_ms=DELAY_MS, **P.engine_kwargs(BASE_PROFILE))


def _adapter(work: Path) -> str:
    """Adapter A of the frozen LoRA recipe (tests/day0/sd3_lora_synth.py), checked against its frozen hash."""
    import hashlib

    import sd3_lora_synth as S

    out = work / "lora"
    if S.main(["generate", "--out", str(out), "--recipe", str(LORA_RECIPE), "--shapes", str(LORA_SHAPES)]) != 0:
        raise RuntimeError("adapter generation failed")
    recipe = json.loads(LORA_RECIPE.read_text())
    p = out / "A" / S.WEIGHT_NAME
    got = hashlib.sha256(p.read_bytes()).hexdigest()
    if got != recipe["expected"]["A"]["file_sha256"]:
        raise RuntimeError(f"adapter A hashes to {got}, the recipe froze {recipe['expected']['A']['file_sha256']}")
    return str(p)


def run_engine(work: Path, reqs: dict) -> dict:
    """Build the engine, serve the sequence, shut it down. Never raises past a reading."""
    import numpy as np
    import native_sequence as N
    import batching_lifecycle as L
    from _common import NoImageProduced, to_image_array

    record, fault_dir, tap = work / "record.jsonl", work / "fault", work / "tap"
    fault_dir.mkdir(parents=True, exist_ok=True)
    tap.mkdir(parents=True, exist_ok=True)
    keys = ("PYTHONPATH", "INFB_SD3_COMPONENT_RECORD_FILE", "INFB_SD3_FAULT_DIR", "INFB_SD3_COMPONENT_TAP_DIR",
            "INFB_SD3_BATCH_OBSERVE", "INFB_SD3_PARAM_CENSUS", "INFB_SD3_PARALLEL_EVIDENCE", "INFB_SD3_COMPUTE_EVERY_FORWARD")
    saved = {k: os.environ.get(k) for k in keys}
    os.environ["PYTHONPATH"] = ":".join([str(RECORDER)] + [q for q in (saved["PYTHONPATH"] or "").split(":") if q])
    os.environ.update({"INFB_SD3_COMPONENT_RECORD_FILE": str(record), "INFB_SD3_FAULT_DIR": str(fault_dir),
                       "INFB_SD3_COMPONENT_TAP_DIR": str(tap), "INFB_SD3_BATCH_OBSERVE": "1", "INFB_SD3_PARAM_CENSUS": "1", "INFB_SD3_PARALLEL_EVIDENCE": "1", "INFB_SD3_COMPUTE_EVERY_FORWARD": "1"})
    run = {"model": _resolve_model(), "engine_kwargs": _engine_kwargs(), "readings": {}, "timing_s": {},
           "record": str(record), "built": False, 'verifier_run_id':os.environ.get('SD3_VERIFIER_RUN_ID')}
    N.save_run(work, run)
    t0 = time.time()
    try:
        from sglang.multimodal_gen import DiffGenerator
        with L.operation(work, run, 'engine_build', kind='setup'):
            gen = DiffGenerator.from_pretrained(model_path=run["model"], **run["engine_kwargs"], **_engine_ports())
        run["built"] = True
        N.save_run(work, run)
    except Exception as exc:  # noqa: BLE001
        run["build_error"] = f"{type(exc).__name__}: {exc}"[:600]
        run["trace"] = traceback.format_exc()[-2000:]
        N.save_run(work, run)
        return run
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        run["timing_s"]["engine_build"] = round(time.time() - t0, 1)
    images: dict = {}
    lock = threading.RLock()

    def ask(tag: str, key: str, **extra) -> dict:
        kw = dict(reqs[key], **extra)
        t = time.time()
        r = {"tag": tag, "key": key, "t0": t, "raised": False, "ok": False, "n": 0}
        try:
            out = gen.generate(sampling_params_kwargs=dict(num_frames=1, return_frames=True, save_output=False, **kw))
            items = out if isinstance(out, list) else [out]
            imgs = []
            for o in items:
                try:
                    imgs.append(to_image_array(o))
                except NoImageProduced:
                    imgs.append(None)
            r.update(n=len(items), ok=bool(imgs) and all(i is not None for i in imgs), returned=type(out).__name__)
            r['image_files'] = []
            with lock:
                images[f"{tag}/{key}"] = imgs
            for j, img in enumerate(imgs):
                path = work/f'{tag}.{key}.{j}.npy'
                if img is not None:
                    np.save(path, img)
                r['image_files'].append(str(path) if img is not None else None)
                if img is not None:
                    try:
                        from PIL import Image
                        IMG_DIR.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(img).save(IMG_DIR / f"{tag}.{key}.{j}.png")
                    except Exception:  # noqa: BLE001 - eyeballing only
                        pass
        except Exception as exc:  # noqa: BLE001 - a failed request is a reading
            r.update(raised=True, type=type(exc).__name__, error=f"{type(exc).__name__}: {exc}"[:600])
            with lock:
                images[f"{tag}/{key}"] = []
        r["t1"] = time.time()
        with lock:
            run["readings"][f"{tag}/{key}"] = r
            N.save_run(work, run)
        return r

    def single(tag, key, **extra):
        return L.bounded(work, run, tag+'/'+key, lambda: ask(tag,key,**extra), seconds=WAIT_S)

    def burst(tag: str, members: list, extra: dict | None = None) -> list:
        ths = []
        for i, key in enumerate(members):
            th = threading.Thread(target=ask, args=(tag, key), kwargs=dict((extra or {}).get(key) or {}), daemon=True)
            th.start()
            ths.append(th)
            time.sleep(STAGGER_S)
        return ths

    def join(ths, tag) -> None:
        with L.operation(work, run, tag, seconds=WAIT_S):
            t0 = time.time()
            deadline = time.monotonic() + WAIT_S
            for th in ths:
                th.join(max(0.0, deadline - time.monotonic()))
            if any(th.is_alive() for th in ths):
                N.expire(work, run, tag, t0, WAIT_S)

    def wait_rows(t_from: float, rows: int, ths, name) -> bool:
        return L.wait_rows(work, run, name,
            lambda: any(int(e.get("rows") or 0) >= rows
                        for e in _forwards(record, t_from, event="batch_dit_start")),
            ths, seconds=WAIT_S)

    try:
        run["t_solo"] = time.time()
        if ADAPTER_FENCE_REQUIRED:
            adapter = _adapter(work)
            run["adapter"] = adapter
            gen.set_lora("A", lora_path=adapter, strength=1.0)
            single("solo", "x2_after")
            gen.unmerge_lora_weights()


            run["solo_base_regime"] = "after_adapter_unmerge"
        else:
            run["solo_base_regime"] = "untouched_base_no_adapter"
            run["excluded_clauses"] = ["adapter_fence"]
        (tap / "BATCH_CAPTURE").write_text(json.dumps({"phase": "solo"}), encoding="utf-8")
        for key in reqs:
            if key != "x2_after" or not ADAPTER_FENCE_REQUIRED:
                if MULTISTEP:

                    mark(tap, {"phase": "solo", "key": key, "multistep": key in MULTISTEP_KEYS})
                single("solo", key)
        if ADAPTER_FENCE_REQUIRED:
            single("base", "x2_after")
        run["t_burst"] = time.time()
        (tap / "BATCH_CAPTURE").write_text(json.dumps({"phase": "burst"}), encoding="utf-8")
        if MULTISTEP:
            mark(tap, {"phase": "burst", "key": "burst", "multistep": True, "members": sorted(MULTISTEP_KEYS)})
        join(burst("burst", GROUPS["G7"]["members"] + GROUPS["G1"]["members"]), 'burst')
        run["t_burst_end"] = time.time()
        (tap / "BATCH_CAPTURE").unlink(missing_ok=True)
        N.checkpoint(work, run, 'basic', tuple('solo/'+k for k in reqs) +
                     tuple('burst/'+k for g in GROUPS.values() for k in g['members']))
        import batching_gates as G
        observed_solo = solo_readings(record, run["readings"], images, reqs)
        plan = G.consumption_plan(_forwards(record, run["t_burst"], run["t_burst_end"]), observed_solo, GROUPS)
        (tap / "BATCH_ARM").write_text(json.dumps(plan), encoding="utf-8")
        try:
            run["t_armed"] = time.time()
            join(burst("armed", GROUPS["G7"]["members"] + GROUPS["G1"]["members"]), 'armed')
            run["t_armed_end"] = time.time()
        finally:
            (tap / "BATCH_ARM").unlink(missing_ok=True)
        run["armed_plan"] = plan

        (fault_dir / "SLOW").write_text(str(SLOW_S), encoding="utf-8")
        try:
            run["t_cancel"] = t_c = time.time()
            ths = burst("cancel", ["c_candles", "c_origami", "c_teapot"],
                        {"c_origami": {"request_id": "infb-batch-cancel-origami"}})
            fused = wait_rows(t_c, 2, ths, 'cancel.wait_fused_rows')
            t_abort = time.time()
            try:
                answer = L.bounded(work, run, 'abort', lambda: gen.abort("infb-batch-cancel-origami"), seconds=WAIT_S)
                run["cancel_abort"] = {"t": t_abort, "fused_seen_before": fused,
                                       "answer": answer if isinstance(answer, (dict, str, bool, int, type(None)))
                                       else repr(answer)[:200]}
            except Exception as exc:  # noqa: BLE001
                run["cancel_abort"] = {"t": t_abort, "fused_seen_before": fused, "error": f"{type(exc).__name__}: {exc}"}
            join(ths, 'cancel')
        finally:
            (fault_dir / "SLOW").unlink(missing_ok=True)
        run["t_cancel_end"] = time.time()
        single("cancel_after", "g1_astronaut")
        N.checkpoint(work, run, 'before_failure', tuple(run['readings']))

        (fault_dir / "FAULT").write_text("dynamic batching: fused forward", encoding="utf-8")
        try:
            run["t_failure"] = time.time()
            join(burst("failure", ["c_candles", "c_teapot"]), 'failure')
        finally:
            (fault_dir / "FAULT").unlink(missing_ok=True)
        run["t_failure_end"] = time.time()
        single("failure_after", "g1_astronaut")
        if ADAPTER_FENCE_REQUIRED:

            (fault_dir / "SLOW").write_text(str(SLOW_S), encoding="utf-8")
            try:
                run["t_fence"] = t_f = time.time()
                ths = burst("fence", ["blocker"])
                wait_rows(t_f, 1, ths, 'fence.wait_blocker_rows')
                ths += burst("fence", ["x1_before"])
                fence_set = {}

                def set_adapter():
                    fence_set["t0"] = time.time()
                    try:
                        gen.set_lora("A", lora_path=adapter, strength=1.0)
                        fence_set["ok"] = True
                    except Exception as exc:  # noqa: BLE001
                        fence_set["error"] = f"{type(exc).__name__}: {exc}"[:300]
                    fence_set["t1"] = time.time()
                th = threading.Thread(target=set_adapter, daemon=True)
                th.start()
                ths.append(th)
                time.sleep(STAGGER_S)
                run["t_fence_x2_sent"] = time.time()
                ths += burst("fence", ["x2_after"])
                time.sleep(STAGGER_S)
                (fault_dir / "SLOW").unlink(missing_ok=True)
                join(ths, 'fence')
                run["fence_set_lora"] = fence_set
            finally:
                (fault_dir / "SLOW").unlink(missing_ok=True)
            run["t_fence_end"] = time.time()
            try:
                gen.unmerge_lora_weights()
            except Exception:  # noqa: BLE001
                pass
    except Exception as exc:  # noqa: BLE001 - the readings so far stand
        run["sequence_error"] = f"{type(exc).__name__}: {exc}"[:600]
        run["sequence_trace"] = traceback.format_exc()[-2000:]
    finally:
        try:
            L.bounded(work, run, 'shutdown', gen.shutdown, seconds=N.SHUTDOWN_TIMEOUT_S)
            run['shutdown_ok'] = True
        except Exception as exc:  # noqa: BLE001
            run.update(shutdown_ok=False, shutdown_error=str(exc)[:500])
    run["timing_s"]["requests"] = round(time.time() - t0 - run["timing_s"]["engine_build"], 1)
    run["images"] = images
    run["events"] = _events(record)
    run["image_shapes"] = {k: [None if i is None else list(np.asarray(i).shape) for i in v] for k, v in images.items()}
    run['producer_complete'] = True
    N.save_run(work, run)
    return run


def run_reference(work: Path, reqs: dict) -> dict:
    """Reuse the fixed serial diffusers pool across both batching topologies.

    x2_after is a LoRA state-control request, independently checked by the LoRA
    verifier. Its native state-control image is never put into the base pool.
    """
    import reference_contract
    base = {k: v for k, v in reqs.items() if k != "x2_after" or not ADAPTER_FENCE_REQUIRED}
    return reference_contract.load_reference(work / "reference", base, _resolve_model(), ARTIFACTS, BASE_PROFILE)


def solo_readings(rec, readings, images, reqs):
    import t2_gates as T2

    out = {}
    for key in reqs:
        rd = readings.get(f"solo/{key}") or {}
        fw = _forwards(rec, rd.get("t0", 0), rd.get("t1")) if rd.get("t0") else []
        out[key] = {"images": images.get(f"solo/{key}") or [], "first": fw[0] if fw else None,
                    "forwards": fw, "guidance": reqs[key]["guidance_scale"], "state": T2.request_state(rd)}
    return out


def collect(run: dict, ref: dict, reqs: dict) -> dict:
    """The judges' readings (tests/batching_gates.py) from one engine run."""
    import t2_gates as T2
    import batching_gates as G

    rec = Path(run.get("record") or "")
    imgs, rd = run.get("images") or {}, run.get("readings") or {}

    solo = solo_readings(rec, rd, imgs, reqs)

    def round_of(tag, t0, t1):
        return {"forwards": _forwards(rec, run.get(t0, 0), run.get(t1)) if run.get(t0) else [],
                "requests": {k.split("/", 1)[1]: {"images": v, "state": T2.request_state(rd.get(k))}
                             for k, v in imgs.items() if k.startswith(tag + "/")}}

    burst, armed = round_of("burst", "t_burst", "t_burst_end"), round_of("armed", "t_armed", "t_armed_end")
    armed["plan"] = run.get("armed_plan")
    ab = run.get("cancel_abort") or {}
    cancel = {"fused_at_abort": bool(ab.get("fused_seen_before") and
                  (rd.get("cancel/c_origami") or {}).get("t1", 0) >= ab.get("t", float("inf"))),
              "expected_survivors": ["c_candles", "c_teapot"],
              "aborted_state": T2.request_state(rd.get("cancel/c_origami")),
              "survivors": {k: imgs.get(f"cancel/{k}") or [] for k in ("c_candles", "c_teapot")},
              "after": (imgs.get("cancel_after/g1_astronaut") or [None])[0],
              "after_key": "g1_astronaut", "after_solo": (imgs.get("solo/g1_astronaut") or [None])[0], "abort": ab}
    cancel["observed_stop"] = G.cancelled_rows_evidence(
        _forwards(rec, run.get("t_cancel", 0), run.get("t_cancel_end")),
        {key: solo[key] for key in ("c_candles", "c_origami", "c_teapot")}, "c_origami",
        ab.get("t"), run.get("t_cancel_end"))
    fired = [e for e in (run.get("events") or []) if e.get("event") == "fault_fired"
             and float(e.get("t") or 0) >= float(run.get("t_failure") or 0)]
    failure = {"expected_members": ["c_candles", "c_teapot"], "fired_rows": max((int(e.get("rows") or 0) for e in fired), default=0),
               "members": {k: {**T2.request_state(rd.get(f"failure/{k}")), "images": imgs.get(f"failure/{k}") or [],
                               "error": (rd.get(f"failure/{k}") or {}).get("error")} for k in ("c_candles", "c_teapot")},
               "after": (imgs.get("failure_after/g1_astronaut") or [None])[0],
               "after_key": "g1_astronaut", "after_solo": (imgs.get("solo/g1_astronaut") or [None])[0]}
    fence_fw = _forwards(rec, run.get("t_fence", 0), run.get("t_fence_end")) if run.get("t_fence") else []
    x1 = rd.get("fence/x1_before") or {}
    att = [G.attribute(f, solo) for f in fence_fw]
    x1_start = next((float(f.get("t_start") or f.get("t") or 0) for f, a in zip(fence_fw, att) if any(x and x[0] == "x1_before" for x in a)),
                    None)
    fence = {"queued_behind": bool(x1_start and run.get("t_fence_x2_sent") and run["t_fence_x2_sent"] < x1_start),
             "shared_forward": any({"x1_before", "x2_after"} <= {x[0] for x in a if x} for a in att),
             "before_image": (imgs.get("fence/x1_before") or [None])[0],
             "before_solo": (imgs.get("solo/x1_before") or [None])[0],
             "after_image": (imgs.get("fence/x2_after") or [None])[0],
             "after_solo": (imgs.get("solo/x2_after") or [None])[0],
             "adapter_effect_db": G.psnr((imgs.get("solo/x2_after") or [None])[0],
                                         (imgs.get("base/x2_after") or [None])[0]),
             "set_lora": run.get("fence_set_lora"), "x1_first_forward_t": x1_start,
             "x2_sent_t": run.get("t_fence_x2_sent"), "x1_state": T2.request_state(x1)}
    steps = {r["num_inference_steps"] for r in reqs.values()}
    if len(steps) != 1:
        raise ValueError("Batching fixture groups require one common step count")
    return {"steps": steps.pop(), "requests": reqs, "solo": solo, "groups": GROUPS, "roles": ROLES, "burst": burst, "armed": armed,
            "cancel": cancel, "failure": failure, "fence": fence, "reference": (ref or {}).get("images") or {}}


def parallel_evidence(run: dict) -> dict:
    from parallel_gates import rank_of
    lo, hi = run.get("t_burst", 0), run.get("t_burst_end", 0)
    events = run.get("events") or []
    ranks = {}
    for rank in range(TP_SIZE):
        own = [e for e in events if rank_of(e) == rank and lo <= float(e.get("t", 0)) <= hi]
        ranks[str(rank)] = {
            "compute": [e for e in own if e.get("event") == "compute_census"],
            "forwards": [e for e in own if e.get("event") == "dit_call"],
            "batch": [e for e in own if e.get("event") == "batch_dit"]}
    return {"ranks": ranks}


def hydrate(run):
    import numpy as np
    import native_sequence as N
    if 'events' not in run:
        run['events'] = N.events(run['record'])
    run['images'] = {key:[np.load(p, allow_pickle=False) if p else None for p in rd.get('image_files',[])]
                     for key,rd in run.get('readings',{}).items()}
    return run


def main() -> dict:

    reqs = _requests()
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="evidence_", dir=IMG_DIR))
    result = {"run_id": os.environ.get("SD3_VERIFIER_RUN_ID"), "measured": False, "requests": reqs, "groups": GROUPS, "engine_kwargs": None, "work": str(work)}
    _write_json(OUT, result)
    try:
        import batching_gates as G
        import native_sequence as N
        execution = N.supervise(__file__, work, ['--native-producer', work], reserve_s=120)
        run = N.read_supervised(work, execution)
        result["engine_kwargs"] = run.get("engine_kwargs")
        result["run"] = {k: v for k, v in run.items() if k not in ("images", "events")}
        if run.get('built') is not True:
            if not run.get('build_error'):
                result['reason'] = 'producer did not complete engine construction; no candidate attribution'
                return result
            v = {"verdict": "fail", "reasons": [f"the engine did not build with {run.get('engine_kwargs')}: "
                                                f"{run.get('build_error')}"], "details": {}}
            import qfe_typed
            v["details"]["qfe"] = qfe_typed.block({"engine_built": "fail"}, [], [], [], "fail")
            result["verdicts"] = {GATE_BATCHING: v, GATE_ISOLATION: v}
            return result
        ref = run_reference(work, reqs)
        result["reference"] = {k: v for k, v in ref.items() if k != "images"}
        result["measured"] = True
        result['verdicts'] = {}
        try:
            basic = hydrate(N.load_checkpoint(work, 'basic', result['run_id']))


            result['verdicts'][GATE_ISOLATION] = G.batching_x_multioutput_isolation(
                collect(basic, ref, reqs), require_adapter_fence=ADAPTER_FENCE_REQUIRED)
            result['parallel_evidence'] = parallel_evidence(basic)
            result['basic_stage'] = basic['stage_manifest']
        except (OSError, ValueError, KeyError) as exc:
            import qfe_typed
            result['verdicts'][GATE_ISOLATION] = {'verdict':'not_evaluated',
                'reasons':[f'complete batching stage unavailable: {exc}'],
                'details':{'qfe': qfe_typed.block({'basic_stage': 'not_evaluated'}, [], [], [], 'not_evaluated')}}
            result['parallel_evidence'] = {}
        if run.get('producer_complete') is True and execution.get('status') == 'completed':
            result['verdicts'][GATE_BATCHING] = G.dynamic_batching_is_real(collect(hydrate(run),ref,reqs))
        else:
            import batching_lifecycle as L
            import qfe_typed
            result['verdicts'][GATE_BATCHING] = L.interruption_verdict(work, run, world=TP_SIZE)


            try:
                result['verdicts'][GATE_BATCHING]['details']['partial_reading'] = G.dynamic_batching_is_real(
                    collect(hydrate(run),ref,reqs))
            except (OSError, ValueError, KeyError) as exc:
                result['verdicts'][GATE_BATCHING]['details']['partial_error'] = str(exc)


            iv = result['verdicts'][GATE_BATCHING]
            prefix = ((iv['details'].get('partial_reading') or {}).get('details') or {}).get('qfe') or {}
            structural = {**(prefix.get('structural') or {}), 'producer_completed': iv['verdict']
                          if iv['verdict'] in ('fail', 'not_evaluated') else 'fail'}
            iv['details']['qfe'] = qfe_typed.block(structural, prefix.get('comparisons') or [],
                                                   prefix.get('obligations') or [], prefix.get('pairs') or [],
                                                   iv['verdict'], prefix.get('readouts'))
        if TP_SIZE == 2:
            dense_path = ARTIFACTS / "sd3_day0_batching.json"
            dense = json.loads(dense_path.read_text()) if dense_path.is_file() else {}
            same = bool(result["run_id"] and dense.get("run_id") == result["run_id"] and dense.get("requests") == reqs)
            pv = G.tp2_execution(result["parallel_evidence"], dense.get("parallel_evidence") if same else None)
            result["tp2_execution"] = pv
            if pv["verdict"] != "pass":
                for v in result["verdicts"].values():
                    v["details"]["tp2_execution"] = pv
                    if v["verdict"] != "fail":
                        v["verdict"] = pv["verdict"]
                    v["reasons"] += pv.get("reasons", [])

            import qfe_typed
            for v in result["verdicts"].values():
                qfe_typed.add_structural((v.get("details") or {}).get("qfe"), "tp2_execution", pv["verdict"], v["verdict"])
        result["batch_dit_lines"] = len([e for e in run.get("events") or [] if e.get("event") == "batch_dit"])
        return result
    finally:

        result["evidence_dir"] = str(work)


def write_gates(result: dict) -> None:
    import batching_gates as BG
    import day0_gates as DG
    from _common import GateResult, write_gate

    missing = {"verdict": "not_evaluated", "reasons": [result.get("reason", "not run")]}
    for gate in GATE_IDS:
        if gate not in PROFILES[TP_SIZE]:
            continue
        v = (result.get("verdicts") or {}).get(gate) or missing
        ok = v.get("verdict") == "pass"
        write_gate(GateResult(gate=gate, passed=ok, score=1.0 if ok else 0.0, is_hard_gate=False, level=LAYER,
                              error=None if ok else "; ".join(v.get("reasons", []))[:500] or v.get("verdict"),
                              details=DG.gate_details(CAPABILITY[gate], LAYER, v, dict(BG.POLICY), artifact=str(OUT),
                                                      policy_state=BG.POLICY_STATE, policy_basis=dict(BG.POLICY_BASIS),
                                                      base_profile=BASE_PROFILE, engine_kwargs=result.get("engine_kwargs"),
                                                      qfe=(v.get("details") or {}).get("qfe"))))


def _engine_ports() -> dict:
    """Return this verifier queue's assigned engine port block."""
    from _common import engine_port_kwargs

    return engine_port_kwargs()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == '--native-producer':
        run_engine(Path(sys.argv[2]), _requests())
        raise SystemExit(0)
    try:
        res = main()
    except Exception as exc:  # noqa: BLE001
        res = {"measured": False, "reason": f"not_evaluated: {type(exc).__name__}: {exc}"[:400],
               "trace": traceback.format_exc()[-2000:]}
    try:
        _write_json(OUT, res)
    except Exception:  # noqa: BLE001
        pass
    write_gates(res)
    for g, v in (res.get("verdicts") or {}).items():
        print(f"[day0-batching] {g}: {v.get('verdict')}; " + "; ".join(v.get("reasons") or [])[:600], flush=True)
    if not res.get("verdicts"):
        print(f"[day0-batching] not run: {res.get('reason')}", flush=True)
    sys.exit(0)
