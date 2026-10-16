#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""SD3 formal TP2 / SP2 verification on two GPUs, using one dense anchor.

The SP engine also supplies CLIP captures. Every engine serves its whole
sequence before shutdown; a rank-drop arm is cleared before recovery checks.
Each configuration is evaluated through the candidate's native engine.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "day0"))
ARTIFACTS = Path(os.environ.get("INFB_ARTIFACTS_DIR", "/logs/artifacts"))
OUT = ARTIFACTS / "sd3_day0_parallel.json"
GATE_IDS = ("tensor_parallel_dit", "sequence_parallel_dit", "sp2_x_cfg_x_nondivisible",
            "lifecycle_under_parallel_tp2", "lifecycle_under_parallel_sp2")
PROFILES = {"dense": {"num_gpus": 1, "tp_size": 1, "sp_degree": 1},
            "tp2": {"num_gpus": 2, "tp_size": 2, "sp_degree": 1},
            "sp2": {"num_gpus": 2, "tp_size": 1, "sp_degree": 2, "ulysses_degree": 2}}
WAIT_S = 120


def hydrate(run):
    import numpy as np
    import native_sequence as N
    if 'events' not in run:
        run['events'] = N.events(run['record']) if Path(run.get('record', '')).is_file() else []
    run['images'] = {key: np.load(rd['image'], allow_pickle=False)
                     for key, rd in run.get('readings', {}).items() if rd.get('image')}
    return run


def supervised_engine(work, mode, reqs, model):
    import native_sequence as N
    execution = N.supervise(__file__, work/mode, ['--native-producer', work, mode, model],
                            reserve_s=15 if mode == 'dense' else 45)
    run = N.read_supervised(work/mode, execution)
    if execution.get('status') != 'completed':
        run['shutdown_ok'] = False
    return run


def requests():
    from reference_contract import request
    def req(fid, h=1024, w=1024):
        return request(fid, height=h, width=w)
    return {"a": req("astronaut_cfg1"), "b": req("astronaut"),
            "odd_rows_cfg1": req("astronaut_cfg1", 1008),
            "odd_tokens_cfg1": req("astronaut_cfg1", 1008, 1008),
            "odd_rows_cfg7": req("astronaut", 1008),
            "odd_tokens_cfg7": req("astronaut", 1008, 1008)}


def reference(work, reqs, model):
    import reference_contract
    ref = reference_contract.load_reference(work / "reference", reqs, model, ARTIFACTS)
    return {**ref, "images": {k: images[0] for k, images in ref["images"].items()}}


def persist(path, value):
    """Atomic receipt; every completed request survives a later blocked one."""
    import uuid
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str))
    os.replace(tmp, path)


def persist_run(root, run):
    persist(root / "run.json", {k: v for k, v in run.items() if k not in ("images", "events")})


def load_run(work, mode):
    from verify_cfg_parallel import _events
    import numpy as np
    root = work / mode
    path = root / "run.json"
    run = json.loads(path.read_text()) if path.is_file() else {"built": False, "build_error": "producer left no receipt"}
    run["events"] = _events(root / "record.jsonl")
    run["images"] = {key: np.load(rd["image"], allow_pickle=False) for key, rd in run.get("readings", {}).items()
                     if rd.get("image") and Path(rd["image"]).is_file()}
    return run


def owned_gate_ids(mode):
    return {"dense": (), "tp2": ("tensor_parallel_dit", "lifecycle_under_parallel_tp2"),
            "sp2": ("sequence_parallel_dit", "sp2_x_cfg_x_nondivisible", "lifecycle_under_parallel_sp2")}.get(mode, GATE_IDS)


def phase_main(mode):
    """Each configuration is a separate outer-clock-owned process segment.

    A failed TP2 segment owns only TP2 gates. SP2 runs afterward regardless;
    fixed reference/dense artifacts are reused only in this exact invocation.
    """
    import parallel_gates as G
    from verify_cfg_parallel import _resolve_model
    reqs, model = requests(), _resolve_model()
    run_id = os.environ.get("SD3_VERIFIER_RUN_ID")
    if not run_id:
        raise RuntimeError("parallel phases require the verifier invocation identity")
    if mode == "dense":
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="sd3_parallel_", dir=ARTIFACTS))
        res = {"run_id": run_id, "work": str(work), "requests": reqs, "verdicts": {}, "runs": {}, "phase": mode}
        persist(OUT, res)
        ref = reference(work, reqs, model)
        res["reference"] = {k: v for k, v in ref.items() if k != "images"}
        persist(OUT, res)
        dense = supervised_engine(work, mode, reqs, model)
        res["runs"][mode] = {k: v for k, v in dense.items() if k not in ("images", "events")}
        persist(OUT, res)
        return res
    if mode not in ("tp2", "sp2"):
        raise ValueError("unknown SD3 parallel mode: " + str(mode))
    res = json.loads(OUT.read_text()) if OUT.is_file() else {}
    if res.get("run_id") != run_id or res.get("requests") != reqs:
        raise RuntimeError("dense/reference producer missing or belongs to a different invocation")
    work = Path(res["work"])
    res["phase"] = mode
    par = supervised_engine(work, mode, reqs, model)
    res["runs"][mode] = {k: v for k, v in par.items() if k not in ("images", "events")}
    persist(OUT, res)
    import native_sequence as N
    ref = reference(work, reqs, model)
    gate = "tensor_parallel_dit" if mode == "tp2" else "sequence_parallel_dit"
    basic_ids = [gate] + (["sp2_x_cfg_x_nondivisible"] if mode == 'sp2' else [])
    try:
        dense = hydrate(N.load_checkpoint(work/'dense', 'basic', run_id))
        basic = hydrate(N.load_checkpoint(work/mode, 'basic', run_id))
        rd = {"mode": mode, "requests": reqs, "dense": dense, "parallel": basic,
              "drop": {"images": {"a": basic.get("images", {}).get("drop_a")},
                       "events": G.window(basic, "drop_a", "rank_drop_active")},
              "drop_key": "a", "reference": ref["images"]}
        res['runs'][mode]['basic_stage'] = basic['stage_manifest']
        res['runs'][mode]['basic_stage_sha256'] = basic['stage_manifest_sha256']
        res["verdicts"][gate] = G.parallel_is_real(rd, ("a", "b"))
        if mode == "sp2":
            res["verdicts"]["sp2_x_cfg_x_nondivisible"] = G.parallel_is_real(rd, tuple(k for k in reqs if k.startswith("odd_")))
    except (OSError, ValueError, KeyError) as exc:
        for key in basic_ids:
            res['verdicts'][key] = {'verdict':'not_evaluated', 'reasons':[f'completed basic stage unavailable: {exc}']}
    life = f'lifecycle_under_parallel_{mode}'
    try:
        if par.get('producer_complete') is True:
            res['verdicts'][life] = G.lifecycle_under_parallel({'parallel':hydrate(par), 'requests':reqs})
        else:
            res['verdicts'][life] = N.timeout_verdict(par, world=2)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        res['verdicts'][life] = {'verdict':'not_evaluated', 'reasons':[f'lifecycle evidence invalid: {exc}']}
    persist(OUT, res)
    return res


def run_engine(work, mode, reqs, model):
    import native_sequence as N
    import numpy as np
    from _common import NoImageProduced, to_image_array, engine_port_kwargs
    from sd3_precision_profiles import engine_kwargs
    from verify_cfg_parallel import _events
    root = work / mode
    root.mkdir(exist_ok=True)
    fault = root / "fault"
    fault.mkdir()
    cap = root / "clip"
    cap.mkdir()
    drop = root / "RANK_DROP"
    record = root / "record.jsonl"
    env = {"INFB_SD3_COMPONENT_RECORD_FILE": str(record), "INFB_SD3_PARAM_CENSUS": "1",
           "INFB_SD3_PARALLEL_EVIDENCE": "1", "INFB_SD3_PARALLEL_DROP_FILE": str(drop),
           "INFB_SD3_BATCH_OBSERVE": "1", "INFB_SD3_FAULT_DIR": str(fault),
           "INFB_SD3_CLIP_OBSERVE": "1" if mode == "sp2" else "", "INFB_SD3_CLIP_OBSERVE_DIR": str(cap),
           "PYTHONPATH": str(HERE / "component_recorder") + os.pathsep + os.environ.get("PYTHONPATH", "")}


    multistep = os.environ.get("SD3_QFE_CAPTURE") == "multistep"
    common = root / "common_input" if os.environ.get("SD3_QFE_COMMON_INPUT") == "1" or multistep else None
    if common is not None:
        common.mkdir()
        env["INFB_SD3_COMMON_INPUT_DIR"] = str(common)
    saved = {k: os.environ.get(k) for k in env}
    kwargs = {**PROFILES[mode], **engine_kwargs("mixed"), "backend": "sglang", "dist_timeout": 30}
    run = {"mode": mode, "model": model, "engine_kwargs": kwargs, "built": False,
           "verifier_run_id": os.environ.get('SD3_VERIFIER_RUN_ID'),
           "readings": {}, "images": {}, "record": str(record), "cap_dir": str(cap)}
    if common is not None:
        run["common_input_dir"] = str(common)
    persist_run(root, run)
    os.environ.update(env)
    try:
        from sglang.multimodal_gen import DiffGenerator
        gen = DiffGenerator.from_pretrained(model_path=model, **kwargs, **engine_port_kwargs())
        run["built"] = True
        persist_run(root, run)
    except Exception as exc:
        run["build_error"] = f"{type(exc).__name__}: {exc}"[:700]
        persist_run(root, run)
        return run
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def generate(key, req):
        rd = {"t0": time.time(), "raised": False}
        run['inflight'] = {'key': key, 't0': rd['t0']}
        persist_run(root, run)
        try:
            response = N.bounded(root, run, key, lambda: gen.generate(sampling_params_kwargs={**req,
                "request_id": f"parallel-{mode}-{key}", "num_frames": 1,
                "return_frames": True, "save_output": False}), seconds=WAIT_S)
            items = response if isinstance(response, list) else [response]
            image = to_image_array(items[0]) if items else None
            if image is None:
                rd["no_output"] = True
                raise NoImageProduced("no image returned")
            run["images"][key] = image
            np.save(root / f"{key}.npy", image)
            rd["image"] = str(root / f"{key}.npy")
        except NoImageProduced as exc:
            rd.update(no_output=True, error=str(exc)[:700])
        except Exception as exc:
            rd.update(raised=True, error=f"{type(exc).__name__}: {exc}"[:700])
        finally:
            rd["t1"] = time.time()
            run["readings"][key] = rd
            run.pop('inflight', None)
            persist_run(root, run)

    marker = common / "BATCH_CAPTURE" if common is not None else None

    def capture(key):

        tmp = marker.with_name(marker.name + ".tmp")
        tmp.write_text(json.dumps({"phase": "parallel", "mode": mode, "key": key, "multistep": multistep}),
                       encoding="utf-8")
        os.replace(tmp, marker)

    try:
        for key, req in reqs.items():
            if marker is not None:
                capture(key)
            generate(key, req)
        if marker is not None:
            marker.unlink(missing_ok=True)
        if mode == 'dense':
            N.checkpoint(root, run, 'basic', tuple(reqs))
        if mode != "dense":
            generate("a_repeat", reqs["a"])
            generate("invalid", {**reqs["a"], "height": 0})
            generate("after_invalid", reqs["a"])

            drop.write_text("1")
            try:
                generate("drop_a", reqs["a"])
            finally:
                drop.unlink(missing_ok=True)
            N.checkpoint(root, run, 'basic', (*reqs, 'drop_a'))
            (fault / "FAULT").write_text("parallel post-DiT recovery")
            generate("fault", reqs["a"])
            (fault / "FAULT").unlink(missing_ok=True)
            generate("after_fault", reqs["a"])
            (fault / "SLOW").write_text("0.4")
            start = time.time()
            thread = threading.Thread(target=generate, args=("cancel", reqs["a"]), daemon=True)
            thread.start()
            try:
                active = False
                deadline = time.monotonic() + WAIT_S
                while thread.is_alive() and time.monotonic() < deadline:
                    active = any(e.get("event") == "batch_dit_start" and float(e.get("t", 0)) >= start
                                 for e in _events(record))
                    if active:
                        break
                    time.sleep(0.05)
                run["abort"] = {"t": time.time(), "active_forward": active}
                if active:
                    try:
                        N.bounded(root, run, 'abort', lambda: gen.abort(f"parallel-{mode}-cancel"), seconds=WAIT_S)
                    except Exception as exc:
                        run["abort"]["error"] = str(exc)[:500]
            finally:
                (fault / "SLOW").unlink(missing_ok=True)
            thread.join(WAIT_S)
            if thread.is_alive():
                raise TimeoutError("cancelled request never replied; further requests were not started")
            generate("after_cancel", reqs["a"])
    except Exception as exc:
        run["sequence_error"] = f"{type(exc).__name__}: {exc}"[:700]
    finally:
        for arm in (drop, fault / "SLOW", fault / "FAULT", *((marker,) if marker is not None else ())):
            arm.unlink(missing_ok=True)
        try:
            N.bounded(root, run, 'shutdown', gen.shutdown, seconds=N.SHUTDOWN_TIMEOUT_S)
            run["shutdown_ok"] = True
        except Exception as exc:
            run.update(shutdown_ok=False, shutdown_error=str(exc)[:500])
        run["events"] = _events(record)
        run["producer_complete"] = True
        persist_run(root, run)
    return run


def main():
    if os.environ.get("SD3_PARALLEL_MODE"):
        return phase_main(os.environ["SD3_PARALLEL_MODE"])
    import parallel_gates as G
    from verify_cfg_parallel import _resolve_model
    reqs = requests()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="sd3_parallel_", dir=ARTIFACTS))
    model = _resolve_model()
    res = {"run_id": os.environ.get("SD3_VERIFIER_RUN_ID"), "work": str(work), "requests": reqs, "verdicts": {}, "runs": {}}
    try:
        ref = reference(work, reqs, model)
        res["reference"] = {k: v for k, v in ref.items() if k != "images"}
        dense = run_engine(work, "dense", reqs, model)
        res["runs"]["dense"] = {k: v for k, v in dense.items() if k not in ("images", "events")}
        for mode, gate in (("tp2", "tensor_parallel_dit"), ("sp2", "sequence_parallel_dit")):
            par = run_engine(work, mode, reqs, model)
            res["runs"][mode] = {k: v for k, v in par.items() if k not in ("images", "events")}
            rd = {"mode": mode, "requests": reqs, "dense": dense, "parallel": par,
                  "drop": {"images": {"a": par.get("images", {}).get("drop_a")},
                           "events": G.window(par, "drop_a", "rank_drop_active")},
                  "drop_key": "a", "reference": ref["images"]}
            res["verdicts"][gate] = G.parallel_is_real(rd, ("a", "b"))
            res["verdicts"][f"lifecycle_under_parallel_{mode}"] = G.lifecycle_under_parallel(rd)
            if mode == "sp2":
                res["verdicts"]["sp2_x_cfg_x_nondivisible"] = G.parallel_is_real(rd, tuple(k for k in reqs if k.startswith("odd_")))
        return res
    finally:
        OUT.write_text(json.dumps(res, indent=2, default=str))


def write_gates(res):
    from _common import GateResult, write_gate
    import parallel_gates as G
    for gate in owned_gate_ids(os.environ.get("SD3_PARALLEL_MODE")):
        v = res.get("verdicts", {}).get(gate) or {"verdict": "not_evaluated", "reasons": [res.get("error", "not run")]}
        ok = v["verdict"] == "pass"
        write_gate(GateResult(gate=gate, passed=ok, score=float(ok), is_hard_gate=False,
            level="T3" if gate in ("tensor_parallel_dit", "sequence_parallel_dit") else "T4",
            error=None if ok else "; ".join(v.get("reasons", []))[:600],
            details={"verdict": v, "policy": G.POLICY, "policy_state": G.POLICY_STATE,
                     "policy_basis": G.POLICY_BASIS, "artifact": str(OUT), "base_profile": "mixed"}))


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == '--native-producer':
        run_engine(Path(sys.argv[2]), sys.argv[3], requests(), sys.argv[4])
        raise SystemExit(0)
    try:
        result = main()
    except Exception as exc:
        result = json.loads(OUT.read_text()) if OUT.is_file() else {}
        result.update(error=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc()[-3000:])
        mode = os.environ.get("SD3_PARALLEL_MODE")
        result.setdefault("phase_errors", {})[mode or "combined"] = result["error"]
        for gate in owned_gate_ids(mode):
            result.setdefault("verdicts", {})[gate] = {"verdict": "not_evaluated", "reasons": [result["error"]]}
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(result, indent=2))
    write_gates(result)
