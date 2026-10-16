#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
'SD3 verify t2 lifecycle utilities.'


from __future__ import annotations

import json
import hashlib
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
OUT = ARTIFACTS / "sd3_day0_t2.json"
IMG_DIR = ARTIFACTS / "sd3_day0_t2"
LAYER = "T2"
GATES = ("request_isolation_abab", "error_recovery", "in_request_batching", "cfg1_single_row", "prompt_semantics",
         "request_cancellation")
CAPABILITY = {g: g for g in GATES}
GOLDEN_SPEC = Path(os.environ.get("SD3_DAY0_GOLDEN_SPEC", HERE / "sd3_golden.json"))
NEGATIVE_PROMPT = "blurry, low quality, watermark, text, deformed"

TAIL = " and far behind them a second moon of polished brass rises over a city of glass towers"

CANCEL_STEPS = 40
SLOW_S = 0.25
CANCEL_AFTER_CALLS = 3


TIMEOUT_S = 4.5
CANCEL_WAIT_S = 120.0


def _long_prompt(base: str) -> str:
    """A prompt past 256 T5 tokens: the base prompt, then distinct clauses (T5 turns each word into 1-3 tokens)."""
    clauses = [f"detail {i}: {w} light on the {w} surface" for i, w in enumerate(
        ("silver", "amber", "crimson", "cobalt", "ivory", "jade", "copper", "slate", "violet", "ochre", "teal",
         "scarlet", "umber", "cerulean", "saffron", "indigo", "pearl", "bronze", "olive", "rose", "onyx", "sand",
         "coral", "plum", "moss", "flint", "honey", "lilac", "rust", "frost"))]
    return base + ", " + ", ".join(clauses)


def _requests() -> dict:
    fx = {f["id"]: f for f in json.loads(GOLDEN_SPEC.read_text())["fixtures"]}
    a, b = fx["astronaut_cfg1"], fx["teapot"]
    A = dict(prompt=a["prompt"], seed=int(a["seed"]), num_inference_steps=8, guidance_scale=1.0, height=512, width=512)
    B = dict(prompt=b["prompt"], seed=int(b["seed"]), num_inference_steps=12, guidance_scale=7.0, height=512,
             width=768)
    return {"A": A, "B": B, "LONG": _long_prompt(a["prompt"])}


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


def _calls(record: Path, t0: float, t1: float | None = None) -> list:
    """The recorder's dit_call lines stamped at or after t0 (and at or before t1)."""
    return [e for e in _events(record) if e.get("event") == "dit_call" and float(e.get("t") or 0.0) >= t0
            and (t1 is None or float(e.get("t") or 0.0) <= t1)]


def _wait_calls(record: Path, t0: float, n: int, limit_s: float) -> int:
    end = time.time() + limit_s
    while time.time() < end:
        k = len(_calls(record, t0))
        if k >= n:
            return k
        time.sleep(0.05)
    return len(_calls(record, t0))


def run_cancellation(gen, ask, run: dict, record: Path, fault_dir: Path, A: dict) -> dict:
    """request_cancellation's requests (t2_gates.request_cancellation says what each must do), every DiT forward
    slowed by SLOW_S while the recorder's SLOW arm is armed; the arm is always removed again."""
    import threading
    import uuid

    import t2_gates as G

    rd = run["readings"]
    c = {"steps": CANCEL_STEPS, "slow_s": SLOW_S, "timeout_s": TIMEOUT_S, "after_calls": CANCEL_AFTER_CALLS,
         "abort_api": callable(getattr(gen, "abort", None))}

    def abort(rid: str) -> dict:
        t = time.time()
        if not c["abort_api"]:
            return {"t": t, "error": "the engine has no DiffGenerator.abort"}
        try:
            ans = gen.abort(rid)
            return {"t": t, "s": round(time.time() - t, 2), "answer": ans if isinstance(ans, (dict, str, bool, int,
                                                                                                 type(None))) else repr(ans)[:200]}
        except Exception as exc:  # noqa: BLE001 - a failed abort is a reading
            return {"t": t, "s": round(time.time() - t, 2), "error": f"{type(exc).__name__}: {exc}"[:300]}

    def state(key: str) -> dict:
        r = rd.get(key) or {}
        st = G.request_state(r)
        st.update(calls_total=len(_calls(record, r["t0"], r.get("t1"))) if r.get("t0") else None, s=r.get("s"))
        return st

    def start(key: str, kw: dict):
        th = threading.Thread(target=ask, args=(key,), kwargs=kw, daemon=True)
        th.start()
        return th

    long_a = dict(A, num_inference_steps=CANCEL_STEPS)
    nonce = uuid.uuid4().hex
    t_arm = time.time()
    (fault_dir / "SLOW").write_text(str(SLOW_S), encoding="utf-8")
    try:


        control_t0 = time.time()
        control_long = start("qc_long", dict(A, num_inference_steps=CANCEL_STEPS // 2,
                                            request_id=f"{nonce}-control-long"))
        _wait_calls(record, control_t0, 2, CANCEL_WAIT_S)
        control_queued = start("qc_queued", dict(A, request_id=f"{nonce}-control-queued"))
        control_long.join(CANCEL_WAIT_S)
        control_queued.join(CANCEL_WAIT_S)
        c['queued_control'] = {'long': state('qc_long'), 'queued': state('qc_queued'),
                              'joined': not (control_long.is_alive() or control_queued.is_alive())}


        t0 = time.time()
        t_long = start("ql", dict(A, num_inference_steps=CANCEL_STEPS // 2, request_id=f"{nonce}-long"))
        _wait_calls(record, t0, 2, CANCEL_WAIT_S)
        queued_id = f"{nonce}-queued"
        submitted = threading.Event()
        def submit_queued():
            submitted.set()
            ask('qq', **dict(A, request_id=queued_id))
        t_queued = threading.Thread(target=submit_queued, daemon=True)
        t_queued.start()
        submitted.wait(CANCEL_WAIT_S)
        import request_receipts


        _wait_calls(record, time.time(), 2, CANCEL_WAIT_S)
        a = abort(queued_id)
        t_long.join(CANCEL_WAIT_S)
        t_queued.join(CANCEL_WAIT_S)
        ql, qq = rd.get("ql") or {}, rd.get("qq") or {}
        receipt = request_receipts.queued_abort_receipt(_events(record), queued_id,
                  qq.get("t0"), a.get("t"), ql.get("t1"), qq.get("t1"))
        peer_work = request_receipts.observed_peer_work(_events(record), ql.get("t0"), ql.get("t1"))
        c["queued"] = {"long": state("ql"), "queued": state("qq"), "abort": a,
                       "public_readings": {'long': ql, 'queued': qq},
                       "wire_receipt": receipt, "peer_expected_steps": CANCEL_STEPS // 2,
                       "peer_row_work": peer_work["rows"] if peer_work["measured"] else None,
                       "peer_row_evidence": peer_work,
                       "joined": not (t_long.is_alive() or t_queued.is_alive()),
                       "calls_after_long": len(_calls(record, ql["t1"] + 1e-6, qq["t1"]))
                       if ql.get("t1") and qq.get("t1") else None}
        t0 = time.time()
        th = start("x", dict(long_a, request_id="infb-t2-cancel-running"))
        seen = _wait_calls(record, t0, CANCEL_AFTER_CALLS, CANCEL_WAIT_S)
        a = abort("infb-t2-cancel-running")
        th.join(CANCEL_WAIT_S)
        x = state("x")
        x.update(calls_before_abort=seen, abort=a, joined=not th.is_alive(),
                 calls_after_abort=len(_calls(record, a["t"], (rd.get("x") or {}).get("t1")))
                 if (rd.get("x") or {}).get("t1") else None)
        c["running"] = x
        ask("d", **dict(long_a, timeout_s=TIMEOUT_S, request_id="infb-t2-cancel-deadline"))
        c["deadline"] = state("d")
    finally:
        (fault_dir / "SLOW").unlink(missing_ok=True)
    c["slow_fired"] = any(e.get("slow_s") for e in _calls(record, t_arm))
    return c


def run_engine(work: Path, req: dict) -> dict:
    """Build the engine with the recorder, run the sequence, return every reading (images kept in memory)."""
    import numpy as np
    from _common import NoImageProduced, to_image_array

    record, fault_dir = work / "record.jsonl", work / "fault"
    fault_dir.mkdir()
    saved = {k: os.environ.get(k) for k in ("PYTHONPATH", "INFB_SD3_COMPONENT_RECORD_FILE", "INFB_SD3_FAULT_DIR", "INFB_SD3_REQUEST_OBSERVE", "INFB_SD3_BATCH_OBSERVE")}
    os.environ["PYTHONPATH"] = ":".join([str(RECORDER)] + [q for q in (saved["PYTHONPATH"] or "").split(":") if q])
    os.environ["INFB_SD3_COMPONENT_RECORD_FILE"] = str(record)
    os.environ["INFB_SD3_FAULT_DIR"] = str(fault_dir)
    os.environ["INFB_SD3_REQUEST_OBSERVE"] = "1"
    os.environ["INFB_SD3_BATCH_OBSERVE"] = "1"
    run = {"model": _resolve_model(), "readings": {}, "timing_s": {}, "record": str(record)}
    t0 = time.time()
    try:
        from sglang.multimodal_gen import DiffGenerator
        gen = DiffGenerator.from_pretrained(model_path=run["model"], num_gpus=1, **_engine_ports())
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

    A, B, LONG = req["A"], req["B"], req["LONG"]
    try:
        ask("r0", **A)
        ask("r1", **B)
        ask("r2", **A)
        ask("r3", **B)
        ask("v", **dict(A, height=-16))
        (fault_dir / "FAULT").write_text("error_recovery: request f", encoding="utf-8")
        ask("f", **A)
        (fault_dir / "FAULT").unlink(missing_ok=True)
        ask("r6", **A)
        ask("l", **dict(A, prompt=[A["prompt"], B["prompt"]]))
        ask("s", **dict(A, prompt=B["prompt"]))
        ask("n", **dict(A, num_outputs_per_prompt=2))
        ask("a", **dict(A, seed=A["seed"] + 1))
        ask("g", **dict(B, negative_prompt=NEGATIVE_PROMPT))
        ask("e", **dict(B, negative_prompt=""))
        ask("t", **dict(A, prompt=LONG))
        ask("t2", **dict(A, prompt=LONG + TAIL))
        run["cancel"] = run_cancellation(gen, ask, run, record, fault_dir, A)
        ask("r7", **A)
    finally:
        try:
            gen.shutdown()
        except Exception:  # noqa: BLE001
            pass
    run["timing_s"]["requests"] = round(time.time() - t0 - run["timing_s"]["engine_build"], 1)
    run["images"] = images
    run["events"] = _events(record)
    run["fault_fired"] = [e for e in run["events"] if e.get("event") == "fault_fired"]
    run["image_shapes"] = {k: [None if i is None else list(np.asarray(i).shape) for i in v] for k, v in images.items()}
    return run


def judge(run: dict, req: dict) -> dict:
    import t2_gates as G

    im, rd, ev = run["images"], run["readings"], run["events"]

    def first(key, j=0):
        v = im.get(key) or []
        return v[j] if len(v) > j else None

    def p(k1, k2, j1=0, j2=0):
        return G.psnr(first(k1, j1), first(k2, j2))

    out = {}
    out["request_isolation_abab"] = G.request_isolation_abab({"A": p("r0", "r2"), "B": p("r1", "r3")})
    v, f = rd.get("v") or {}, rd.get("f") or {}
    out["error_recovery"] = G.error_recovery(
        {"raised": v.get("raised"), "type": v.get("type"), "returned": v.get("returned")},
        {"raised": f.get("raised"), "error": f.get("error"), "none": f.get("none"), "returned": f.get("returned"),
         "fired": bool(run.get("fault_fired"))},
        p("r0", "r6"))
    lst, npp = rd.get("l") or {}, rd.get("n") or {}
    out["in_request_batching"] = G.in_request_batching(
        [p("l", "r0", 0, 0), p("l", "s", 1, 0)], lst.get("n") if not lst.get("raised") else 0,
        [p("n", "r0", 0, 0), p("n", "a", 1, 0)], npp.get("n") if not npp.get("raised") else 0)
    r0, r1 = rd.get("r0") or {}, rd.get("r1") or {}
    calls = [e for e in ev if e.get("event") == "dit_call"]
    c1 = G.count_calls_between(calls, r0["t0"], r0["t1"]) if r0 and calls else None
    c7 = G.count_calls_between(calls, r1["t0"], r1["t1"]) if r1 and calls else None
    out["cfg1_single_row"] = G.cfg1_single_row(c1, req["A"]["num_inference_steps"], c7,
                                               req["B"]["num_inference_steps"])
    pairs = {"negative_effect": ("g", "r1"), "truncation": ("t", "t2"), "default_negative": ("e", "r1")}
    states = {name: {k: G.request_state(rd.get(k)) for k in keys} for name, keys in pairs.items()}
    out["prompt_semantics"] = G.prompt_semantics(p("g", "r1"), p("t", "t2"), p("e", "r1"), requests=states)
    if run.get('cancel', {}).get('queued_control') is not None:
        control = run['cancel']['queued_control']
        control['queued_reference_psnr'] = p('r0', 'qc_queued')
        control['long_reference_psnr'] = p('ql', 'qc_long')
        public = run['cancel']['queued']['public_readings']
        import request_receipts
        control_ok = (control.get('joined') is True
                      and all(control.get(k, {}).get('state') == 'image' for k in ('long', 'queued'))
                      and all(G.equal_pair(control.get(k), G.POLICY['ISO_PSNR_DB'])[0] is True
                              for k in ('queued_reference_psnr', 'long_reference_psnr')))
        run['cancel']['queued']['public_outcome'] = request_receipts.queued_public_outcome(
            public['long'], public['queued'], run['cancel']['queued']['abort'], control_ok=control_ok)
    out["request_cancellation"] = G.request_cancellation(run.get("cancel"), p("r0", "r7"))
    return out


def write_gates(result: dict) -> None:
    import day0_gates as DG
    import t2_gates as G
    from _common import GateResult, write_gate

    verdicts = result.get("verdicts") or {}
    for gate in GATES:
        v = verdicts.get(gate) or {"verdict": "not_evaluated", "reasons": [result.get("reason", "not run")]}
        ok = v.get("verdict") == "pass"
        write_gate(GateResult(gate=gate, passed=ok, score=1.0 if ok else 0.0, is_hard_gate=False, level=LAYER,
                              error=None if ok else "; ".join(v.get("reasons", []))[:500] or v.get("verdict"),
                              details=DG.gate_details(CAPABILITY[gate], LAYER, v, dict(G.POLICY), artifact=str(OUT))))


def main() -> dict:


    req = _requests()
    work = Path(tempfile.mkdtemp(prefix="sd3_day0_t2_"))
    result = {"measured": False, "requests": {k: (v if k != "LONG" else v[:80] + "...") for k, v in req.items()}}
    try:
        try:
            run = run_engine(work, req)
        except Exception as exc:  # noqa: BLE001
            from _common import save_stderr_log
            save_stderr_log("day0_t2_engine", traceback.format_exc())
            result.update(reason=f"the engine did not build or serve: {type(exc).__name__}: {exc}"[:400],
                        engine_failed=True,
                        verdicts={g: {"verdict": "fail", "reasons": ["the engine did not build or serve"]}
                                  for g in GATES})
            return result
        result.update(measured=True, model=run["model"], timing_s=run["timing_s"],
                      readings={k: {kk: vv for kk, vv in v.items()} for k, v in run["readings"].items()},
                      image_shapes=run["image_shapes"], fault_fired=run["fault_fired"], cancel=run.get("cancel"),
                      dit_calls=sum(1 for e in run["events"] if e.get("event") == "dit_call"))
        result["verdicts"] = judge(run, req)
        return result
    finally:


        source = work / "record.jsonl"
        destination = ARTIFACTS / "sd3_day0_t2_record.jsonl"
        result["native_record"] = {"copied": False, "path": str(destination)}
        try:
            if source.is_file():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                result["native_record"].update(copied=True, bytes=destination.stat().st_size,
                    sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
            else:
                result["native_record"]["error"] = "recorder file missing"
        except OSError as exc:
            result["native_record"]["error"] = f"{type(exc).__name__}: {exc}"
        shutil.rmtree(work, ignore_errors=True)


def _engine_ports() -> dict:
    """Return this verifier queue's assigned engine port block."""
    from _common import engine_port_kwargs

    return engine_port_kwargs()

if __name__ == "__main__":
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
    print("[day0-t2] " + "; ".join(f"{g}: {((res.get('verdicts') or {}).get(g) or {}).get('verdict', res.get('reason'))}"
                                   for g in GATES), flush=True)
    sys.exit(0)
