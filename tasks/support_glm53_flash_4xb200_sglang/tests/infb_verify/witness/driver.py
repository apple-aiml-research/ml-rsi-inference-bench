#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The witness launch: one untimed, instrumented launch of a deployment class's own recipe, before its graded group."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import importlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
for p in (str(TESTS_DIR), "/app/test"):
    if p not in sys.path:
        sys.path.insert(0, p)
from glm53_protocols.limits import (ACTION_S, HTTP_S, INVOCATION_S, STARTUP_S, TEARDOWN_KILL_AFTER_S,  # noqa: E402
                                     TEARDOWN_S)
from infb_verify.records import write_record  # noqa: E402

MODEL = "/model/zai-org/GLM-5.3-Flash"
SITE = TESTS_DIR / "infb_verify" / "witness" / "site"
ROOT = Path(os.environ.get("INFB_WITNESS_ROOT", "/logs/verifier/witness"))
PORT = int(os.environ.get("INFB_WITNESS_PORT", "32000"))
N_RANKS = 4
UPSTREAM = {"TestHiddenHighThroughput": "TestGLM53FlashB200HighThroughput",
            "TestHiddenLowLatency": "TestGLM53FlashB200LowLatency"}
FEATURES = {"TestHiddenHighThroughput": "runtime,attention", "TestHiddenLowLatency": "runtime,attention,adaptive"}
CLASS_KEY = {"TestHiddenHighThroughput": "high_throughput", "TestHiddenLowLatency": "low_latency"}
WARM_INPUTS = ("Explain in plain words how a radix tree stores strings that share a prefix.",
               "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number iteratively.\"\"\"\n")
DECODE_POSITIONS, PREFILL_ROWS = 16, 32
DENSE_PROMPT = 2032


def say(msg: str) -> None:
    print(f"[witness] {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


class Channel:
    def __init__(self, wdir: Path):
        self.wdir, self.seq = wdir, int(time.time() * 1000)

    def send(self, phase: str, args: dict | None = None, timeout: float = ACTION_S) -> dict:
        self.seq += 1
        tmp = self.wdir / ".control.tmp"
        tmp.write_text(json.dumps({"seq": self.seq, "phase": phase, "args": args or {}}))
        os.replace(tmp, self.wdir / "control.json")
        t0 = time.time()
        while time.time() - t0 < timeout:
            acks = [json.loads(Path(f).read_text()) for f in glob.glob(str(self.wdir / "acks" / "*.json"))]
            ranked = [a for a in acks if a.get("seq") == self.seq and a.get("rank")]
            if len(ranked) >= N_RANKS:
                return {"seq": self.seq, "acked": len(ranked), "s": round(time.time() - t0, 3)}
            time.sleep(0.2)
        raise TimeoutError(f"phase {phase}: {len(ranked)} of {N_RANKS} ranks acknowledged within {timeout} s")

    def events(self) -> list[dict]:
        time.sleep(1.0)
        out = []
        for f in glob.glob(str(self.wdir / "events" / "*.jsonl")):
            with open(f) as fh:
                for line in fh:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        pass
        return sorted(out, key=lambda e: (e.get("pid"), e.get("n")))


def _post(base: str, path: str, body: dict, timeout: float = HTTP_S) -> dict:
    import requests
    r = requests.post(base + path, json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _flush(base: str) -> None:
    import requests
    requests.post(base + "/flush_cache", params={"timeout": 30}, timeout=ACTION_S).raise_for_status()


def _generate(base: str, ids: list[int], new: int, dp_rank: int | None) -> dict:
    body = {"input_ids": ids, "sampling_params": {"temperature": 0.0, "max_new_tokens": new},
            "return_logprob": True, "logprob_start_len": 0}
    if dp_rank is not None:
        body["routed_dp_rank"] = dp_rank
    return _post(base, "/generate", body)


def _teacher_forced(base: str, ids: list[int], dp_rank: int | None) -> list[float]:
    r = _generate(base, ids, 0, dp_rank)
    ent = r["meta_info"]["input_token_logprobs"]
    assert len(ent) == len(ids) and all(e[1] == t for e, t in zip(ent, ids)), "teacher-forced answer misaligned"
    return [e[0] for e in ent]


def _k3(served, ref) -> float:
    from glm53_protocols.fidelity import _k3_term
    return sum(_k3_term(r - s) for r, s in zip(ref, served)) / len(ref)


def launch(cls: str, wdir: Path):
    from sglang.test.test_utils import popen_launch_server
    upstream = importlib.import_module("registered.models_e2e.test_glm53_flash_b200")
    server_args = list(getattr(upstream, UPSTREAM[cls]).server_args)
    env = dict(os.environ, PYTHONPATH=f"{SITE}:{os.environ.get('PYTHONPATH', '')}".rstrip(":"),
               INFB_GLM53_WITNESS_DIR=str(wdir), INFB_GLM53_WITNESS_FEATURES=FEATURES[cls])
    base = f"http://127.0.0.1:{PORT}"
    t0 = time.time()
    proc = popen_launch_server(MODEL, base, timeout=STARTUP_S, other_args=server_args, env=env)
    return proc, base, server_args, round(time.time() - t0, 1)


def stop(proc) -> dict:
    """The server's whole process tree: TERM, then KILL after 10 s (the teardown operation), and reaped."""
    import psutil
    t0 = time.time()
    out: dict = {"pid": proc.pid}
    try:
        tree = [proc_ for proc_ in psutil.Process(proc.pid).children(recursive=True)] + [psutil.Process(proc.pid)]
    except psutil.Error:
        tree = []
    for p_ in tree:
        try:
            p_.send_signal(signal.SIGTERM)
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(tree, timeout=TEARDOWN_KILL_AFTER_S)
    for p_ in alive:
        try:
            p_.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(alive, timeout=TEARDOWN_S - TEARDOWN_KILL_AFTER_S)
    try:
        proc.wait(timeout=TEARDOWN_KILL_AFTER_S)
    except subprocess.TimeoutExpired:
        pass
    out.update(killed=len(alive), s=round(time.time() - t0, 2))
    return out


def phase_runtime(ch: Channel, base: str, cls: str, bar: float) -> dict:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    dp = 0 if cls == "TestHiddenHighThroughput" else None
    out: dict = {"warm": []}
    ch.send("idle")
    for text in WARM_INPUTS:
        _generate(base, tok(text, add_special_tokens=False)["input_ids"], 16, dp)
    out["arm"] = ch.send("runtime", {"record_forwards": True, "profile_forwards": 12})
    for text in WARM_INPUTS:
        ids = tok(text, add_special_tokens=False)["input_ids"]
        _flush(base)
        r = _generate(base, ids, 48, dp)
        gen = [e[1] for e in r["meta_info"]["output_token_logprobs"]]
        served = [e[0] for e in r["meta_info"]["output_token_logprobs"]]
        tf = _teacher_forced(base, ids + gen, dp)[len(ids):]
        out["warm"].append({"prompt_len": len(ids), "generated": len(gen), "k3_vs_teacher_forced": _k3(served, tf),
                            "bar": bar, "text": tok.decode(gen)[:200]})
    out["disarm"] = ch.send("idle")
    return out


def _valid_output(base: str, ids: list[int], r: dict, bar: float) -> dict:
    """An output under a substituted tier is valid when every requested token came back with a finite log-probability
    and the generation-time log-probabilities agree with the same tokens' teacher-forced ones under the class's bar."""
    import math
    ent = r["meta_info"]["output_token_logprobs"]
    gen, served = [e[1] for e in ent], [e[0] for e in ent]
    out = {"generated": len(gen)}
    if not gen or any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in served):
        return {**out, "valid": False, "problem": "a generated token has no finite log-probability"}
    k3 = _k3(served, _teacher_forced(base, ids + gen, None)[len(ids):])
    return {**out, "k3_vs_teacher_forced": k3, "bar": bar, "valid": k3 < bar}


def phase_adaptive(ch: Channel, base: str, bar: float) -> dict:
    """GLM-3: ordinary private C1/C16 workloads; the min/max substitution; the counterfactual replay."""
    out: dict = {"ordinary_arm": ch.send("adaptive_ordinary")}
    client = TESTS_DIR / "infb_efficiency" / "client.py"
    eff = json.loads((TESTS_DIR / "config.json").read_text())["infb"]["efficiency"]
    seed_file = str(eff["seed_file"])
    if seed_file.startswith("/tests/"):
        seed_file = str(TESTS_DIR / seed_file[len("/tests/"):])
    for c in (1, 16):
        argv = [sys.executable, "-I", str(client), "measure", "--engine", "sglang", "--base-url", base,
                "--tokenizer", eff["tokenizer"], "--texts", str(TESTS_DIR / "infb_efficiency" / "pool.jsonl.xz"),
                "--seed-file", seed_file, "--concurrency", str(c), "--axis", "decode_c1" if c == 1 else "throughput",
                "--deadline-s", str(INVOCATION_S - 60), "--side", "dev", "--quiet"]
        t0 = time.time()
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=INVOCATION_S, cwd="/tmp")
        out[f"ordinary_c{c}"] = {"rc": proc.returncode, "s": round(time.time() - t0, 1),
                                 "tail": (proc.stdout + proc.stderr)[-600:] if proc.returncode else None}
    events = ch.events()
    slots = [e for e in events if e["k"] == "adaptive_init"]
    ladder = sorted({s for e in slots for v in (e.get("slots") or {}).values() for s in (v.get("candidate_steps") or [])})
    c1 = next(((e.get("slots") or {}).get("1") for e in slots if (e.get("slots") or {}).get("1")), None)
    admitted = sorted(t for t in ((c1 or {}).get("candidate_steps") or ladder) if t > 0)
    out["admitted_c1"] = admitted
    if len(admitted) >= 2:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL)
        ids = tok(WARM_INPUTS[0], add_special_tokens=False)["input_ids"]
        for tier in (admitted[0], admitted[-1]):
            out[f"substitute_{tier}_arm"] = ch.send("adaptive_substitute", {"tier": tier, "record_forwards": True})
            outs = [_generate(base, ids, 64, None) for _ in range(3)]
            ch.send("adaptive_hold", {"record_forwards": True})
            out[f"substitute_{tier}_outputs"] = [_valid_output(base, ids, o, bar) for o in outs]
        out["restore"] = ch.send("adaptive_restore")
    occupied = sorted({e["bs"] for e in events if e["k"] == "adaptive_feedback" and e.get("bs")
                       and e.get("cseq") == out["ordinary_arm"]["seq"]})
    out["occupied_batch_sizes"] = occupied
    out["replay"] = ch.send("adaptive_replay", {"slots": occupied}, timeout=INVOCATION_S)
    ch.send("idle")
    return out


def phase_attention(ch: Channel, base: str, cls: str, inputs: list[dict], repeats: int = 3, ab: bool = False) -> dict:
    dp = 0 if cls == "TestHiddenHighThroughput" else None
    out: dict = {"inputs": []}
    for inp in inputs:
        ch.send("idle")
        _flush(base)
        arm = ch.send("attention_capture", {"capture_attention": True, "record_forwards": True})
        t0 = time.time()
        r = _generate(base, inp["prompt_ids"], DECODE_POSITIONS + 1, dp)
        out["inputs"].append({"id": inp["id"], "total": inp["total"], "cseq": arm["seq"],
                              "output_ids": [e[1] for e in r["meta_info"]["output_token_logprobs"]],
                              "s": round(time.time() - t0, 2)})
    ch.send("idle")
    out["consumption"] = []
    dense = {"id": f"dense-{DENSE_PROMPT + DECODE_POSITIONS}", "total": DENSE_PROMPT + DECODE_POSITIONS,
             "prompt_ids": list(inputs[0]["prompt_ids"][:DENSE_PROMPT])}
    _flush(base)
    r = _generate(base, dense["prompt_ids"], DECODE_POSITIONS + 1, dp)
    dense_cap = {"output_ids": [e[1] for e in r["meta_info"]["output_token_logprobs"]]}
    for inp, cap in list(zip(inputs, out["inputs"])) + [(dense, dense_cap)]:
        forced = list(cap["output_ids"][:DECODE_POSITIONS + 1])
        fixed = list(inp["prompt_ids"]) + forced
        P = len(inp["prompt_ids"])
        window = [P - PREFILL_ROWS, P + DECODE_POSITIONS]
        body = {"input_ids": inp["prompt_ids"], "sampling_params": {"temperature": 0.0, "max_new_tokens": len(forced)},
                "return_logprob": True, "logprob_start_len": len(inp["prompt_ids"]) - PREFILL_ROWS}
        if dp is not None:
            body["routed_dp_rank"] = dp
        rec = {"id": inp["id"], "total": inp["total"], "prompt_len": P, "window": window,
               "observable": "mlp_input_branches", "fixed_tail": fixed[-(PREFILL_ROWS + DECODE_POSITIONS + 1):]}
        if inp is dense:
            runs = []
            for zero in [False] * repeats + [True]:
                _flush(base)
                arm = ch.send("consumption", {"capture_logits": True, "force_ids": fixed, "zero_attention": zero,
                                              "record_forwards": True})
                r = _post(base, "/generate", body)
                runs.append({"zero": zero, "cseq": arm["seq"],
                             "served_is_forced": [e[1] for e in r["meta_info"]["output_token_logprobs"]] == forced})
                ch.send("idle", {"zero_attention": False})
            rec["dense_runs"] = runs
        for armed in ([True, False] if ab else [True]):
            _flush(base)
            arm = ch.send("consumption", {"consumption": armed, "capture_next": window, "force_ids": fixed,
                                          "record_forwards": True, **({"capture_logits": True} if ab else {})})
            r = _post(base, "/generate", body)
            served = [e[1] for e in r["meta_info"]["output_token_logprobs"]]
            ch.send("idle")
            if armed:
                rec.update(cseq=arm["seq"], served_is_forced=served == forced)
            else:
                rec.update(cseq_unarmed=arm["seq"], unarmed_served_is_forced=served == forced)
        out["consumption"].append(rec)
    ch.send("idle")
    return out


def phase_chat(base: str, cls: str, bar: float) -> dict:
    """A maker-only phase (never part of the graded witness): the GLM-2 chat check on this launch, for qualifying chat
    controls on an otherwise unchanged server."""
    from infb_verify.chat import chat_protocols
    return {"chat": chat_protocols(base, f"{cls}-witness", bar)}


def prefill_observer(wdir: Path) -> list[dict]:
    """v62: every rank's `prefill_observer` events, for the launch record -- whether the observer patched the
    engine's prefill graph runner, each runner's largest captured size and what the observer's buffers took once every
    shape was captured, or why prefill observation was skipped or stopped. A file that cannot be read is listed with
    its error."""
    out: list[dict] = []
    for f in sorted(glob.glob(str(Path(wdir) / "events" / "*.jsonl"))):
        try:
            with open(f) as fh:
                for line in fh:
                    if '"prefill_observer"' not in line:
                        continue
                    try:
                        e = json.loads(line)
                    except ValueError:
                        continue
                    if e.get("k") == "prefill_observer":
                        out.append({k: v for k, v in e.items() if k != "k"})
        except OSError as exc:
            out.append({"file": f, "error": repr(exc)[:300]})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--class", dest="cls", required=True, choices=sorted(UPSTREAM))
    ap.add_argument("--inputs-file", default=str(TESTS_DIR / "hidden" / "attention" / "heldout.json"))
    ap.add_argument("--phases", default=None, help="comma-separated subset of the class's phases (default: all)")
    ap.add_argument("--consumption-ab", action="store_true",
                    help="maker-only: each consumption run again unarmed, both recording the served logits")
    ap.add_argument("--consumption-repeats", type=int, default=3,
                    help="unmodified consumption runs before the zeroed one (the rule's 3; more only for a maker "
                         "diagnostic, never a verdict)")
    a = ap.parse_args(argv)
    phases = [p for p in (a.phases or "runtime,adaptive,attention").split(",") if p]
    wdir = ROOT / a.cls
    wdir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((TESTS_DIR / "config.json").read_text())
    bar = float(cfg["infb"]["bars"]["fidelity_T"][CLASS_KEY[a.cls]])
    inputs_doc = json.loads(Path(a.inputs_file).read_text())
    inputs = inputs_doc["sequences"]
    summary: dict = {"class": a.cls, "inputs_file": a.inputs_file, "inputs_role": inputs_doc.get("role"),
                     "inputs": [{k: i.get(k) for k in ("id", "total", "sha256")} for i in inputs],
                     "features": FEATURES[a.cls], "requested_phases": phases,
                     "consumption_repeats": a.consumption_repeats, "phases": {}}
    proc = None
    try:
        say(f"launch {a.cls}")
        proc, base, server_args, startup_s = launch(a.cls, wdir)
        summary.update(server_args=server_args, startup_s=startup_s, port=PORT)
        ch = Channel(wdir)
        for name, fn in (("runtime", lambda: phase_runtime(ch, base, a.cls, bar)),
                         ("adaptive", (lambda: phase_adaptive(ch, base, bar)) if a.cls == "TestHiddenLowLatency" else None),
                         ("attention", lambda: phase_attention(ch, base, a.cls, inputs, a.consumption_repeats,
                                                               a.consumption_ab)),
                         ("chat", lambda: phase_chat(base, a.cls, bar))):
            if fn is None or name not in phases:
                continue
            say(f"phase {name}")
            t0 = time.time()
            try:
                res = {"completed": True, **fn()}
            except Exception as exc:  # noqa: BLE001 - recorded; the judge reads a failed phase as missing evidence
                res = {"completed": False, "error": f"{type(exc).__name__}: {str(exc)[:800]}"}
            res["s"] = round(time.time() - t0, 1)
            summary["phases"][name] = res
            write_record("witness", a.cls, f"phase-{name}", res)
            if proc.poll() is not None:
                summary["server_exited"] = {"after_phase": name, "returncode": proc.returncode, "t": time.time()}
                break
    except Exception as exc:  # noqa: BLE001
        summary["launch_error"] = f"{type(exc).__name__}: {str(exc)[:800]}"
        summary["launch_error_t"] = time.time()
    finally:
        if proc is not None and proc.poll() is None:
            summary["teardown"] = stop(proc)
    summary["prefill_observer"] = prefill_observer(wdir)
    write_record("witness", a.cls, "launch", summary)
    say("replay")
    from infb_verify.witness import assemble
    summary["replay"] = assemble.replay_all(wdir, a.cls, summary.get("phases", {}).get("attention", {}), inputs)
    write_record("witness", a.cls, "replay", summary["replay"])
    if os.environ.get("INFB_GLM53_WITNESS_KEEP_TENSORS") != "1":
        tensors = Path(wdir) / "tensors"
        size = sum(f.stat().st_size for f in tensors.rglob("*") if f.is_file()) if tensors.exists() else 0
        shutil.rmtree(tensors, ignore_errors=True)
        say(f"witness tensors removed after assembly ({size} bytes)")
    say("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
