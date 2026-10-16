#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Generation judge: the pristine target scores what the candidate's speculative decoder EMITTED."""

from __future__ import annotations

import json
import math
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

NUM_PROMPTS = 32
MAX_TOKENS = 256
EPS_GRID = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
JUDGE_SANE_MAX_PRISTINE_NOT_TOP1 = 0.5
JUDGE_SANE_MAX_PRISTINE_ABS_Z = 8.0
SEED_BASE = {"cand": 1_000_003, "pristine": 2_000_003}
REQUEST_TIMEOUT_S = 600
SAMPLED_CONCURRENCY = 4
_NEUTRAL = {"top_p": 1.0, "min_p": 0.0, "repetition_penalty": 1.0,
            "presence_penalty": 0.0, "frequency_penalty": 0.0}
_SGLANG_TOP_K_ALL = 1 << 30


def _post(port: int, path: str, body: dict, timeout: float = REQUEST_TIMEOUT_S) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _model_id(port: int) -> str:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=60) as resp:
        return json.loads(resp.read().decode())["data"][0]["id"]


def _tokenize(port: int, model: str, text: str) -> list[int]:
    return [int(t) for t in _post(port, "/tokenize", {
        "model": model, "prompt": text, "add_special_tokens": True})["tokens"]]


def _sampling(engine: str, temperature: float) -> dict:
    sp = dict(_NEUTRAL, temperature=temperature)
    sp["top_k"] = _SGLANG_TOP_K_ALL if engine == "sglang" else 0
    return sp


def _generate(engine: str, port: int, model: str, ids: list[int], *, temperature: float,
              seed: int | None) -> list[int]:
    if engine == "sglang":
        sp = _sampling(engine, temperature)
        sp.update(max_new_tokens=MAX_TOKENS, ignore_eos=True)
        out = _post(port, "/generate", {"input_ids": ids, "sampling_params": sp})
        return [int(t) for t in out["output_ids"]]
    body = {"model": model, "prompt": ids, "max_tokens": MAX_TOKENS, "ignore_eos": True,
            "return_token_ids": True, "stream": False, **_sampling(engine, temperature)}
    if seed is not None:
        body["seed"] = seed
    choice = _post(port, "/v1/completions", body)["choices"][0]
    return [int(t) for t in choice["token_ids"]]


def _teacher_force(engine: str, port: int, model: str, prompt_ids: list[int],
                   out_ids: list[int]) -> dict:
    """Per generated position: logprob of the emitted token, and the target's top-1 id/logprob."""
    full = list(prompt_ids) + list(out_ids)
    n0 = len(prompt_ids)
    lp, top_lp, top_id = [], [], []
    if engine == "sglang":
        meta = _post(port, "/generate", {
            "input_ids": full,
            "sampling_params": dict(_sampling(engine, 1.0), max_new_tokens=1),
            "return_logprob": True, "logprob_start_len": n0 - 1, "top_logprobs_num": 1,
        })["meta_info"]
        toks = meta["input_token_logprobs"][1:]
        tops = meta["input_top_logprobs"][1:]
        if not (len(toks) == len(tops) == len(out_ids)):
            raise ValueError(f"sglang input logprobs misaligned: {len(toks)}/{len(tops)} entries "
                             f"for {len(out_ids)} generated tokens")
        for want, tok, top in zip(out_ids, toks, tops):
            if int(tok[1]) != int(want):
                raise ValueError(f"sglang input logprobs misaligned: id {tok[1]} != {want}")
            lp.append(float(tok[0]))
            top_lp.append(float(top[0][0]))
            top_id.append(int(top[0][1]))
    else:
        choice = _post(port, "/v1/completions", {
            "model": model, "prompt": full, "max_tokens": 1, "prompt_logprobs": 1,
            "stream": False, **_sampling(engine, 1.0),
        })["choices"][0]
        plp = choice["prompt_logprobs"]
        if len(plp) != len(full):
            raise ValueError(f"vllm prompt_logprobs length {len(plp)} != prompt length {len(full)}")
        for pos in range(n0, len(full)):
            entry = plp[pos]
            want = str(full[pos])
            if want not in entry:
                raise ValueError(f"vllm prompt_logprobs[{pos}] lacks the emitted token {want}")
            lp.append(float(entry[want]["logprob"]))
            best = next((k for k, v in entry.items() if int(v["rank"]) == 1), None)
            if best is None:
                raise ValueError(f"vllm prompt_logprobs[{pos}] has no rank-1 entry")
            top_lp.append(float(entry[best]["logprob"]))
            top_id.append(int(best))
    if len(lp) != len(out_ids):
        raise ValueError(f"judged {len(lp)} positions for {len(out_ids)} generated tokens")
    return {"lp": [round(x, 5) for x in lp],
            "gap": [round(max(0.0, t - x), 5) for t, x in zip(top_lp, lp)],
            "top1": [int(i == o) for i, o in zip(top_id, out_ids)]}


def load_prompts(path: Path, n: int = NUM_PROMPTS) -> list[str]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = rows if isinstance(rows, list) else rows.get("prompts", [])
    prompts = [r if isinstance(r, str) else r.get("prompt", "") for r in rows]
    prompts = [p for p in prompts if p.strip()][:n]
    if len(prompts) < n:
        raise ValueError(f"{path}: {len(prompts)} usable prompts, need {n}")
    return prompts


def probe(*, engine: str, port: int, side: str, prompts: list[str]) -> dict:
    """Generate greedy and sampled continuations on a live arm server; token ids, not text."""
    t0 = time.time()
    model = _model_id(port)
    ids = [_tokenize(port, model, p) for p in prompts]
    rec: dict = {"engine": engine, "side": side, "model": model, "max_tokens": MAX_TOKENS,
                 "prompt_ids": ids, "modes": {}}
    tg = time.time()
    rec["modes"]["greedy"] = [
        _generate(engine, port, model, x, temperature=0.0, seed=None) for x in ids]
    rec["greedy_elapsed_s"] = round(time.time() - tg, 1)
    ts = time.time()
    base = SEED_BASE[side]
    with ThreadPoolExecutor(max_workers=SAMPLED_CONCURRENCY) as pool:
        rec["modes"]["sampled"] = list(pool.map(
            lambda a: _generate(engine, port, model, a[1], temperature=1.0, seed=base + a[0]),
            enumerate(ids)))
    rec["sampled_elapsed_s"] = round(time.time() - ts, 1)
    rec["elapsed_s"] = round(time.time() - t0, 1)
    return rec


def judge(*, engine: str, port: int, probes: dict) -> dict:
    """Teacher-force every arm's generations through THIS (pristine) server."""
    t0 = time.time()
    model = _model_id(port)
    out: dict = {"engine": engine, "model": model, "arms": {}}
    for side, rec in probes.items():
        arm: dict = {}
        for mode, gens in (rec.get("modes") or {}).items():
            arm[mode] = [_teacher_force(engine, port, model, p, g)
                         for p, g in zip(rec["prompt_ids"], gens, strict=True)]
        out["arms"][side] = arm
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


def _probe_path(raw_dir: Path, wl: str, side: str) -> Path:
    return Path(raw_dir) / f"gen_probe_{wl}_{side}.json"


def _judge_path(raw_dir: Path, wl: str) -> Path:
    return Path(raw_dir) / f"gen_judge_{wl}.json"


def after_measure(*, engine: str, port: int, side: str, workload: str, raw_dir: Path,
                  prompts_path: Path, speculative: bool) -> None:
    """The verifier's per-arm call, on each arm's server after its scored measurement."""
    if not speculative:
        return
    side = "cand" if side.lower().startswith("cand") else "pristine"
    try:
        rec = probe(engine=engine, port=port, side=side, prompts=load_prompts(prompts_path))
    except Exception as e:                             # noqa: BLE001
        rec = {"side": side, "error": f"{type(e).__name__}: {e}"}
    _probe_path(raw_dir, workload, side).write_text(json.dumps(rec))
    print(f"  [gen-judge] {workload}/{side}: probe "
          f"{rec.get('error') or 'ok in ' + str(rec.get('elapsed_s')) + 's'}", flush=True)


def record_judge_error(raw_dir: Path, workloads, message: str) -> None:
    """For a judge server that never came up: the reader must see why there is no verdict."""
    for wl in workloads:
        _judge_path(raw_dir, wl).write_text(json.dumps({"error": message}))
        print(f"  [gen-judge] {wl}: judge {message}", flush=True)


def judge_all(*, engine: str, port: int | None, raw_dir: Path, workloads) -> None:
    """The verifier's judge call, on the spec-OFF pristine server after both arms. Never raises."""
    if port is None:
        record_judge_error(raw_dir, workloads, "judge server did not become ready")
        return
    for wl in workloads:
        try:
            probes = {}
            for s in ("cand", "pristine"):
                p = _probe_path(raw_dir, wl, s)
                r = json.loads(p.read_text()) if p.is_file() else {"error": "probe file missing"}
                if not r.get("error"):
                    probes[s] = r
            res = judge(engine=engine, port=port, probes=probes)
        except Exception as e:                         # noqa: BLE001
            res = {"error": f"{type(e).__name__}: {e}"}
        _judge_path(raw_dir, wl).write_text(json.dumps(res))
        print(f"  [gen-judge] {wl}: judge "
              f"{res.get('error') or 'ok in ' + str(res.get('elapsed_s')) + 's'}", flush=True)


def strip_speculative_argv(cmd: list[str]) -> list[str]:
    """The serve argv with every speculative flag and its value removed -- the judge server's."""
    out, i, removed = [], 0, 0
    while i < len(cmd):
        a = str(cmd[i])
        if a.startswith("--speculative"):
            removed += 1
            i += 2 if (i + 1 < len(cmd) and not str(cmd[i + 1]).startswith("--")) else 1
            continue
        out.append(cmd[i])
        i += 1
    if not removed:
        raise ValueError("no --speculative* flag in the serve argv; refusing to call it spec-off")
    return out




def j1_stats(judged: list[dict]) -> dict:
    """Violation counts at every eps in EPS_GRID over one arm's greedy generations."""
    gaps = [g for r in judged for g, t in zip(r["gap"], r["top1"]) if not t]
    n = sum(len(r["gap"]) for r in judged)
    viol = {str(e): sum(1 for g in gaps if g > e) for e in EPS_GRID}
    return {"n_tokens": n, "n_not_top1": len(gaps),
            "violations": viol,
            "rate": {k: (v / n if n else None) for k, v in viol.items()},
            "largest_gaps": sorted(gaps, reverse=True)[:10]}


def _per_prompt(judged: list[dict]) -> tuple[list[float], list[float]]:
    top1 = [sum(r["top1"]) / len(r["top1"]) for r in judged if r["top1"]]
    mlp = [sum(r["lp"]) / len(r["lp"]) for r in judged if r["lp"]]
    return top1, mlp


def _paired_z(a: list[float], b: list[float]) -> float | None:
    d = [x - y for x, y in zip(a, b)]
    if len(d) < 3:
        return None
    m = sum(d) / len(d)
    var = sum((x - m) ** 2 for x in d) / (len(d) - 1)
    if var <= 0:
        return 0.0 if m == 0 else math.copysign(math.inf, m)
    return m / math.sqrt(var / len(d))


def j2_stats(cand: list[dict], pris: list[dict]) -> dict:
    """Paired per-prompt shift of the top-1 share and of mean log p, candidate minus pristine."""
    ct, cm = _per_prompt(cand)
    pt, pm = _per_prompt(pris)
    mean = lambda v: (sum(v) / len(v)) if v else None      # noqa: E731
    return {"n_prompts": min(len(ct), len(pt)),
            "cand_top1_share": mean(ct), "pristine_top1_share": mean(pt),
            "cand_mean_logp": mean(cm), "pristine_mean_logp": mean(pm),
            "z_top1_share": _paired_z(ct, pt), "z_mean_logp": _paired_z(cm, pm)}


def calibration_z(judged: list[dict]) -> dict:
    """Within-arm top-1 calibration of a SAMPLED set."""
    num = var = 0.0
    n = 0
    for r in judged:
        for lp, gap, top1 in zip(r["lp"], r["gap"], r["top1"]):
            q = math.exp(min(0.0, lp + gap))
            num += top1 - q
            var += q * (1.0 - q)
            n += 1
    return {"n_tokens": n, "observed_minus_expected_top1": (num / n) if n else None,
            "z": (num / math.sqrt(var)) if var > 0 else None, "_num": num, "_var": var}


def two_sample_z(cand: dict, pris: dict) -> float | None:
    """Candidate's (observed - expected) top-1 share against the pristine arm's; see J2."""
    if not cand.get("n_tokens") or not pris.get("n_tokens"):
        return None
    nc, np_ = cand["n_tokens"], pris["n_tokens"]
    var = cand["_var"] / nc ** 2 + pris["_var"] / np_ ** 2
    return ((cand["_num"] / nc - pris["_num"] / np_) / math.sqrt(var)) if var > 0 else None


def summarise(res: dict) -> dict:
    arms = res.get("arms") or {}
    out: dict = {"j1": {}, "j2": None}
    for side in ("cand", "pristine"):
        g = (arms.get(side) or {}).get("greedy")
        out["j1"][side] = j1_stats(g) if g else None
    c, p = (arms.get("cand") or {}).get("sampled"), (arms.get("pristine") or {}).get("sampled")
    if c and p:
        out["j2"] = j2_stats(c, p)
    out["j2_cal"] = {side: calibration_z(x) for side, x in (("cand", c), ("pristine", p)) if x}
    if "cand" in out["j2_cal"] and "pristine" in out["j2_cal"]:
        out["j2_cal"]["z_diff"] = two_sample_z(out["j2_cal"]["cand"], out["j2_cal"]["pristine"])
    pj = out["j1"].get("pristine")
    out["judge_sane"] = bool(pj and pj["n_tokens"] and pj["n_not_top1"] / pj["n_tokens"]
                             <= JUDGE_SANE_MAX_PRISTINE_NOT_TOP1)
    return out


def read_from_raw_dir(raw_dir: Path, workload_names) -> dict:
    out: dict = {}
    for wl in workload_names:
        entry: dict = {}
        for side in ("cand", "pristine"):
            p = _probe_path(raw_dir, wl, side)
            if p.is_file():
                r = json.loads(p.read_text())
                entry[f"probe_{side}"] = {k: r.get(k) for k in (
                    "error", "elapsed_s", "greedy_elapsed_s", "sampled_elapsed_s", "model")}
        jp = _judge_path(raw_dir, wl)
        if jp.is_file():
            res = json.loads(jp.read_text())
            entry["judge_error"] = res.get("error")
            entry["judge_elapsed_s"] = res.get("elapsed_s")
            if not res.get("error"):
                entry.update(summarise(res))
        out[wl] = entry
    return out


def _env_float(name: str) -> float | None:
    raw = os.environ.get(name, "").strip()
    try:
        return float(raw) if raw else None
    except ValueError:
        return None


def gate(per_workload: dict, workload_names, *, speculative: bool) -> dict:
    """Verdict + the numbers behind it. Never raises."""
    eps = _env_float("INFB_GEN_JUDGE_J1_EPS")
    max_excess = _env_float("INFB_GEN_JUDGE_J1_MAX_EXCESS")
    j1_z = _env_float("INFB_GEN_JUDGE_J1_Z")
    j1_z = 4.0 if j1_z is None else j1_z
    zmax = _env_float("INFB_GEN_JUDGE_J2_Z")
    j1_armed = eps is not None and max_excess is not None
    verdict: dict = {"per_workload": per_workload, "j1_eps": eps, "j1_max_excess": max_excess,
                     "j1_z": j1_z, "j2_z": zmax, "armed": j1_armed or zmax is not None,
                     "failed": False, "reason": None}
    if not speculative:
        verdict["status"] = "not_a_speculative_task"
        return verdict
    rejected, unavailable = [], []
    for name in workload_names:
        e = per_workload.get(name) or {}
        j1 = e.get("j1") or {}
        c1, p1 = j1.get("cand"), j1.get("pristine")
        if verdict["armed"] and p1 and p1["n_tokens"] and not e.get("judge_sane"):
            unavailable.append(f"{name}: the judge disagrees with the pristine arm's own greedy "
                               f"tokens on {p1['n_not_top1']}/{p1['n_tokens']}")
            continue
        if j1_armed:
            if not c1 or not p1 or not c1["n_tokens"] or not p1["n_tokens"]:
                unavailable.append(f"{name}: J1 judged no tokens "
                                   f"({e.get('judge_error') or 'probe/judge missing'})")
            else:
                k = str(eps) if str(eps) in c1["rate"] else None
                if k is None:
                    unavailable.append(f"{name}: J1 eps {eps} not on the recorded grid")
                else:
                    c, p = c1["violations"][k], p1["violations"][k]
                    bar = max(max_excess * c1["n_tokens"], j1_z * math.sqrt(c + p + 1))
                    e["j1_excess"], e["j1_bar"] = c - p, round(bar, 1)
                    if c - p > bar:
                        rejected.append(f"{name}: J1 {c}/{c1['n_tokens']} tokens more than {eps} "
                                        f"nats below the target's top-1, pristine {p}; excess "
                                        f"{c - p} > {bar:.1f}")
        if zmax is not None:
            cal = e.get("j2_cal") or {}
            zc, zp = (cal.get("cand") or {}).get("z"), (cal.get("pristine") or {}).get("z")
            zd = cal.get("z_diff")
            if zc is None or zp is None or zd is None:
                unavailable.append(f"{name}: J2 has no sampled set on one arm")
            elif abs(zp) > JUDGE_SANE_MAX_PRISTINE_ABS_Z:
                unavailable.append(f"{name}: the PRISTINE arm's own calibration z={zp:.2f} is "
                                   f"beyond {JUDGE_SANE_MAX_PRISTINE_ABS_Z} -- the judge's p is not "
                                   f"the sampler's p")
            elif abs(zd) > zmax:
                rejected.append(f"{name}: J2 two-sample calibration z={zd:.2f} beyond {zmax} "
                                f"(candidate {zc:.2f}, pristine {zp:.2f})")
    if rejected:
        verdict.update(failed=True, reason="spec_output_rejected", failures=rejected)
    elif unavailable:
        verdict.update(failed=True, reason="generation_judge_unavailable", failures=unavailable)
    verdict["status"] = "judged" if verdict["armed"] else "recorded"
    return verdict
