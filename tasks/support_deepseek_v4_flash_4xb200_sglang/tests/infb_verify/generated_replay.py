#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The private driver of the two
generated-history nodes, `test_generation_replay` and `test_prefix_branch_replay`. The protocol, the parser and the
metrics are the public `eval/deepseek_v4_flash/fidelity/generated_replay.py` (its trusted copy under /tests/eval);
this module supplies the fixed private schedule and the frozen bars, and records everything."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
RECORDS = Path(os.environ.get("VERIFIER_DIR", "/logs/verifier")) / "records"
GEN_TOKENS = 512
SINGLE_CUTS = (4093, 4095, 4096, 4097, 4223, 4224)
C3 = ((2, 4095), (3, 4096), (4, 4097))
C5 = ((3, 4093), (4, 4095), (5, 4096), (6, 4097), (7, 4223))
PREFIX_LENGTHS = (4000, 4321)
SUFFIX_TOKENS = 129

ALPHA = 0.0025
Z_ALPHA = 2.807
TAIL_U = 0.25
T2_FLOOR = 3
T3_FLOOR = 2.0



def public_module(tests_dir: Path = TESTS_DIR):
    """The trusted copy of the public protocol, loaded by path (the grader's module names)."""
    name = "deepseek_v4_flash_eval_generated_replay"
    if name not in sys.modules:
        path = tests_dir / "eval" / "deepseek_v4_flash" / "fidelity" / "generated_replay.py"
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def digest(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def corpus(fixture_dir: Path, band: int = 8192, count: int = 8) -> list[dict]:
    """The first `count` fixtures of length `band` by id: the schedule's corpus."""
    recs = []
    for path in sorted(fixture_dir.glob(f"*-{band}-*.json")):
        rec = json.loads(path.read_text())
        if len(rec["input_ids"]) == band:
            recs.append(rec)
    recs.sort(key=lambda r: r["id"])
    if len(recs) < count:
        raise RuntimeError(f"the schedule needs {count} fixtures of {band} tokens, {len(recs)} found in {fixture_dir}")
    return recs[:count]


def build_schedule(fixture_dir: Path) -> dict:
    """The two nodes' schedules as explicit prompt lists (the manifest's content, before digests are compared)."""
    fx = corpus(fixture_dir)
    ids = [f["input_ids"] for f in fx]
    names = [f["id"] for f in fx]
    gen = []
    for i in (0, 1):
        for p in SINGLE_CUTS:
            gen.append({"batch": f"single-{names[i]}-{p}", "items": [{"fixture": names[i], "cut": p}]})
    gen.append({"batch": "c3", "items": [{"fixture": names[i], "cut": p} for i, p in C3]})
    gen.append({"batch": "c5", "items": [{"fixture": names[i], "cut": p} for i, p in C5]})
    for b in gen:
        for it in b["items"]:
            it["prompt_digest"] = digest(ids[names.index(it["fixture"])][: it["cut"]])
    pre = []
    for length in PREFIX_LENGTHS:
        for i in (0, 1):
            a, b_ = names[i + 2], names[i + 4]
            series = {"series": f"prefix-{names[i]}-{length}", "prefix": {"fixture": names[i], "cut": length},
                      "suffixes": {"A": {"fixture": a, "cut": SUFFIX_TOKENS}, "B": {"fixture": b_, "cut": SUFFIX_TOKENS}},
                      "order": ["A", "B", "A", "B"]}
            series["prefix"]["digest"] = digest(ids[i][:length])
            for key, s in series["suffixes"].items():
                s["digest"] = digest(ids[names.index(s["fixture"])][:SUFFIX_TOKENS])
            pre.append(series)
    return {"corpus": [{"id": n, "input_digest": digest(i)} for n, i in zip(names, ids)],
            "generation": {"tokens": GEN_TOKENS, "batches": gen},
            "prefix": {"tokens": GEN_TOKENS, "series": pre}}


def check_manifest(schedule: dict, manifest: dict) -> None:
    """The frozen manifest must name exactly this schedule, digests and all."""
    for key in ("corpus", "generation", "prefix"):
        if schedule[key] != manifest[key]:
            raise RuntimeError(f"the replay schedule's {key!r} differs from the frozen manifest")


def _fixture_ids(fixture_dir: Path) -> dict[str, list[int]]:
    return {f["id"]: f["input_ids"] for f in corpus(fixture_dir)}


def _no_op(_kind: str, _label: str):
    return contextlib.nullcontext()


class SpecObserver:
    """The speculative-transition observer's switch and its records, for one node's interval."""

    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.t0 = self.t1 = None

    def __enter__(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        (self.dir / "on").write_text(str(self.t0))
        return self

    def __exit__(self, *exc):
        with contextlib.suppress(OSError):
            (self.dir / "on").unlink()
        self.t1 = time.time()
        return False

    def records(self) -> list[dict]:
        out = []
        for f in sorted(self.dir.glob("spec-*.jsonl")):
            for line in f.read_text(errors="replace").splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if self.t0 is not None and self.t0 <= r.get("t", 0) <= (self.t1 or time.time()):
                    r["process"] = f.stem
                    out.append(r)
        return out

    def coverage(self) -> dict:
        """Steps observed (one process's view: the lowest pid's records, every TP rank records the same steps),
        partial acceptances, rejected suffixes followed by more generated tokens of the same request, errors."""
        recs = self.records()
        procs = sorted({r["process"] for r in recs})
        mine = [r for r in recs if procs and r["process"] == procs[0]]
        rows = [row for r in mine for row in r.get("rows", [])]
        n = max([r.get("draft_tokens", 0) for r in mine] or [0])
        drafts = n - 1
        partial = [x for x in rows if x.get("accepted_drafts") is not None and 0 < x["accepted_drafts"] < drafts]
        rejected = [x for x in rows if x.get("accepted_drafts") is not None and x["accepted_drafts"] < drafts]
        later = {}
        for x in rows:
            later.setdefault(x.get("request"), []).append(x.get("pre_len") or 0)
        followed = [x for x in rejected if any(p > (x.get("pre_len") or 0) for p in later.get(x.get("request"), []))]
        return {"processes": len(procs), "steps": len(mine), "request_steps": len(rows), "draft_tokens": n,
                "accepted_drafts": {str(k): sum(1 for x in rows if x.get("accepted_drafts") == k) for k in range(n)},
                "partial_acceptances": len(partial), "rejected_suffixes": len(rejected),
                "rejected_then_continued": len(followed),
                "inconsistent_counts": sum(1 for x in rows if x.get("consistent") is False),
                "errors": [r.get("observer_error") or r.get("patch_error") for r in recs
                           if r.get("observer_error") or r.get("patch_error")][:5]}


def _record(node: str, rec: dict) -> None:
    RECORDS.mkdir(parents=True, exist_ok=True)
    with (RECORDS / f"{node}.jsonl").open("a") as fh:
        fh.write(json.dumps(rec) + "\n")


def run_generation(base_url: str, fixture_dir: Path, schedule: dict, node: str = "generation_replay",
                   on_observation=None, op=_no_op) -> list[dict]:
    """The generation node's schedule: per single request or batch, flush, generate together, then R = 3 cold replay
    passes of the batch's histories, each pass after its own flush and in the registered order."""
    g = public_module()
    fixtures = _fixture_ids(fixture_dir)
    n = schedule["generation"]["tokens"]
    out = []
    for batch in schedule["generation"]["batches"]:
        prompts = []
        for it in batch["items"]:
            ids = fixtures[it["fixture"]][: it["cut"]]
            if digest(ids) != it["prompt_digest"]:
                raise RuntimeError(f"prompt {it['fixture']}@{it['cut']} does not match its frozen digest")
            prompts.append((f"{batch['batch']}/{it['fixture']}@{it['cut']}", ids))
        with op("action", f"{batch['batch']}: flush"):
            flush_before = g.flush(base_url)
        with op("http_request", f"{batch['batch']}: generate {len(prompts)} together"):
            gens = g.generate_together(base_url, prompts, n)
        items = [(label, ids, gen) for (label, ids), gen in zip(prompts, gens)]
        passes, flush_after = g.replay_group(base_url, items, op=op)
        for i, (label, ids, gen) in enumerate(items):
            obs = g.observe(label, ids, gen, passes[i])
            obs.update({"node": node, "batch": batch["batch"], "concurrency": len(prompts),
                        "flush_s": [flush_before, *flush_after]})
            _record(node, obs)
            if on_observation:
                on_observation(obs)
            out.append(obs)
    return out


def run_prefix(base_url: str, fixture_dir: Path, schedule: dict, node: str = "prefix_branch_replay",
               on_observation=None, op=_no_op) -> list[dict]:
    """The prefix node's schedule: per series, one flush, A B A B warm, then R = 3 cold replay passes of all four,
    each pass after its own flush and in the registered order."""
    g = public_module()
    fixtures = _fixture_ids(fixture_dir)
    n = schedule["prefix"]["tokens"]
    out = []
    for s in schedule["prefix"]["series"]:
        prefix = fixtures[s["prefix"]["fixture"]][: s["prefix"]["cut"]]
        if digest(prefix) != s["prefix"]["digest"]:
            raise RuntimeError(f"series {s['series']}: the prefix does not match its frozen digest")
        suffix = {}
        for key, spec in s["suffixes"].items():
            ids = fixtures[spec["fixture"]][: spec["cut"]]
            if digest(ids) != spec["digest"]:
                raise RuntimeError(f"series {s['series']}: suffix {key} does not match its frozen digest")
            suffix[key] = ids
        with op("action", f"{s['series']}: flush"):
            flush_before = g.flush(base_url)
        warm = []
        for visit, key in enumerate(s["order"]):
            prompt = prefix + suffix[key]
            label = f"{s['series']}/{key}{visit}"
            with op("http_request", f"{label}: warm generation"):
                warm.append((label, key, visit, prompt, g.generate(base_url, prompt, n, label)))
        items = [(label, prompt, gen) for label, _k, _v, prompt, gen in warm]
        passes, flush_after = g.replay_group(base_url, items, op=op)
        for i, (label, key, visit, prompt, gen) in enumerate(warm):
            obs = g.observe(label, prompt, gen, passes[i])
            obs.update({"node": node, "series": s["series"], "branch": key, "visit": visit,
                        "prefix_tokens": len(prefix), "flush_s": [flush_before, *flush_after]})
            _record(node, obs)
            if on_observation:
                on_observation(obs)
            out.append(obs)
    return out


def summarize(observations: list[dict]) -> dict:
    """The node's four statistics (the verdict), with each sequence's diagnostics and the engine-reported coverage,
    and without the per-token arrays."""
    g = public_module()
    seqs = [{k: o.get(k) for k in ("label", "prompt_tokens", "generated_tokens", "mean_k3", "max_abs_d",
                                   "max_excess", "replay_passes", "batch", "series", "branch", "visit",
                                   "concurrency")}
            | {"cached_tokens": o["engine_counts"].get("cached_tokens"),
               "spec_verify_ct": o["engine_counts"].get("spec_verify_ct"),
               "spec_accepted_drafts": o["engine_counts"].get("spec_accepted_drafts"),
               "completion_tokens": o["engine_counts"].get("completion_tokens")} for o in observations]
    return {"statistics": g.node_statistics(observations) if observations else None,
            "sequences": seqs, "n": len(seqs),
            "max_mean_k3": max((s["mean_k3"] for s in seqs), default=None),
            "max_abs_d": max((s["max_abs_d"] for s in seqs), default=None),
            "max_excess": max((s["max_excess"] for s in seqs), default=None)}



def _count_quantile(values: list[int], alpha: float = ALPHA) -> tuple[int, str]:
    """The smallest t with P(S2' > t) <= alpha, from a negative binomial fitted by moments (a Poisson when the
    variance is at most the mean). All-zero calibration readings give 0, and T2's floor then decides (9.7)."""
    mu = statistics.mean(values)
    var = statistics.pvariance(values) if len(values) > 1 else 0.0
    if mu == 0:
        return 0, "every calibration run read 0; the floor decides"
    if var <= mu:
        pmf, cdf, k = math.exp(-mu), 0.0, 0
        while True:
            cdf += pmf
            if 1 - cdf <= alpha:
                return k, f"Poisson(mean {mu:.3f})"
            k += 1
            pmf *= mu / k
    r = mu * mu / (var - mu)
    p = r / (r + mu)
    pmf, cdf, k = p ** r, 0.0, 0
    while True:
        cdf += pmf
        if 1 - cdf <= alpha:
            return k, f"negative binomial (mean {mu:.3f}, variance {var:.3f}, r {r:.3f})"
        pmf *= (k + r) / (k + 1) * (1 - p)
        k += 1


def _gpd_fit(excesses: list[float]) -> tuple[float, float]:
    """Maximum-likelihood (xi, beta) of a generalized Pareto for exceedances y > 0: a grid over xi, golden section
    over log beta."""
    def ll(xi: float, b: float) -> float:
        if b <= 0:
            return -1e300
        if abs(xi) <= 1e-9:
            return -len(excesses) * math.log(b) - sum(excesses) / b
        s = 0.0
        for v in excesses:
            z = 1 + xi * v / b
            if z <= 0:
                return -1e300
            s += math.log(z)
        return -len(excesses) * math.log(b) - (1 + 1 / xi) * s
    best = (-1e300, 0.0, statistics.mean(excesses))
    xi = -0.45
    while xi <= 1.5:
        lo, hi = math.log(1e-5), math.log(1e3)
        for _ in range(80):
            a = lo + (hi - lo) * 0.382
            c = lo + (hi - lo) * 0.618
            if ll(xi, math.exp(a)) < ll(xi, math.exp(c)):
                lo = a
            else:
                hi = c
        b = math.exp((lo + hi) / 2)
        v = ll(xi, b)
        if v > best[0]:
            best = (v, xi, b)
        xi = round(xi + 0.005, 3)
    return best[1], best[2]


def _tail_bar(runs: list[dict], alpha: float = ALPHA) -> tuple[float, str]:
    """T3': the peaks-over-threshold alpha-quantile of the per-run maximum excess, with the threshold at u = 0.25
    nats; at least 2 x the observed max of S3' and at least 2 nats (9.5, 9.7)."""
    exceedances = [e - TAIL_U for r in runs for e in r["excesses_over_u"]]
    total_tokens = sum(r["tokens"] for r in runs)
    mean_tokens = statistics.mean([r["tokens"] for r in runs])
    observed = max(r["S3p"] for r in runs)
    floor = max(2 * observed, T3_FLOOR)
    if not exceedances:
        return floor, f"no calibration token at e >= {TAIL_U}: the floors decide (2 x {observed:.3f}, {T3_FLOOR})"
    lam = len(exceedances) / total_tokens
    q = 1 - (1 - alpha) ** (1 / mean_tokens)
    if len(exceedances) >= 10:
        xi, beta = _gpd_fit([v if v > 0 else 1e-9 for v in exceedances])
        how = f"generalized Pareto (xi {xi:.3f}, beta {beta:.4f}) over {len(exceedances)} exceedances"
    else:
        xi, beta = 0.0, max(statistics.mean(exceedances), 1e-6)
        how = f"exponential tail (beta {beta:.4f}) over {len(exceedances)} exceedances"
    if q / lam >= 1:
        x = TAIL_U
    elif abs(xi) < 1e-9:
        x = TAIL_U - beta * math.log(q / lam)
    else:
        x = TAIL_U + beta / xi * ((q / lam) ** (-xi) - 1)
    return max(x, floor), f"{how}, lambda {lam:.3e}; the alpha-quantile {x:.3f}, floors 2 x {observed:.3f} and {T3_FLOOR}"


def bars_from_calibration(runs: list[dict]) -> dict:
    """The calibration rule (with T1' as rule Q) from the correct runs of THIS schedule with R = 3."""
    s1 = [r["S1p"] for r in runs]
    logs = [math.log(v) for v in s1 if v > 0 and math.isfinite(v)]
    m, sd = statistics.mean(logs), (statistics.stdev(logs) if len(logs) > 1 else 0.0)
    t1 = math.exp(m + Z_ALPHA * sd)
    s2 = [r["S2p"] for r in runs]
    q2, how2 = _count_quantile(s2)
    t3, how3 = _tail_bar(runs)
    s4 = [r["S4"] for r in runs]
    m4, sd4 = statistics.mean(s4), (statistics.stdev(s4) if len(s4) > 1 else 0.0)
    return {"T1p": t1,
            "T2p": max(q2, 2 * max(s2), T2_FLOOR),
            "T3p": t3,
            "G": min(m4 - 4 * sd4, min(s4) - 0.05),
            "from": {"runs": len(runs), "alpha_per_part": ALPHA,
                     "T1p": {"rule": "Q: the lognormal quantile alone, no 3x floor", "lognormal_quantile": t1,
                             "mean_log": m, "sd_log": sd, "superseded_3x_floor_not_applied": 3 * max(s1)},
                     "T2p": {"quantile": q2, "model": how2, "2x_observed_max": 2 * max(s2), "floor": T2_FLOOR},
                     "T3p": how3,
                     "G": {"mean_minus_4sd": m4 - 4 * sd4, "observed_min_minus_0.05": min(s4) - 0.05},
                     "observed": {"S1p_max": max(s1), "S2p_max": max(s2), "S3p_max": max(r["S3p"] for r in runs),
                                  "S4_min": min(s4)}}}


def failures(observations: list[dict], bars: dict) -> list[str]:
    """The node's verdict: S1' > T1', S2' > T2', S3' > T3' or S4 < G, each named with its reading and its bar."""
    g = public_module()
    if not observations:
        return ["the node produced no observation"]
    s = g.node_statistics(observations)
    out = []
    if not (s["S1p"] <= bars["T1p"]):
        out.append(f"S1' {s['S1p']:.6g} exceeds the node's bar {bars['T1p']:.6g}: over the {s['decisive_tokens']} "
                   f"decisive tokens of {s['tokens']}, the generation's median distance from its replays' median is "
                   f"{s['decisive_median_abs_g_minus_m']:.6g} nats against {s['decisive_median_abs_r2_minus_r1']:.6g} "
                   f"between two replays of the same history")
    if not (s["S2p"] <= bars["T2p"]):
        worst = sorted(((t["excess"], o["label"], t["position"]) for o in observations for t in o["tokens"]
                        if t["excess"] >= g.EXCESS_NAT), reverse=True)[:5]
        out.append(f"S2' {s['S2p']} tokens lie at least {g.EXCESS_NAT:g} nat outside their replays' range, over the "
                   f"node's bar {bars['T2p']}: worst "
                   + ", ".join(f"{lb} position {p} by {e:.6g}" for e, lb, p in worst))
    if not (s["S3p"] <= bars["T3p"]):
        worst = max(((t["excess"], o["label"], t) for o in observations for t in o["tokens"]), key=lambda x: x[0])
        e, label, t = worst
        out.append(f"S3' {s['S3p']:.6g} exceeds the node's bar {bars['T3p']:.6g}: {label} position {t['position']} "
                   f"(C4 offset {t['c4']}, C128 offset {t['c128']}) generated {t['generated']:.6g}, its "
                   f"{len(t['replays'])} cold replays {[round(v, 6) for v in t['replays']]}")
    if not (s["S4"] >= bars["G"]):
        out.append(f"S4 {s['S4']:.6g} is below the node's floor {bars['G']:.6g}: only {s['decisive_tokens']} of "
                   f"{s['tokens']} generated tokens carry probability {g.DECISIVE_P} or more, so the consistency "
                   f"statistic rests on too few positions")
    return out
