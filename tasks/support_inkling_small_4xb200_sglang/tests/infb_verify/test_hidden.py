#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The hidden nodes of `support_inkling_small_4xb200_sglang`, run inside the registered classes' own launches.

Both classes INHERIT the registered test's own classes, which `test.patch` applied to the candidate tree: the launch,
the recipe flags and the environment stay upstream's. The production class runs its nodes in a fixed order
(conftest.py orders them); the deterministic class runs the three inherited helpers with frozen thresholds.

  production, `TestHiddenInklingProduction`:
    test_0_efficiency_decode_c1, test_0_efficiency_throughput   whole-serving efficiency, paired, rho 0.8
    test_served_configuration   /server_info and the launch environment against the registered recipe
    test_api_probes             structure of /generate and /v1/completions
    test_gsm8k_hidden_slice     upstream's evaluator on the hidden 200-row slice, the calibrated floor
    test_generation_replay      17 generated continuations against their cold replay
    test_prefix_branch_replay   reuse on wave 4 and the cold replay of waves 1, 3 and 4
    test_conv_state_restore     76 windows: the first token generated after a prefix-cache restore against a cold
                                run, the reuse evidence on every window
    test_author_fidelity        24 sequences against the author recording, bands and per-sequence caps
    test_long_context           two 262,143-token sequences, engine agreement
    test_execution_witness      the component's kernels observed in the served model on every rank
  deterministic, `TestHiddenInklingDeterministic`: the three inherited helpers, and
    test_conv_state_history     76 windows: each window's restored request run twice and its cold request run
                                twice, under different slot histories; each pair must be exactly identical

Calibration runs these same nodes on the reference with `INFB_CALIBRATION_OUT=<dir>`: each node
writes its measurements there and applies no bar, so the bars come from exactly the code that grades.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import lzma
import math
import os
import shutil
import statistics
import sys
import time
from pathlib import Path

TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
APP_DIR = Path(os.environ.get("APP_DIR", "/app"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(APP_DIR / "eval"))

from infb_efficiency import operations  # noqa: E402
from infb_efficiency.nodes import EfficiencyMixin  # noqa: E402

import records  # noqa: E402

CONFIG = json.loads((TESTS_DIR / "config.json").read_text())
X = CONFIG["infb"]
HIDDEN = TESTS_DIR / "hidden"
MODEL = X.get("model_dir", "/model/thinkingmachines/Inkling-Small-NVFP4")
CALIBRATION = os.environ.get("INFB_CALIBRATION_OUT")

# The launch timeout is the public startup limit, set before the registered module binds it.
import sglang.test.test_utils as _tu  # noqa: E402

_tu.DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH = operations.limit_s("startup")
os.environ.setdefault("INKLING_SMALL_TEST_MODEL_PATH", MODEL)

# The deterministic helpers read their LongBench prompts from the helper's own cache path; the hidden draw goes there
# before the class runs, so nothing is fetched and the prompts are the hidden ones.
_HELPER = APP_DIR / "python" / "sglang" / "test" / "kl_test_utils.py"
_CACHE = _HELPER.parent / ".longbench_cache" / f"input_ids_{MODEL.replace('/', '_')}_3000_48.json"
if (HIDDEN / "longbench_hidden.json").is_file():
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(HIDDEN / "longbench_hidden.json", _CACHE)

_UPSTREAM = APP_DIR / "test" / "registered" / "models_e2e" / "test_inkling_small_nvfp4.py"


def _load_upstream():
    spec = importlib.util.spec_from_file_location("infb_upstream_registered", _UPSTREAM)
    if spec is None or spec.loader is None:  # pragma: no cover - a missing file is a verifier fault
        raise RuntimeError(f"cannot import the registered test at {_UPSTREAM}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    port = os.environ.get("INFB_TEST_PORT")  # calibration only: side-by-side launches on one host
    if port:
        module.DEFAULT_URL_FOR_TEST = f"http://127.0.0.1:{port}"
    return module


_upstream = _load_upstream()

from inkling_small.serving import api, configuration, conv_state, fidelity, prefix_branch, replay, server  # noqa: E402


def policy() -> dict:
    """The frozen bars (`hidden/policy.json`); none in calibration."""
    if CALIBRATION:
        return {}
    return json.loads((HIDDEN / "policy.json").read_text())


def calibrate(node: str, payload: dict) -> None:
    out = Path(CALIBRATION)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{node}.json").write_text(json.dumps(records._finite_or_text(payload), indent=1))


def _server_pid(case) -> int | None:
    proc = getattr(type(case), "process", None) or getattr(case, "process", None)
    return getattr(proc, "pid", None)


def _load_json(path: Path):
    opener = lzma.open if path.suffix == ".xz" else open
    with opener(path, "rt") as fh:
        return json.load(fh)


def _record(check: str, rows: list[dict]) -> None:
    d = records.record_dir()
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{check}.jsonl", "a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(records._finite_or_text(r), sort_keys=True) + "\n")


def token_k3(ref: list[float], served: list) -> list[float]:
    """Per-token k3 = expm1(ref - served) - (ref - served) over positions 1..n-1 (served[0] scores no token)."""
    out = []
    for r, v in zip(ref, served[1:]):
        d = r - v
        try:
            out.append(math.expm1(d) - d)
        except OverflowError:
            out.append(math.inf)
    return out


def c32(k3: list[float], trim: int, chaos: float) -> dict:
    """The mean of per-token k3 after leaving out the `trim` largest terms, and the tokens above the chaos level;
    the plain mean and the median are diagnostics."""
    srt = sorted(k3)
    kept = srt[: max(0, len(srt) - trim)] or srt[:1]
    return {"trimmed_mean_k3": sum(kept) / len(kept), "trim": trim, "outliers": sum(1 for x in k3 if x > chaos),
            "tokens": len(k3), "mean_k3": sum(k3) / len(k3), "median_k3": statistics.median(k3)}


class TestHiddenInklingProduction(EfficiencyMixin, _upstream.TestInklingSmallNvfp4):
    """The registered production class plus this task's hidden nodes, in one server launch."""

    @classmethod
    def setUpClass(cls):
        with operations.operation("startup", "the candidate server's launch (production)"):
            super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        with operations.operation("teardown", "the candidate server's teardown (production)"):
            super().tearDownClass()

    def test_0_efficiency_decode_c1(self):
        if CALIBRATION:
            self.skipTest("calibration: efficiency is paired within the trial and has no calibrated bar")
        super().test_0_efficiency_decode_c1()

    def test_0_efficiency_throughput(self):
        if CALIBRATION:
            self.skipTest("calibration: efficiency is paired within the trial and has no calibrated bar")
        super().test_0_efficiency_throughput()

    def test_gsm8k(self):
        """Upstream's GSM8K on its public rows: a diagnostic in the same launch, never scored."""
        with operations.operation("registered_gsm8k", "upstream GSM8K on the public rows (diagnostic)"):
            try:
                super().test_gsm8k()
            except AssertionError as exc:  # diagnostic: printed, not raised
                print(f"[inkling diagnostic] upstream test_gsm8k on the public rows: {exc}")

    def test_served_configuration(self):
        with operations.operation("action", "GET /server_info"):
            info = server.server_info(self.base_url, timeout=operations.limit_s("action"))
        rec = configuration.evaluate(info)
        failures = [f"{i['key']} = {i['observed']!r}, the recipe has {i['expected']}" for i in rec["checks"]
                    if not i["matches"]]
        pid = _server_pid(self)
        env = {}
        with contextlib.suppress(OSError):
            env = dict(kv.split("=", 1) for kv in Path(f"/proc/{pid}/environ").read_bytes().decode(errors="replace")
                       .split("\0") if "=" in kv)
        unified = env.get("SGLANG_ENABLE_UNIFIED_RADIX_TREE")
        rec["launch_environment"] = {"SGLANG_ENABLE_UNIFIED_RADIX_TREE": unified, "server_pid": pid}
        if unified != "1":
            failures.append(f"the server was not launched with SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 (saw {unified!r})")
        print(f"[inkling configuration] {json.dumps(rec['diagnostics'], sort_keys=True)}; "
              f"unified radix tree {unified!r}; mismatches {failures}")
        _record("configuration", [{**rec, "failures": failures}])
        if CALIBRATION:
            calibrate("test_served_configuration", {**rec, "failures": failures})
            return
        self.assertFalse(failures, "; ".join(failures))

    def test_api_probes(self):
        with operations.operation("http_request", "the API structure probes"):
            rec = api.run(self.base_url)
        failures = [f"{k}: {p}" for k, v in rec.items() if isinstance(v, dict) for p in v["problems"]]
        _record("api", [{**rec, "failures": failures}])
        if CALIBRATION:
            calibrate("test_api_probes", rec)
            return
        self.assertFalse(failures, "; ".join(failures))

    def test_gsm8k_hidden_slice(self):
        from sglang.test.few_shot_gsm8k import run_eval  # noqa: PLC0415
        from types import SimpleNamespace  # noqa: PLC0415
        from urllib.parse import urlparse  # noqa: PLC0415

        url = urlparse(self.base_url)
        args = SimpleNamespace(num_shots=10, data_path=str(HIDDEN / "gsm8k_slice.jsonl"), num_questions=200,
                               max_new_tokens=512, parallel=128, host=f"http://{url.hostname}", port=int(url.port))
        t0 = time.monotonic()
        with operations.operation("registered_gsm8k", "the hidden GSM8K slice"):
            metrics = run_eval(args)
        rec = {"accuracy": metrics["accuracy"], "invalid": metrics.get("invalid"), "wall_s": round(time.monotonic() - t0, 1)}
        print(f"[inkling GSM8K hidden slice] {rec}")
        _record("gsm8k_hidden", [rec])
        if CALIBRATION:
            calibrate("test_gsm8k_hidden_slice", rec)
            return
        floor = float(policy()["gsm8k_floor"])
        self.assertGreaterEqual(metrics["accuracy"], floor, f"accuracy {metrics['accuracy']} below the floor {floor}")

    def test_generation_replay(self):
        doc = _load_json(HIDDEN / "replay.json")
        groups = doc["groups"]
        t0 = time.monotonic()
        with operations.operation(operations.request_kind(max(len(s["token_ids"]) for g in groups for s in g)),
                                  "the replay workload"):
            res = replay.run(self.base_url, [[s["token_ids"] for s in g] for g in groups], new_tokens=doc["new_tokens"],
                             timeout=operations.limit_s("http_request_long"))
        ids = [s["id"] for g in groups for s in g]
        rows = [{"id": i, **{k: v for k, v in r.items() if k != "d"}} for i, r in zip(ids, res)]
        _record("replay", rows)
        print("[inkling replay] " + ", ".join(f"{r['id']} {r.get('mean_k3', float('nan')):.4f}/{r.get('max_abs_d', float('nan')):.3f}"
                                               for r in rows) + f" ({time.monotonic() - t0:.0f} s)")
        if CALIBRATION:
            calibrate("test_generation_replay", {"sequences": rows})
            return
        bars = policy()["replay"]
        failures = [f"{r['id']}: " + "; ".join(r["structural_problems"]) for r in rows if not r.get("valid")]
        failures += [f"{r['id']}: mean k3 {r['mean_k3']:.6f} > {bars['T_mean']}" for r in rows
                     if r.get("valid") and not r["mean_k3"] <= bars["T_mean"]]
        failures += [f"{r['id']}: max|d| {r['max_abs_d']:.4f} > {bars['T_max']}" for r in rows
                     if r.get("valid") and not r["max_abs_d"] <= bars["T_max"]]
        self.assertFalse(failures, "; ".join(failures))

    def test_prefix_branch_replay(self):
        doc = _load_json(HIDDEN / "prefix.json")
        with operations.operation("http_request", "the prefix-branch workload"):
            res = prefix_branch.run(self.base_url, doc["waves"], new_tokens=doc["new_tokens"],
                                    timeout=operations.limit_s("http_request"))
        served = [{k: s.get(k) for k in ("wave", "id", "prompt_len", "cached_tokens")} for s in res["served"]]
        reps = [{k: v for k, v in r.items() if k != "d"} for r in res["replays"]]
        _record("prefix_branch", [{"served": served, "replays": reps}])
        print("[inkling prefix branch] cached " + ", ".join(f"{s['wave']}/{s['id']} {s['cached_tokens']}" for s in served)
              + "; replay " + ", ".join(f"{r['wave']}/{r['id']} {r.get('mean_k3', float('nan')):.4f}/"
                                        f"{r.get('max_abs_d', float('nan')):.3f}" for r in reps))
        if CALIBRATION:
            calibrate("test_prefix_branch_replay", {"served": served, "replays": reps})
            return
        bars = policy()["prefix"]
        failures = []
        for s in served:
            if s["wave"] == "w4":  # reuse is graded on wave 4 only; wave 3's counts are diagnostics
                need = int(bars["H"][s["id"]])
                if not (s["cached_tokens"] or 0) >= need:
                    failures.append(f"w4/{s['id']}: cached_tokens {s['cached_tokens']} < {need}: the recipe's prefix "
                                    f"caching is not active")
        failures += [f"{r['wave']}/{r['id']}: " + "; ".join(r["structural_problems"]) for r in reps if not r.get("valid")]
        failures += [f"{r['wave']}/{r['id']}: mean k3 {r['mean_k3']:.6f} > {bars['T_mean']}" for r in reps
                     if r.get("valid") and not r["mean_k3"] <= bars["T_mean"]]
        failures += [f"{r['wave']}/{r['id']}: max|d| {r['max_abs_d']:.4f} > {bars['T_max']}" for r in reps
                     if r.get("valid") and not r["max_abs_d"] <= bars["T_max"]]
        self.assertFalse(failures, "; ".join(failures))

    def test_conv_state_restore(self):
        """The convolution state restored with a cached prefix. Per window (L0 + 2 tokens): flush,
        a cold greedy continuation of the first L0 + 1 tokens, then the identical request, which restores the cached
        prefix (c = its cached tokens; re-issued at c + 1 when c < L0). F = the windows whose first generated token
        differs from the cold run's; pass iff F <= B, and every window must show the reuse (c >= 0.5 x L0) and be
        structurally valid (both runs return the 16 tokens). The first-token and prompt-mode log-probability
        differences and the first divergence are recorded, not graded."""
        doc = _load_json(HIDDEN / "conv_state.json.xz")
        rows = []
        for w in doc["windows"]:
            # A window's requests (two flushes, two continuations and three prompt scorings; a flush and two
            # continuations more on a re-issue) are each far below http_request's limit; the window runs as one
            # operation under that limit.
            with operations.operation(operations.request_kind(len(w["token_ids"])), f"the conv-state window {w['id']}"):
                rec = conv_state.probe(self.base_url, w["token_ids"], int(w["L0"]), new_tokens=int(doc["new_tokens"]),
                                       timeout=operations.limit_s("http_request"))
            rec["id"] = w["id"]
            rows.append(rec)
        summary = conv_state.summarize(rows)
        _record("conv_state", rows)
        print(f"[inkling conv state] F {summary['flips']} of {summary['windows']}; below the reuse floor "
              f"{summary['below_reuse_floor']}; structurally invalid {summary['structurally_invalid']}; re-issued "
              f"{summary['reissued']}; first-token |d| median "
              f"{summary['first_token_abs_d_median']}; prompt-mode |d| median {summary['prompt_mode_abs_d_median']}")
        if CALIBRATION:
            calibrate("test_conv_state_restore", {"summary": summary, "windows": rows})
            return
        bar = policy()["conv_state"]
        failures = [f"{r['id']}: " + "; ".join(r["structural_problems"]) for r in rows if r["structural_problems"]]
        failures += [f"{r['id']}: cached_tokens {r['cached_tokens']} < 0.5 x L0 = {0.5 * r['L0']:g}: the recipe's prefix "
                     f"caching is not active" for r in rows if not r["cached_tokens"] >= 0.5 * r["L0"]]
        if not summary["flips"] <= int(bar["B"]):
            failures.append(f"{summary['flips']} of {summary['windows']} windows: the first token generated after a "
                            f"prefix-cache restore differs from the cold run's, more than the bar {bar['B']}")
        self.assertFalse(failures, "; ".join(failures))

    def _fidelity_reading(self, seqs: list[dict], key: str, check: str, draw: int) -> list[dict]:
        rows = []
        for s in seqs:
            with operations.operation(operations.request_kind(len(s["token_ids"])), f"{check} {s['id']} draw {draw}"):
                served, meta = fidelity.prompt_logprobs(self.base_url, s["token_ids"])
            rec = {"id": s["id"], "band": s.get("band"), "family": s.get("family"), "length": len(s["token_ids"]),
                   "draw": draw, "cached_tokens": meta.get("cached_tokens")}
            if s.get(key) is not None:
                rec.update(fidelity.score(s["token_ids"], s[key], served))
            rec["served_logprobs"] = [x if x is None else round(float(x), 6) for x in served]
            rows.append(rec)
        _record(check, [{k: v for k, v in r.items() if k != "served_logprobs"} for r in rows])
        return rows

    def test_author_fidelity(self):
        seqs = _load_json(HIDDEN / "fidelity.json.xz")["sequences"]
        rows = self._fidelity_reading(seqs, "author_logprobs", "fidelity", 1)
        print("[inkling fidelity] " + ", ".join(f"{r['id']} {r.get('mean_k3', float('nan')):.5f}" for r in rows))
        if CALIBRATION:
            calibrate("test_author_fidelity", {"sequences": rows})
            return
        bars = policy()["fidelity"]
        chaos = float(bars["chaos_level"])
        failures = [f"{r['id']}: " + "; ".join(r["problems"]) for r in rows if not r.get("valid")]
        if failures:
            self.fail("; ".join(failures))
        ref = {sq["id"]: sq for sq in seqs}

        def stat(r):
            band = str(r["band"])
            return c32(token_k3(ref[r["id"]]["author_logprobs"], r["served_logprobs"]), int(bars["trim"][band]), chaos)

        by_band: dict[str, list[dict]] = {}
        for r in rows:
            r.update(stat(r))
            by_band.setdefault(str(r["band"]), []).append(r)
        _record("fidelity_c32", [{k: v for k, v in r.items() if k != "served_logprobs"} for r in rows])
        for band, rs in sorted(by_band.items()):
            T, C = float(bars["T"][band]), float(bars["C"][band])
            R = float(bars["outlier_rate"][band])
            rate = sum(r["outliers"] for r in rs) / sum(r["tokens"] for r in rs)
            if not rate <= R:  # the band's pooled fraction; measured once more, fails only if both
                again = [dict(a, **stat(a)) for a in self._fidelity_reading(
                    [sq for sq in seqs if str(sq.get("band")) == band], "author_logprobs", "fidelity", 2)]
                rate2 = sum(a["outliers"] for a in again) / sum(a["tokens"] for a in again)
                if not rate2 <= R:
                    failures.append(f"band {band}: {rate:.5f} then {rate2:.5f} of tokens above the chaos level, over {R} "
                                    f"on both readings")
            med = statistics.median(r["trimmed_mean_k3"] for r in rs)
            if not med <= T:  # a band over its bar is measured once more and fails only if both readings fail
                again = [dict(a, **stat(a)) for a in self._fidelity_reading(
                    [sq for sq in seqs if str(sq.get("band")) == band], "author_logprobs", "fidelity", 2)]
                med2 = statistics.median(a["trimmed_mean_k3"] for a in again)
                print(f"[inkling fidelity] band {band}: trimmed median {med:.6f} then {med2:.6f} (bar {T})")
                if not med2 <= T:
                    failures.append(f"band {band}: median trimmed k3 {med:.6f} then {med2:.6f}, over {T} on both readings")
            for r in rs:
                if not r["trimmed_mean_k3"] <= C:  # one re-send
                    a = self._fidelity_reading([ref[r["id"]]], "author_logprobs", "fidelity", 2)[0]
                    a.update(stat(a))
                    if not a["trimmed_mean_k3"] <= C:
                        failures.append(f"{r['id']}: trimmed k3 {r['trimmed_mean_k3']:.6f} then {a['trimmed_mean_k3']:.6f}, "
                                        f"over the per-sequence cap {C} on both readings")
        self.assertFalse(failures, "; ".join(failures))

    def test_long_context(self):
        seqs = _load_json(HIDDEN / "long.json.xz")["sequences"]
        rows = self._fidelity_reading(seqs, "reference_logprobs", "long_context", 1)
        print("[inkling long context] " + ", ".join(
            f"{r['id']} {r.get('mean_k3', float('nan')):.5f} finite {r.get('valid')}" for r in rows))
        if CALIBRATION:
            calibrate("test_long_context", {"sequences": rows})
            return
        bars = policy()["long"]
        T, chaos, trim = float(bars["T"]), float(bars["chaos_level"]), int(bars["trim"])
        ref = {sq["id"]: sq for sq in seqs}
        failures = [f"{r['id']}: " + "; ".join(r["problems"]) for r in rows if not r.get("valid")]
        stats = {}
        for r in rows:
            if r.get("valid"):
                stats[r["id"]] = c32(token_k3(ref[r["id"]]["reference_logprobs"], r["served_logprobs"]), trim, chaos)
        _record("long_c32", [{"id": k, **v} for k, v in stats.items()])
        R = float(bars["outlier_rate"])
        rate = sum(v["outliers"] for v in stats.values()) / (sum(v["tokens"] for v in stats.values()) or 1)
        if not rate <= R:  # the two sequences pooled
            failures.append(f"{rate:.5f} of tokens above the chaos level {chaos}, over {R}")
        for k, v in stats.items():
            if not v["trimmed_mean_k3"] <= T:
                a = self._fidelity_reading([ref[k]], "reference_logprobs", "long_context", 2)[0]
                v2 = c32(token_k3(ref[k]["reference_logprobs"], a["served_logprobs"]), trim, chaos) if a.get("valid") else None
                if not (v2 and v2["trimmed_mean_k3"] <= T):
                    failures.append(f"{k}: trimmed k3 {v['trimmed_mean_k3']:.6f} then "
                                    f"{v2['trimmed_mean_k3'] if v2 else 'invalid'}, over {T} on both readings")
        self.assertFalse(failures, "; ".join(failures))

    def test_execution_witness(self):
        import witness_node  # noqa: PLC0415 - sits next to this module

        with operations.operation("invocation", "the serving-side attribution profile"):
            failures, facts = witness_node.observe(self.base_url, _server_pid(self))
        _record("witness", [{**facts, "failures": failures}])
        print(f"[inkling witness] {json.dumps(witness_node.summary(facts), sort_keys=True)}")
        if CALIBRATION:
            calibrate("test_execution_witness", {**facts, "failures": failures})
            return
        self.assertFalse(failures, "; ".join(failures))


_HELPER_KEYS = {
    "test_input_output_logprobs_match_helper": "match",
    "test_input_output_logprobs_match_prefill_cache_hit_helper": "prefill_cache_hit",
    "test_input_output_logprobs_match_decode_cache_hit_helper": "decode_cache_hit",
}


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)


class TestHiddenInklingDeterministic(_upstream.TestInklingSmallNvfp4Deterministic):
    """The registered deterministic class; its three helpers run verbatim with frozen thresholds."""

    @classmethod
    def setUpClass(cls):
        with operations.operation("startup", "the candidate server's launch (deterministic)"):
            super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        with operations.operation("teardown", "the candidate server's teardown (deterministic)"):
            super().tearDownClass()

    def _run(self, helper, **kwargs):
        key = _HELPER_KEYS[helper.__name__]
        if CALIBRATION:
            threshold, kwargs = math.inf, {**kwargs, **({"min_cache_hit_ratio": 0.0} if "min_cache_hit_ratio" in kwargs else {})}
        else:
            bars = policy()["deterministic"]
            threshold = float(bars["kl_div"][key])
            if "min_cache_hit_ratio" in kwargs:
                kwargs = {**kwargs, "min_cache_hit_ratio": float(bars["decode_cache_hit_ratio"])}
        buf = io.StringIO()
        t0 = time.monotonic()
        with operations.operation("deterministic_helper", f"the {key} helper"), \
                contextlib.redirect_stdout(_Tee(sys.stdout, buf)):
            helper(self.base_url, {self.model: {"kl_div": threshold}}, self.model, max_samples=32,
                   max_new_tokens=_upstream.KL_MAX_NEW_TOKENS, trust_remote_code=True, **kwargs)
        text = buf.getvalue()
        kls = [float(v) for v in __import__("re").findall(r"avg_kl_div=([0-9eE+\-.naninf]+)", text)]
        misses = text.count("cache miss for prompt")
        rec = {"helper": key, "avg_kl_div": kls[-1] if kls else None, "cache_misses": misses,
               "cache_hit_ratio": None if key == "match" else (32 - misses) / 32, "threshold": threshold,
               "wall_s": round(time.monotonic() - t0, 1)}
        _record("deterministic", [rec])
        if CALIBRATION:
            calibrate(f"deterministic_{key}", rec)

    def test_conv_state_history(self):
        """the convolution state restored with a cached prefix must not depend on what ran before it.

        Per window (L0 + 2 tokens), at this deterministic launch: flush, a cold greedy continuation of the first
        L0 + 1 tokens (cold A), the identical request, which restores the cached prefix (restored A, c_A), a decoy
        from another window, the identical request again (restored B, c_B), flush, a second decoy, and the
        identical request once more (cold B). **restored A and restored B must be identical, and -- when the
        policy grades the cold pair -- cold A and cold B must be identical**: the same token ids and the same
        output log-probabilities, compared exactly. A request handed the cached state cannot depend on its
        history; one handed the slot's leftovers does.

        There is no tolerance and nothing calibrated: any difference is a failure of the deterministic contract,
        which the instruction already requires. A window is structurally invalid if a run returns other than the
        requested tokens with one log-probability each, if the two restores reuse different prefixes
        (c_A != c_B), if the reuse is below the floor (c_A < 0.5 x L0), or if a decoy is cached; it is re-issued
        once and then named. The first differing position and its |difference|, and the cold-against-restored flip
        this launch reads, are recorded, not graded."""
        doc = _load_json(HIDDEN / "conv_state.json.xz")
        windows = doc["windows"]
        rows = []
        for wi, w in enumerate(windows):
            # A window's requests (two flushes, six continuations; the same again on a re-issue) are each far
            # below http_request's limit; the window runs as one operation under that limit.
            with operations.operation(operations.request_kind(len(w["token_ids"])),
                                      f"the conv-state history window {w['id']}"):
                rec = conv_state.history_probe(self.base_url, windows, wi, new_tokens=int(doc["new_tokens"]),
                                               timeout=operations.limit_s("http_request"))
            rows.append(rec)
        summary = conv_state.history_summarize(rows)
        _record("conv_state_history", rows)
        print(f"[inkling conv state history] restored pairs differing {summary['restored_pair_differing']} of "
              f"{summary['windows']}; cold pairs differing {summary['cold_pair_differing']}; structurally invalid "
              f"{summary['structurally_invalid']}; re-issued {summary['reissued']}; c_A = c_B "
              f"{summary['c_A_equals_c_B']}; decoys cached {summary['decoys_cached']}; flips cold against restored "
              f"{summary['flips_cold_vs_restored']}")
        if CALIBRATION:
            calibrate("test_conv_state_history", {"summary": summary, "windows": rows})
            return
        bar = policy()["conv_state_history"]
        failures = [f"{r['id']}: " + "; ".join(r["structural_problems"]) for r in rows if r["structural_problems"]]
        for r in rows:
            if not r["restored_pair_identical"]:
                failures.append(
                    f"{r['id']}: the two restored runs differ at generated position {r['restored_first_diff']} "
                    f"({r['restored_diff_kind']}, |d| {r['restored_abs_d']}): the request's outputs depend on the "
                    f"requests before it, so the restore is not reproducing the cached state")
        if bar.get("grade_cold_pair"):
            for r in rows:
                if not r["cold_pair_identical"]:
                    failures.append(
                        f"{r['id']}: the two cold runs differ at generated position {r['cold_first_diff']} "
                        f"({r['cold_diff_kind']}, |d| {r['cold_abs_d']}): a request that restores nothing still "
                        f"depends on the requests before it, so its slot is not being cleared")
        self.assertFalse(failures, "; ".join(failures))
