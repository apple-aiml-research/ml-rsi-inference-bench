#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The hidden nodes, run inside the registered class's own server launch."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
APP_DIR = Path(os.environ.get("APP_DIR", "/app"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(TESTS_DIR))

from infb_efficiency import operations  # noqa: E402 - the task's operation registry, one instance with the plugin's
from infb_efficiency.nodes import EfficiencyMixin  # noqa: E402 - the task's own, under /tests

X = json.loads((TESTS_DIR / "config.json").read_text())["infb"]

_UPSTREAM = APP_DIR / "test" / X["registered_test_path"]


def _load_upstream():
    spec = importlib.util.spec_from_file_location("infb_upstream_registered", _UPSTREAM)
    if spec is None or spec.loader is None:  # pragma: no cover - a missing file is a verifier fault
        raise RuntimeError(f"cannot import the registered test at {_UPSTREAM}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_upstream = _load_upstream()

_KIT = sys.modules.get("sglang.test.kits.server_sanity_kit")
if _KIT is not None:
    _KIT._REQUEST_TIMEOUT = operations.limit_s("http_request")  # noqa: SLF001 - the kit's own module constant

EVAL_DIR = APP_DIR / "eval" / "deepseek_v4_flash"

OBSERVER_DIR = Path(__file__).resolve().parent / "observer"
SPEC_RECORDS = Path(os.environ.get("VERIFIER_DIR", "/logs/verifier")) / "records" / "spec-observer"
if (OBSERVER_DIR / "sitecustomize.py").is_file():
    os.environ["INFB_SPEC_OBSERVER_DIR"] = str(SPEC_RECORDS)
    os.environ["PYTHONPATH"] = os.pathsep.join([str(OBSERVER_DIR), *filter(None, [os.environ.get("PYTHONPATH")])])


def _eval_module(name: str, relative: str):
    """A module of the shared runtime materials, loaded once by path under a fixed name."""
    full = f"deepseek_v4_flash_eval_{name}"
    if full not in sys.modules:
        spec = importlib.util.spec_from_file_location(full, EVAL_DIR / relative)
        if spec is None or spec.loader is None:  # pragma: no cover - a missing file is a verifier fault
            raise RuntimeError(f"cannot import {EVAL_DIR / relative}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full] = module
        spec.loader.exec_module(module)
    return sys.modules[full]


def _verifier_fault(node_id: str, reason: str) -> None:
    """A scored node whose result is a verifier fault, by id, for disposition.py (`verifier-faults.json`)."""
    path = Path(os.environ.get("VERIFIER_DIR", "/logs/verifier")) / "verifier-faults.json"
    faults = json.loads(path.read_text()) if path.is_file() else {}
    faults[node_id] = reason[:2000]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(faults, indent=1, sort_keys=True) + "\n")


def _server_pid(case) -> int | None:
    """The candidate server's PID for the attempt records (the registered class keeps its process as `process`)."""
    proc = getattr(type(case), "process", None) or getattr(case, "process", None)
    return getattr(proc, "pid", None)


class TestHiddenDSV4(EfficiencyMixin, getattr(_upstream, X["registered_test_class"])):
    """Upstream's registered class plus this task's hidden nodes, in one server launch."""

    @classmethod
    def setUpClass(cls):
        with operations.operation("startup", "the candidate server's launch"):
            super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        with operations.operation("teardown", "the candidate server's teardown"):
            super().tearDownClass()

    def _sanity_generate(self, prompt: str, max_new_tokens: int, stop=None) -> str:
        with operations.operation("http_request", f"sanity request ({len(prompt)} chars)"):
            return super()._sanity_generate(prompt, max_new_tokens, stop)

    def test_health(self):
        """Upstream's probe, with the public HTTP limit in place of its 10 s literal."""
        import requests  # noqa: PLC0415

        with operations.operation("http_request", "GET /health"):
            resp = requests.get(self.base_url + "/health", timeout=operations.limit_s("http_request"))
        self.assertEqual(resp.status_code, 200)

    def test_health_generate(self):
        """Upstream's probe, with the public HTTP limit in place of its 60 s literal."""
        import requests  # noqa: PLC0415

        with operations.operation("http_request", "GET /health_generate"):
            resp = requests.get(self.base_url + "/health_generate", timeout=operations.limit_s("http_request"))
        self.assertEqual(resp.status_code, 200)

    def test_streaming_response(self):
        with operations.operation("http_request", "streaming request"):
            super().test_streaming_response()

    def test_gsm8k(self):
        """Upstream's own GSM8K on its default rows (a diagnostic): one concurrent batch, one operation."""
        with operations.operation("http_request", "upstream GSM8K batch"):
            super().test_gsm8k()

    def test_gsm8k_hidden_slice(self):
        from sglang.test.run_eval import run_eval  # noqa: PLC0415 - after the applied install

        slice_path = TESTS_DIR / "hidden" / "gsm8k_slice.jsonl"
        assert slice_path.is_file(), f"the hidden slice is missing at {slice_path}"
        args = _eval_module("check_accuracy", "tests/check_accuracy.py").gsm8k_args(
            self.base_url, self.model, str(slice_path), X["hidden_slice"]["num_examples"],
            X["hidden_slice"]["num_shots"])
        with operations.operation("http_request", "hidden GSM8K batch"):
            metrics = run_eval(args)
        floor = float(X["bars"]["gsm8k_hidden_floor"])
        print(f"[DSV4 hidden slice] GSM8K {metrics=} floor={floor}")
        self.assertGreaterEqual(metrics["score"], floor)

    def test_spec_accept_length(self):
        """Diagnostic: upstream's own acceptance-length measurement, not a gate."""
        spec = _eval_module("check_speculation", "tests/check_speculation.py")
        values = []
        for i in range(3):
            with operations.operation("http_request", f"acceptance-length request {i + 1}"):
                values += spec.acceptance_lengths(self.base_url, 1)
        best = max(values, key=lambda v: v[0])
        print(
            f"[DSV4 diagnostic] speculative acceptance length {best[0]:.3f} "
            f"(three attempts: {[round(v[0], 3) for v in values]}), "
            f"bs=1 decode speed {best[1]:.1f} tok/s "
            f"(attempts: {[round(v[1], 1) for v in values]})"
        )

    def test_fidelity_kl(self):
        """The served log-probabilities against the model-author implementation, on held-out text."""
        import fidelity  # noqa: PLC0415 - sits next to this module; see the sys.path line above
        import records  # noqa: PLC0415 - sits next to this module

        hidden = TESTS_DIR / "hidden" / "fidelity"
        bars = X.get("bars", {})
        if not hidden.is_dir() or not any(hidden.glob("*.json")):
            self.skipTest("no fidelity reference shipped yet")
        log = records.AttemptLog("test_fidelity_kl", "fidelity", _server_pid(self))
        value, per_sequence, positions = fidelity.measure(self.base_url, hidden, on_attempt=log, check="fidelity")
        worst = max(per_sequence.items(), key=lambda kv: kv[1])
        by_kind: dict[str, list[float]] = {}
        for name, v in per_sequence.items():
            by_kind.setdefault(name.split("-", 1)[0], []).append(v)
        bands = fidelity.band_medians(per_sequence)
        print(
            f"[DSV4 fidelity] median k3 {value:.6f} over {len(per_sequence)} sequences "
            f"({positions} positions); worst {worst[0]} {worst[1]:.6f}; "
            f"per band " + ", ".join(f"{b} {v:.6f}" for b, v in bands.items()) + "; "
            f"per kind " + ", ".join(f"{k} {fidelity.median(v):.4f}" for k, v in sorted(by_kind.items()))
        )
        T = bars.get("fidelity_T")
        if T is None:
            self.skipTest("fidelity_T is not published yet; the value above is the measurement")
        failures = self._bands_failing(fidelity, bands, bars["fidelity_T_bands"], fidelity.load_reference(hidden),
                                       "fidelity", log)
        cap = float(bars["fidelity_per_sequence_cap_factor"]) * float(T)
        over = fidelity.worst_over_cap(per_sequence, cap)
        if over:
            failures.append(f"{len(over)} sequence(s) above the per-sequence cap {cap:.6f}: "
                            + ", ".join(f"{k} {v:.6f}" for k, v in over[:5]))
        self.assertFalse(failures, "; ".join(failures))

    def test_generation_replay(self):
        """20 generation histories of the candidate's own greedy output, each
        replayed cold three times, judged by S1'-S4 against the node's frozen bars."""
        self._generated_replay("test_generation_replay", "generation")

    def test_prefix_branch_replay(self):
        """16 shared-prefix branch pairs, A B A B warm without a flush, then three
        cold replay passes of all four, judged by S1'-S4 against the node's frozen bars."""
        self._generated_replay("test_prefix_branch_replay", "prefix")

    def _generated_replay(self, node: str, which: str) -> None:
        import generated_replay  # noqa: PLC0415 - the private driver beside this module
        import records  # noqa: PLC0415

        fixtures = TESTS_DIR / "hidden" / "fidelity"
        schedule = generated_replay.build_schedule(fixtures)
        try:
            generated_replay.check_manifest(
                schedule, json.loads((TESTS_DIR / "hidden" / "kernel_state_v49.json").read_text())["replay"])
        except (RuntimeError, KeyError, OSError) as exc:
            _verifier_fault(f"infb_verify.test_hidden.TestHiddenDSV4.{node}", f"the replay schedule: {exc}")
            raise AssertionError(f"censored: a verifier fault, not the candidate's result: {exc}") from exc
        run = generated_replay.run_generation if which == "generation" else generated_replay.run_prefix
        observer = generated_replay.SpecObserver(SPEC_RECORDS)
        with observer:
            obs = run(self.base_url, fixtures, schedule, node=node,
                      op=lambda kind, label: operations.operation(kind, label))
        summary = generated_replay.summarize(obs)
        coverage = observer.coverage()
        d = records.record_dir()
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{node}-summary.json").write_text(json.dumps({"summary": summary, "speculation": coverage}, indent=1,
                                                           default=str))
        s = summary["statistics"]
        print(f"[DSV4 {which} replay] {summary['n']} sequences, {s['replay_passes']} cold replays each: "
              f"S1' {s['S1p']:.6g}, S2' {s['S2p']}, S3' {s['S3p']:.6g}, S4 {s['S4']:.6g} over {s['tokens']} tokens "
              f"({s['decisive_tokens']} decisive); diagnostics max mean k3 {summary['max_mean_k3']:.6g}, max |d| "
              f"{summary['max_abs_d']:.6g}; cached tokens on warm requests "
              f"{[s2['cached_tokens'] for s2 in summary['sequences'] if s2.get('visit') is not None]}; speculative "
              f"coverage {json.dumps(coverage, sort_keys=True)}")
        bars = (X.get("generated_replay") or {}).get("bars", {}).get(node)
        if not bars:
            self.skipTest("the node's bars are not frozen yet (the calibration rule asks for at least 20 "
                          "correct runs of this schedule with R = 3, each alone, on at least 8 hosts, over both "
                          "routes); the values above are the measurement")
        failures = generated_replay.failures(obs, bars)
        self.assertFalse(failures, "; ".join(failures[:12]))

    def _bands_failing(self, fidelity, bands, band_bars, records, label, log):
        """Each band at or over its bar is drawn once more, whole; it fails only if both draws are.
        Both draws are printed. Returns one failure line per band that failed both. A malformed answer in the
        re-draw raises from `redraw_band` and is decisive; only a structurally valid band is ever re-drawn."""
        not_below = fidelity.bands_not_below(bands, band_bars)
        if not not_below:
            return []
        again = {b: fidelity.redraw_band(self.base_url, records, b, on_attempt=log, check=log.check)
                 for b, _, _ in not_below}
        print(f"[DSV4 {label}] band(s) at or over the bar on the first draw, drawn once more: "
              + ", ".join(f"{b} {v:.6f} then {again[b]:.6f} (bar {bar:.6f})" for b, v, bar in not_below))
        return [f"band {b}: median k3 {v:.6f} then {again[b]:.6f}, at or over its bar {bar:.6f} on both draws"
                for b, v, bar in not_below if not again[b] < bar]

    def test_long_context_fidelity(self):
        """The held-out long set against the model-author implementation, each at exactly its own length."""
        import fidelity  # noqa: PLC0415
        import records  # noqa: PLC0415

        long_dir = TESTS_DIR / "hidden" / "long"
        bars = X["bars"]
        leave_out = int(bars["long_per_sequence_cap_leave_out"])
        log = records.AttemptLog("test_long_context_fidelity", "long_context", _server_pid(self))
        value, per_sequence, trimmed, positions = fidelity.measure_leaving_out(self.base_url, long_dir, leave_out,
                                                                               on_attempt=log, check="long_context")
        worst = max(per_sequence.items(), key=lambda kv: kv[1])
        bands = fidelity.band_medians(per_sequence)
        print(
            f"[DSV4 long context] median k3 {value:.6f} over {len(per_sequence)} sequences "
            f"({positions} positions); worst {worst[0]} {worst[1]:.6f}; "
            f"per band " + ", ".join(f"{b} {v:.6f}" for b, v in bands.items()) + "; "
            + ", ".join(f"{k} {v:.4f}" for k, v in sorted(per_sequence.items()))
            + f"; leaving out {leave_out} terms, for the cap: "
            + ", ".join(f"{k} {v:.4f}" for k, v in sorted(trimmed.items()))
        )
        refs = fidelity.load_reference(long_dir)
        failures = self._bands_failing(fidelity, bands, bars["long_T_bands"], refs, "long context", log)
        cap = float(bars["long_per_sequence_cap"])
        over = fidelity.worst_over_cap(trimmed, cap)
        if over:
            by_id = {r["id"]: r for r in refs}
            again = {k: fidelity.rescore_leaving_out(self.base_url, by_id[k], leave_out, draw=2, on_attempt=log,
                                                     check="long_context") for k, _ in over}
            print(f"[DSV4 long context] above the cap {cap:.6f} (k3 leaving out its {leave_out} largest terms) on the "
                  f"first reading, sent once more: " + ", ".join(f"{k} {v:.6f} then {again[k]:.6f}" for k, v in over))
            still = [(k, again[k]) for k, _ in over if not again[k] <= cap]
            if still:
                failures.append(f"{len(still)} long sequence(s) above the per-sequence cap {cap:.6f} (k3 leaving out "
                                f"its {leave_out} largest terms) on both readings: "
                                + ", ".join(f"{k} {dict(over)[k]:.6f} then {v:.6f}" for k, v in still))
        self.assertFalse(failures, "; ".join(failures))

    def test_long_reach_fidelity(self):
        """Two held-out sequences of 262,144 tokens, prose and code, each against the
        landing's recorded log-probabilities -- the oracle fallback, since the model-author implementation cannot score
        this length on four B200s -- by k3 leaving out its `long_reach_leave_out` largest terms, below its kind's bar
        `long_reach_T`. A sequence at or over its bar is sent once more and fails only if it is at or over it again."""
        import fidelity  # noqa: PLC0415
        import records  # noqa: PLC0415

        bars, leave_out = X["bars"]["long_reach_T"], int(X["bars"]["long_reach_leave_out"])
        readings, failures = {}, []
        log = records.AttemptLog("test_long_reach_fidelity", "long_reach", _server_pid(self))
        refs = fidelity.load_reference(TESTS_DIR / "hidden" / "long-reach")
        for record in refs:
            bar = float(bars[record["kind"]])
            first = fidelity.rescore_leaving_out(self.base_url, record, leave_out, on_attempt=log, check="long_reach")
            again = None if first < bar else fidelity.rescore_leaving_out(self.base_url, record, leave_out, draw=2,
                                                                          on_attempt=log, check="long_reach")
            readings[record["id"]] = (first, again, bar)
            if again is not None and not again < bar:
                failures.append(f"{record['id']}: k3 leaving out its {leave_out} largest terms {first:.6f} then "
                                f"{again:.6f}, at or over its bar {bar:.6f} on both readings")
        print("[DSV4 long reach] " + ", ".join(
            f"{k} {first:.6f}" + ("" if again is None else f" then {again:.6f}") + f" (bar {bar:.6f})"
            for k, (first, again, bar) in sorted(readings.items())))
        self.assertFalse(failures, "; ".join(failures))

    def test_prefix_cache(self):
        """Prefix caching on, reused, and changing nothing beyond the engine's own noise (prefix_cache.py)."""
        import prefix_cache  # noqa: PLC0415 - sits next to this module
        import records  # noqa: PLC0415

        bars = X["bars"]
        log = records.AttemptLog("test_prefix_cache", "prefix", _server_pid(self))
        result = prefix_cache.measure(self.base_url, TESTS_DIR / "hidden" / "prefix", on_attempt=log)
        per_item = result["per_item"]
        print(
            f"[DSV4 prefix cache] warm-against-cold median k3 {result['median_k3']:.6f} over {len(per_item)} items; "
            f"cached tokens " + ", ".join(f"{k} {v['cached_tokens']}" for k, v in sorted(per_item.items()))
        )
        short = {k: v["cached_tokens"] for k, v in per_item.items() if v["cached_tokens"] < int(bars["prefix_min_cached"])}
        self.assertFalse(short, f"a repeated {X['prefix']['prefix_tokens']}-token prefix was not served from the "
                                f"cache: cached_tokens {short}, below {bars['prefix_min_cached']}")
        T = float(bars["prefix_T"])
        if not result["median_k3"] <= T:
            again = prefix_cache.measure(self.base_url, TESTS_DIR / "hidden" / "prefix", draw=2, on_attempt=log)
            print(f"[DSV4 prefix cache] over the bar {T:.6f} on the first draw, drawn once more: "
                  f"{result['median_k3']:.6f} then {again['median_k3']:.6f}")
            self.assertLessEqual(again["median_k3"], T, f"warm-against-cold median k3 {result['median_k3']:.6f} "
                                                        f"then {again['median_k3']:.6f}, over the bar {T:.6f} on both draws")

    def test_recipe_conformance(self):
        """The release recipe as served, on every rank (served_recipe.py), and
        what the recipe actually executes during ordinary warm serving, on every rank (recipe_witness.py): the target's
        verify forwards replayed from captured graphs doing the model's work, the draft model's work, and returned
        tokens beyond one per executed verify step. One node, one score; each witness profile is one instrumented
        invocation (dispatch through all ranks' records read back), the traces read outside it."""
        import recipe_witness  # noqa: PLC0415 - sits next to this module
        import records  # noqa: PLC0415
        import served_recipe  # noqa: PLC0415 - sits next to this module

        declared = int(json.loads((Path(X["model_dir"]) / "config.json").read_text())["max_position_embeddings"])
        cmap_path = Path(X["kernels"]["records"]) / "candidate" / "component-map.json"
        cmap = json.loads(cmap_path.read_text()) if cmap_path.is_file() else None
        roles, notes = recipe_witness.role_families(cmap)
        attention = sorted(roles["packed_attention"].names) if "packed_attention" in roles else None
        with operations.operation("invocation", "served configuration and kernel families"):
            failures, facts = served_recipe.check(self.base_url, declared, attention)
        facts["attribution_notes"] = notes
        print(f"[DSV4 recipe] {json.dumps(facts, sort_keys=True, default=str)}")
        families = {**served_recipe.TRTLLM_MXFP4, **roles}
        workload = recipe_witness.load_workload(TESTS_DIR / "hidden" / "recipe_witness.json")
        info = served_recipe.server_info(self.base_url)
        facts_w: dict = {"source": recipe_witness.source_identity(APP_DIR), "server_pid": _server_pid(self)}
        try:
            wfail, wfacts = recipe_witness.witness(
                self.base_url, workload, int(info.get("tp_size") or 4), families, info,
                invocation=lambda label: operations.operation("invocation", label),
                request=lambda label: operations.operation("http_request", label),
                draft_families=(*served_recipe.TRTLLM_MXFP4, *recipe_witness.DRAFT_ROLES))
            wfacts["component_map"] = {"path": str(cmap_path), "roles": {k: sorted(v.names) for k, v in roles.items()},
                                       "unattributable": notes}
            facts_w.update(wfacts)
            failures += wfail
        except recipe_witness.WitnessFault as exc:
            facts_w["witness_fault"] = str(exc)
            if not failures:
                _verifier_fault("infb_verify.test_hidden.TestHiddenDSV4.test_recipe_conformance",
                                f"the recipe witness could not observe the served execution: {exc}")
                raise AssertionError(f"censored: a verifier fault, not the candidate's result: {exc}") from exc
            failures.append(f"(the execution witness was not observed: {exc})")
        finally:
            d = records.record_dir()
            d.mkdir(parents=True, exist_ok=True)
            (d / "recipe_witness.json").write_text(json.dumps({**facts_w, "failures": failures}, indent=1,
                                                              sort_keys=True, default=str))
        print(f"[DSV4 recipe witness] {json.dumps(recipe_witness.summary(facts_w), sort_keys=True)}")
        self.assertFalse(failures, "; ".join(failures))
