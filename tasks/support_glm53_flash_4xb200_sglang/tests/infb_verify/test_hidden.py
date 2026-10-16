#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The hidden deployment classes."""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import requests

from sglang.test.run_eval import run_eval

TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
HIDDEN_SLICE = TESTS_DIR / "hidden" / "gsm8k_slice.jsonl"
BARS = json.loads((TESTS_DIR / "config.json").read_text())["infb"]["bars"]
RECIPE_RESOLVED = json.loads((TESTS_DIR / "config.json").read_text())["infb"].get("recipe_resolved", {})

from infb_efficiency.nodes import EfficiencyMixin  # noqa: E402

_upstream = importlib.import_module("registered.models_e2e.test_glm53_flash_b200")


def _server_info(base_url: str) -> dict:
    try:
        info = requests.get(base_url + "/server_info", timeout=60).json()
        states = info.get("internal_states") or [{}]
        return {"avg_spec_accept_length": states[0].get("avg_spec_accept_length")}
    except Exception as exc:  # noqa: BLE001 - a diagnostic, never a verdict
        return {"error": repr(exc)[:200]}


class _HiddenSliceMixin:
    hidden_class: str
    fidelity_dp_ranks: tuple = ()
    SERVICE_FREE_NODES = frozenset({"test_zz_adaptive_schedule", "test_attention_semantics"})

    def setUp(self):
        super().setUp()
        from infb_verify.service import ServiceTracker
        cls = type(self)
        if "_service" not in cls.__dict__:
            cls._service = ServiceTracker(cls.__name__)
        if self._testMethodName in self.SERVICE_FREE_NODES:
            return
        status = cls._service.before(f"{cls.__name__}.{self._testMethodName}", getattr(cls, "process", None),
                                     self.base_url)
        if status == "candidate_fail":
            print(f"[INFB] {cls.__name__}.{self._testMethodName} status=candidate_fail (the server is gone)")
            self.fail("candidate_fail: the class's one server is gone before this node: the primary continued-service "
                      "failure; every later node of the class is blocked by it")
        if status:
            print(f"[INFB] {cls.__name__}.{self._testMethodName} status={status}")
            self.fail(f"{status}: the class's one server is gone, so this node cannot be observed; the primary "
                      f"failure is the node it died in")

    def tearDown(self):
        cls = type(self)
        if self._testMethodName not in self.SERVICE_FREE_NODES and "_service" in cls.__dict__:
            cls._service.after(f"{cls.__name__}.{self._testMethodName}", getattr(cls, "process", None), self.base_url)
        super().tearDown()

    def test_00_recipe_conformance(self):
        """R-RUNTIME: the served configuration equals the recipe as launched, and the attention
        backends the reference resolved to, at the top level and on every data-parallel scheduler (`/server_info`);
        and the class's witness launch observed every GPU rank of the recipe's topology running that configuration with
        the FP8 KV cache, decode steps replayed from CUDA graphs that compute the model and produce its logits on two
        numerically valid warm responses, and (LowLatency) verified draft tokens (`witness/judge.py`)."""
        from infb_verify.conformance import conformance
        from infb_verify.records import write_record
        from infb_verify.witness import judge
        label = type(self).__name__
        result = conformance(self.base_url, label, list(type(self).server_args), RECIPE_RESOLVED.get(label))
        witness = judge.runtime(label, list(type(self).server_args), float(BARS["fidelity_T"][self.hidden_class]))
        status = "candidate_fail" if result["mismatches"] else witness["status"]
        write_record("recipe_conformance", label, "verdict", {"status": status, "readback": result, "witness": witness})
        print(f"[INFB] {label} recipe_conformance={json.dumps(result, default=str)} "
              f"runtime_witness={json.dumps(witness, default=str)[:6000]}")
        if result["mismatches"]:
            self.fail(f"candidate_fail: the served configuration departs from the recipe: {result['mismatches'][:8]}")
        if witness["status"] != "pass":
            self.fail(f"{witness['status']}: {witness['reason']}")

    def test_attention_semantics(self):
        """R-ATTENTION: in the class's witness launch, every designated complete-attention tensor of the held-out inputs
        agrees with the author computation on the same captured history within the public allowance, and the observed
        attention result is what serving consumes (`witness/judge.py`, `infb_verify/attention_replay.py`)."""
        from infb_verify.records import write_record
        from infb_verify.witness import judge
        label = type(self).__name__
        result = judge.attention(label, list(type(self).server_args))
        write_record("attention_semantics", label, "verdict", result)
        print(f"[INFB] {label} attention_semantics={json.dumps(result, default=str)[:8000]}")
        if result["status"] != "pass":
            self.fail(f"{result['status']}: {result['reason']}")

    def test_fidelity_kl(self):
        """The fidelity statistic over two context-length bands, the hidden 32 (782-1,530 tokens) and the long group
        (2,600-8,192 tokens), each sent as its own grader request: per sequence k3 against the independent reference,
        and the band median below that band's bar (config.json infb.bars.fidelity_T and fidelity_T_long). The fixed
        policy (`infb_verify/fidelity.py`): the answer's structure is checked before any statistic, a malformed,
        misaligned or non-finite value fails the band at once and is never re-drawn, and a structurally valid band that
        misses its bar is re-drawn once whole, either valid reading passing. Every reading is persisted first; the node
        passes only if both bands pass."""
        from infb_verify.fidelity import FIDELITY_DIR, LONG_DIR, fidelity_band
        label = type(self).__name__
        bands = (("short", "fidelity_kl", FIDELITY_DIR, BARS.get("fidelity_T", {}).get(self.hidden_class),
                  self.fidelity_dp_ranks, label),
                 ("long", "fidelity_kl_long", LONG_DIR, BARS.get("fidelity_T_long", {}).get(self.hidden_class), (),
                  f"{label}-long"))
        failures = []
        for band, key, directory, bar, dp_ranks, store_label in bands:
            if bar is None:
                failures.append(f"verifier_fault: no bar published for the {band} band")
                continue
            result = fidelity_band(self.base_url, store_label, band, directory, float(bar), dp_ranks)
            summary = {"status": result["status"], "reason": result["reason"], "diagnostics": result["diagnostics"],
                       "readings": [{k: r.get(k) for k in ("attempt", "structural", "numerical", "median_k3", "max_k3",
                                                            "per_kind", "elapsed_s", "error", "record")}
                                    for r in result["readings"]]}
            print(f"[INFB] {label} {key}={json.dumps(summary, default=str)}")
            if result["status"] != "pass":
                failures.append(f"{result['status']}: {result['reason']}")
        if failures:
            failures.sort(key=lambda f: not f.startswith("candidate_fail"))
            self.fail("; ".join(failures))

    def test_chat_protocols(self):
        """R-CHAT: the chat-completions protocols on this class's live server, after its earlier workloads --
        ordinary chat, a streamed chat, a subsequent turn and a concurrent pair repeated alone, each checked for
        protocol, template and token-identity structure, then its generation-time log-probabilities against the same
        tokens' solo teacher-forced ones under the fidelity statistic and this class's short-band bar
        (`infb_verify/chat.py`)."""
        from infb_verify.chat import chat_protocols
        label = type(self).__name__
        bar = BARS.get("fidelity_T", {}).get(self.hidden_class)
        self.assertIsNotNone(bar, "verifier_fault: no fidelity bar published for this class")
        result = chat_protocols(self.base_url, label, float(bar))
        print(f"[INFB] {label} chat_protocols={json.dumps(result, default=str)}")
        if result["status"] != "pass":
            self.fail(f"{result['status']}: {result['reason']}")

    def test_gsm8k_hidden_slice(self):
        assert HIDDEN_SLICE.is_file(), f"hidden slice missing: {HIDDEN_SLICE}"
        requests.get(self.base_url + "/flush_cache")
        metrics = run_eval(
            SimpleNamespace(
                base_url=self.base_url,
                model=self.model,
                eval_name="gsm8k",
                num_examples=500,
                num_threads=128,
                api="completion",
                max_tokens=512,
                num_shots=20,
                gsm8k_data_path=str(HIDDEN_SLICE),
            )
        )
        print(f"[INFB] {type(self).__name__} gsm8k_hidden_slice metrics={metrics} server_info={_server_info(self.base_url)}")
        self.assertGreaterEqual(float(metrics["score"]), float(BARS["gsm8k_hidden_floor"][self.hidden_class]))


class TestHiddenHighThroughput(EfficiencyMixin, _HiddenSliceMixin, _upstream.TestGLM53FlashB200HighThroughput):
    hidden_class = "high_throughput"
    fidelity_dp_ranks = (0, 1, 2, 3)


class TestHiddenLowLatency(EfficiencyMixin, _HiddenSliceMixin, _upstream.TestGLM53FlashB200LowLatency):
    hidden_class = "low_latency"

    def test_spec_accept_length(self):
        from glm53_protocols.speculation import acceptance_probe
        bar = float(BARS["spec_accept_length"])
        acc_length, speed, attempts = acceptance_probe(self.base_url, bar)
        print(f"[INFB] {type(self).__name__} spec_accept_length={acc_length:.3f} bs1_speed_tok_s={speed:.2f} attempts={attempts} (this node asserts acceptance only; the speed is asserted by upstream's own test_bs_1_speed)")
        self.assertGreater(acc_length, bar)

    def test_zz_adaptive_schedule(self):
        """R-ADAPTIVE: in the class's witness launch, acceptance feedback
        updates the active controller and the tier it returns drives the draft and verify work under the ordinary C1
        and C16 workloads; the work follows the admitted minimum and maximum tiers when the controller's return value
        is substituted by them, and the controller's state is restored; and the controller replayed from its pre-use
        state decides differently on all-rejected and all-accepted histories in an occupied slot (`witness/judge.py`)."""
        from infb_verify.records import write_record
        from infb_verify.witness import judge
        label = type(self).__name__
        result = judge.adaptive(label)
        write_record("adaptive", label, "verdict", result)
        print(f"[INFB] {label} adaptive_schedule={json.dumps(result, default=str)[:6000]}")
        if result["status"] != "pass":
            self.fail(f"{result['status']}: {result['reason']}")
