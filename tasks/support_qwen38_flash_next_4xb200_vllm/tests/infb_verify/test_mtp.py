#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""MTP3: API and independent fidelity, ordinary/MTP3 equivalence, useful drafting and native
component obligations.

One MTP3 grading launch carries whole-serving performance, the API probes, independent fidelity,
complete fixed-length target distributions, acceptance counters and component math/state, speed,
memory and binding. The ordinary session records the same committed source's greedy series. A
separate diagnostic launch carries the retained production/state gate (test_native). Private inputs
and calibrated policies live outside the public protocol implementation; every observation is
validated structurally before any statistic.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from infb_verify import mtp_series, serve, served
from qwen38.measurement import budget  # noqa: E402

TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
X = json.loads((TESTS_DIR / "config.json").read_text())["infb"]
MTP = X["bars"]["mtp"]
LAUNCH_LOG = Path(os.environ.get("INFB_LAUNCH_LOG_DIR", "/logs/verifier")) / "candidate-launch-mtp.log"

if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
from infb_efficiency.nodes import run_axis  # noqa: E402

#: The task keys a recipe by the last component of the scoring nodes' junit classname, which for module-level
#: functions is this module's name (see test_hidden.RECIPE).
RECIPE = "test_mtp"
assert Path(__file__).stem == RECIPE, f"this module is {Path(__file__).stem!r} but RECIPE is {RECIPE!r}"
assert RECIPE in (X.get("efficiency", {}).get("recipes") or {}), f"config.json has no recipe {RECIPE!r}"
#: The candidate's launch and the reference's differ by nothing but the form: the reference phase serves the
#: reference at the recipe's argv, which is test_hidden's plus `speculative_form`.
_FORM = X["efficiency"]["recipes"][RECIPE]["launch"]["speculative_form"]
assert [a for kv in _FORM.items() for a in kv] == serve.MTP_ARGS, (
    f"serve.MTP_ARGS {serve.MTP_ARGS} is not the recipe's speculative_form {_FORM}"
)


@pytest.fixture(scope="session")
def server_mtp():
    yield from serve.launch(extra_args=serve.MTP_ARGS, launch_log=LAUNCH_LOG)


def test_served_configuration(server_mtp):
    report = serve.served_configuration(LAUNCH_LOG.read_text(), "mtp3")
    print("INFB_SERVED_CONFIGURATION " + json.dumps(report))
    assert not report["departures"], report["departures"]


def test_0_efficiency_decode_c1(server_mtp):
    """S1 at the MTP recipe: per-user decode speed at concurrency 1, at least 0.8 of the reference's."""
    ok, why, _ = run_axis(server_mtp.url_root, "decode_c1", RECIPE)
    assert ok, why


def test_0_efficiency_throughput(server_mtp):
    """S_C at the MTP recipe: output throughput at C = 32, at least 0.8 of the reference's."""
    ok, why, _ = run_axis(server_mtp.url_root, "throughput", RECIPE)
    assert ok, why


def test_api_probes(server_mtp):
    """The ordinary recipe's API properties at the MTP recipe: repeat and streamed greedy agreement
    by the near-tie rule, length termination and concurrent first-token isolation, every record validated."""
    result = served.api_probes(server_mtp, "mtp3")
    assert result["holds"], {k: result[k] for k in ("repeat", "stream", "length", "isolation")}


def test_fidelity_kl(server_mtp):
    """The independent 40-sequence teacher data at the MTP recipe, under the unchanged bands, caps and
    reread policy (1.9403 at 12,288 included)."""
    result = served.fidelity(server_mtp, "mtp3")
    report = {key: value for key, value in result.items() if key not in ("reads", "sequences")}
    assert result["passed"], json.dumps(report, allow_nan=False)


def test_mtp_equivalence(server_mtp):
    """Greedy with MTP equals the target's own greedy: every record valid and greedy-consistent (a token outside
    its own run's top-1 is structural), and near-ties only against the series the same tree gave without MTP, at
    `bars.mtp.equivalence_min_prompts` of the sixteen prompts or more."""
    if not mtp_series.SERIES_FILE.is_file():
        pytest.fail(f"the main session recorded no non-speculative series at {mtp_series.SERIES_FILE} "
                    f"(test_hidden.test_greedy_series), so there is nothing to compare the MTP session against")
    base = json.loads(mtp_series.SERIES_FILE.read_text())
    need = int(MTP["equivalence_min_prompts"])
    reports = []
    with budget.operation("mtp_equivalence"):
        for draw in (1, 2):
            s = mtp_series.series(server_mtp.url_root, int(MTP["series_tokens"]), recipe="mtp3")
            mtp_series.record(f"equivalence-{draw}.json", s)
            # Structural errors propagate immediately. They are never a numerical
            # disagreement and cannot consume an allowance or earn a redraw.
            rows = mtp_series.compare(base, s)
            broken = [r for r in rows if not r["rule_holds"]]
            holding = len(rows) - len(broken)
            report = {"draw": draw, "identical": sum(1 for r in rows if r["identical"]), "holding": holding,
                      "need": need, "rows": rows, "broken": broken}
            print("INFB_MTP_EQUIVALENCE " + json.dumps(report))
            reports.append(report)
            if holding >= need:
                return
    last = reports[-1]
    pytest.fail(
        f"greedy decoding with MTP is not the target's own greedy decoding on either draw: the near-tie rule holds "
        f"at the first divergence of {last['holding']} of {len(base['cases'])} prompts against the non-speculative "
        f"series, where {need} are required (broken: {last['broken'][:3]})"
    )


def test_mtp_acceptance(server_mtp):
    """The draft head's proposals are accepted at or above the floor, from the server's own counters over an
    isolated interval (finite, monotone, positive draft activity, accepted <= proposed; an unavailable counter is
    structural, never a zero)."""
    floor = float(MTP["acceptance_floor"])
    draws = []
    with budget.operation("mtp_acceptance"):
        for draw in (1, 2):
            a = mtp_series.acceptance(server_mtp.url_root, int(MTP["acceptance_tokens"]))
            a["draw"] = draw
            mtp_series.record(f"acceptance-{draw}.json", a)
            print("INFB_MTP_ACCEPTANCE " + json.dumps({k: v for k, v in a.items() if k != "series"}))
            draws.append(a)
            if a["acceptance_rate"] >= floor:
                return
    pytest.fail(
        f"the MTP head's acceptance rate is {draws[0]['acceptance_rate']} then {draws[1]['acceptance_rate']} over "
        f"the sixteen hidden prompts at greedy, under the floor {floor}"
    )


from infb_verify import expanded  # noqa: E402


@pytest.fixture(scope='session')
def component_records(server_mtp):
    return expanded.components(server_mtp, 'mtp3')


def test_qsa_mathematics_and_state(component_records):
    expanded.mathematics(component_records, 'qsa')


def test_qsa_decode_efficiency(component_records):
    expanded.speed(component_records, 'qsa', 'decode')


def test_ple_mathematics_and_state(component_records):
    expanded.mathematics(component_records, 'ple')


def test_ple_prefill_efficiency(component_records):
    expanded.speed(component_records, 'ple', 'prefill')


def test_ple_decode_efficiency(component_records):
    expanded.speed(component_records, 'ple', 'decode')


def test_ple_workspace_memory(component_records):
    expanded.memory(component_records)


def test_native_component_binding(component_records):
    expanded.native_binding(component_records)
