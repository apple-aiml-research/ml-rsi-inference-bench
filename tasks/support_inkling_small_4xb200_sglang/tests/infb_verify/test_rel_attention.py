#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The operator contract's four scored nodes: correctness on the private cases and component efficiency.

Expected verdicts, stated first: the oracle adapter passes all four; an adapter that fails any private case under the
numerical policy fails its correctness node; a correct operator slower than 0.8 of the E2 arm on any family of a node,
by GPU or by wall time, fails that efficiency node.

  test_prefill_correctness   RG1-RG16 and PE1-PE6, both kinds, every family (rel_cases.py)
  test_decode_correctness    PD1-PD11, both kinds, every family
  test_prefill_efficiency    prefill_full and prefill_local: ratio = median_E2 / median_candidate >= 0.8, GPU and wall
  test_decode_efficiency     decode_full and decode_local, likewise

Each (node, kind, family) group of cases is one component invocation (limits.json `invocation`, 1,800 s) in its own
process, with the candidate arm's private caches; every case's record is written before the next case runs. The E2
arm's medians were measured in the reference phase by the same protocol (`rel_attention/measure.py`), on GPU 0 with no
server resident, from a private seed identical in both arms; they are read from `INFB_E2_MEASURE`.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
APP_DIR = Path(os.environ.get("APP_DIR", "/app"))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(TESTS_DIR))

from infb_efficiency import operations  # noqa: E402

import rel_cases  # noqa: E402

RECORDS = Path(os.environ.get("INFB_RECORD_DIR", "/logs/verifier/records")) / "rel_attention"
PY = os.environ.get("INFB_PY", "/opt/sglang/bin/python")
ADAPTER = str(APP_DIR / "inkling_rel_attention.py")
RHO = 0.8
TIMING_SEED = json.loads((TESTS_DIR / "hidden" / "efficiency_seed.json").read_text())["seed"]


def _candidate_env() -> dict:
    env = dict(os.environ)
    root = Path(os.environ.get("INFB_CANDIDATE_CACHE", "/tmp/infb-candidate-cache"))
    for var, sub in (("TRITON_CACHE_DIR", "triton"), ("TORCH_EXTENSIONS_DIR", "torch_extensions"),
                     ("SGLANG_CACHE_DIR", "sglang"), ("CUTE_DSL_CACHE_DIR", "cute_dsl"),
                     ("FLASH_ATTENTION_CUTE_DSL_CACHE_DIR", "fa4"), ("TILELANG_CACHE_DIR", "tilelang"),
                     ("FLASHINFER_WORKSPACE_BASE", "flashinfer"), ("XDG_CACHE_HOME", "xdg")):
        env[var] = str(root / sub)
    env["CUDA_VISIBLE_DEVICES"] = env.get("INFB_COMPONENT_GPU", "0")
    return env


def _invoke(argv: list[str], label: str) -> subprocess.CompletedProcess:
    limit = operations.limit_s("invocation")
    with operations.operation("invocation", label):
        try:
            return subprocess.run(argv, env=_candidate_env(), cwd="/tmp", capture_output=True, text=True, timeout=limit)
        except subprocess.TimeoutExpired as exc:
            raise operations.OperationLimitExceeded(f"{label}: over the {limit:g} s invocation limit") from exc


def _correctness(node: str) -> None:
    RECORDS.mkdir(parents=True, exist_ok=True)
    groups = sorted({(s["kind"], s["family"]) for s in rel_cases.matrix() if s["node"] == node})
    failures = []
    for kind, family in groups:
        out = RECORDS / f"{node}-{kind}-{family}.json"
        p = _invoke([PY, str(HERE / "rel_cases.py"), "run", "--node", node, "--kind", kind, "--family", family,
                     "--adapter", ADAPTER, "--out", str(out)], f"{node} correctness {kind} {family}")
        (RECORDS / f"{node}-{kind}-{family}.log").write_text(p.stdout[-200000:] + p.stderr[-200000:])
        results = json.loads(out.read_text())["results"] if out.is_file() else []
        expected = sum(1 for s in rel_cases.matrix() if s["node"] == node and s["kind"] == kind and s["family"] == family)
        bad = [r for r in results if not r.get("passed")]
        if len(results) != expected:
            failures.append(f"{kind} {family}: {len(results)} of {expected} cases recorded (rc {p.returncode}): "
                            f"{(p.stderr or p.stdout)[-400:]}")
        failures += [f"{r['case']}: {r.get('reason') or 'over the bound'} (max|o-r| {r.get('max_abs_err')}, bound "
                     f"{r.get('bound')})" for r in bad]
    assert not failures, f"{len(failures)} failure(s): " + "; ".join(failures[:20])


def _efficiency(families: tuple[str, ...]) -> None:
    e2 = {r["family"]: r for r in json.loads(Path(os.environ.get(
        "INFB_E2_MEASURE", "/logs/verifier/records/e2-measure.json")).read_text())["results"]}
    RECORDS.mkdir(parents=True, exist_ok=True)
    failures, rows = [], []
    for fam in families:
        out = RECORDS / f"measure-{fam}.json"
        p = _invoke([PY, str(APP_DIR / "eval" / "inkling_small" / "rel_attention" / "measure.py"), "--family", fam,
                     "--adapter", ADAPTER, "--seed", TIMING_SEED, "--output", str(out)], f"component timing {fam}")
        if not out.is_file():
            failures.append(f"{fam}: no measurement (rc {p.returncode}): {(p.stderr or p.stdout)[-400:]}")
            continue
        cand = json.loads(out.read_text())["results"][0]
        ref = e2[fam]
        gpu = ref["gpu_s_median"] / cand["gpu_s_median"]
        wall = ref["wall_s_median"] / cand["wall_s_median"]
        rows.append({"family": fam, "gpu_ratio": gpu, "wall_ratio": wall, "candidate": cand, "e2": ref})
        if not (gpu >= RHO and wall >= RHO):
            failures.append(f"{fam}: GPU ratio {gpu:.3f}, wall ratio {wall:.3f} (both must be >= {RHO})")
    (RECORDS / f"efficiency-{'-'.join(families)}.json").write_text(json.dumps(rows, indent=1))
    print("[inkling component efficiency] " + ", ".join(f"{r['family']} gpu {r['gpu_ratio']:.3f} wall {r['wall_ratio']:.3f}"
                                                         for r in rows))
    assert not failures, "; ".join(failures)


def test_prefill_correctness():
    _correctness("prefill")


def test_decode_correctness():
    _correctness("decode")


def test_prefill_efficiency():
    _efficiency(("prefill_full", "prefill_local"))


def test_decode_efficiency():
    _efficiency(("decode_full", "decode_local"))


@pytest.fixture(scope="session", autouse=True)
def _binding_identity():
    """The witness's component side: the candidate adapter's four identity sets, recorded for
    `test_execution_witness`; not a node of its own."""
    yield
    out = Path(os.environ.get("INFB_BINDING_IDS", "/logs/verifier/records/binding.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    p = _invoke([PY, str(APP_DIR / "eval" / "inkling_small" / "rel_attention" / "binding.py"), "--adapter", ADAPTER,
                 "--output", str(out)], "component attribution profile")
    if p.returncode:
        (out.parent / "binding.log").write_text(p.stdout[-100000:] + p.stderr[-100000:])
