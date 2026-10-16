#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Task-owned KDA/GDN measurement orchestration; imports no inference engine.

Two sources, one per responsibility. The supplied evaluation tree (`/tests/eval/kimi_linear/kda/`, the
grading copy, byte-identical to the supplied `/app/eval/kimi_linear/kda/`) provides the mathematics,
the native-layer mapping, the numerical interpretation, the timing RPC and its client. The private
package (`/tests/infb_kda_private/`) provides the correctness schedule, the GDN preservation cases,
the binding instrumentation and workload, the family timing loop and the verdicts. The same loaded
worker entry is measured by both reference and candidate. This module adds artifact locations, hidden
seeds, the trial's shared canary baseline and the private acceptance values, never a second KDA
implementation.

Each check measures first (`run_client`: a client exit, a timeout or an incomplete record keeps its
classification) and then analyses the recorded measurement under `records.analysis`: an exception
that escapes the analysis is a verifier fault carrying `checker_exception`, never the candidate's.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from infb_verify import records

TESTS = Path(os.environ.get("TESTS_DIR", "/tests"))
PUBLIC = TESTS / "eval/kimi_linear/kda"
PRIVATE = TESTS / "infb_kda_private"
RECORDS = Path(os.environ.get("INFB_KDA_DIR", "/logs/verifier/kda"))
#: Client exit codes that still mean a complete record: 1 is a supplied client's numerical failure,
#: which the private verdicts decide themselves.
COMPLETE_RECORD_EXITS = {"correctness": {0}, "binding": {0}, "gdn": {0}, "timing": {0, 1}, "sample": {0, 1},
                         "inspect": {0}}


def _load(directory: Path, prefix: str, name: str):
    spec = importlib.util.spec_from_file_location(prefix + name, directory / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def public_module(name):
    """A module of the trusted copy of the supplied evaluation tree."""
    return _load(PUBLIC, "infb_kda_", name)


def private_module(name):
    """A module of the private KDA package."""
    return _load(PRIVATE, "infb_kda_private_", name)


def protocol():
    """The supplied measurement protocol: shapes, the sample, the timing families, warmups, draws and repetitions."""
    return json.loads((PUBLIC / "protocol.json").read_text())


def specification():
    """The private specification: the correctness schedule, timing validity and pairing, binding and GDN cases."""
    return json.loads((PRIVATE / "specification.json").read_text())


def criteria():
    """The private acceptance values, unchanged from the qualified criteria."""
    cfg = json.loads((TESTS / "config.json").read_text())
    values = cfg["infb"]["kda"]["acceptance"]
    if values["qualification"] != "qualified":
        raise RuntimeError("KDA criteria have not been GPU-qualified")
    return values


def write_record(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run_client(mode, path, cfg, *, url=None, extra=(), env=None, timeout=None):
    """A bounded, separate trusted client with a durable record even on an RPC failure.

    timing (and sample/inspect) run the supplied client; correctness and binding the private one; gdn the
    private GDN cases. Every invocation is bounded by the published component-invocation limit."""
    settings = cfg["infb"]["kda"]
    command = [cfg["infb"]["efficiency"]["python"], "-B"]
    if mode == "gdn":
        command += [str(PRIVATE / "gdn.py"), "--seed", str(settings["gdn_seed"])]
    else:
        script = PRIVATE / "orchestration.py" if mode in ("correctness", "binding") else PUBLIC / "client.py"
        command += [str(script), "--mode", mode, "--url", url,
                    "--seed", str(settings["seed"]), "--timeout", str(settings["rpc_timeout_s"])]
    command += ["--output", str(path), *map(str, extra)]
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with path.with_suffix(".log").open("w") as log:
        try:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd="/",
                                     env=env, timeout=timeout or settings["client_timeout_s"])
        except subprocess.TimeoutExpired:
            write_record(path.with_suffix(".timeout.json"), {
                "command": command, "elapsed_s": time.monotonic() - started, "timeout": True})
            raise
    if process.returncode not in COMPLETE_RECORD_EXITS[mode]:
        raise RuntimeError(f"{mode} client exited {process.returncode}; {path.with_suffix('.log')}")
    result = json.loads(path.read_text())
    if result.get("completed") is not True:
        raise RuntimeError(f"incomplete {mode} measurement: {path}")
    return result


def check_gdn(cfg, side, env=None):
    if env is None:
        cache = RECORDS / "jit" / "candidate"
        env = dict(os.environ, TRITON_CACHE_DIR=str(cache / "triton"),
                   TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"), VLLM_CACHE_ROOT=str(cache / "vllm"))
    path = RECORDS / f"{side}-gdn.json"
    result = run_client("gdn", path, cfg, env=env)
    with records.analysis("gdn"):
        problems = private_module("verdicts").gdn_problems(result, criteria()["gdn"])
        write_record(RECORDS / f"{side}-gdn-assessment.json", {"problems": problems})
    return problems


def check_math(cfg, side, url):
    result = run_client("correctness", RECORDS / f"{side}-math.json", cfg, url=url)
    with records.analysis("math"):
        problems = private_module("verdicts").correctness_problems(result["result"], criteria()["kda"])
        write_record(RECORDS / f"{side}-math-assessment.json", {"problems": problems})
    return problems


def measure_timing(cfg, side, url, axis, reference=None):
    """Bracket each family's five steady-state samples; repeat an invalid family once.

    The loop is the private `measure_families`. The first available whole-model canary is the trial
    baseline on both sides. Canary evidence only requests a verifier rerun; it never changes a measured
    speed verdict. Candidate work stops after a fully measured failing family. That is a recorded
    numerical/speed failure, not a timeout while needlessly timing larger failing work.
    """
    from infb_efficiency import canary

    values, timing = criteria(), protocol()["timing"]
    orchestration, verdicts = private_module("orchestration"), private_module("verdicts")
    core_path = Path(os.environ.get("INFB_EFFICIENCY_DIR", "/logs/verifier/efficiency")) / "test_hidden/efficiency.json"
    core = json.loads(core_path.read_text())
    core_canary = core.get("canary", {})
    first = core_canary.get("first")
    baseline = core_canary["readings"][first] if first is not None else None
    tolerance = cfg["infb"]["efficiency"]["canary"]["tolerance"]
    python = cfg["infb"]["efficiency"]["python"]
    command = [python, "-I", str(TESTS / "infb_efficiency/canary.py"), "read"]
    environment = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    path = RECORDS / f"{side}-{axis}.json"
    latest = {}

    def sample(family, attempt):
        sample_path = RECORDS / "samples" / f"{side}-family-{family}-attempt-{attempt}.json"
        result = run_client("timing", sample_path, cfg, url=url, extra=(
            "--family", family, "--draws", timing["draws"], "--repeats", timing["repeats"]))
        return {**result, "path": str(sample_path)}

    def assess(family, evidence):
        with records.analysis(f"{axis} timing"):
            report = verdicts.timing_ratios(
                reference["result"], evidence["result"], values["timing"]["floor"],
                values["kda"], axis=axis, family=family)
            evidence["family_assessments"].append(report)
            return bool(report["problems"])

    def write(evidence):
        latest["evidence"] = evidence
        write_record(path, evidence)

    try:
        evidence = orchestration.measure_families(
            axis, sample, lambda: canary.take(command, environment),
            lambda first_reading, reading: canary.compare(first_reading, reading, tolerance),
            baseline=baseline, after_family=assess if reference is not None else None,
            write=write, side=side)
    finally:
        evidence = latest.get("evidence")
        if evidence is not None and evidence["environment_invalid"]:
            request = {"side": side, "axis": axis, "families": evidence["environment_invalid"],
                       "evidence": str(path), "request": "rerun this verifier on the same committed tree"}
            write_record(RECORDS / f"{side}-{axis}-rerun-request.json", request)
            print("INFB_KDA_RERUN_REQUEST " + json.dumps(request), flush=True)
    return evidence


def check_timing(cfg, url, axis):
    path = RECORDS / f"reference-{axis}.json"
    if not path.is_file():
        raise records.ReferenceFault(f"the reference KDA {axis} timing left no record")
    reference = json.loads(path.read_text())
    if not reference.get("completed"):
        raise records.ReferenceFault(f"the reference KDA {axis} timing was not completed")
    candidate = measure_timing(cfg, "candidate", url, axis, reference=reference)
    with records.analysis(f"{axis} timing"):
        verdicts = private_module("verdicts")
        if "stopped_after_failure" in candidate:
            report = candidate["family_assessments"][-1]
            if not report["problems"]:
                raise RuntimeError("a shortened candidate measurement must carry a failing family verdict")
            stopped = candidate["stopped_after_failure"]
            families = [f for f in (range(4) if axis == "prefill" else range(4, 8)) if f > stopped]
            report = {**report, "not_measured": [{"family": f, "status": f"blocked_by:family-{stopped}"}
                                                 for f in families]}
            write_record(RECORDS / f"candidate-{axis}-assessment.json", report)
            return report["problems"]
        values = criteria()
        report = verdicts.timing_ratios(
            reference["result"], candidate["result"], values["timing"]["floor"], values["kda"], axis=axis)
        write_record(RECORDS / f"candidate-{axis}-assessment.json", report)
    return report["problems"]


def check_binding(cfg, url, trace_dir):
    result = run_client("binding", RECORDS / "candidate-binding.json", cfg, url=url,
                        extra=("--trace-dir", trace_dir), timeout=cfg["infb"]["kda"]["binding_timeout_s"])
    with records.analysis("binding"):
        verdicts = private_module("verdicts")
        report = verdicts.binding_problems(result["result"], **criteria()["binding"])
        report["problems"] += private_module("traces").trace_problems(result["result"].get("traces", []))
        for name in ("math", "prefill", "decode"):
            # An earlier measurement that left no complete record is stated, never read.
            path = RECORDS / f"candidate-{name}.json"
            measured = json.loads(path.read_text()) if path.is_file() else None
            if measured is None or not measured.get("completed"):
                report["problems"].append(f"the {name} measurement "
                                          f"{'left no record' if measured is None else 'was incomplete'}")
                continue
            report["problems"] += verdicts.identity_problems(measured["result"], result["result"]["execution"])
        write_record(RECORDS / "candidate-binding-assessment.json", report)
    return report["problems"]
