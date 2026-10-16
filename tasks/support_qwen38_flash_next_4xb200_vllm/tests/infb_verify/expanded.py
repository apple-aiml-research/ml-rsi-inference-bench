#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Private inputs and verdicts over the public native-region measurement protocol.

The public protocol (`qwen38.measurement`) captures each case through the engine's own runner,
times it by the three-matrix protocol and records memory and use evidence; `run.py components` is
its command. This module supplies the private cases and token content to that same command, and
turns the complete records of the candidate and of the reference (measured earlier on the same GPUs
by the reference phase, with the same command) into verdicts under the private policy. The server's
worker extension is the private `GradingWorker`, whose independent comparator checks every capture
and every timed draw. Missing or malformed evidence fails; it is never a pass.

The retained MTP3 production/state gate (`production`) runs on its own diagnostic launch.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("TESTS_DIR", "/tests"))
for path in (str(ROOT / "eval"), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
from qwen38.measurement import timing  # noqa: E402
from qwen38.serving import recipes  # noqa: E402
from infb_verify import budgets  # noqa: E402
from infb_verify.production_noise import paired_evidence  # noqa: E402
from infb_verify.qwen38_private import cases as inventory  # noqa: E402
from infb_verify.qwen38_private import production as native  # noqa: E402
from infb_verify.qwen38_private import stateful  # noqa: E402
from infb_verify.qwen38_private.observer_client import lifecycle, observation  # noqa: E402

DATA = ROOT / "hidden/expanded"
LOGS = Path(os.environ.get("INFB_EXPANDED_DIR", "/logs/verifier/expanded"))
EFFICIENCY = Path(os.environ.get("INFB_EFFICIENCY_DIR", "/logs/verifier/efficiency"))
RECIPE_KEY = {"ordinary": "test_hidden", "mtp3": "test_mtp"}
GRADING_WORKER = "infb_verify.qwen38_private.worker.GradingWorker"
DIAGNOSTIC_WORKER = "infb_verify.qwen38_private.worker.DiagnosticWorker"# Restoration tolerances of the use witness (QSA 0.016/0.004, PLE exact): unperturbed repeats agree
# within them; the perturbed run leaves them.
RESTORATION = {"qsa": {"rtol": 0.016, "atol": 0.004}, "ple": {"rtol": 0.0, "atol": 0.0}}


def policy() -> dict:
    result = json.loads((DATA / "policy.json").read_text())
    if result.get("status") != "qualified":
        raise RuntimeError("expanded policy has not been qualified and frozen")
    return result


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, allow_nan=False))

# ── collection through the public command ───────────────────────────────────────────────


def candidate_enclosing(tree: str = "/app") -> dict | None:
    """The enclosures the submission's committed declaration measures (read as data, never imported);
    the reference is measured over the same ones. An invalid declaration fails the submission's own
    collection, and the reference is then measured over its defaults."""
    from qwen38.measurement import native
    try:
        return {r["component"]: list(r["encloses"]) for r in native.declaration(tree)}
    except (native.DeclarationError, ValueError, OSError):
        return None


def inputs(recipe: str, *, tag: str, enclosing: dict | None = None) -> dict:
    """The recipe's private cases with their token content, in the public command's input form."""
    cases = inventory.inventory(recipe)
    pool = inventory._pool()
    tokens = {c["id"]: inventory.tokens(c, tag=tag, pool=pool) for c in cases}
    return {"schema": "qwen38-component-inputs/1", "recipe": recipe, "cases": cases, "tokens": tokens,
            "use_tolerance": RESTORATION, "enclosing": enclosing}


def collect(base_url: str, recipe: str, directory: Path, *, tag: str, python: str = sys.executable,
            env: dict | None = None, cwd: str | None = None, enclosing: dict | None = None) -> dict:
    """Run the public `components` command against a server under the collection budget."""
    directory.mkdir(parents=True, exist_ok=True)
    source, output = directory / "component-inputs.json", directory / f"components-{tag}.json"
    write(source, inputs(recipe, tag=tag, enclosing=enclosing))
    limit = budgets.seconds("component_collection")
    killed = False
    with (directory / f"components-{tag}-client.log").open("w") as log:
        try:
            subprocess.run([python, "-I", str(ROOT / "eval/qwen38/run.py"), "components", "--base-url", base_url,
                            "--recipe", recipe, "--inputs", str(source), "--output", str(output),
                            "--budget-s", str(limit), "--matrix-budget-s", str(budgets.seconds("component_matrix"))],
                           stdout=log, stderr=subprocess.STDOUT, env=env, cwd=cwd, timeout=limit + 30, check=False)
        except subprocess.TimeoutExpired:
            killed = True
    if not output.is_file():
        raise AssertionError(f"the component collection wrote no record (see {directory})")
    record = json.loads(output.read_text())
    if killed:
        # The client keeps within its own allowance; a kill here means it did not, and what it
        # observed so far stays on record as incomplete.
        record.update(status="failed", killed_at_budget_s=limit + 30)
    return record


def components(server, recipe: str) -> dict:
    record = collect(server.url_root, recipe, LOGS / recipe, tag="candidate")
    write(LOGS / recipe / "components.json", record)
    return record


def reference(recipe: str) -> dict:
    return json.loads((EFFICIENCY / RECIPE_KEY[recipe] / "components-reference.json").read_text())

# ── record access ───────────────────────────────────────────────────────────────


def _ranks(answers) -> dict:
    if not isinstance(answers, list) or len(answers) != 4 or {a.get("rank") for a in answers} != {0, 1, 2, 3}:
        raise AssertionError("incomplete TP rank answers")
    if any(a.get("status") != "complete" for a in answers):
        raise AssertionError("a TP rank answer failed")
    return {a["rank"]: a["value"] for a in answers}


def rows(record: dict, component: str | None = None, phase: str | None = None, *,
         timed: bool | None = None) -> list[tuple[dict, dict]]:
    """Complete cases of a record in inventory order, optionally one component and phase family, and
    optionally only the timed (or untimed) cases."""
    expected = [c["id"] for c in inventory.inventory(record["recipe"])]
    if list(record.get("cases", {})) != expected:
        raise AssertionError("component case inventory differs from the frozen inputs")
    selected = []
    for cid in expected:
        row = record["cases"][cid]
        case = row["case"]
        if component is not None and case["component"] != component:
            continue
        family = "prefill" if case["mode"] == "prefill" else "decode"
        if phase is not None and family != phase:
            continue
        if timed is not None and case.get("timed", True) != timed:
            continue
        if row.get("status") != "complete":
            raise AssertionError(f"{cid}: {row.get('errors')}")
        selected.append((case, row))
    if not selected:
        raise AssertionError("empty required component family")
    return selected


def mathematics(record: dict, component: str) -> dict:
    """Every case captured, agreeing with the independent comparator on every rank, and every timed
    draw validated (outputs and the probe continuation); the worker refuses anything less."""
    out = []
    for case, row in rows(record, component):
        capture = _ranks(row["capture"][0])
        for rank, value in capture.items():
            # The worker compares every measured and probe output before answering; a disagreement
            # fails the capture itself, so a complete answer carries only agreeing checks.
            steps = [s for s in value["steps"] if s["phase"] != "history"]
            if value.get("expected_source") != "independent" or len(value.get("checks", [])) != len(steps):
                raise AssertionError(f"{case['id']}/rank {rank}: no independent comparison of every call")
        searches = [s for v in capture.values() for c in v["checks"] for s in c.get("selection_searches", [])]
        if any(not s["agreed"] for s in searches):
            raise AssertionError(f"{case['id']}: an output agrees with no admissible selection")
        if case.get("timed", True):
            _timed(case, row)
        out.append({"case": case["id"], "max_abs": max(c["max_abs"] for v in capture.values() for c in v["checks"]),
                    "selection_searches": len(searches)})
    return {"component": component, "cases": out}


def _timed(case: dict, row: dict) -> None:
    """Three complete matrices of validated draws in the case's execution mode on every rank."""
    want = "eager" if case["mode"] == "prefill" else "cuda_graph"
    if len(row.get("matrices", [])) != timing.MATRICES:
        raise AssertionError(f"{case['id']}: {len(row.get('matrices', []))} of {timing.MATRICES} matrices")
    for matrix in row["matrices"]:
        for rank, value in _ranks(matrix).items():
            draws = value.get("draws") or []
            if (value.get("execution") != want or value.get("warmups") != timing.WARMUPS
                    or len(draws) != timing.DRAWS or any(not isinstance(d.get("validation"), dict) for d in draws)):
                raise AssertionError(f"{case['id']}/rank {rank}: a matrix entry is not {timing.WARMUPS} warm-ups "
                                     f"and {timing.DRAWS} validated {want} draws")


def _statistic(row: dict) -> dict:
    return timing.case_statistic([[dict(v, rank=r) for r, v in _ranks(m).items()] for m in row["matrices"]])


def _work(capture: dict) -> list:
    return [{"phase": s["phase"], "scheduled": [v["scheduled"] for v in s["requests"].values()],
             "emitted": [len(v["emitted"]) for v in s["requests"].values()]} for s in capture["executed"]]


def _paired(case: dict, reference_row: dict, candidate_row: dict) -> None:
    """Reference and candidate ran the same logical work: requests, histories, active tokens and
    scheduled progress (speculative acceptance included), in request order."""
    ref = _ranks(reference_row["capture"][0])
    cand = _ranks(candidate_row["capture"][0])
    ref_case = {k: v for k, v in reference_row["case"].items()}
    if ref_case != candidate_row["case"]:
        raise AssertionError(f"{case['id']}: reference and candidate cases differ")
    for rank in range(4):
        if _work(ref[rank]) != _work(cand[rank]):
            raise AssertionError(f"{case['id']}/rank {rank}: reference and candidate did not execute the same "
                                 "logical work")


def speed(record: dict, component: str, phase: str) -> dict:
    """Paired ratio reference/candidate of the three-matrix statistic, every applicable case."""
    bars = policy()
    recipe = record["recipe"]
    if recipe == "mtp3" and component == "qsa" and phase == "prefill":
        raise AssertionError("MTP3 QSA prefill component timing is excluded from this profile")
    ref = reference(recipe)
    if ref.get("protocol") != record.get("protocol") or ref.get("recipe") != recipe:
        raise AssertionError("component reference protocol/recipe differs")
    ref_rows = {case["id"]: row for case, row in rows(ref, component, phase, timed=True)}
    comparisons = []
    for case, row in rows(record, component, phase, timed=True):
        _paired(case, ref_rows[case["id"]], row)
        encloses = {tuple(_ranks(r["capture"][0])[0]["identity"]["encloses"]) for r in (ref_rows[case["id"]], row)}
        if len(encloses) != 1:
            raise AssertionError(f"{case['id']}: reference and candidate were measured over different enclosures "
                                 f"{sorted(encloses)}")
        _timed(case, ref_rows[case["id"]])
        _timed(case, row)
        r, c = _statistic(ref_rows[case["id"]]), _statistic(row)
        comparisons.append({"case": case["id"], "reference_s": r["statistic_s"], "candidate_s": c["statistic_s"],
                            "reference_matrices_s": r["matrix_statistics_s"],
                            "candidate_matrices_s": c["matrix_statistics_s"],
                            "ratio": timing.ratio(r, c)})
    floor = bars["component_ratio_floor"]
    result = {"component": component, "phase": phase, "floor": floor, "comparisons": comparisons,
              "order": {"reference": ref.get("started"), "candidate": record.get("started")}}
    write(LOGS / recipe / f"{component}-{phase}-speed.json", result)
    failures = [row for row in comparisons if not row["ratio"] >= floor]
    assert not failures, failures
    return result


def memory_cap(case: dict, limits: dict, *, channels: int, positions: int) -> int:
    lengths = case["lengths"]
    steps = case.get("steps", 1)
    if case["mode"] == "prefill":
        n = sum(lengths)
    else:
        n = len(lengths) * (4 if case["mode"] == "spec" else 1)
    return (limits["fixed_bytes"] + limits["token_channel_bytes"] * n * channels
            + limits["request_state_bytes"] * len(lengths) * channels * positions
            + limits["retained_output_bytes"] * n * channels * steps)


def memory(record: dict) -> list[dict]:
    limits = policy()["ple_memory"]
    scope = inventory.scope()["ple_memory"]
    out = []
    for case, row in rows(record, "ple", timed=True):
        cap = memory_cap(case, limits, channels=scope["C"], positions=scope["H"])
        for rank, value in _ranks(row["memory"][0]).items():
            if value.get("observer_errors"):
                raise AssertionError(f"{case['id']}/rank {rank}: the allocation observer lost events")
            out.append({"case": case["id"], "rank": rank, "bytes": value["total_bytes"], "cap_bytes": cap,
                        "transient_peak_bytes": value["transient_peak_bytes"],
                        "persistent_auxiliary_bytes": value["persistent_auxiliary_bytes"],
                        "state_excess_bytes": value["state_excess_bytes"],
                        "driver_events": value["driver_events"]})
    write(LOGS / record["recipe"] / "ple-memory.json", out)
    assert all(0 <= row["bytes"] <= row["cap_bytes"] for row in out), [r for r in out if r["bytes"] > r["cap_bytes"]]
    return out


def native_binding(record: dict) -> dict:
    """One loaded native owner per component layer, bound identically in every record, invoked by
    the runner in every measured step, validated outputs and state progression, and its output
    consumed (a perturbed run changes the next layer's input beyond the variation of an unperturbed
    repeat)."""
    result = {}
    for component in ("qsa", "ple"):
        owners = set()
        for case, row in rows(record, component):
            for rank, value in _ranks(row["capture"][0]).items():
                ident = value["identity"]
                owners.add((ident["owner"], ident["class"], ident["source_sha256"], ident["method"]))
                if not ident.get("state"):
                    raise AssertionError(f"{case['id']}/rank {rank}: the region has no bound state")
        if len({o[0] for o in owners}) != len(owners):
            raise AssertionError(f"{component}: one owner bound under two identities: {owners}")
        use = record["use"].get(component)
        if not use or use.get("status") != "complete":
            raise AssertionError(f"{component}: no use evidence")
        for rank, value in _ranks(use["answers"][0]).items():
            if not value.get("steps"):
                raise AssertionError(f"{component}/rank {rank}: empty use evidence")
            for step in value["steps"]:
                if step.get("tolerance") != RESTORATION[component]:
                    raise AssertionError(f"{component}/rank {rank}: use evidence under another tolerance")
                if step["repeat_beyond_tolerance"]:
                    raise AssertionError(f"{component}/rank {rank}: unperturbed repeats disagree ({step})")
                if not step["perturbed_beyond_tolerance"]:
                    raise AssertionError(f"{component}/rank {rank}: the region's output is not consumed ({step})")
        result[component] = {"owners": sorted(owners), "use_case": use["case"]}
    return {"component_binding": "complete", **result}

# ── the retained MTP3 production/state gate ──────────────────────────────────────────────


def production_mapping(recipe: str) -> list[dict]:
    """The production observers' rows, from the loaded native owners the component record bound."""
    record = json.loads((LOGS / recipe / "components.json").read_text())
    out = []
    for component in ("qsa", "ple"):
        case, row = rows(record, component)[0]
        ident = _ranks(row["capture"][0])[0]["identity"]
        output = ident.get("output", "return")
        out.append({"component": component, "class": ident["class"], "method": ident["method"],
                    "output_argument": None if output == "return" else output["argument"]})
    return out


def owners(record: dict) -> dict:
    """The owner each component's intervention targets: the first inventory case's bound owner."""
    return {component: _ranks(rows(record, component)[0][1]["capture"][0])[0]["identity"]["owner"]
            for component in ("qsa", "ple")}


def join(record: dict, probe: dict) -> list[dict]:
    """Every component case's bound owner, on every rank, observed serving in production."""
    for field in ("source", "recipe"):
        if not record.get(field) or record[field] != probe.get(field):
            raise AssertionError(f"component/production {field} differs")
    ranks = probe["phases"]["normal"]["observation"]["ranks"]
    if len(ranks) != 4 or {r["rank"] for r in ranks} != set(range(4)):
        raise AssertionError("incomplete production ranks")
    serving = {r["rank"]: r["sites"] for r in ranks}
    matches = []
    for case, row in rows(record):
        for rank, value in _ranks(row["capture"][0]).items():
            ident = value["identity"]
            found = [s for s in serving[rank] if s["device_calls"] > 0 and s["identity"]["component"] == case["component"]
                     and s["identity"]["owner"] == ident["owner"] and s["identity"]["method"] == ident["method"]
                     and s["identity"]["source_sha256"] == ident["source_sha256"]]
            if not found:
                raise AssertionError(f"{case['id']}/rank {rank}: {ident['owner']} was not observed serving")
            matches.append({"case": case["id"], "rank": rank, "owner": ident["owner"],
                            "shapes": [s["shape"] for s in found]})
    return matches


def check_lifecycle(report: dict, recipe: str) -> None:
    assert report["status"] == "complete", report
    for rank in report["ranks"]:
        row = rank["value"]
        assert not row.get("failure_count", 0), row
        assert row["nonempty_history_rows"] > 0 and row["restored_prefill_rows"] > 0, row
        if recipe == "mtp3":
            assert row["mixed_spec_prefill_steps"] > 0, row


def state(server, recipe: str, launch_log: Path) -> dict:
    data = json.loads((ROOT / "hidden/fidelity.json").read_text())["sequences"]
    tokens = next(row["token_ids"] for row in data if len(row["token_ids"]) >= 18432)
    lifecycle(server.url_root, action="install")
    report = {"recipe": recipe}
    try:
        report["branches"] = stateful.restored_branches(server.url_root, recipes.MODEL, tokens,
            [tokens[8000:8008], tokens[9000:9010]], 64, case_id="restored-state", recipe=recipe)
        report["mixed"] = stateful.mixed_turnover(server.url_root, recipes.MODEL,
            [row["token_ids"][:200] for row in data[:4]],
            [tokens[:n] for n in (1, 16, 676, 4096)], [tokens[0]], launch_log=launch_log)
        report["mixed_turnover_diagnostic"] = stateful.turnover_diagnostic(report["mixed"]["aligned"])
    finally:
        report["lifecycle"] = lifecycle(server.url_root, action="remove")
        write(LOGS / recipe / "state.json", report)
    check_lifecycle(report["lifecycle"], recipe)
    # Local aligned-history observations accompany the lifecycle coverage and the component state
    # checks; they do not compare diverged histories. An overflowed statistic is +infinity.
    bars = policy()["state"]
    ungraded = set(bars.get("ungraded_mixed_records", ()))
    for family, records in [("branches", report["branches"]["branches"]), ("mixed", report["mixed"]["aligned"])]:
        for record in records:
            if family == "mixed" and record["id"] in ungraded:
                continue
            row = record["statistics"]
            identity = f"{recipe}/{family}/{record['id']}"
            mean = stateful.statistic(row["mean"])
            assert mean <= bars["mean_k3"], (
                f"{identity}: mean k3 {mean} exceeds {bars['mean_k3']}; complete observations are in state.json")
            for width, window in row["windows"].items():
                local = stateful.statistic(window["max_mean"])
                assert local <= bars["max_local_k3"], (
                    f"{identity}: width-{width} k3 {local} at position {window['start']} exceeds "
                    f"{bars['max_local_k3']}; complete observations are in state.json")
    return report


def production(server, recipe: str, component_record: dict, launch_log: Path) -> dict:
    data = json.loads((ROOT / "hidden/fidelity.json").read_text())["sequences"]
    long = next(row["token_ids"] for row in data if len(row["token_ids"]) >= 18432)
    cases = [("use-short", data[0]["token_ids"][:512]), ("use-chunk", long[:4096]), ("use-long", long[:18432])]
    targets = owners(component_record)
    bars = policy()["production"]
    probes = []
    for name, sign in [("positive", 1), ("negative", -1)]:
        probe = native.collect(server.url_root, cases, recipe=recipe,
            output=LOGS / recipe / f"production-{name}.json",
            evaluation_id=os.environ["INFB_EVALUATION_ID"], tokens=32, owners=targets,
            amplitude=sign * bars["intervention_amplitude"])
        probe["join"] = join(component_record, probe)
        for rank in probe["phases"]["normal"]["observation"]["ranks"]:
            for component in ("qsa", "ple"):
                sites = [s for s in rank["sites"] if s["identity"]["component"] == component
                         and s["identity"]["owner"] == targets[component]]
                assert sites and any(s["shape"][0] >= 4096 for s in sites), (rank["rank"], component, "no prefill")
                assert any(s["device_calls"] > s["python_calls"] for s in sites), (rank["rank"], component,
                                                                                   "no graph replay")
        probes.append(probe)
    sensitivity = paired_evidence(probes, bars)
    write(LOGS / recipe / "production-sensitivity.json", sensitivity)
    assert all(row["sensitive"] for row in sensitivity["components"].values()), sensitivity
    result = {"schema": "qwen38-production-qualification/3", "status": "running", "source": probes[0]["source"],
              "protocol": probes[0]["protocol"], "recipe": recipe, "probes": probes, "sensitivity": sensitivity}
    observation(server.url_root, recipe=recipe, reset=True)
    result["state"] = state(server, recipe, launch_log)
    result["mixed_observation"] = observation(server.url_root, recipe=recipe)
    for rank in result["mixed_observation"]["ranks"]:
        assert {s["identity"]["component"] for s in rank["sites"] if s["device_calls"] > 0} == {"qsa", "ple"}
    result["status"] = "complete"
    write(LOGS / recipe / "production-complete.json", result)
    return result
