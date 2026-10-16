#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Unscored common-input DiT readout: the independent diffusers transformer on a native forward's exact inputs.

`budget` decides how long the readout may take: at most CAP_S, and only out of work time that remains after RESERVE_S
(the clock's TERM/KILL/recovery allowance plus margin), so it never reaches the publication tail. `run` lists this
verifier run's first-step snapshots (parallel dense / TP2 / SP2 when SD3_QFE_COMMON_INPUT=1, batching TP1 / TP2),
binds each to its run, request and rank, then executes the frozen reference generator text
(generate_goldens_inrun._gen_script(), byte-identical, with an empty fixture spec) followed by
common_input_reference_body.py in a separate reference interpreter with the candidate repo off sys.path. Every item
ends with an explicit outcome, including time_budget_exhausted. Stdlib only. Nothing here is a gate: no verdict, bar,
reward or Q/F/E axis reads its files, and it always exits 0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
SCHEMA = "sd3-qfe-common-input-readout/1"
MANIFEST_SCHEMA = "sd3-qfe-common-input-manifest/1"
BODY = HERE / "common_input_reference_body.py"
CAP_S, RESERVE_S, MIN_S = 600, 120, 90

TEARDOWN_S, MIN_CHILD_S = 20, 45
FILE = re.compile(r"rank(\d+)_n(\d+)\.pt")

RUNS = (("parallel_tp2", "parallel", "tp2"), ("parallel_sp2", "parallel", "sp2"), ("parallel_dense", "parallel", "dense"),
        ("batching_tp1", "sd3_day0_batching.json", None), ("batching_tp2", "sd3_day0_batching_tp2.json", None))
LIMITS = {"item_reserve_s": 5.0, "save_reference_bytes_max": 512 * 1024 * 1024}


def sha_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))
    os.replace(tmp, path)


def budget(clock_root: Path, out_dir: Path, now: float | None = None) -> int:
    """Seconds the readout segment may hold, or 0 to skip it. Never more than the work time left minus RESERVE_S."""
    now = time.monotonic() if now is None else now
    try:
        clock = json.loads((clock_root / "clock.json").read_text())
        remaining = float(clock["work_deadline"]) - now
    except (OSError, ValueError, KeyError, TypeError) as exc:
        atomic_json(out_dir / "budget.json", {"scored": False, "cap_s": 0, "decision": "skipped",
                                              "reason": f"clock unreadable: {type(exc).__name__}"})
        return 0
    cap = int(min(CAP_S, remaining - RESERVE_S))
    decision = "run" if cap >= MIN_S else "skipped"
    atomic_json(out_dir / "budget.json", {
        "scored": False, "remaining_work_s": remaining, "cap_limit_s": CAP_S, "reserve_s": RESERVE_S,
        "minimum_s": MIN_S, "cap_s": cap if decision == "run" else 0, "decision": decision,
        "reason": None if decision == "run" else "insufficient remaining work budget; the readout is not started",
        "note": "CPU clock arithmetic only; it does not establish that a GPU readout fits"})
    return cap if decision == "run" else 0


def remap(path: str, prefix: tuple[str, str] | None) -> Path:
    if prefix and path.startswith(prefix[0]):
        return Path(prefix[1] + path[len(prefix[0]):])
    return Path(path)


def events(record: Path) -> list:
    out = []
    if record.is_file():
        for line in record.read_text(errors="replace").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def rank_of(event) -> int | None:
    r = event.get("rank")
    if isinstance(r, dict):
        r = r.get("dist_rank", r.get("rank"))
    try:
        return int(r)
    except (TypeError, ValueError):
        return None


def latent_hw(req: dict) -> list | None:
    try:
        return [int(req["height"]) // 8, int(req["width"]) // 8]
    except (KeyError, TypeError, ValueError):
        return None


def windows(readings: dict, t: float | None) -> list:
    if t is None:
        return []
    return sorted(k for k, r in readings.items()
                  if isinstance(r, dict) and r.get("t0") is not None and r.get("t1") is not None
                  and float(r["t0"]) <= t <= float(r["t1"]))


def run_items(label: str, files_dir: Path, record: Path, readings: dict, requests: dict, mode: str | None) -> list:
    """Items for one run's snapshots: rank 0 is replayed, every other rank is compared with rank 0 on CPU."""
    starts = {}
    for e in events(record):
        if e.get("event") == "batch_dit_start" and e.get("n") is not None:
            starts[(rank_of(e), int(e["n"]))] = e.get("t_start", e.get("t"))
    found = []
    for p in sorted(files_dir.glob("rank*_n*.pt")) if files_dir.is_dir() else []:
        m = FILE.fullmatch(p.name)
        if m:
            found.append((int(m.group(2)), int(m.group(1)), p))
    items, primary = [], {}
    for n, rank, p in sorted(found, key=lambda x: (x[1] != 0, x[0], x[1])):
        t = starts.get((rank, n))
        if t is None:
            t = starts.get((None, n))
        inside = windows(readings, float(t) if t is not None else None)
        keys = [k.split("/", 1)[-1] for k in inside] if mode is None else [k for k in inside if k in requests]

        binding = {"readings_in_flight": inside, "requests": {k: requests.get(k) for k in keys if k in requests}}
        expect = latent_hw(requests[keys[0]]) if len(keys) == 1 and keys[0] in requests else None
        item = {"id": f"{label}/{p.name}", "run": label, "rank": rank, "forward_n": n, "path": str(p),
                "sha256": sha_file(p), "bytes": p.stat().st_size, "t_start": t, "binding": binding,
                "expect_latent_hw": expect, "mode": mode}
        if rank == 0:
            item["kind"] = "replay"
            primary[n] = item
        else:
            base = primary.get(n)
            item.update(kind="rank_agreement", compare_to=base["id"] if base else None,
                        compare_path=base["path"] if base else None, compare_sha256=base["sha256"] if base else None)
        items.append(item)
    return items


def build_manifest(artifacts: Path, run_id: str | None, prefix: tuple[str, str] | None = None) -> dict:
    """This verifier run's snapshots, located through the run's own result files, never by a free directory scan."""
    items, runs = [], {}
    for label, source, mode in RUNS:
        name = "sd3_day0_parallel.json" if source == "parallel" else source
        try:
            res = json.loads((artifacts / name).read_text())
        except (OSError, ValueError):
            runs[label] = {"status": "result_file_missing", "file": name}
            continue
        if run_id is not None and res.get("run_id") != run_id:
            runs[label] = {"status": "foreign_run", "file": name}
            continue
        if source == "parallel":
            run = (res.get("runs") or {}).get(mode) or {}
            if not run.get("common_input_dir"):
                runs[label] = {"status": "capture_not_enabled"}
                continue
            files = remap(run["common_input_dir"], prefix) / "common_inputs"
            record = remap(run.get("record", ""), prefix)
            run_json = record.parent / "run.json"
            try:
                readings = json.loads(run_json.read_text()).get("readings") or {}
            except (OSError, ValueError):
                readings = {}
            requests = res.get("requests") or {}
        else:
            evidence = remap(str(res.get("evidence_dir") or res.get("work") or ""), prefix)
            files = evidence / "tap" / "common_inputs"
            record = evidence / "record.jsonl"
            readings = ((res.get("run") or {}).get("readings")) or {}
            requests = res.get("requests") or {}
        got = run_items(label, files, record, readings, requests, mode)
        runs[label] = {"status": "listed", "files": len(got), "dir": str(files)}
        items.extend(got)
    order = {label: i for i, (label, _, _) in enumerate(RUNS)}
    items.sort(key=lambda it: (it["kind"] != "replay", order[it["run"]], it["forward_n"]))
    recipe = None
    try:
        recipe = (json.loads((artifacts / "sd3_day0_parallel.json").read_text()).get("reference") or {}).get("recipe")
    except (OSError, ValueError):
        pass
    return {"schema": MANIFEST_SCHEMA, "scored": False, "run_id": run_id, "runs": runs, "recipe": recipe,
            "limits": dict(LIMITS), "items": items}


def resolve_model(recipe: dict | None) -> str | None:
    explicit = os.environ.get("SD3_MODEL")
    if explicit:
        return explicit
    for name in ("INFB_MODEL_PATH",):
        staged = os.environ.get(name)
        if staged and Path(staged).is_dir():
            return staged
    link = Path(os.environ.get("INFB_WORKSPACE", "/app")) / "model"
    if link.is_dir():
        return str(link.resolve())
    model = (recipe or {}).get("model")
    return model if model and Path(model).is_dir() else None


def child_env(work: Path, model: str, manifest_path: Path, deadline: float) -> dict:
    """The reference_contract child environment: no INFB_SD3_* knobs, no workspace or tests dir on PYTHONPATH."""
    import generate_goldens_inrun as GI
    ws = Path(os.environ.get("INFB_WORKSPACE", "/app")).resolve()
    env = {k: v for k, v in os.environ.items() if not k.startswith("INFB_SD3_")}
    env["PYTHONPATH"] = os.pathsep.join(
        q for q in env.get("PYTHONPATH", "").split(os.pathsep)
        if q and not Path(q).resolve().is_relative_to(ws) and not Path(q).resolve().is_relative_to(HERE))
    spec = work / "spec.json"
    spec.write_text(json.dumps({"fixtures": [], "smoke_ids": [], "diagnostic_ids": []}))
    env.update(SD3_SPEC=str(spec), SD3_MODEL=model, SD3_OUT=str(work), SD3_REF_DTYPE=GI.REF_SETS["mixed"][0],
               SD3_NEG=" ", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               SD3_READOUT_MANIFEST=str(manifest_path), SD3_READOUT_DEADLINE=repr(deadline))
    return env


def collect(manifest: dict, log: Path, status: str) -> tuple[list, dict]:
    seen = {}
    if log.is_file():
        for line in log.read_text(errors="replace").splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if "id" in rec:
                seen[rec["id"]] = rec
    missing = "time_budget_exhausted" if status in ("killed_at_budget", "not_started_budget") else "not_reached_child_" + status
    items, counts = [], {}
    for item in manifest["items"]:
        rec = seen.get(item["id"]) or {"id": item["id"], "kind": item["kind"], "outcome": missing}
        rec["binding"] = {k: item.get(k) for k in ("run", "rank", "forward_n", "sha256", "bytes", "t_start",
                                                    "binding", "expect_latent_hw", "mode", "trial")}
        meta = rec.get("meta") or {}
        if item.get("mode") and meta.get("request_key") is not None:

            rec["binding_consistent"] = list((item.get("binding") or {}).get("requests") or {}) == [meta["request_key"]]
        counts[rec["outcome"]] = counts.get(rec["outcome"], 0) + 1
        items.append(rec)
    return items, counts


def run(artifacts: Path, out_dir: Path, manifest_in: Path | None = None) -> int:
    import generate_goldens_inrun as GI
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    deadline = float(os.environ.get("SD3_COMMAND_DEADLINE") or (start + CAP_S)) - TEARDOWN_S
    report = {"schema": SCHEMA, "scored": False, "status": "error", "items": [],
              "limits": "Readout only: no tolerance, bar, verdict or credit. A replay compares the native DiT with the "
                        "independent diffusers DiT on the native inputs at the first step; it is component evidence, "
                        "not proof of a complete feature."}
    try:
        if manifest_in is None:
            manifest = build_manifest(artifacts, os.environ.get("SD3_VERIFIER_RUN_ID"))
        else:
            manifest = json.loads(manifest_in.read_text())
            base = manifest_in.parent
            for item in manifest["items"]:
                for key in ("path", "compare_path"):
                    if item.get(key) and not Path(item[key]).is_absolute():
                        item[key] = str(base / item[key])
        if manifest.get("schema") != MANIFEST_SCHEMA:
            raise ValueError("unknown manifest schema")
        manifest_path = out_dir / "manifest.json"
        atomic_json(manifest_path, manifest)
        script = GI._gen_script()
        prefix_sha = hashlib.sha256(script.encode()).hexdigest()
        recipe = manifest.get("recipe") or {}
        model = resolve_model(recipe)
        report.update(manifest_sha256=sha_file(manifest_path), runs=manifest.get("runs"),
                      reference={"profile": "mixed", "generator_prefix_sha256": prefix_sha,
                                 "recipe_generator_sha256": recipe.get("generator_sha256"),
                                 "recipe_model": recipe.get("model"), "model": model,
                                 "body_sha256": sha_file(BODY), "python": os.environ.get("SD3_PYTHON", sys.executable)})
        work = out_dir / "reference_child"
        work.mkdir(exist_ok=True)
        child_status, rc = "not_started_budget", None
        if recipe.get("generator_sha256") not in (None, prefix_sha):
            child_status = "reference_recipe_mismatch"
        elif model is None:
            child_status = "model_unresolved"
        elif recipe.get("model") and Path(model).resolve() != Path(recipe["model"]).resolve():
            child_status = "reference_model_mismatch"
        elif not manifest["items"]:
            child_status = "no_items"
        elif deadline - time.monotonic() >= MIN_CHILD_S:
            code = script + "\n" + BODY.read_text() + "\n_sd3_readout_entry()\n"

            env = child_env(work, model, manifest_path, deadline - TEARDOWN_S)
            with (work / "log.txt").open("w") as log:
                try:
                    proc = subprocess.run([os.environ.get("SD3_PYTHON", sys.executable), "-P", "-B", "-c", code],
                                          cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT,
                                          timeout=max(1.0, deadline - time.monotonic()))
                    rc = proc.returncode
                    child_status = "completed" if rc == 0 else "failed"
                except subprocess.TimeoutExpired:
                    child_status = "killed_at_budget"
        items, counts = collect(manifest, work / "readout.jsonl", child_status)
        replay = [it for it in items if it.get("kind") == "replay"]
        complete = child_status == "completed" and all(it.get("outcome") == "replayed" for it in replay)
        report.update(status="complete" if complete else "incomplete", child_status=child_status,
                      child_returncode=rc, counts=counts, items=items,
                      elapsed_s=time.monotonic() - start, deadline_monotonic=deadline)
    except Exception as exc:  # noqa: BLE001 - an unscored readout reports its own failure and never fails the run
        report.update(status="error", error=f"{type(exc).__name__}: {exc}"[:600])
    atomic_json(out_dir / "readout.json", report)
    print(f"[sd3-qfe-readout] {report['status']} "
          f"{json.dumps(report.get('counts', {}), sort_keys=True)} -> {out_dir / 'readout.json'}", flush=True)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    b = sub.add_parser("budget")
    b.add_argument("--clock-root", type=Path, required=True)
    b.add_argument("--out-dir", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--artifacts", type=Path, required=True)
    r.add_argument("--out-dir", type=Path, required=True)
    r.add_argument("--manifest", type=Path, default=None)
    args = p.parse_args(argv)
    if args.mode == "budget":
        print(budget(args.clock_root, args.out_dir))
        return 0
    return run(args.artifacts, args.out_dir, args.manifest)


if __name__ == "__main__":
    raise SystemExit(main())
