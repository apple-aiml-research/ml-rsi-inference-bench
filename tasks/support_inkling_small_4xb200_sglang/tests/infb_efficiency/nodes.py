#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""This task's two scored efficiency nodes, as a mixin for its hidden class.

    class TestHiddenX(EfficiencyMixin, getattr(_upstream, "TestUpstreamClass")):
        ...

The upstream class launches the candidate's server in `setUpClass` and sets `self.base_url`; the two nodes below then
run first in the class, because unittest orders methods by name and `test_0_` sorts before every upstream node:

  * `test_0_efficiency_decode_c1`  -- S1, per-user decode speed at concurrency 1;
  * `test_0_efficiency_throughput` -- S_C, output throughput at the recipe's load C.

The recipe is the `config.json` `infb.efficiency.recipes` entry keyed by the hidden class's name, with its C, its
reference launch and its record directory, `/logs/verifier/efficiency/<recipe>/`.

Each draw starts the client in its own process -- `python -I`, an allow-listed environment, a working directory
outside `/app`, its own process group -- so no candidate code is imported where the timing is taken, and that group is
killed on the way out whatever happened. The client measures the candidate on the same hidden draw the reference phase
measured and updates the core record.

Each axis is drawn `decode_c1_draws` or `candidate_throughput_draws` times (3) on the candidate's serving server, every
draw on the same seed file and so the same draw, warm-ups included, with no cache flush between draws; the node asserts
that every draw measured the same workload. The reference phase kept its median draw per axis. The candidate's median
draw decides: for decode_c1 by the paired statistic (the median over prompts of candidate speed over the kept reference
draw's speed on the same prompt), for throughput by value; the node passes iff that draw's verdict is `pass`, the
comparison in exact arithmetic against rho.

One axis's draws are ONE measurement invocation, an operation with the public 1,800 s limit (`limits.json`
`invocation`). Every prescribed draw is attempted; each draw's client gets the invocation's remaining allowance; there
is no pace-derived stop and no forecast. A median of three is never formed from a subset: an invocation that cannot
complete its three draws is censored as a measurement and fails the node, the draw that did not complete deciding. The
node's own wall (`infb.efficiency.deadlines.nodes`, generated from the deadline DAG) is larger than the invocation's
limit.
"""
from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
RECORD_DIR = Path(os.environ.get("INFB_EFFICIENCY_DIR", "/logs/verifier/efficiency"))
CLIENT_MARGIN_S = 15.0
#: The reference is measured this many times per axis on the already-serving reference, and the median draw is kept.
#: Odd counts only, so the kept draw is one that was measured.
REFERENCE_DRAWS_DEFAULT = 1
REFERENCE_DRAWS_ALLOWED = (1, 3, 5)
#: The candidate's throughput is drawn this many times on its serving server, and the median draw decides.
CANDIDATE_THROUGHPUT_DRAWS_DEFAULT = 1
CANDIDATE_THROUGHPUT_DRAWS_ALLOWED = (1, 3, 5)
#: decode_c1 is drawn this many times on both sides, and the median draw decides.
DECODE_C1_DRAWS_ALLOWED = (3, 5)
#: decode_c1's paired statistic: the median over prompts of candidate speed / reference speed.
DECODE_C1_STATISTIC = "median_of_ratios"
#: What a draw's client needs beyond its measured work: its process start-up and record write (`sizing`).
TEARDOWN_ALLOWANCE_S = 10.0
#: What each draw keeps from the client's axis entry (the full entry survives for the deciding draw only).
_ATTEMPT_FIELDS = ("started_at", "finished_at", "duration_s", "value", "value_exact", "status",
                   "per_request_tok_s", "ttft_s", "tpot_s", "flags")

_AXIS_NODE = {"decode_c1": "test_0_efficiency_decode_c1", "throughput": "test_0_efficiency_throughput"}


def _config() -> dict:
    return json.loads((TESTS_DIR / "config.json").read_text(encoding="utf-8"))["infb"]["efficiency"]


def recipe_of(eff: dict, name: str) -> dict:
    recipes = eff.get("recipes") or {}
    if name not in recipes:
        raise KeyError(f"config.json infb.efficiency.recipes has no entry {name!r}; it has {sorted(recipes)}")
    return recipes[name]


def record_dir(name: str) -> Path:
    """Where one recipe's core record, sidecar and reference-server log live."""
    return RECORD_DIR / name


def _clean_env() -> dict:
    # LD_LIBRARY_PATH is kept for the environment snapshot's nvidia-smi, not for the client, which loads
    # nothing through it. Where the host driver's libraries, NVML among them, are bind-mounted into
    # the container, only that path finds them: without it nvidia-smi cannot read the GPUs and prints
    # an NVML error instead of a device list.
    keep = ("PATH", "HOME", "LANG", "LC_ALL", "TZ", "TMPDIR", "CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "LD_LIBRARY_PATH")
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env.setdefault("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
    return env


def client_argv(eff: dict, recipe: dict, base_url: str, axis: str, side: str, limit_s: float, record: Path,
                provenance: Path | None = None, python: str | None = None) -> list[str]:
    """The one client invocation both sides use, so the two measurements cannot differ in form."""
    py = python or eff.get("python") or sys.executable
    argv = [py, "-I", str(HERE / "client.py"), "measure", "--engine", eff["engine"], "--base-url", base_url,
            "--tokenizer", eff["tokenizer"], "--texts", str(HERE / eff.get("texts", "pool.jsonl.xz")),
            "--seed-file", eff["seed_file"], "--concurrency", str(recipe["C"]), "--axis", axis,
            "--deadline-s", f"{max(1.0, limit_s - 2 * CLIENT_MARGIN_S):g}", "--record", str(record),
            "--side", side, "--rho", str(eff["rho"]), "--quiet"]
    if eff.get("model_name"):
        argv += ["--model", eff["model_name"]]
    if provenance is not None:
        argv += ["--provenance", str(provenance)]
    return argv


def run_client(argv: list[str], limit_s: float, cwd: Path) -> tuple[int, str, bool]:
    """(exit code, output, killed). The client stops itself `CLIENT_MARGIN_S` before this bound; if
    it has not, its whole process group is killed here, and in every case on the way out."""
    cwd.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(argv, env=_clean_env(), cwd=str(cwd), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, start_new_session=True)
    killed = False
    try:
        out, _ = proc.communicate(timeout=max(1.0, limit_s - CLIENT_MARGIN_S))
    except subprocess.TimeoutExpired:
        killed = True
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        out, _ = proc.communicate()
    finally:
        if proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    return proc.returncode, out or "", killed


def _client():
    """This package's client module, for its record lock, summary and sidecar reader. The measurement itself stays in
    the client's own process; this is the verifier's bookkeeping, with the client's own arithmetic."""
    try:
        from . import client  # noqa: PLC0415 - imported as infb_efficiency.nodes
    except ImportError:
        import client  # noqa: PLC0415 - a top-level sibling, as reference.py imports this module
    return client


def reference_draws(eff: dict) -> int:
    """`infb.efficiency.reference_draws`: how many times the reference is measured per axis; the median draw is kept."""
    n = eff.get("reference_draws", REFERENCE_DRAWS_DEFAULT)
    if isinstance(n, bool) or not isinstance(n, int) or n not in REFERENCE_DRAWS_ALLOWED:
        raise ValueError(f"config.json infb.efficiency.reference_draws must be one of {REFERENCE_DRAWS_ALLOWED}, "
                         f"got {n!r}")
    return n


def candidate_throughput_draws(eff: dict) -> int:
    """`infb.efficiency.candidate_throughput_draws`: how many times the candidate's throughput is drawn; the median decides."""
    n = eff.get("candidate_throughput_draws", CANDIDATE_THROUGHPUT_DRAWS_DEFAULT)
    if isinstance(n, bool) or not isinstance(n, int) or n not in CANDIDATE_THROUGHPUT_DRAWS_ALLOWED:
        raise ValueError(f"config.json infb.efficiency.candidate_throughput_draws must be one of "
                         f"{CANDIDATE_THROUGHPUT_DRAWS_ALLOWED}, got {n!r}")
    return n


def decode_c1_draws(eff: dict) -> int:
    """`infb.efficiency.decode_c1_draws`: 3 or 5 draw decode_c1 that many times on both sides and the median draw
    decides."""
    n = eff.get("decode_c1_draws")
    if isinstance(n, bool) or not isinstance(n, int) or n not in DECODE_C1_DRAWS_ALLOWED:
        raise ValueError(f"config.json infb.efficiency.decode_c1_draws must be one of {DECODE_C1_DRAWS_ALLOWED}, "
                         f"got {n!r}")
    return n


def reference_draws_for(eff: dict, axis: str) -> int:
    """How many times the reference is drawn on one axis: `decode_c1_draws` for decode_c1 when it is above 1 (the two
    sides take the same number), else `reference_draws`."""
    k = decode_c1_draws(eff) if axis == "decode_c1" else 1
    return k if k > 1 else reference_draws(eff)


def decode_c1_statistic(eff: dict) -> str:
    """`infb.efficiency.decode_c1_statistic`: this task's is `median_of_ratios`, the one the client computes."""
    stat = eff.get("decode_c1_statistic")
    if stat != DECODE_C1_STATISTIC:
        raise ValueError(f"config.json infb.efficiency.decode_c1_statistic must be {DECODE_C1_STATISTIC!r}, "
                         f"got {stat!r}")
    return stat


def bar_of(summary: dict, axis: str):  # noqa: ARG001 - one bar for both axes
    """The bar an axis is judged against: rho."""
    return summary.get("rho")


def keep_median_draw(entries: list[dict]) -> dict:
    """The reference entry kept from an odd number of draws of one axis: the draw with the median value, whole (its
    per-prompt figures included, which the candidate's attempts are compared against), with its index and every draw's
    figures beside it. Each draw is a complete measurement; the reference phase fails before this on any that is not."""
    if len(entries) % 2 != 1:
        raise ValueError(f"the median draw needs an odd number of draws, got {len(entries)}")
    order = sorted(range(len(entries)), key=lambda i: entries[i]["value"])
    med = order[len(entries) // 2]
    return {**entries[med], "draw_index": med + 1,
            "draws": [{k: e.get(k) for k in _ATTEMPT_FIELDS} for e in entries]}


def _sidecar_rows(record: Path, rec: dict) -> int:
    name = (rec.get("sidecar") or {}).get("requests") or "requests.jsonl.gz"
    return len(_client().read_requests(record.parent / name))


def axis_sizing(eff: dict, recipe_name: str) -> dict[str, dict]:
    """Per axis of the recipe, the time the axis's stated draws need for a tree at the bar, and the time the
    reference's draws need at the pace `sizing` records, beside the measurement invocation's public limit that holds
    them (one invocation per axis on either side, `limits.json` `invocation`). The task's own tests refuse a config
    whose invocation limit falls short.

    The inputs are `infb.efficiency.sizing`:
        {"draw_s": {"<recipe>": {"decode_c1": d, "throughput": t}}, "start_s": s, "measured": "where and how"}
    d and t are the longest `duration_s` of a draw of each axis. s is everything else a node spends besides its draws:
    its setup, and each draw's client start-up.

    k draws of a tree at the bar, each d / bar long, need
        s + k x (d / bar + TEARDOWN_ALLOWANCE_S) + 2 x CLIENT_MARGIN_S,
    where k is decode_c1_draws or candidate_throughput_draws, and the bar is rho; the reference's k draws at the
    recorded pace need s + k x (d + TEARDOWN_ALLOWANCE_S) + 2 x CLIENT_MARGIN_S. Both must fit the invocation
    limit."""
    sizing = eff.get("sizing")
    if not isinstance(sizing, dict):
        raise ValueError("config.json infb.efficiency.sizing is missing: the measured draw durations per "
                         "recipe and axis (`draw_s`), `start_s` and `measured`")
    draws = (sizing.get("draw_s") or {}).get(recipe_name)
    if not isinstance(draws, dict):
        raise ValueError(f"config.json infb.efficiency.sizing.draw_s has no entry for {recipe_name!r}")
    fields = {"start_s": (sizing.get("start_s"), 0.0, "a number of seconds, 0 or more"),
              **{f"draw_s.{recipe_name}.{a}": (draws.get(a), None, "a number of seconds above 0") for a in _AXIS_NODE}}
    for key, (v, floor, what) in fields.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or (v < floor if floor is not None else v <= 0):
            raise ValueError(f"config.json infb.efficiency.sizing.{key} must be {what}, got {v!r}")
    if not str(sizing.get("measured") or "").strip():
        raise ValueError("config.json infb.efficiency.sizing.measured must say where and how the durations were "
                         "measured")
    start = float(sizing["start_s"])
    limit = float(invocation_limit_s())
    out = {}
    for axis in _AXIS_NODE:
        k = decode_c1_draws(eff) if axis == "decode_c1" else candidate_throughput_draws(eff)
        kr = reference_draws_for(eff, axis)
        bar = str(eff.get("rho"))
        d = float(draws[axis])
        d_bar = d / float(Fraction(bar))
        out[axis] = {
            "k": k, "unit": "draws", "bar": bar, "d_s": d, "d_bar_s": round(d_bar, 3), "limit_s": limit,
            "need_s": round(start + k * (d_bar + TEARDOWN_ALLOWANCE_S) + 2 * CLIENT_MARGIN_S, 3),
            "reference_k": kr, "reference_s": limit,
            "reference_need_s": round(start + kr * (d + TEARDOWN_ALLOWANCE_S) + 2 * CLIENT_MARGIN_S, 3)}
    return out


def invocation_limit_s() -> float:
    """The public limit of one measurement invocation (`limits.json` `invocation`): one axis's prescribed draws."""
    return _operations().limit_s("invocation")


def _operations():
    """The package's operation registry, one instance per process: imported as `infb_efficiency.operations` under
    pytest, as a top-level sibling where reference.py imports this module."""
    try:
        from . import operations  # noqa: PLC0415
    except ImportError:
        try:
            import operations  # noqa: PLC0415
        except ModuleNotFoundError:  # loaded by path, with this directory not on sys.path: the sibling by path
            import importlib.util  # noqa: PLC0415

            spec = importlib.util.spec_from_file_location("operations", HERE / "operations.py")
            operations = importlib.util.module_from_spec(spec)
            sys.modules["operations"] = operations
            spec.loader.exec_module(operations)
    return operations


def run_axis(base_url: str, axis: str, recipe_name: str, python: str | None = None) -> tuple[bool, str, dict]:
    """Measure the candidate on one axis of one recipe; (passed, message, the axis's summary)."""
    operations = _operations()
    eff = _config()
    recipe = recipe_of(eff, recipe_name)
    limit = invocation_limit_s()
    median_of = decode_c1_draws(eff) if axis == "decode_c1" else candidate_throughput_draws(eff)
    with operations.operation("invocation", f"{axis} measurement ({median_of} draws)", cap_s=limit):
        return _run_axis(base_url, axis, recipe_name, eff, recipe, limit, median_of, python)


def _run_axis(base_url, axis, recipe_name, eff, recipe, limit, median_of, python):
    record = record_dir(recipe_name) / "efficiency.json"
    if not record.is_file():
        return False, f"no efficiency record at {record}: the reference phase did not run", {}
    stat = decode_c1_statistic(eff) if axis == "decode_c1" else None
    if stat:
        # the record states its own statistic before the first draw, so the client's summarize uses it
        _client().update_record(record, lambda rec: rec["summary"].__setitem__("decode_c1_statistic", stat))
    t0 = time.monotonic()
    attempts: list[dict] = []
    entries: list[dict] = []
    draws: list = []
    stopped = "drawn"
    while True:
        rows_before = _sidecar_rows(record, json.loads(record.read_text(encoding="utf-8")))
        remaining = limit - (time.monotonic() - t0)
        if remaining <= 2 * CLIENT_MARGIN_S + 1.0:
            # the invocation's allowance is spent before its prescribed draws: censored, never a median of a subset
            stopped = "allowance_spent"
            break
        rc, out, killed = run_client(client_argv(eff, recipe, base_url, axis, "candidate", remaining, record,
                                                 python=python), remaining, record.parent)
        print(out, flush=True)
        if rc == _client().EXIT_WORKLOAD_MISMATCH:
            raise AssertionError(f"{axis}: attempt {len(attempts) + 1} refused a workload different from the record's "
                                 f"(client exit {rc}); attempts on different draws are not comparable")
        rec = json.loads(record.read_text(encoding="utf-8"))
        summ = rec.get("summary", {})
        entry = {k: v for k, v in rec.get("axes", {}).get("candidate", {}).get(axis, {}).items()
                 if k not in ("attempts", "attempt_index")}
        entries.append(entry)
        draws.append((rec.get("workload") or {}).get("digest"))
        attempts.append({"index": len(attempts) + 1, **{k: entry.get(k) for k in _ATTEMPT_FIELDS},
                         "ratio": summ.get("ratios", {}).get(axis), "verdict": summ.get("verdicts", {}).get(axis),
                         "client_exit": rc, "killed": killed,
                         "sidecar_rows": [rows_before, _sidecar_rows(record, rec)]})
        last = attempts[-1]
        if killed or last["status"] != "complete":
            stopped = "not_complete"
            break
        if len(attempts) >= median_of:
            stopped = "drawn"
            break
    if len(set(draws)) != 1:
        raise AssertionError(f"{axis}: the candidate's attempts drew different workloads {draws}; the attempts "
                             f"would not be comparable, so none of them is a verdict")
    # the paired statistic's median draw is by ratio; throughput's by value
    by = "ratio" if stat == DECODE_C1_STATISTIC else "value"
    if not attempts:
        raise AssertionError(f"{axis}: the measurement invocation's {limit:g} s allowance was spent before any draw")
    pick = len(attempts) - 1  # a draw that did not complete decides, as a single draw's failure does
    complete = (len(attempts) == median_of and
                all(a["status"] == "complete" and not a["killed"] and isinstance(a[by], (int, float)) for a in attempts))
    if complete:
        # the median of the prescribed draws, never of a subset
        pick = sorted(range(len(attempts)), key=lambda i: attempts[i][by])[(median_of - 1) // 2]

    def keep(rec: dict) -> None:
        s = rec["summary"]
        rec["axes"].setdefault("candidate", {})[axis] = {
            **entries[pick], "draw_index": pick + 1,
            "draws": [{k: e.get(k) for k in _ATTEMPT_FIELDS} for e in entries]}
        _client().summarize(rec)
        s.setdefault("candidate_draws", {})[axis] = len(attempts)
        s.setdefault("draws_stopped", {})[axis] = stopped
        s.setdefault("verdict_from", {})[axis] = f"median_of_{len(attempts)}"
        s.setdefault("first_attempt_ratios", {})[axis] = attempts[0]["ratio"]
        s.setdefault("reference_draws", {})[axis] = reference_draws_for(eff, axis)
    rec = _client().update_record(record, keep)
    summ = rec.get("summary", {})
    cand = rec.get("axes", {}).get("candidate", {}).get(axis, {})
    ref = rec.get("axes", {}).get("reference", {}).get(axis, {})
    verdict = summ.get("verdicts", {}).get(axis)
    ratio = summ.get("ratios", {}).get(axis)
    info = {"recipe": recipe_name, "axis": axis, "reference_tok_s": ref.get("value"), "candidate_tok_s": cand.get("value"),
            "ratio": ratio, "rho": bar_of(summ, axis), "verdict": verdict, "status": cand.get("status"),
            "bar_from": "rho",
            "client_exit": attempts[-1]["client_exit"], "killed": attempts[-1]["killed"],
            "attempts": len(attempts), "first_attempt_ratio": attempts[0]["ratio"],
            "retry_stopped": None}
    print("[efficiency] " + json.dumps(info, sort_keys=True), flush=True)
    if attempts[-1]["killed"] or stopped == "allowance_spent":
        return False, (f"{axis}: the measurement invocation could not complete its {median_of} draws within its "
                       f"{limit:g} s limit ({len(attempts)} drawn): censored, not a verdict from a subset"), info
    if not complete:
        why = attempts[-1].get("status") or "no measurement"
        return False, f"{axis}: the candidate's measurement did not complete: {why}", info
    if verdict == "pass":
        return True, "", info
    if cand.get("status") != "complete":
        why = cand.get("invalid") or cand.get("status") or "no measurement"
        return False, f"{axis}: the candidate's measurement did not complete: {why}", info
    return False, (f"{axis}: the candidate measures {cand.get('value'):.2f} tok/s against the reference's "
                   f"{ref.get('value'):.2f} tok/s, a ratio of {ratio:.4f}, below rho = {bar_of(summ, axis)}"), info


class EfficiencyMixin:
    """Mix into the hidden class ahead of the upstream class; `self.base_url` is the candidate's, and the class's own
    name is its recipe's key in `config.json`."""

    def test_0_efficiency_decode_c1(self):
        ok, why, _ = run_axis(self.base_url, "decode_c1", type(self).__name__)
        self.assertTrue(ok, why)

    def test_0_efficiency_throughput(self):
        ok, why, _ = run_axis(self.base_url, "throughput", type(self).__name__)
        self.assertTrue(ok, why)
