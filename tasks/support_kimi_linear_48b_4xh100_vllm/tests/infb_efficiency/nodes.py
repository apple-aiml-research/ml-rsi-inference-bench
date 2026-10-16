#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The measurement behind Kimi Linear's two scored whole-model efficiency nodes.

`infb_verify/test_hidden.py` defines them as module functions, first in the module, on the served
session's one candidate server:

  * `test_0_efficiency_decode_c1`  -- S1, per-user decode speed at concurrency 1;
  * `test_0_efficiency_throughput` -- S_C, output throughput at the recipe's load C.

Each calls `run_axis(base_url, axis, "test_hidden")`. The recipe is the one entry of `config.json`
`infb.efficiency.recipes`, keyed by that module's name, with its C, its reference launch and its record
directory, `/logs/verifier/efficiency/test_hidden/`.

Each starts the shared client in its own process -- `python -I`, an allow-listed environment, a
working directory outside `/app`, its own process group -- so no candidate code is imported where the
timing is taken, and kills that group on the way out whatever happened. The client measures the
candidate on the same hidden draw the reference phase measured, updates the core record, and the node
passes iff the record's verdict for its axis is `pass`: the candidate's value at least rho times the
reference's, compared in exact arithmetic. The node's measurement is one measurement invocation under the
published 1,800 s limit (`config.json` `infb.efficiency.deadlines.operation_s`, from
eval/kimi_linear/operation-limits.json); every attempt or draw receives the invocation's remaining allowance as
its client's deadline, and a client still running `CLIENT_MARGIN_S` after it is ended. The per-node alarm
(`deadlines.nodes`, the limit + 60 s) is the composed bound above both, so the limit always acts first.

The candidate's single-request node takes the best of up
to `infb.efficiency.c1_attempts` attempts (1-3, default 3), with an early exit. This is the maintainers' own
practice for this quantity: the corresponding upstream test loops three attempts and stops at the first that
clears.
- The first attempt that meets rho passes the node.
- A retry starts after an attempt that completed below rho, while fewer than `c1_attempts` have run. No
  forecast decides whether it may start; it receives the remaining allowance, and an
  attempt the limit cuts off is the candidate's measurement that did not complete within its limit.
- Every attempt re-runs the client exactly as the first, on the same seed file and so the same draw, warm-ups
  included. The node asserts that the draw is the same.
- The candidate's throughput takes exactly `infb.efficiency.candidate_throughput_draws` draws and their median
  decides; a draw that does not complete ends the measurement, and a partial set of draws is never a median.
  The reference takes `infb.efficiency.reference_draws` draws per axis and keeps the median draw whole.
- The record keeps every attempt, the best attempt as the axis entry and the verdict, and the first attempt's
  ratio as the unbiased value. A published speed uses the first attempt, never the best.
- The paired statistic is the ratio of the two sides' axis values (a ratio of medians on the single-request axis).

The kit's SGLang cache flush, its per-task bar option, its alternative
single-request statistic and its unittest mixin were removed, none of them reachable at this
task's configuration.
"""
from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
RECORD_DIR = Path(os.environ.get("INFB_EFFICIENCY_DIR", "/logs/verifier/efficiency"))
CLIENT_MARGIN_S = 15.0
#: Candidate attempts on the single-request axis, `infb.efficiency.c1_attempts` (1-3).
C1_ATTEMPTS_DEFAULT = 3
C1_ATTEMPTS_MAX = 3
# The reference may be measured more than once per axis, on the already-serving reference, and the median
# draw is kept. Odd counts only, so the kept draw is one that was measured.
REFERENCE_DRAWS_DEFAULT = 1
REFERENCE_DRAWS_ALLOWED = (1, 3, 5)
# The candidate's throughput may be drawn more than once on its serving server, and the median
# draw decides, as the reference keeps its median (a symmetric rule adds no bias to the ratio). decode_c1 keeps its
# best-of attempts.
CANDIDATE_THROUGHPUT_DRAWS_DEFAULT = 1
CANDIDATE_THROUGHPUT_DRAWS_ALLOWED = (1, 3, 5)
# The environment canary brackets every draw on both sides (canary.py says what it reads
# and why). Off unless `infb.efficiency.canary` is set.
CANARY_TOLERANCE_DEFAULT = 0.10
#: What each attempt keeps from the client's axis entry (the full entry survives for the best attempt only).
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


def _deadline(eff: dict, recipe: str, node: str) -> float:
    """The node's measurement allowance: its published operation limit (`deadlines.operation_s`), else its bound."""
    deadlines = eff.get("deadlines") or {}
    for table in ("operation_s", "nodes"):
        entries = deadlines.get(table) or {}
        for key in (f"{recipe}.{node}", node):
            if key in entries:
                return float(entries[key])
    raise KeyError(f"config.json infb.efficiency.deadlines names no limit for {recipe}.{node}")


def _clean_env() -> dict:
    # LD_LIBRARY_PATH is kept for the environment snapshot's nvidia-smi, not for the client, which loads
    # nothing through it.
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
            "--deadline-s", f"{max(1.0, limit_s):g}", "--record", str(record),
            "--side", side, "--rho", str(eff["rho"]), "--quiet"]
    if eff.get("model_name"):
        argv += ["--model", eff["model_name"]]
    if provenance is not None:
        argv += ["--provenance", str(provenance)]
    return argv


def run_client(argv: list[str], limit_s: float, cwd: Path) -> tuple[int, str, bool]:
    """(exit code, output, killed). The client stops itself at this bound, its deadline; if it has not
    `CLIENT_MARGIN_S` later, its whole process group is killed here, and in every case on the way out."""
    cwd.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(argv, env=_clean_env(), cwd=str(cwd), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, start_new_session=True)
    killed = False
    try:
        out, _ = proc.communicate(timeout=max(1.0, limit_s) + CLIENT_MARGIN_S)
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


def c1_attempts(eff: dict) -> int:
    """`infb.efficiency.c1_attempts`: how many candidate attempts the single-request node may take."""
    n = eff.get("c1_attempts", C1_ATTEMPTS_DEFAULT)
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= C1_ATTEMPTS_MAX:
        raise ValueError(f"config.json infb.efficiency.c1_attempts must be an integer from 1 to {C1_ATTEMPTS_MAX}, "
                         f"got {n!r}")
    return n


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


def bar_of(summary: dict, axis: str):
    """The bar an axis is judged against: the record's rho."""
    return summary.get("rho")


def canary_config(eff: dict) -> dict | None:
    """`infb.efficiency.canary`: absent or false, off; true or an object, on, with `tolerance` (0 < t < 1, default
    0.10), and optionally `gpu_iters`, `cpu_iters`, and `command` (an argv that prints one reading, for tests)."""
    c = eff.get("canary")
    if c is None or c is False:
        return None
    if c is True:
        c = {}
    if not isinstance(c, dict):
        raise ValueError(f"config.json infb.efficiency.canary must be true, false or an object, got {c!r}")
    t = c.get("tolerance", CANARY_TOLERANCE_DEFAULT)
    if isinstance(t, bool) or not isinstance(t, (int, float)) or not 0 < t < 1:
        raise ValueError(f"config.json infb.efficiency.canary.tolerance must be a number between 0 and 1, got {t!r}")
    return {**c, "tolerance": float(t)}


def _canary():
    try:
        from . import canary  # noqa: PLC0415 - imported as infb_efficiency.nodes
    except ImportError:
        import canary  # noqa: PLC0415 - a top-level sibling, as reference.py imports this module
    return canary


class Bracket:
    """The canary readings around one side's draws of one recipe, judged against the trial's first reading
    (the record's first available one, which is the reference phase's first). The reading after one draw is the
    reading before the next, so n draws in a row take n + 1 readings, unless that reading was itself below the
    tolerance: then the next draw, a re-draw included, takes a fresh reading first. Everything goes into
    the core record under `canary`; an axis whose draw and re-draw are both invalid goes into
    `summary.environment_invalid`, and the recipe's record directory gets `rerun-request.json`."""

    def __init__(self, eff: dict, python: str | None, record: Path, side: str):
        self.cfg = canary_config(eff)
        self.py = python or eff.get("python") or sys.executable
        self.record, self.side = record, side
        self.last: int | None = None
        self.last_good = False  # the latest reading was not below the tolerance (an unavailable one judges nothing)
        self.took = 0.0  # the latest reading's own duration, which a re-draw's closing reading will take again

    @property
    def on(self) -> bool:
        return self.cfg is not None

    def argv(self) -> list[str]:
        if self.cfg.get("command"):
            return [str(c) for c in self.cfg["command"]]
        argv = [self.py, "-I", str(HERE / "canary.py"), "read"]
        for key in ("gpu_iters", "cpu_iters"):
            if key in self.cfg:
                argv += [f"--{key.replace('_', '-')}", str(int(self.cfg[key]))]
        return argv

    def read(self, axis: str, label: str) -> int:
        """Take one reading and append it to the record; its index there."""
        reading = _canary().take(self.argv(), _clean_env())
        reading.update({"side": self.side, "axis": axis, "label": label})
        self.took = float(reading.get("took_s") or 0.0)
        where: dict = {}

        def put(rec: dict) -> None:
            c = rec.setdefault("canary", {"tolerance": self.cfg["tolerance"], "first": None, "readings": [],
                                          "draws": []})
            c["readings"].append(reading)
            if c.get("first") is None and reading.get("ok"):
                c["first"] = len(c["readings"]) - 1
            where["i"] = len(c["readings"]) - 1
            first = c["readings"][c["first"]] if c.get("first") is not None else None
            where["good"] = _canary().compare(first, reading, self.cfg["tolerance"])["valid"] is not False
        _client().update_record(self.record, put)
        self.last_good = where["good"]
        state = "ok" if reading.get("ok") else f"unavailable ({reading.get('error')})"
        print(f"[canary] {self.side} {label}: {state} in {reading.get('took_s')} s", flush=True)
        return where["i"]

    def before(self, axis: str, label: str) -> int:
        if self.last is None or not self.last_good:
            self.last = self.read(axis, f"before {label}")
        return self.last

    def after(self, axis: str, label: str) -> int:
        self.last = self.read(axis, f"after {label}")
        return self.last

    def judge(self, axis: str, draw: int, redraw: bool, before: int, after: int, value) -> bool | None:
        """Record the draw and its brackets: False when either bracket reads more than the tolerance below the trial's
        first reading, None when neither could be judged, True otherwise."""
        out: dict = {}

        def put(rec: dict) -> None:
            c = rec["canary"]
            first = c["readings"][c["first"]] if c.get("first") is not None else None
            tol = self.cfg["tolerance"]
            cb = _canary().compare(first, c["readings"][before], tol)
            ca = _canary().compare(first, c["readings"][after], tol)
            valid = None if cb["valid"] is None and ca["valid"] is None else \
                not (cb["valid"] is False or ca["valid"] is False)
            c["draws"].append({"side": self.side, "axis": axis, "draw": draw, "redraw": redraw, "value": value,
                               "before": before, "after": after, "valid": valid, "before_vs_first": cb,
                               "after_vs_first": ca})
            out["valid"], out["cb"], out["ca"] = valid, cb, ca
        _client().update_record(self.record, put)
        if out["valid"] is False:
            worst = min((x for x in (out["cb"], out["ca"]) if x["valid"] is False), key=lambda x: x["ratio"])
            print(f"[canary] {self.side} {axis} draw {draw}{' (re-draw)' if redraw else ''}: environment-invalid, "
                  f"{worst['worst']} at {worst['ratio']:.4f} of the trial's first reading", flush=True)
        return out["valid"]

    def environment_invalid(self, axis: str, draw: int, why: str) -> None:
        """The axis cannot be judged on this host: recorded, and a verifier-only re-run is requested."""
        entry = {"side": self.side, "axis": axis, "draw": draw, "why": why}

        def put(rec: dict) -> None:
            rec["summary"].setdefault("environment_invalid", []).append(entry)
            rec["summary"]["rerun_requested"] = True
        rec = _client().update_record(self.record, put)
        request = {"rule": "re-run this trial verifier-only on the same tree and append the re-run "
                           "beside this one; the canary decided only which draws were re-drawn, never a verdict",
                   "recipe": self.record.parent.name, "environment_invalid": rec["summary"]["environment_invalid"],
                   "tolerance": self.cfg["tolerance"], "at": _client().utcnow()}
        (self.record.parent / "rerun-request.json").write_text(json.dumps(request, indent=1) + "\n", encoding="utf-8")
        print("INFB_EFFICIENCY_RERUN_REQUEST " + json.dumps(request, sort_keys=True), flush=True)


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


def _stop_message(axis: str, attempts: list[dict], ref: dict, rho, stopped: str) -> str:
    ratios = ", ".join(f"{a['ratio']:.4f}" if isinstance(a["ratio"], (int, float)) else "none" for a in attempts)
    ref_v = ref.get("value")
    ref_s = f"{ref_v:.2f} tok/s" if isinstance(ref_v, (int, float)) else "no value"
    return (f"{axis}: {len(attempts)} attempt(s) measured ratio(s) {ratios} against the reference's {ref_s}, "
            f"below rho = {rho}; retrying stopped: {stopped}")


def run_axis(base_url: str, axis: str, recipe_name: str, python: str | None = None) -> tuple[bool, str, dict]:
    """Measure the candidate on one axis of one recipe; (passed, message, the axis's summary)."""
    eff = _config()
    recipe = recipe_of(eff, recipe_name)
    node = _AXIS_NODE[axis]
    limit = _deadline(eff, recipe_name, node)
    median_of = candidate_throughput_draws(eff) if axis == "throughput" else 1
    allowed = c1_attempts(eff) if axis == "decode_c1" else median_of
    record = record_dir(recipe_name) / "efficiency.json"
    if not record.is_file():
        return False, f"no efficiency record at {record}: the reference phase did not run", {}
    t0 = time.monotonic()
    attempts: list[dict] = []
    entries: list[dict] = []
    draws: list = []
    stopped = "disabled"
    bracket = Bracket(eff, python, record, "candidate")
    redrawing = False
    while True:
        rows_before = _sidecar_rows(record, json.loads(record.read_text(encoding="utf-8")))
        n = len(attempts) + 1
        before = bracket.before(axis, f"{axis} draw {n}") if bracket.on else None
        remaining = limit - (time.monotonic() - t0)
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
        if bracket.on and not killed:
            # A draw bracketed by a reading more than the tolerance below the trial's first is re-drawn
            # once, whatever its ratio; a second invalid draw, or no time for the re-draw, asks for a re-run
            after = bracket.after(axis, f"{axis} draw {n}")
            valid = bracket.judge(axis, n, redrawing, before, after, entry.get("value"))
            if valid is False and not redrawing:
                # The existing host policy: an invalid draw is re-drawn once, with the remaining allowance.
                attempts.pop()
                entries.pop()
                draws.pop()
                redrawing = True
                continue
            elif valid is False:
                bracket.environment_invalid(axis, n, "the draw and its re-draw both read the canary more than the "
                                                     "tolerance below the trial's first reading")
        redrawing = False
        if last["verdict"] == "pass" and median_of == 1:  # best-of stops at a pass; a median takes every draw
            stopped = "pass"
            break
        if killed or last["status"] != "complete":
            stopped = "not_complete"
            break
        if allowed == 1:
            stopped = "disabled"
            break
        if len(attempts) >= allowed:
            stopped = "exhausted" if median_of == 1 else "drawn"
            break
    if len(set(draws)) != 1:
        raise AssertionError(f"{axis}: the candidate's attempts drew different workloads {draws}; the attempts "
                             f"would not be comparable, so none of them is a verdict")
    best = max(range(len(attempts)), key=lambda i: attempts[i]["value"]
               if isinstance(attempts[i]["value"], (int, float)) else float("-inf"))
    pick = len(attempts) - 1  # a draw that did not complete decides, as a single draw's failure does
    if median_of > 1 and len(attempts) == median_of and all(
            a["status"] == "complete" and not a["killed"] and isinstance(a["value"], (int, float)) for a in attempts):
        # the median of every prescribed draw; never of a partial set
        pick = sorted(range(len(attempts)), key=lambda i: attempts[i]["value"])[len(attempts) // 2]

    def keep(rec: dict) -> None:
        s = rec["summary"]
        if axis == "decode_c1":
            rec["axes"].setdefault("candidate", {})[axis] = {**entries[best], "attempt_index": best + 1,
                                                             "attempts": attempts}
            _client().summarize(rec)
            s.setdefault("attempts", {})[axis] = len(attempts)
            s.setdefault("retry_stopped", {})[axis] = stopped
            s.setdefault("verdict_from", {})[axis] = f"best_of_{len(attempts)}"
        elif median_of > 1:
            rec["axes"].setdefault("candidate", {})[axis] = {
                **entries[pick], "draw_index": pick + 1,
                "draws": [{k: e.get(k) for k in _ATTEMPT_FIELDS} for e in entries]}
            _client().summarize(rec)
            s.setdefault("candidate_draws", {})[axis] = len(attempts)
            s.setdefault("draws_stopped", {})[axis] = stopped
            s.setdefault("verdict_from", {})[axis] = f"median_of_{len(attempts)}"
        else:
            s.setdefault("verdict_from", {})[axis] = "single_draw"
        s.setdefault("first_attempt_ratios", {})[axis] = attempts[0]["ratio"]
        s.setdefault("reference_draws", {})[axis] = reference_draws(eff)
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
            "retry_stopped": stopped if axis == "decode_c1" else None}
    print("[efficiency] " + json.dumps(info, sort_keys=True), flush=True)
    if attempts[-1]["killed"]:
        return False, f"{axis}: the measurement did not complete within its {limit:g} s measurement-invocation limit", info
    if verdict == "pass":
        return True, "", info
    if cand.get("status") != "complete":
        why = cand.get("invalid") or cand.get("status") or "no measurement"
        return False, f"{axis}: the candidate's measurement did not complete: {why}", info
    if axis == "decode_c1":
        return False, _stop_message(axis, attempts, ref, bar_of(summ, axis), stopped), info
    return False, (f"{axis}: the candidate measures {cand.get('value'):.2f} tok/s against the reference's "
                   f"{ref.get('value'):.2f} tok/s, a ratio of {ratio:.4f}, below rho = {bar_of(summ, axis)}"), info
