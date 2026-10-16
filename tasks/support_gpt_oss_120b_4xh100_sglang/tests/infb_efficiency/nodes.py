#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The two scored efficiency nodes, as a mixin for a model-support hidden class.

The upstream class launches the candidate's server in `setUpClass` and sets `self.base_url`; the two
nodes below then run first in the class, because unittest orders methods by name and `test_0_`
sorts before every upstream node:

  * `test_0_efficiency_decode_c1`  -- S1, per-user decode speed at concurrency 1;
  * `test_0_efficiency_throughput` -- S_C, output throughput at the recipe's load C.

Every scored recipe class is one entry of `config.json` `infb.efficiency.recipes`, keyed by the name
of the hidden class (the last component of its junit classname), with its own C, its own reference
launch and its own record directory, `/logs/verifier/efficiency/<recipe>/`.

Each starts the shared client in its own process -- `python -I`, an allow-listed environment, a
working directory outside `/app`, its own process group -- so no candidate code is imported where the
timing is taken, and kills that group on the way out whatever happened. The client measures the
candidate on the same hidden draw the reference phase measured, updates the core record, and the node
passes iff the record's verdict for its axis is `pass`: the candidate's value at least rho times the
reference's, compared in exact arithmetic.

The record keeps every attempt, the best attempt as the axis entry and the verdict, and the first
attempt's ratio as the unbiased value. A published speed uses the first attempt, never the best.
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
C1_ATTEMPTS_DEFAULT = 3
C1_ATTEMPTS_MAX = 3
REFERENCE_DRAWS_DEFAULT = 1
REFERENCE_DRAWS_ALLOWED = (1,)
CANDIDATE_THROUGHPUT_DRAWS_DEFAULT = 1
CANDIDATE_THROUGHPUT_DRAWS_ALLOWED = (1,)
DECODE_C1_STATISTIC_DEFAULT = "ratio_of_medians"
TASK_BAR_RANGE = (Fraction(0), Fraction(2))
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


def invocation_limit(eff: dict) -> float:
    """The published measurement-invocation allowance every client invocation receives."""
    return float((eff.get("deadlines") or {})["invocation_s"])


def _deadline(eff: dict, recipe: str, node: str) -> float:
    nodes = (eff.get("deadlines") or {}).get("nodes") or {}
    for key in (f"{recipe}.{node}", node):
        if key in nodes:
            return float(nodes[key])
    raise KeyError(f"config.json infb.efficiency.deadlines.nodes names no deadline for {recipe}.{node}")


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
    argv = [py, "-I", "-S", str(HERE.parent / "trusted_python.py"),
            str(HERE / "client.py"), "measure", "--engine", eff["engine"], "--base-url", base_url,
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


def decode_c1_statistic(eff: dict) -> str:
    """`infb.efficiency.decode_c1_statistic`: `ratio_of_medians` (default) or `median_of_ratios`."""
    stat = eff.get("decode_c1_statistic", DECODE_C1_STATISTIC_DEFAULT)
    if stat != "ratio_of_medians":
        raise ValueError(f"config.json infb.efficiency.decode_c1_statistic must be one of "
                         f"ratio_of_medians, got {stat!r}")
    return stat


def task_bars(eff: dict) -> dict[str, str]:
    """Both GPT-OSS serving axes use the declared rho."""
    if eff.get("task_bars"):
        raise ValueError("GPT-OSS uses the declared rho for both serving axes")
    return {}


def bar_of(summary: dict, axis: str):
    """The bar an axis is judged against: the task's own (`summary.task_bars`) or else the default rho."""
    return (summary.get("task_bars") or {}).get(axis, summary.get("rho"))


FLUSH_PATHS = {"sglang": "/flush_cache"}


def flush_between_draws(eff: dict) -> bool:
    """`infb.efficiency.flush_between_draws` (default false, so a task at an earlier release grades as before)."""
    v = eff.get("flush_between_draws", False)
    if not isinstance(v, bool):
        raise ValueError(f"config.json infb.efficiency.flush_between_draws must be true or false, got {v!r}")
    return v


def flush_cache(eff: dict, base_url: str, record: Path, side: str, axis: str, draw: int) -> dict:
    """Flush the serving engine's prefix cache before a draw, and record the outcome under `cache_flushes`. A flush that
    fails is recorded, never raised: the `cached_tokens` flag still reads what a draw prefilled from cache."""
    path = FLUSH_PATHS.get(eff.get("engine"))
    entry: dict = {"side": side, "axis": axis, "draw": draw, "at": _client().utcnow()}
    if path is None:
        entry["status"] = "not needed: every request carries its own cache_salt"
    else:
        import urllib.error  # noqa: PLC0415
        import urllib.request  # noqa: PLC0415
        try:
            with urllib.request.urlopen(urllib.request.Request(base_url + path, data=b"", method="POST"),
                                        timeout=60) as r:
                entry["status"] = r.status
                entry["body"] = r.read(200).decode(errors="replace")
        except urllib.error.HTTPError as exc:
            entry["status"] = exc.code
        except Exception as exc:  # noqa: BLE001
            entry["status"] = f"{type(exc).__name__}: {exc}"[:200]
    _client().update_record(record, lambda rec: rec.setdefault("cache_flushes", []).append(entry))
    return entry


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
        argv = [self.py, "-I", "-S", str(HERE.parent / "trusted_python.py"),
                str(HERE / "canary.py"), "read"]
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
    limit = invocation_limit(eff)
    median_of = candidate_throughput_draws(eff) if axis == "throughput" else 1
    allowed = c1_attempts(eff) if axis == "decode_c1" else median_of
    record = record_dir(recipe_name) / "efficiency.json"
    if not record.is_file():
        return False, f"no efficiency record at {record}: the reference phase did not run", {}
    stat = decode_c1_statistic(eff) if axis == "decode_c1" else None
    if stat and stat != DECODE_C1_STATISTIC_DEFAULT:
        _client().update_record(record, lambda rec: rec["summary"].__setitem__("decode_c1_statistic", stat))
    bars = task_bars(eff)
    if bars:
        _client().update_record(record, lambda rec: rec["summary"].__setitem__("task_bars", bars))
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
        if flush_between_draws(eff):
            flush_cache(eff, base_url, record, "candidate", axis, n)
        # A killed client or missing result must not reuse an earlier attempt.
        def clear_attempt(rec):
            rec["axes"].setdefault("candidate", {}).pop(axis, None)
            rec["summary"].setdefault("verdicts", {})[axis] = "fail"
            rec["summary"].setdefault("ratios", {})[axis] = None
        _client().update_record(record, clear_attempt)
        rc, out, killed = run_client(client_argv(eff, recipe, base_url, axis, "candidate", limit, record,
                                                 python=python), limit, record.parent)
        print(out, flush=True)
        if rc == _client().EXIT_WORKLOAD_MISMATCH:
            raise AssertionError(f"{axis}: attempt {len(attempts) + 1} refused a workload different from the record's "
                                 f"(client exit {rc}); attempts on different draws are not comparable")
        if killed or rc not in (_client().EXIT_OK, _client().EXIT_BELOW_RHO):
            def reject_client(rec):
                rec["axes"].setdefault("candidate", {})[axis] = {
                    "status": "failed_client", "value": None, "value_exact": None,
                    "invalid": f"measurement client exited {rc}; killed={killed}",
                }
                _client().summarize(rec)
            _client().update_record(record, reject_client)
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
            after = bracket.after(axis, f"{axis} draw {n}")
            valid = bracket.judge(axis, n, redrawing, before, after, entry.get("value"))
            if valid is False and not redrawing:
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
    by = "ratio" if stat == "median_of_ratios" else "value"  # the paired statistic's best attempt is its best ratio
    best = max(range(len(attempts)), key=lambda i: attempts[i][by]
               if isinstance(attempts[i][by], (int, float)) else float("-inf"))
    pick = len(attempts) - 1  # a draw that did not complete decides, as a single draw's failure does
    if median_of > 1 and all(a["status"] == "complete" and not a["killed"]
                             and isinstance(a["value"], (int, float)) for a in attempts):
        # the median; the lower middle of an even count (a deadline stop), which is the conservative choice
        pick = sorted(range(len(attempts)), key=lambda i: attempts[i]["value"])[(len(attempts) - 1) // 2]

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
            "bar_from": "task_bars" if axis in (summ.get("task_bars") or {}) else "rho",
            "client_exit": attempts[-1]["client_exit"], "killed": attempts[-1]["killed"],
            "attempts": len(attempts), "first_attempt_ratio": attempts[0]["ratio"],
            "retry_stopped": stopped if axis == "decode_c1" else None}
    print("[efficiency] " + json.dumps(info, sort_keys=True), flush=True)
    if attempts[-1]["killed"]:
        return False, f"{axis}: the measurement exceeded the node's deadline of {limit:g} s", info
    if verdict == "pass":
        return True, "", info
    if cand.get("status") != "complete":
        why = cand.get("invalid") or cand.get("status") or "no measurement"
        return False, f"{axis}: the candidate's measurement did not complete: {why}", info
    if axis == "decode_c1":
        return False, _stop_message(axis, attempts, ref, bar_of(summ, axis), stopped), info
    return False, (f"{axis}: the candidate measures {cand.get('value'):.2f} tok/s against the reference's "
                   f"{ref.get('value'):.2f} tok/s, a ratio of {ratio:.4f}, below rho = {bar_of(summ, axis)}"), info


class EfficiencyMixin:
    """Mix into a Day-0 hidden class ahead of the upstream class; `self.base_url` is the candidate's,
    and the class's own name is its recipe's key in `config.json`."""

    def test_0_efficiency_decode_c1(self):
        ok, why, _ = run_axis(self.base_url, "decode_c1", type(self).__name__)
        self.assertTrue(ok, why)

    def test_0_efficiency_throughput(self):
        ok, why, _ = run_axis(self.base_url, "throughput", type(self).__name__)
        self.assertTrue(ok, why)
