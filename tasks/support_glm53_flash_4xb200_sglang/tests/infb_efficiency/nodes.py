#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The two scored efficiency nodes, as a mixin for a hidden deployment class.

    class TestHiddenX(EfficiencyMixin, getattr(_upstream, "TestUpstreamClass")):
        ...

The upstream class launches the candidate's server in `setUpClass` and sets `self.base_url`; the two
nodes below then run first in the class, because unittest orders methods by name and `test_0_`
sorts before every upstream node:

  * `test_0_efficiency_decode_c1`  -- S1, per-user decode speed at concurrency 1;
  * `test_0_efficiency_throughput` -- S_C, output throughput at the recipe's load C.

Every scored recipe class is one entry of `config.json` `infb.efficiency.recipes`, keyed by the name
of the hidden class (the last component of its junit classname), with its own C, its own reference
launch and its own record directory, `/logs/verifier/efficiency/<recipe>/`. A task whose upstream
acceptance is a plain test function has no class to mix into: it defines two functions of those names
that call `run_axis(base_url, axis, "<the hidden module's name>")`, and keys its one recipe so.

Each starts the shared client in its own process -- `python -I`, an allow-listed environment, a
working directory outside `/app`, its own process group -- so no candidate code is imported where the
timing is taken, and kills that group on the way out whatever happened. The client measures the
candidate on the same hidden draw the reference phase measured, updates the core record, and the node
passes iff the record's verdict for its axis is `pass`: the candidate's value at least rho times the
reference's, compared in exact arithmetic. The node's own deadline comes from `config.json`
(`infb.efficiency.deadlines.nodes`); the client is given a little less, so it stops and records a
censored measurement before the per-node alarm or the group runner has to act.

How the draws run:
- decode_c1 is drawn `infb.efficiency.decode_c1_draws` times on both sides, and throughput
  `candidate_throughput_draws` times against the reference's `reference_draws`; every draw is taken, and the
  median draw decides, by ratio under decode_c1's paired statistic (`median_of_ratios`) and by value otherwise.
- Every prescribed draw is taken; there is no pace-derived cutoff and no forecast that stops collection early.
  Each draw is one measurement invocation under the public 1,800 s limit and receives the node's remaining
  allowance up to it. A draw that exceeds its own 1,800 s on a healthy host is the candidate's resource
  failure; a draw cut off because the node's composed bound ran out while it was still within its own limit is
  censored (`verifier_fault`), and a fixed-N median never decides from a partial subset.
- Before each draw the server's prefix cache is flushed (`flush_between_draws`), and the canary brackets each draw.

The candidate-only best-of attempts (`c1_attempts`) and the relaunch (`infb.efficiency.relaunch`: the
measurement and relaunch nodes, `settle` and `judge`) are not carried.
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
INVOCATION_S = 1800
# The reference may be measured more than once per axis, on the already-serving reference, and the
# median draw is kept. Odd counts only, so the kept draw is one that was measured.
REFERENCE_DRAWS_DEFAULT = 1
REFERENCE_DRAWS_ALLOWED = (1, 3, 5)
# The candidate's throughput may be drawn more than once on its serving server, and the median
# draw decides, as the reference keeps its median (a symmetric rule adds no bias to the ratio). decode_c1 keeps its
# best-of attempts. 1 draws it once.
CANDIDATE_THROUGHPUT_DRAWS_DEFAULT = 1
CANDIDATE_THROUGHPUT_DRAWS_ALLOWED = (1, 3, 5)
# decode_c1's statistic. The default is the ratio of medians; the option is the median over
# prompts of candidate/reference speed, which is the paired statistic of the two sides' per-prompt speeds.
DECODE_C1_STATISTIC_DEFAULT = "ratio_of_medians"
# A task may set its own bar per axis, `infb.efficiency.task_bars` ({"decode_c1": "0.9"}), as
# a decimal string like rho; an axis it does not name keeps the config's rho. The record carries the bars
# (`summary.task_bars`), so the client's summary and every verdict use them and no task needs a side path.
TASK_BAR_RANGE = (Fraction(0), Fraction(2))
# The environment canary brackets every draw on both sides (canary.py says what it reads
# and why). Off unless `infb.efficiency.canary` is set, so a task that does not set it grades exactly as before.
CANARY_TOLERANCE_DEFAULT = 0.10
# The CPU reading has its own tolerance, judged like the GPUs against the running
# median of the trial's readings so far: a whole CPU reading can read 0.836-0.884 of any baseline
# while the draws beside it are normal, where a hog on the serving processes' CPUs reads 0.24-0.39.
CANARY_CPU_TOLERANCE_DEFAULT = 0.25
# decode_c1 may be drawn k times on both sides, and the median draw decides, the same rule
# throughput has. A candidate-only best-of-attempts is asymmetric and biases the ratio upward; a
# symmetric median of k on both sides does not. 1 keeps the attempts.
DECODE_C1_DRAWS_DEFAULT = 1
DECODE_C1_DRAWS_ALLOWED = (1, 3, 5)
#: The sizing rule's allowance for one draw above its measured duration (`sizing`); no draw is stopped by it.
ATTEMPT_HEADROOM = 1.2
TEARDOWN_ALLOWANCE_S = 10.0
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
    nodes = (eff.get("deadlines") or {}).get("nodes") or {}
    for key in (f"{recipe}.{node}", node):
        if key in nodes:
            return float(nodes[key])
    raise KeyError(f"config.json infb.efficiency.deadlines.nodes names no deadline for {recipe}.{node}")


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
    """The client module, for its record lock, summary and sidecar reader. The measurement itself stays in
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
    """`infb.efficiency.decode_c1_draws`: 3 or 5 draw decode_c1 that many times on both sides and the median
    draw decides. The candidate-only attempts (`c1_attempts`) are retired, and a config that still names them is
    refused."""
    n = eff.get("decode_c1_draws", DECODE_C1_DRAWS_DEFAULT)
    if isinstance(n, bool) or not isinstance(n, int) or n not in DECODE_C1_DRAWS_ALLOWED:
        raise ValueError(f"config.json infb.efficiency.decode_c1_draws must be one of {DECODE_C1_DRAWS_ALLOWED}, "
                         f"got {n!r}")
    if "c1_attempts" in eff:
        raise ValueError("config.json infb.efficiency.c1_attempts is retired: decode_c1 is drawn decode_c1_draws times "
                         "on both sides and the median draw decides")
    return n


def reference_draws_for(eff: dict, axis: str) -> int:
    """How many times the reference is drawn on one axis: `decode_c1_draws` for decode_c1 when it is above 1 (the two
    sides take the same number), else `reference_draws`."""
    k = decode_c1_draws(eff) if axis == "decode_c1" else 1
    return k if k > 1 else reference_draws(eff)


def decode_c1_statistic(eff: dict) -> str:
    """`infb.efficiency.decode_c1_statistic`: `ratio_of_medians` (default) or `median_of_ratios`."""
    stat = eff.get("decode_c1_statistic", DECODE_C1_STATISTIC_DEFAULT)
    if stat not in _client().DECODE_C1_STATISTICS:
        raise ValueError(f"config.json infb.efficiency.decode_c1_statistic must be one of "
                         f"{_client().DECODE_C1_STATISTICS}, got {stat!r}")
    return stat


def task_bars(eff: dict) -> dict[str, str]:
    """`infb.efficiency.task_bars`: a task's own bar per axis, each a decimal string in (0, 2]."""
    bars = eff.get("task_bars") or {}
    axes = _client().AXES
    if not isinstance(bars, dict) or not set(bars) <= set(axes):
        raise ValueError(f"config.json infb.efficiency.task_bars must map some of {axes} to a bar, got {bars!r}")
    out = {}
    for axis, bar in bars.items():
        try:
            value = Fraction(str(bar))
        except (ValueError, ZeroDivisionError):
            raise ValueError(f"config.json infb.efficiency.task_bars.{axis} is not a number: {bar!r}") from None
        if isinstance(bar, bool) or not TASK_BAR_RANGE[0] < value <= TASK_BAR_RANGE[1]:
            raise ValueError(f"config.json infb.efficiency.task_bars.{axis} must be in (0, 2], got {bar!r}")
        out[axis] = str(bar)
    return out


def bar_of(summary: dict, axis: str):
    """The bar an axis is judged against: the task's own (`summary.task_bars`) or else the default rho."""
    return (summary.get("task_bars") or {}).get(axis, summary.get("rho"))


def flush_between_draws(eff: dict) -> bool:
    """`infb.efficiency.flush_between_draws` (default false, so a task at an earlier release grades as before)."""
    v = eff.get("flush_between_draws", False)
    if not isinstance(v, bool):
        raise ValueError(f"config.json infb.efficiency.flush_between_draws must be true or false, got {v!r}")
    return v


def flush_cache(eff: dict, base_url: str, record: Path, side: str, axis: str, draw: int) -> dict:
    """Flush the serving engine's prefix cache before a draw, and record the outcome under `cache_flushes`. A flush that
    fails is recorded, never raised: the `cached_tokens` flag still reads what a draw prefilled from cache.

    SGLang's radix cache keeps a draw's prompts, and a later draw of the same prompts prefills from
    it: this task's LowLatency reference prefilled 5,120 new and 322,560 cached tokens on its third
    throughput draw against 327,680 new on its first two. The flush itself is the client's
    `flush_prefix_cache`, the one function the agent's `serving/measure.py` calls too."""
    entry: dict = {"side": side, "axis": axis, "draw": draw, **_client().flush_prefix_cache(base_url)}
    _client().update_record(record, lambda rec: rec.setdefault("cache_flushes", []).append(entry))
    return entry


def canary_config(eff: dict) -> dict | None:
    """`infb.efficiency.canary`: absent or false, off; true or an object, on, with `tolerance` (the GPUs', 0 < t < 1,
    default 0.10), `cpu_tolerance` (the CPU's, default 0.25), and optionally `gpu_iters`, `cpu_iters`, and
    `command` (an argv that prints one reading, for tests)."""
    c = eff.get("canary")
    if c is None or c is False:
        return None
    if c is True:
        c = {}
    if not isinstance(c, dict):
        raise ValueError(f"config.json infb.efficiency.canary must be true, false or an object, got {c!r}")
    out = dict(c)
    for key, default in (("tolerance", CANARY_TOLERANCE_DEFAULT), ("cpu_tolerance", CANARY_CPU_TOLERANCE_DEFAULT)):
        t = c.get(key, default)
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not 0 < t < 1:
            raise ValueError(f"config.json infb.efficiency.canary.{key} must be a number between 0 and 1, got {t!r}")
        out[key] = float(t)
    return out


def _canary():
    try:
        from . import canary  # noqa: PLC0415 - imported as infb_efficiency.nodes
    except ImportError:
        import canary  # noqa: PLC0415 - a top-level sibling, as reference.py imports this module
    return canary


class Bracket:
    """The canary readings around one side's draws of one recipe. The reading after one draw is the
    reading before the next, so n draws in a row take n + 1 readings, unless that reading was itself below the
    tolerance: then the next draw, a re-draw included, takes a fresh reading first. Everything goes into
    the core record under `canary`; an axis whose draw and re-draw are both invalid goes into
    `summary.environment_invalid`, and the recipe's record directory gets `rerun-request.json`.

    A reading is judged against the running median of the trial's readings up to and including it,
    the GPUs at `tolerance` and the CPU at `cpu_tolerance`, where an earlier rule judged it against
    the trial's first reading at one tolerance. That earlier judgement stays in the record beside it,
    as a diagnostic.

    A reading read in part is judged on what it read, where the earlier rule judged nothing of it,
    and `canary.counts[side][axis]` counts the draws judged, those of them judged in part (a bracket
    not read whole, or a bracket or a quantity of one not judged) and the draws not judged at all."""

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

    def against_baseline(self, readings: list[dict], i: int) -> dict:
        """Reading i against the running median of the readings up to and including it."""
        return _canary().compare(_canary().baseline(readings[: i + 1]), readings[i], self.cfg["tolerance"],
                                 self.cfg["cpu_tolerance"])

    def read(self, axis: str, label: str) -> int:
        """Take one reading and append it to the record; its index there."""
        reading = _canary().take(self.argv(), _clean_env())
        reading.update({"side": self.side, "axis": axis, "label": label})
        self.took = float(reading.get("took_s") or 0.0)
        where: dict = {}

        def put(rec: dict) -> None:
            c = rec.setdefault("canary", {"tolerance": self.cfg["tolerance"], "first": None, "readings": [],
                                          "draws": []})
            c.setdefault("cpu_tolerance", self.cfg["cpu_tolerance"])
            c.setdefault("baseline", "running_median")
            c["readings"].append(reading)
            if c.get("first") is None and reading.get("ok"):
                c["first"] = len(c["readings"]) - 1
            where["i"] = len(c["readings"]) - 1
            where["good"] = self.against_baseline(c["readings"], where["i"])["valid"] is not False
        _client().update_record(self.record, put)
        self.last_good = where["good"]
        print(f"[canary] {self.side} {label}: {_canary().describe(reading)} in {reading.get('took_s')} s", flush=True)
        return where["i"]

    def before(self, axis: str, label: str) -> int:
        if self.last is None or not self.last_good:
            self.last = self.read(axis, f"before {label}")
        return self.last

    def after(self, axis: str, label: str) -> int:
        self.last = self.read(axis, f"after {label}")
        return self.last

    def judge(self, axis: str, draw: int, redraw: bool, before: int, after: int, value) -> bool | None:
        """Record the draw and its brackets: False when either bracket reads below its tolerance of the running median
        of the readings up to it, None when neither could be judged, True otherwise."""
        out: dict = {}

        def put(rec: dict) -> None:
            c = rec["canary"]
            cb, ca = self.against_baseline(c["readings"], before), self.against_baseline(c["readings"], after)
            valid = None if cb["valid"] is None and ca["valid"] is None else \
                not (cb["valid"] is False or ca["valid"] is False)
            # the earlier judgement, against the trial's first reading at the one tolerance: kept to compare the rules
            first = c["readings"][c["first"]] if c.get("first") is not None else None
            c["draws"].append({"side": self.side, "axis": axis, "draw": draw, "redraw": redraw, "value": value,
                               "before": before, "after": after, "valid": valid, "before_vs_baseline": cb,
                               "after_vs_baseline": ca,
                               "before_vs_first": _canary().compare(first, c["readings"][before], self.cfg["tolerance"]),
                               "after_vs_first": _canary().compare(first, c["readings"][after], self.cfg["tolerance"])})
            n = c.setdefault("counts", {}).setdefault(self.side, {}).setdefault(
                axis, {"judged": 0, "judged_in_part": 0, "unjudged": 0})
            if valid is None:
                n["unjudged"] += 1
            else:
                n["judged"] += 1
                n["judged_in_part"] += any(x["valid"] is None or x.get("unjudged") for x in (cb, ca)) or \
                    not (c["readings"][before].get("ok") and c["readings"][after].get("ok"))
            out["valid"], out["cb"], out["ca"] = valid, cb, ca
        _client().update_record(self.record, put)
        if out["valid"] is False:
            worst = min((x for x in (out["cb"], out["ca"]) if x["valid"] is False), key=lambda x: x["ratio"])
            print(f"[canary] {self.side} {axis} draw {draw}{' (re-draw)' if redraw else ''}: environment-invalid, "
                  f"{worst['worst']} at {worst['ratio']:.4f} of the running median of the readings", flush=True)
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
                   "tolerance": self.cfg["tolerance"], "cpu_tolerance": self.cfg["cpu_tolerance"],
                   "at": _client().utcnow()}
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


def axis_sizing(eff: dict, recipe_name: str) -> dict[str, dict]:
    """Per axis of one recipe, the least measurement limit that holds the
    axis's stated draws for a tree at the bar, and the least per-draw reference bound, both by `run_axis`'s own stop
    rule, beside the bounds the config gives.

    The inputs are `infb.efficiency.sizing`:
        {"draw_s": {"<recipe>": {"decode_c1": d, "throughput": t}}, "start_s": s, "canary_reading_s": c,
         "measured": "where and how"}
    d and t are the longest `duration_s` of a draw of each axis. s is everything else a node spends besides its draws
    and canary readings: its setup, and each draw's client start-up. c is one canary reading's `took_s`, and is needed
    only with the canary on.

    A further draw starts only when the limit still holds ATTEMPT_HEADROOM x the last draw, the client's two margins,
    the teardown allowance and, with the canary on, the next draw's closing reading. So k draws of a tree at the bar,
    each d / bar long, need
        s + (k - 1 + ATTEMPT_HEADROOM) x d / bar + 2 x CLIENT_MARGIN_S + TEARDOWN_ALLOWANCE_S + (k + 1) x c,
    where k is decode_c1_draws or candidate_throughput_draws, and the bar is the task's own for the axis, or rho. The
    limit is the scored node's `deadlines.nodes` entry, which is also that node's alarm. Each reference draw has its
    own bound (`reference_<axis>_s`), which must hold one draw at the pace `sizing` records:
        s + ATTEMPT_HEADROOM x d + 2 x CLIENT_MARGIN_S + TEARDOWN_ALLOWANCE_S."""
    sizing = eff.get("sizing")
    if not isinstance(sizing, dict):
        raise ValueError("config.json infb.efficiency.sizing is missing: the measured draw durations per "
                         "recipe and axis (`draw_s`), `start_s`, `canary_reading_s` with the canary on, and `measured`")
    draws = (sizing.get("draw_s") or {}).get(recipe_name)
    if not isinstance(draws, dict):
        raise ValueError(f"config.json infb.efficiency.sizing.draw_s has no entry for {recipe_name!r}")
    fields = {"start_s": (sizing.get("start_s"), 0.0, "a number of seconds, 0 or more"),
              **{f"draw_s.{recipe_name}.{a}": (draws.get(a), None, "a number of seconds above 0") for a in _AXIS_NODE}}
    if canary_config(eff):
        fields["canary_reading_s"] = (sizing.get("canary_reading_s"), 0.0, "a number of seconds, 0 or more, with the "
                                                                           "canary on")
    for key, (v, floor, what) in fields.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or (v < floor if floor is not None else v <= 0):
            raise ValueError(f"config.json infb.efficiency.sizing.{key} must be {what}, got {v!r}")
    if not str(sizing.get("measured") or "").strip():
        raise ValueError("config.json infb.efficiency.sizing.measured must say where and how the durations were "
                         "measured")
    start = float(sizing["start_s"])
    reading = float(sizing["canary_reading_s"]) if canary_config(eff) else 0.0
    configured = (eff.get("deadlines") or {}).get("nodes") or {}
    bars = eff.get("task_bars") or {}
    out = {}
    for axis, node in _AXIS_NODE.items():
        k = decode_c1_draws(eff) if axis == "decode_c1" else candidate_throughput_draws(eff)
        unit = "draws"
        bar = str(bars.get(axis, eff.get("rho")))
        d = float(draws[axis])
        d_bar = d / float(Fraction(bar))
        key = next((x for x in (f"{recipe_name}.{node}", node) if x in configured), None)
        ref_key = f"reference_{axis}_s"
        out[axis] = {
            "k": k, "unit": unit, "bar": bar, "d_s": d, "d_bar_s": round(d_bar, 3),
            "limit_key": key, "limit_s": float(configured[key]) if key else None,
            "need_s": round(start + (k - 1 + ATTEMPT_HEADROOM) * d_bar + 2 * CLIENT_MARGIN_S + TEARDOWN_ALLOWANCE_S
                            + (k + 1) * reading, 3),
            "reference_key": ref_key, "reference_s": (eff.get("deadlines") or {}).get(ref_key),
            "reference_need_s": round(start + ATTEMPT_HEADROOM * d + 2 * CLIENT_MARGIN_S + TEARDOWN_ALLOWANCE_S, 3)}
    return out


def run_axis(base_url: str, axis: str, recipe_name: str, python: str | None = None) -> tuple[bool, str, dict]:
    """Measure the candidate on one axis of one recipe; (passed, message, the axis's summary)."""
    eff = _config()
    recipe = recipe_of(eff, recipe_name)
    node = _AXIS_NODE[axis]
    limit = _deadline(eff, recipe_name, node)
    median_of = decode_c1_draws(eff) if axis == "decode_c1" else candidate_throughput_draws(eff)
    allowed = median_of
    record = record_dir(recipe_name) / "efficiency.json"
    if not record.is_file():
        return False, f"verifier_fault: no efficiency record at {record}: the reference phase did not run", {}
    stat = decode_c1_statistic(eff) if axis == "decode_c1" else None
    if stat and stat != DECODE_C1_STATISTIC_DEFAULT:
        # the record states its own statistic before the first attempt, so the client's summarize uses it
        _client().update_record(record, lambda rec: rec["summary"].__setitem__("decode_c1_statistic", stat))
    bars = task_bars(eff)
    if bars:
        # the task's bars go into the record before the first attempt, for the same reason
        _client().update_record(record, lambda rec: rec["summary"].__setitem__("task_bars", bars))
    t0 = time.monotonic()
    attempts: list[dict] = []
    entries: list[dict] = []
    draws: list = []
    stopped = "disabled"
    bracket = Bracket(eff, python, record, "candidate")  # off unless infb.efficiency.canary is set
    redrawing = False
    while True:
        rows_before = _sidecar_rows(record, json.loads(record.read_text(encoding="utf-8")))
        n = len(attempts) + 1
        before = bracket.before(axis, f"{axis} draw {n}") if bracket.on else None
        if flush_between_draws(eff):
            flush_cache(eff, base_url, record, "candidate", axis, n)
        remaining = limit - (time.monotonic() - t0)
        op_limit = min(float(INVOCATION_S), remaining)  # the draw's own limit, within the node's remaining allowance
        rc, out, killed = run_client(client_argv(eff, recipe, base_url, axis, "candidate", op_limit, record,
                                                 python=python), op_limit, record.parent)
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
                         "client_exit": rc, "killed": killed, "op_limit_s": round(op_limit, 1),
                         "censored": bool(killed or entry.get("status") == "censored")
                         and op_limit < float(INVOCATION_S),
                         "sidecar_rows": [rows_before, _sidecar_rows(record, rec)]})
        last = attempts[-1]
        if bracket.on and not killed:
            # a draw bracketed by a reading more than the tolerance below the trial's first is re-drawn
            # once, whatever its ratio; a second invalid draw, or no time for the re-draw, asks for a re-run
            after = bracket.after(axis, f"{axis} draw {n}")
            valid = bracket.judge(axis, n, redrawing, before, after, entry.get("value"))
            if valid is False and not redrawing:
                # the host-fault policy's one re-draw, always taken (no forecast decides it); a re-draw the
                # node's bound cannot hold is censored like any other draw it cuts off
                attempts.pop()
                entries.pop()
                draws.pop()
                redrawing = True
                continue
            elif valid is False:
                bracket.environment_invalid(axis, n, "the draw and its re-draw both read the canary below its "
                                                     "tolerance of the running median of the readings")
        redrawing = False
        if killed or last["status"] != "complete":
            stopped = "not_complete"
            break
        if allowed == 1:
            stopped = "disabled"
            break
        if len(attempts) >= allowed:
            stopped = "drawn"
            break
    if len(set(draws)) != 1:
        raise AssertionError(f"{axis}: the candidate's attempts drew different workloads {draws}; the attempts "
                             f"would not be comparable, so none of them is a verdict")
    # the paired statistic's median draw is by ratio; every other one's by value
    by = "ratio" if stat == "median_of_ratios" else "value"
    pick = len(attempts) - 1  # a draw that did not complete decides, as a single draw's failure does
    if median_of > 1 and len(attempts) == median_of and all(a["status"] == "complete" and not a["killed"]
                                                            and isinstance(a[by], (int, float)) for a in attempts):
        # the median of the prescribed odd number of draws, never of a partial subset
        pick = sorted(range(len(attempts)), key=lambda i: attempts[i][by])[(len(attempts) - 1) // 2]

    def keep(rec: dict) -> None:
        s = rec["summary"]
        if median_of > 1:
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
        s.setdefault("reference_draws", {})[axis] = reference_draws_for(eff, axis)
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
            "retry_stopped": None}
    print("[efficiency] " + json.dumps(info, sort_keys=True), flush=True)
    if attempts[-1]["censored"]:
        return False, (f"verifier_fault: censored: {axis} draw {len(attempts)} of {median_of} was cut off by the "
                       f"node's bound of {limit:g} s while within its own {INVOCATION_S} s limit; the prescribed draws "
                       f"did not complete, so no verdict is taken from a subset"), info
    if attempts[-1]["killed"]:
        return False, (f"candidate_fail: {axis} draw {len(attempts)} exceeded the measurement invocation's "
                       f"{INVOCATION_S} s limit"), info
    if verdict == "pass":
        return True, "", info
    if cand.get("status") != "complete":
        why = cand.get("invalid") or cand.get("status") or "no measurement"
        return False, f"candidate_fail: {axis}: the candidate's measurement did not complete: {why}", info
    return False, (f"candidate_fail: {axis}: the candidate measures {cand.get('value'):.2f} tok/s against the reference's "
                   f"{ref.get('value'):.2f} tok/s, a ratio of {ratio:.4f}, below rho = {bar_of(summ, axis)}"), info


class EfficiencyMixin:
    """Mix into a hidden deployment class ahead of the upstream class; `self.base_url` is the candidate's,
    and the class's own name is its recipe's key in `config.json`."""

    def test_0_efficiency_decode_c1(self):
        ok, why, _ = run_axis(self.base_url, "decode_c1", type(self).__name__)
        if not ok:
            self.fail(why)

    def test_0_efficiency_throughput(self):
        ok, why, _ = run_axis(self.base_url, "throughput", type(self).__name__)
        if not ok:
            self.fail(why)
