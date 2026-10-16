#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The efficiency nodes' logic, as this task runs it.

`infb_verify/test_hidden.py` defines the nodes as module-level functions and calls these:

  * `test_00_efficiency_measure`   -- both axes of the session server's first launch (`measure_first_launch`);
  * `test_0_efficiency_relaunch`   -- the relaunch of both sides near the bar (`settle`);
  * `test_0_efficiency_decode_c1`  -- S1, per-user decode speed at concurrency 1 (`scored`);
  * `test_0_efficiency_throughput` -- S_C, output throughput at the recipe's load C (`scored`).

The one scored recipe is `config.json` `infb.efficiency.recipes["test_hidden"]`, with its C, its reference launch and
its record directory, `/logs/verifier/efficiency/test_hidden/`. decode_c1 is drawn five times and throughput three
times on both sides, and the median draw decides (decode_c1 by the median over prompts of candidate / reference speed);
this copy carries no other statistic, no per-axis bar and no best-of attempts.

Each starts the shared client in its own process -- `python -I`, an allow-listed environment, a
working directory outside `/app`, its own process group -- so no candidate code is imported where the
timing is taken, and kills that group on the way out whatever happened. The client measures the
candidate on the same hidden draw the reference phase measured, updates the core record, and the node
passes iff the record's verdict for its axis is `pass`: the candidate's value at least rho times the
reference's, compared in exact arithmetic. The node's own deadline comes from `config.json`
(`infb.efficiency.deadlines.nodes`); the client is given a little less, so it stops and records a
censored measurement before the per-node alarm or the group runner has to act.

Every draw re-runs the client exactly as the first, on the same seed file and so the same draw, warm-ups included; the
node asserts that the draw is the same. An axis's prescribed draws are ONE invocation under the public
one-invocation limit (`deadlines.nodes` for the axis, 1,800 s), every draw is made, and each draw's client receives
what remains of that limit -- no pace-derived per-draw cutoff and no forecast that stops collection early. An
invocation that cannot complete its prescribed draws inside its limit is not a verdict from a subset: its draw is
`not_complete`, which fails the axis (a candidate's resource failure on a healthy host).

Two options this copy carries:
- `infb.efficiency.decode_c1_draws` (3 or 5) draws decode_c1 that many times on both sides, and the median draw
  decides, as throughput's draws do. The candidate-only attempts retire with it.
- `infb.efficiency.relaunch` relaunches both sides once where a first launch's reading lies close enough to the bar
  for a launch-level low mode to have decided it, and keeps each side's better valid launch. Three nodes carry it,
  ordered by the deadline plugin: `test_00_efficiency_measure` measures the first launch before every other node;
  `test_0_efficiency_relaunch` runs after every other node, when the session knows whether the reward is already 0;
  and the scored nodes then judge what it settled (`settle`, `judge`). Unset, the plugin deselects the two relaunch
  nodes and the scored nodes measure themselves.
"""
from __future__ import annotations

import contextlib
import copy
import json
import os
import re
import signal
import subprocess
import sys
import time
import unittest
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
RECORD_DIR = Path(os.environ.get("INFB_EFFICIENCY_DIR", "/logs/verifier/efficiency"))
CLIENT_MARGIN_S = 15.0
#: Candidate attempts on the single-request axis, `infb.efficiency.c1_attempts` (1-3).
C1_ATTEMPTS_DEFAULT = 3
C1_ATTEMPTS_MAX = 3
# The reference may be measured more than once per axis, on the already-serving reference, and the median
# draw is kept, so that a single slow reference draw cannot decide an axis. Odd counts only, so the
# kept draw is one that was measured; 1 draws it once.
REFERENCE_DRAWS_DEFAULT = 1
REFERENCE_DRAWS_ALLOWED = (1, 3, 5)
# The candidate's throughput may be drawn more than once on its serving server, and the median
# draw decides, as the reference keeps its median (a symmetric rule adds no bias to the ratio). decode_c1 keeps its
# best-of attempts. 1 draws it once.
CANDIDATE_THROUGHPUT_DRAWS_DEFAULT = 1
CANDIDATE_THROUGHPUT_DRAWS_ALLOWED = (1, 3, 5)
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
#: A retry starts only if the node's remaining time holds this multiple of the attempt just measured, plus the
#: client's two margins, plus `TEARDOWN_ALLOWANCE_S`.
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


def decode_c1_draws(eff: dict) -> int:
    """`infb.efficiency.decode_c1_draws`: 3 or 5 draw decode_c1 that many times on both sides and the median
    draw decides; the candidate-only attempts retire with it, so `c1_attempts` may then only be 1 or unset."""
    n = eff.get("decode_c1_draws", DECODE_C1_DRAWS_DEFAULT)
    if isinstance(n, bool) or not isinstance(n, int) or n not in DECODE_C1_DRAWS_ALLOWED:
        raise ValueError(f"config.json infb.efficiency.decode_c1_draws must be one of {DECODE_C1_DRAWS_ALLOWED}, "
                         f"got {n!r}")
    if n > 1 and "c1_attempts" in eff and c1_attempts(eff) != 1:
        raise ValueError(f"config.json infb.efficiency.decode_c1_draws = {n} takes the median of {n} draws on both "
                         f"sides, and the candidate-only attempts retire with it: c1_attempts must be 1 or unset, "
                         f"got {eff['c1_attempts']!r}")
    return n


def reference_draws_for(eff: dict, axis: str) -> int:
    """How many times the reference is drawn on one axis: `decode_c1_draws` for decode_c1 when it is above 1 (the two
    sides take the same number), else `reference_draws`."""
    k = decode_c1_draws(eff) if axis == "decode_c1" else 1
    return k if k > 1 else reference_draws(eff)


def decode_c1_statistic(eff: dict) -> str:
    """`infb.efficiency.decode_c1_statistic`: this task's is `median_of_ratios`, the only one it carries."""
    stat = eff.get("decode_c1_statistic")
    if stat not in _client().DECODE_C1_STATISTICS:
        raise ValueError(f"config.json infb.efficiency.decode_c1_statistic must be one of "
                         f"{_client().DECODE_C1_STATISTICS}, got {stat!r}")
    return stat


def no_other_options(eff: dict) -> None:
    """The options this task's copy does not carry: a per-axis bar in place of rho, and the prefix-cache flush between
    draws (every vLLM request carries its own `cache_salt`)."""
    for key in ("task_bars", "flush_between_draws"):
        if eff.get(key):
            raise ValueError(f"config.json infb.efficiency.{key} is set, and this task's efficiency copy does not "
                             f"carry it: {eff[key]!r}")


def bar_of(summary: dict, axis: str):
    """The bar an axis is judged against: rho."""
    return summary.get("rho")


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
    `summary.environment_invalid`, and the recipe's record directory gets `rerun-request.json`."""

    def __init__(self, eff: dict, python: str | None, record: Path, side: str):
        self.cfg = canary_config(eff)
        self.py = python or eff.get("python") or sys.executable
        self.record, self.side = record, side
        self.last: int | None = None
        self.last_good = False
        self.took = 0.0

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
    axis's stated draws or attempts for a tree at the bar, and the least per-draw reference bound, both by `run_axis`'s
    own stop rule, beside the bounds the config gives. The vendoring guard refuses a config whose bounds fall short.

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
    where k is decode_c1_draws, or c1_attempts under best-of, or candidate_throughput_draws, and the bar is the task's
    own for the axis, or rho. The limit is the scored node's `deadlines.nodes` entry. It is also that node's alarm with
    the relaunch off, and with the relaunch on it bounds the axis's draws inside the measurement and relaunch nodes.
    The reference's draws of an axis are one invocation under `deadlines.invocation_s` (no per-draw bound), which must
    hold all of its prescribed draws at the pace `sizing` records:
        s + (k_ref - 1 + ATTEMPT_HEADROOM) x d + 2 x CLIENT_MARGIN_S + TEARDOWN_ALLOWANCE_S."""
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
    out = {}
    for axis, node in _AXIS_NODE.items():
        if axis == "decode_c1":
            k = decode_c1_draws(eff)
            k, unit = (k, "draws") if k > 1 else (c1_attempts(eff), "attempts")
        else:
            k, unit = candidate_throughput_draws(eff), "draws"
        bar = str(eff.get("rho"))
        d = float(draws[axis])
        d_bar = d / float(Fraction(bar))
        key = next((x for x in (f"{recipe_name}.{node}", node) if x in configured), None)
        ref_key = "invocation_s"
        k_ref = reference_draws_for(eff, axis)
        out[axis] = {
            "k": k, "unit": unit, "bar": bar, "d_s": d, "d_bar_s": round(d_bar, 3),
            "limit_key": key, "limit_s": float(configured[key]) if key else None,
            "need_s": round(start + (k - 1 + ATTEMPT_HEADROOM) * d_bar + 2 * CLIENT_MARGIN_S + TEARDOWN_ALLOWANCE_S
                            + (k + 1) * reading, 3),
            "reference_key": ref_key, "reference_s": (eff.get("deadlines") or {}).get(ref_key),
            "reference_need_s": round(start + (k_ref - 1 + ATTEMPT_HEADROOM) * d + 2 * CLIENT_MARGIN_S
                                      + TEARDOWN_ALLOWANCE_S, 3)}
    return out


def run_axis(base_url: str, axis: str, recipe_name: str, python: str | None = None) -> tuple[bool, str, dict]:
    """Measure the candidate on one axis of one recipe; (passed, message, the axis's summary)."""
    eff = _config()
    recipe = recipe_of(eff, recipe_name)
    node = _AXIS_NODE[axis]
    limit = _deadline(eff, recipe_name, node)
    no_other_options(eff)
    if axis == "decode_c1":
        median_of = decode_c1_draws(eff)  # above 1, a symmetric median replaces the attempts
    else:
        median_of = candidate_throughput_draws(eff)
    if median_of == 1:
        raise ValueError(f"config.json infb.efficiency: this task draws {axis} 3 or 5 times on both sides and keeps "
                         f"the median draw; a single draw or best-of attempts are not carried")
    allowed = median_of
    record = record_dir(recipe_name) / "efficiency.json"
    if not record.is_file():
        return False, f"no efficiency record at {record}: the reference phase did not run", {}
    stat = decode_c1_statistic(eff) if axis == "decode_c1" else None
    if stat:
        # the record states its statistic before the first draw, as the record has always named it
        _client().update_record(record, lambda rec: rec["summary"].__setitem__("decode_c1_statistic", stat))
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
            # a draw bracketed by a reading more than the tolerance below the trial's first is re-drawn
            # once, whatever its ratio; a second invalid draw, or no time for the re-draw, asks for a re-run
            after = bracket.after(axis, f"{axis} draw {n}")
            valid = bracket.judge(axis, n, redrawing, before, after, entry.get("value"))
            if valid is False and not redrawing:
                # the existing host-fault policy: an invalid draw is re-drawn once, inside what remains of the
                # invocation's limit (no forecast decides whether it may start)
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
        if len(attempts) >= allowed:
            stopped = "drawn"
            break
    if len(set(draws)) != 1:
        raise AssertionError(f"{axis}: the candidate's attempts drew different workloads {draws}; the attempts "
                             f"would not be comparable, so none of them is a verdict")
    # the paired statistic's median draw is by ratio; throughput's by value
    by = "ratio" if stat == "median_of_ratios" else "value"
    pick = len(attempts) - 1  # a draw that did not complete decides, as a single draw's failure does
    if median_of > 1 and all(a["status"] == "complete" and not a["killed"]
                             and isinstance(a[by], (int, float)) for a in attempts):
        # the median of the prescribed draws (every draw is made, so the count is the prescribed odd count)
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
            "bar_from": "rho",
            "client_exit": attempts[-1]["client_exit"], "killed": attempts[-1]["killed"],
            "attempts": len(attempts), "first_attempt_ratio": attempts[0]["ratio"],
            "retry_stopped": None}
    print("[efficiency] " + json.dumps(info, sort_keys=True), flush=True)
    if attempts[-1]["killed"]:
        return False, f"{axis}: the measurement exceeded the node's deadline of {limit:g} s", info
    if verdict == "pass":
        return True, "", info
    if cand.get("status") != "complete":
        why = cand.get("invalid") or cand.get("status") or "no measurement"
        return False, f"{axis}: the candidate's measurement did not complete: {why}", info
    return False, (f"{axis}: the candidate measures {cand.get('value'):.2f} tok/s against the reference's "
                   f"{ref.get('value'):.2f} tok/s, a ratio of {ratio:.4f}, below rho = {bar_of(summ, axis)}"), info


# ── one relaunch of both sides ───────────────────────────────────────────────────────────────────────────────────
#
# A launch can serve in a low mode for its whole life, and draws inside one launch cannot see it. So
# where a first launch's reading
# lies close enough to the bar for such a mode to have decided it, both sides are launched once more and each side keeps
# its best launch. Where the reward is already 0, the relaunch is skipped: this task's grader is binary and
# worst-status-wins, so a graded node that has failed or been skipped decides the
# reward whatever the efficiency nodes say, and skipping then never changes a reward.

#: The nodes the deadline plugin orders when `infb.efficiency.relaunch` is set: the measurement
#: first, before every other node of its class or module, as the scored nodes ran before; the relaunch after every other
#: node; the scored nodes after it, to judge what it settled. Unset, the plugin deselects the two, and the scored nodes
#: measure themselves.
MEASURE_NODE = "test_00_efficiency_measure"
RELAUNCH_NODE = "test_0_efficiency_relaunch"
RELAUNCH_OFF = "infb.efficiency.relaunch is not set: the scored nodes measure"
RELAUNCH_RERUN_RULE = ("the relaunch did not fit in what the verifier's cap left this session; "
                       "re-run this trial verifier-only on the same tree and append the re-run beside this one; the "
                       "first launch decided")
RELAUNCH_KEYS = ("b", "m", "reference_figures", "axes", "force", "prior_reports", "order")
RELAUNCH_ORDERS = ("parent", "session")
#: The IPC reap's own pattern: the benchmark's `_paired_helper.cleanup_stale_ipc` runs `pkill -9 -f 'vllm.*serve'`.
REAP_PATTERN = re.compile(r"vllm.*serve")
#: The line `reference.py --host-state` prints its reading on.
HOST_STATE_PREFIX = "INFB_HOST_STATE "
#: What one launch's summary holds per axis: saved with each launch, and restored for the launch kept.
_SUMMARY_AXIS_FIELDS = ("first_attempt_ratios", "attempts", "retry_stopped", "verdict_from", "reference_draws",
                        "candidate_draws", "draws_stopped")
_RANK = {"passed": 0, "skipped": 1, "failed": 2}


def _plugin():
    """The deadline plugin's module, as pytest loaded it (`-p infb_efficiency.deadline`): the session's outcomes, the
    environment it started with, and whether it ordered the nodes."""
    try:
        from . import deadline  # noqa: PLC0415 - imported as infb_efficiency.nodes, the plugin's own package
    except ImportError:
        import deadline  # noqa: PLC0415
    return deadline


def _unit_interval(value, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < 1:
        raise ValueError(f"config.json infb.efficiency.relaunch.{what} must be a number between 0 and 1, got {value!r}")
    return float(value)


def relaunch_config(eff: dict) -> dict | None:
    """`infb.efficiency.relaunch`: absent or false, off. On, an object:
      b, m               the reference's and the candidate's launch-level low mode, each a number in (0, 1) or
                         {axis: number}. A first launch whose ratio lies in [bar (1 - m), bar / (1 - b)) -- the zone
                         where either side's low mode could have decided the verdict -- is relaunched;
      reference_figures  optional {recipe: {axis: tok/s}}: a reference whose kept value is below (1 - b) of its figure
                         is relaunched whatever the ratio (the band);
      axes               the axes it applies to (default both);
      force              relaunch every axis it applies to, for the task's own trials (default false);
      prior_reports      junit reports of groups that ran before this one, read with the session's own outcomes for
                         the reward-0 skip (default none);
      order              where the deadline plugin orders the nodes: `parent` (default), within their class or
                         module, the scope of a class's launch or a module fixture's; `session`, within the whole
                         session, for a server fixture of session scope that several modules share (deepseek-v4.1's),
                         so the relaunch runs after every node served by the first launch."""
    c = eff.get("relaunch")
    if c is None or c is False:
        return None
    if not isinstance(c, dict):
        raise ValueError(f"config.json infb.efficiency.relaunch must be an object or false, got {c!r}")
    unknown = sorted(set(c) - set(RELAUNCH_KEYS))
    if unknown:
        raise ValueError(f"config.json infb.efficiency.relaunch has unknown keys {unknown}; it takes {RELAUNCH_KEYS}")
    axes = _client().AXES
    on = c.get("axes", list(axes))
    if not isinstance(on, list) or not on or not set(on) <= set(axes) or len(set(on)) != len(on):
        raise ValueError(f"config.json infb.efficiency.relaunch.axes must list some of {axes} once each, got {on!r}")
    out: dict = {"axes": [a for a in axes if a in on], "force": c.get("force", False),
                 "prior_reports": c.get("prior_reports", []), "order": c.get("order", "parent")}
    if out["order"] not in RELAUNCH_ORDERS:
        raise ValueError(f"config.json infb.efficiency.relaunch.order must be one of {RELAUNCH_ORDERS}, "
                         f"got {out['order']!r}")
    if not isinstance(out["force"], bool):
        raise ValueError(f"config.json infb.efficiency.relaunch.force must be true or false, got {out['force']!r}")
    if not isinstance(out["prior_reports"], list) or not all(isinstance(p, str) for p in out["prior_reports"]):
        raise ValueError(f"config.json infb.efficiency.relaunch.prior_reports must be a list of paths, "
                         f"got {out['prior_reports']!r}")
    for key in ("b", "m"):
        v = c.get(key)
        if isinstance(v, dict):
            if not v or not set(v) <= set(axes):
                raise ValueError(f"config.json infb.efficiency.relaunch.{key} must map some of {axes}, got {v!r}")
            out[key] = {a: _unit_interval(x, f"{key}.{a}") for a, x in v.items()}
        else:
            out[key] = {a: _unit_interval(v, key) for a in axes}
    figures = c.get("reference_figures") or {}
    if not isinstance(figures, dict) or not all(
            isinstance(f, dict) and set(f) <= set(axes) and all(
                not isinstance(x, bool) and isinstance(x, (int, float)) and x > 0 for x in f.values())
            for f in figures.values()):
        raise ValueError(f"config.json infb.efficiency.relaunch.reference_figures must map recipes to "
                         f"{{axis: tok/s > 0}}, got {figures!r}")
    out["reference_figures"] = figures
    return out


def relaunch_zone(bar, b: float, m: float) -> tuple[Fraction, Fraction]:
    """[bar (1 - m), bar / (1 - b)): the first-launch ratios a launch-level low mode could have put on the wrong side of
    the bar -- a candidate at the bar whose launch ran m low, or a reference whose launch ran b low."""
    bar = Fraction(str(bar))
    return bar * (1 - Fraction(str(m))), bar / (1 - Fraction(str(b)))


def relaunch_triggers(cfg: dict, recipe_name: str, rec: dict) -> dict[str, list[str]]:
    """Per axis the relaunch applies to, why this first launch is relaunched -- `zone`, `band`, `force` -- leaving out
    the axes with no reason. Both read kept values, never a single draw: `zone` the verdict ratio in the record's
    summary (from the kept entries, under the record's statistic), `band` the reference entry's `value`, which with
    several draws is the median draw's (`keep_median_draw`)."""
    s = rec.get("summary") or {}
    out: dict[str, list[str]] = {}
    for axis in cfg["axes"]:
        why = []
        bar, ratio = bar_of(s, axis), (s.get("ratios") or {}).get(axis)
        b, m = cfg["b"].get(axis), cfg["m"].get(axis)
        if bar is not None and b is not None and m is not None and isinstance(ratio, (int, float)):
            lo, hi = relaunch_zone(bar, b, m)
            if lo <= Fraction(ratio) < hi:
                why.append("zone")
        figure = (cfg["reference_figures"].get(recipe_name) or {}).get(axis)
        kept = (((rec.get("axes") or {}).get("reference") or {}).get(axis) or {}).get("value")
        if figure is not None and b is not None and isinstance(kept, (int, float)) and \
                Fraction(kept) < (1 - Fraction(str(b))) * Fraction(str(figure)):
            why.append("band")
        if cfg["force"]:
            why.append("force")
        if why:
            out[axis] = why
    return out


def _junit_outcomes(path: Path) -> dict[str, str]:
    """classname.name -> passed, skipped or failed, from one junit report, as the grader reads one: a failure or an
    error fails a case, a skip skips it, and a case reported twice keeps its worst."""
    import xml.etree.ElementTree as ET  # noqa: PLC0415

    out: dict[str, str] = {}
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return out
    for tc in root.iter("testcase"):
        outcome = "passed"
        for ch in tc:
            if ch.tag in ("failure", "error"):
                outcome = "failed"
            elif ch.tag == "skipped" and outcome == "passed":
                outcome = "skipped"
        nid = f"{tc.get('classname')}.{tc.get('name')}" if tc.get("classname") else str(tc.get("name"))
        if _RANK[outcome] > _RANK.get(out.get(nid), -1):
            out[nid] = outcome
    return out


def reward_bound(cfg: dict) -> tuple[int | None, list[str]]:
    """(0, the graded nodes that decide it) when a graded node -- config.json's f2p_node_ids and p2p_node_ids -- has
    already failed or been skipped, in this session (the deadline plugin's record of it) or in a prior group's report;
    else (None, []): the reward is not known yet."""
    known: dict[str, str] = {}

    def merge(outcomes: dict[str, str]) -> None:
        for nid, outcome in outcomes.items():
            if _RANK.get(outcome, -1) > _RANK.get(known.get(nid), -1):
                known[nid] = outcome
    for report in cfg["prior_reports"]:
        merge(_junit_outcomes(Path(report)))
    merge(dict(getattr(_plugin(), "_OUTCOMES", None) or {}))
    graded = json.loads((TESTS_DIR / "config.json").read_text(encoding="utf-8"))
    ids = list(graded.get("f2p_node_ids") or []) + list(graded.get("p2p_node_ids") or [])
    lost = [nid for nid in ids if known.get(nid) in ("failed", "skipped")]
    return (0 if lost else None), lost


def _relaunch_estimate(eff: dict, rec: dict, axes: list[str]) -> tuple[float, float]:
    """(seconds the whole relaunch should take, seconds its candidate half should take), from this trial's own first
    launch: the reference's launch, draws and teardown, and the candidate's launch and draws, each side's canary
    readings for those axes with its draws -- the candidate's launch is not recorded, and the reference's, of the same
    model on the same GPUs, stands for it -- with the draws' headroom and the client's margins."""
    deadlines = eff.get("deadlines") or {}
    phase = next((p for p in rec.get("phases", []) if p.get("side") == "reference" and not p.get("relaunch")), {})
    launch_s = float((phase.get("launch") or {}).get("time_to_healthy_s") or deadlines.get("launch_s") or 0.0)
    teardown_s = float((phase.get("teardown") or {}).get("seconds") or deadlines.get("teardown_s") or 0.0)
    readings = (rec.get("canary") or {}).get("readings") or []
    drawn = {}
    for side in ("reference", "candidate"):
        drawn[side] = sum(float(r.get("took_s") or 0.0) for r in readings
                          if r.get("side") == side and r.get("axis") in axes)
        for axis in axes:
            e = ((rec.get("axes") or {}).get(side) or {}).get(axis) or {}
            drawn[side] += sum(float(d.get("duration_s") or 0.0) for d in (e.get("draws") or e.get("attempts") or [e]))
    candidate = ATTEMPT_HEADROOM * (launch_s + drawn["candidate"]) + 2 * CLIENT_MARGIN_S + TEARDOWN_ALLOWANCE_S
    reference = ATTEMPT_HEADROOM * (launch_s + drawn["reference"] + teardown_s) + 2 * CLIENT_MARGIN_S + TEARDOWN_ALLOWANCE_S
    return reference + candidate, candidate


def _stop_group(proc: subprocess.Popen, grace_s: float) -> None:
    """SIGTERM a process group and wait, so reference.py tears its server down; SIGKILL what is left after the grace."""
    if proc.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def _reference_argv(py: str, *mode: str) -> list[str]:
    return [py, "-I", str(HERE / "reference.py"), "--config", str(TESTS_DIR / "config.json"), "--python", py, *mode]


def _reference_env() -> dict:
    """The environment this session started with (the deadline plugin keeps it), less the plugin's and pytest's own."""
    base = getattr(_plugin(), "_ENV_AT_START", None) or os.environ
    env = {k: v for k, v in base.items() if not k.startswith(("INFB_DEADLINE_", "PYTEST_"))}
    env.setdefault("TESTS_DIR", str(TESTS_DIR))
    env.setdefault("INFB_EFFICIENCY_DIR", str(RECORD_DIR))
    return env


def relaunch_reference(recipe_name: str, axes: list[str], seconds: float, python: str | None = None) -> tuple[bool, dict]:
    """The reference's second launch: reference.py itself (`--relaunch`), which captures the argv, launches, draws the
    axes and tears down exactly as for the first launch, into the same record. It runs in its own process group with
    the environment this session started with (the deadline plugin keeps it), so a class that changed the process's
    environment in its setUpClass does not reach the reference."""
    eff = _config()
    py = python or eff.get("python") or sys.executable
    argv = _reference_argv(py, "--relaunch", recipe_name, "--axes", ",".join(axes), "--deadline-s",
                           f"{max(1.0, seconds):g}")
    env = _reference_env()
    grace = float((eff.get("deadlines") or {}).get("teardown_s", 120)) + 30.0
    t0 = time.monotonic()
    proc = subprocess.Popen(argv, env=env, cwd=str(record_dir(recipe_name)), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, start_new_session=True)
    pgid_file = record_dir(recipe_name) / _plugin().RELAUNCH_PGID_FILE  # for the runner, should it kill the group
    killed = False
    try:
        pgid_file.write_text(f"{proc.pid}\n", encoding="utf-8")
        out, _ = proc.communicate(timeout=max(1.0, seconds) + grace)
    except subprocess.TimeoutExpired:
        killed = True
        _stop_group(proc, grace)
        out, _ = proc.communicate()
    finally:
        _stop_group(proc, grace)  # a deadline raised in this node lands here too: the reference tears down first
        pgid_file.unlink(missing_ok=True)
    print(out[-6000:], flush=True)
    return proc.returncode == 0 and not killed, {"exit": proc.returncode, "killed": killed,
                                                 "seconds": round(time.monotonic() - t0, 3)}


def read_host_state(recipe_name: str, seconds: float, python: str | None = None) -> dict:
    """The host as the reference's teardown reads it -- the IPC reap, /dev/shm, the GPUs free within teardown_s --
    from `reference.py --host-state` in its own process, within `seconds`."""
    eff = _config()
    py = python or eff.get("python") or sys.executable
    if seconds < 2 * CLIENT_MARGIN_S:
        return {"ok": None, "skipped": f"{max(seconds, 0.0):.0f} s left in the relaunch node"}
    try:
        p = subprocess.run(_reference_argv(py, "--host-state", recipe_name), env=_reference_env(),
                           cwd=str(record_dir(recipe_name)), capture_output=True, text=True, timeout=seconds,
                           start_new_session=True)
    except subprocess.TimeoutExpired:
        return {"ok": None, "error": f"no reading within {seconds:g} s"}
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith(HOST_STATE_PREFIX)]
    if not lines:
        return {"ok": None, "error": f"exit {p.returncode}: {(p.stdout + p.stderr)[-600:]}"}
    return json.loads(lines[-1][len(HOST_STATE_PREFIX):])


def _cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return ""


def _parent_pid(pid: int) -> int:
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
    except (OSError, ValueError, IndexError):
        return 0


def reap_hazards(argv: list[str]) -> list[str]:
    """What of this session the reference's second launch could kill, one line each. Its teardown
    (`teardown_and_confirm_free`) SIGKILLs every process computing on the GPUs but its own, and its IPC reap
    (`cleanup_stale_ipc`) runs `pkill -9 -f 'vllm.*serve'`: right in a verifier container, and both see this session.
    So this process must hold no CUDA context, and neither it, its ancestors nor the reference's own command line
    (`argv`) may match the pattern."""
    out: list[str] = []
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_initialized():
            out.append(f"torch has initialised CUDA in this session's process ({os.getpid()}), which the teardown's "
                       "GPU reap would kill")
    except Exception:  # noqa: BLE001 - a torch that cannot say has initialised nothing here
        pass
    pid, seen = os.getpid(), set()
    while pid > 1 and pid not in seen:
        seen.add(pid)
        cmd = _cmdline(pid)
        if REAP_PATTERN.search(cmd):
            out.append(f"pid {pid}'s command line matches the IPC reap's pkill -9 -f 'vllm.*serve': {cmd[:300]}")
        pid = _parent_pid(pid)
    if REAP_PATTERN.search(" ".join(argv)):
        out.append("the reference's own command line matches the IPC reap's pattern: " + " ".join(argv)[:300])
    return out


def _snapshot(rec: dict, axes: list[str]) -> dict:
    """One launch's entries and per-axis summary fields, whole."""
    s = rec.get("summary") or {}
    return {"reference": {a: copy.deepcopy(rec["axes"].get("reference", {}).get(a)) for a in axes},
            "candidate": {a: copy.deepcopy(rec["axes"].get("candidate", {}).get(a)) for a in axes},
            "summary": {f: {a: s[f][a] for a in axes if a in (s.get(f) or {})} for f in _SUMMARY_AXIS_FIELDS}}


def _put_launch(rec: dict, launch: dict, axes: list[str], sides=("reference", "candidate")) -> None:
    s = rec["summary"]
    for a in axes:
        for side in sides:
            entry = (launch.get(side) or {}).get(a)
            if entry is not None:
                rec["axes"].setdefault(side, {})[a] = copy.deepcopy(entry)
        if "candidate" in sides:
            for f in _SUMMARY_AXIS_FIELDS:
                v = ((launch.get("summary") or {}).get(f) or {}).get(a)
                if v is None:
                    (s.get(f) or {}).pop(a, None)
                else:
                    s.setdefault(f, {})[a] = v
    _client().summarize(rec)


def _decide(record: Path, decision: str, **detail) -> dict:
    out = {"decision": decision, **detail}
    if record.is_file():
        def put(rec: dict) -> None:
            rec.setdefault("relaunch", {"launches": []}).update(out)
            rec["summary"]["relaunch"] = {"decision": decision, **({"flags": detail["flags"]} if detail.get("flags")
                                                                   else {})}
        _client().update_record(record, put)
    print("[efficiency] relaunch " + json.dumps(out, sort_keys=True, default=str), flush=True)
    return out


def _ratio_against(rec: dict, axis: str, cand: dict | None, ref: dict | None) -> float | None:
    """A candidate launch's ratio against a reference entry, by the record's own statistic and bar."""
    if not cand or not ref:
        return None
    s = rec["summary"]
    scratch = {"summary": {**{k: copy.deepcopy(s[k]) for k in ("rho", "task_bars", "decode_c1_statistic") if k in s},
                           "ratios": {}, "verdicts": {}},
               "axes": {"reference": {axis: ref}, "candidate": {axis: cand}}}
    _client().summarize(scratch)
    return scratch["summary"]["ratios"].get(axis)


def _second_reference_phase(record: Path) -> dict:
    """The reference phase the second launch appended, or {}."""
    rec = _client().read_record(record)
    return next((p for p in reversed(rec.get("phases") or []) if p.get("side") == "reference" and p.get("relaunch")),
                {})


def _measured(info: dict) -> bool:
    return not info.get("killed") and info.get("status") == "complete"


def measure_first_launch(base_url: str, recipe_name: str, python: str | None = None) -> dict:
    """The measurement node (with `infb.efficiency.relaunch` set): both axes of the first launch, measured
    before every other node exactly as the scored nodes measure them without it, each axis's outcome recorded for its
    scored node to report. It asserts no verdict; it fails only where a measurement raised."""
    eff = _config()
    if relaunch_config(eff) is None:
        raise unittest.SkipTest(RELAUNCH_OFF)
    _deadline(eff, recipe_name, MEASURE_NODE)  # named, or the plugin's alarm would be node_default_s: refused here
    record = record_dir(recipe_name) / "efficiency.json"
    if not record.is_file():
        raise AssertionError(f"no efficiency record at {record}: the reference phase did not run")
    _client().update_record(record, lambda rec: rec.__setitem__(
        "relaunch", {"decision": "pending", "launches": [{"index": 1, "results": {}}]}))
    results, raised = {}, []
    for axis in _client().AXES:
        try:
            ok, why, info = run_axis(base_url, axis, recipe_name, python)
            result = {"passed": ok, "why": why[:4000], "valid": _measured(info)}
        except AssertionError as exc:
            result = {"passed": False, "why": str(exc)[:4000], "valid": False}
            raised.append(f"{axis}: {exc}")
        _client().update_record(record, lambda rec, a=axis, r=result: rec["relaunch"]["launches"][0]["results"]
                                .__setitem__(a, r))
        results[axis] = result
    if raised:
        raise AssertionError("; ".join(raised))
    return results


def session_clock(eff: dict) -> dict | None:
    """The session's end (`INFB_SESSION_END_BY`, which the runner sets from
    test.sh's END_BY) and what the relaunch must leave before it (`reserve_s`): every node the session runs after this
    one, at the alarm the plugin arms for its call (the scored nodes at `node_default_s`); a set-up and a
    tear-down for each class or module still to come; this node's own class's tear-down; and the grace in which the
    reference's second launch tears down should the alarm cut it (`relaunch_reference`). None where the session has no
    end: the relaunch then does not run."""
    plugin = _plugin()
    try:
        end = float(os.environ[plugin.END_BY_ENV])
    except (KeyError, ValueError):
        return None
    deadlines = eff.get("deadlines") or {}
    teardown_s = float(deadlines.get("teardown_s", 120))
    after = plugin.session_after(plugin._CURRENT.get("nodeid"))  # noqa: SLF001 - the plugin's own record of the order
    if after is None:  # not in a session the plugin ordered: what the relaunch order puts after it, the scored nodes
        after = {"items": [{"nodeid": n, "call_s": plugin.node_deadline(deadlines, "", n, "call", relaunch=True)}
                           for n in plugin.SCORED_NODES], "other_parents": 0, "from": "the scored nodes"}
    later_s = sum(float(i["call_s"] or 0.0) for i in after["items"])
    parents_s = after["other_parents"] * (float(deadlines.get("launch_s") or 0.0) + teardown_s)
    reserve = {"later_nodes_s": later_s, "later_nodes": [i["nodeid"] for i in after["items"]],
               "later_parents_s": parents_s, "own_teardown_s": teardown_s,
               "reference_stop_grace_s": teardown_s + 30.0}
    try:
        cfg = plugin.verifier_clock(TESTS_DIR / "config.json") or {}
    except ValueError:
        cfg = {}
    return {"session_end_by": end, "reserve_s": round(later_s + parents_s + 2 * teardown_s + 30.0, 3),
            "reserve": reserve, **{k: cfg.get(k) for k in ("cap_s", "before_s", "tail_s")},
            "order_from": after.get("from", "the plugin's order")}


def _request_rerun_for_time(record: Path, clock: dict) -> None:
    """The relaunch had no room in what the cap left, so the first launch decides and a
    verifier-only re-run on the same tree is asked for, in the shape of the canary's request."""
    if record.is_file():
        _client().update_record(record, lambda rec: rec["summary"].__setitem__("rerun_requested", True))
    path = record.parent / "rerun-request.json"
    try:
        request = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        request = {"rule": RELAUNCH_RERUN_RULE, "recipe": record.parent.name}
    request["relaunch"] = {"rule": RELAUNCH_RERUN_RULE, "clock": clock, "at": _client().utcnow()}
    path.write_text(json.dumps(request, indent=1) + "\n", encoding="utf-8")
    print("INFB_EFFICIENCY_RERUN_REQUEST " + json.dumps(request, sort_keys=True), flush=True)


def settle(recipe_name: str, stop, start, python: str | None = None, reference=None, check=None) -> dict:
    """The relaunch node. `stop()` ends the candidate's server; `start()` launches it again exactly as the
    class or the fixture did and returns its base URL; `check(base_url)`, where the task gives one, reads that second
    launch as the task reads its first -- its served configuration -- and returns a dict whose `ok` is True only when
    it is the same; `reference(recipe, axes, seconds)` relaunches and measures the reference (default
    `relaunch_reference`). The decision, recorded under the record's `relaunch`:
      not_triggered        no axis in the zone or the band, and nothing forced;
      skipped_reward_zero  a graded node has already failed or been skipped, so the reward is 0 whatever this decides;
      skipped_reap_hazard  the reference's teardown could kill this session (`reap_hazards`);
      (`skipped_deadline` and `skipped_verifier_time`, the earlier forecast skips, are retired: the composed
                           bound holds the relaunch, and no forecast stops collection early)
      skipped_no_verifier_clock  the session has no end (`INFB_SESSION_END_BY` unset: the runner was not
                           given `--end-by`), so nothing says the relaunch would end before the cap;
      host_not_ready       once the candidate's server stopped, the GPUs were not free within teardown_s or /dev/shm
                           had no room for the reference's second launch: the first launch decides, flagged
                           `environment`;
      reference_failed     the reference's second launch or its draws failed: the first launch decides;
      relaunched           each triggered axis is judged on the best valid candidate launch against the best reference
                           launch (the highest value; for the candidate, the highest ratio against that reference); an
                           axis not triggered keeps its first launch. A candidate second launch that did not start
                           (flagged `candidate_start_failed`, with the host read right after it), that the check did
                           not pass (`candidate_configuration_differs`; not drawn) or that drew no valid measurement
                           does not count: the candidate keeps its first launch against the best reference, so nothing
                           done to its own second launch can raise its outcome.
    A relaunch cut off by this node's alarm leaves `running`; the scored node then restores the first launch."""
    eff = _config()
    cfg = relaunch_config(eff)
    if cfg is None:
        raise unittest.SkipTest(RELAUNCH_OFF)
    t0 = time.monotonic()
    limit = _deadline(eff, recipe_name, RELAUNCH_NODE)
    sc = session_clock(eff)

    def now() -> dict | None:
        if sc is None:
            return None
        return {**sc, "clock_left_s": round(sc["session_end_by"] - time.time() - sc["reserve_s"], 1),
                "node_left_s": round(limit - (time.monotonic() - t0), 1)}
    record = record_dir(recipe_name) / "efficiency.json"
    rec = _client().read_record(record) if record.is_file() else {}
    if not ((rec.get("relaunch") or {}).get("launches") or [{}])[0].get("results"):
        return _decide(record, "not_measured", why=f"{MEASURE_NODE} recorded no first launch", clock=now())
    triggers = relaunch_triggers(cfg, recipe_name, rec)
    if not triggers:
        return _decide(record, "not_triggered", clock=now())
    bound, lost = reward_bound(cfg)
    if bound == 0:
        return _decide(record, "skipped_reward_zero", triggers=triggers, reward_lost_by=lost[:20], clock=now())
    axes = [a for a in _client().AXES if a in triggers]
    py = python or eff.get("python") or sys.executable
    hazards = reap_hazards(_reference_argv(py, "--relaunch", recipe_name, "--axes", ",".join(axes)))
    if hazards:
        return _decide(record, "skipped_reap_hazard", triggers=triggers, hazards=hazards, clock=now())
    need, candidate_need = _relaunch_estimate(eff, rec, axes)  # recorded only (no forecast decides)
    node_left = limit - (time.monotonic() - t0)
    if sc is None:
        return _decide(record, "skipped_no_verifier_clock", triggers=triggers, estimate_s=round(need, 1),
                       deadline_s=limit, clock=None)
    clock_left = sc["session_end_by"] - time.time() - sc["reserve_s"]
    working = min(node_left, clock_left)
    clock = {**sc, "clock_left_s": round(clock_left, 1), "node_left_s": round(node_left, 1),
             "working_s": round(working, 1), "estimate_s": round(need, 1)}
    # No time-based forecast skips or trims the relaunch. The
    # composed bound of this node holds both sides' full second launches, and each half runs within its own composed
    # bound; a clock that still runs out ends the node by its alarm -- censored, never a verdict from a partial relaunch.
    until = time.monotonic() + working
    reference_half = float((eff.get("deadlines") or {}).get("relaunch_reference_s") or 0.0) or (working - candidate_need)
    left = _plugin().shorten_alarm(working)  # the plugin's alarm, brought forward to the clock where the clock binds
    clock["alarm_left_before_s"] = None if left is None else round(left, 1)
    first = {**rec["relaunch"]["launches"][0], **_snapshot(rec, axes)}
    first_ratios = {a: (rec["summary"].get("ratios") or {}).get(a) for a in axes}

    def begin(r: dict) -> None:
        r["relaunch"].update({"decision": "running", "triggers": triggers, "axes": axes, "estimate_s": round(need, 1),
                              "started_at": _client().utcnow(), "clock": clock})
        r["relaunch"]["launches"][0] = first
        r["summary"]["relaunch"] = {"decision": "running"}
    _client().update_record(record, begin)
    stopped: dict = {"started_at": _client().utcnow()}
    t_stop = time.monotonic()
    try:
        stop()
    except Exception as exc:  # noqa: BLE001 - recorded; the host read before the reference's launch decides
        stopped["error"] = f"{type(exc).__name__}: {exc}"[:2000]
    stopped["seconds"] = round(time.monotonic() - t_stop, 3)
    ok, info = (reference or relaunch_reference)(recipe_name, axes, min(reference_half, until - time.monotonic()), python)
    if not ok:
        _client().update_record(record, lambda r: _put_launch(r, first, axes))
        host = _second_reference_phase(record).get("before_launch") or {}
        if host.get("ok") is False:
            return _decide(record, "host_not_ready", flags=["environment"], candidate_stop=stopped, host=host,
                           reference=info, clock=clock)
        return _decide(record, "reference_failed", candidate_stop=stopped, reference=info, clock=clock)
    rec = _client().read_record(record)
    second = {"index": 2, "candidate_stop": stopped,
              "reference": {a: copy.deepcopy(rec["axes"]["reference"].get(a)) for a in axes}, "reference_run": info}
    best_ref = {}
    for a in axes:
        r1, r2 = first["reference"][a], second["reference"][a]
        j = 2 if r2 and isinstance(r2.get("value"), (int, float)) and r2.get("status") == "complete" and \
            r2["value"] > float((r1 or {}).get("value") or 0.0) else 1
        best_ref[a] = {**(r2 if j == 2 else r1), "launch": j}

    def with_best_reference(r: dict) -> None:
        r["relaunch"]["launches"].append(second)
        _put_launch(r, {"reference": best_ref}, axes, sides=("reference",))
    _client().update_record(record, with_best_reference)
    results: dict[str, dict] = {}
    flags: list[str] = []
    try:
        base_url = start()
    except Exception as exc:  # noqa: BLE001 - a candidate that does not serve again keeps its first launch
        base_url, second["start_error"] = None, f"{type(exc).__name__}: {exc}"[:2000]
        flags.append("candidate_start_failed")
        left = until - time.monotonic() - TEARDOWN_ALLOWANCE_S
        teardown_s = float((eff.get("deadlines") or {}).get("teardown_s", 120))
        second["host_after_start"] = read_host_state(recipe_name, min(left, teardown_s + 60.0), python)
    if base_url and check is not None:
        try:
            got = check(base_url)
            second["check"] = {**got, "ok": got.get("ok") is True} if isinstance(got, dict) else \
                {"ok": False, "error": f"the check returned {got!r}"[:600]}
        except Exception as exc:  # noqa: BLE001 - a second launch the check cannot read is not shown to be the same
            second["check"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:2000]}
        if not second["check"]["ok"]:
            base_url = None
            flags.append("candidate_configuration_differs")
    for a in axes if base_url else []:
        try:
            ok_a, why_a, info_a = run_axis(base_url, a, recipe_name, python)
            results[a] = {"passed": ok_a, "why": why_a[:4000], "valid": _measured(info_a)}
        except AssertionError as exc:
            results[a] = {"passed": False, "why": str(exc)[:4000], "valid": False}
    rec = _client().read_record(record)
    later = _snapshot(rec, axes)
    second.update({"candidate": {a: later["candidate"][a] for a in results},
                   "summary": {f: {a: v for a, v in per.items() if a in results} for f, per in later["summary"].items()},
                   "results": results})
    kept: dict[str, dict] = {}
    for a in axes:
        options = [(1, first["candidate"][a], first["summary"])] if first["results"].get(a, {}).get("valid") else []
        if results.get(a, {}).get("valid"):
            options.append((2, later["candidate"][a], later["summary"]))
        ratios = {j: _ratio_against(rec, a, c, best_ref[a]) for j, c, _ in options}
        j, cand, summ = max(options, key=lambda o: (ratios[o[0]] if isinstance(ratios[o[0]], (int, float))
                                                    else float("-inf"), -o[0]), default=(1, first["candidate"][a],
                                                                                         first["summary"]))
        kept[a] = {"launch": j, "candidate": cand, "summary": summ, "ratios": ratios}

    def finish(r: dict) -> None:
        r["relaunch"]["launches"][1:] = [second]
        for a, k in kept.items():
            launch = {"candidate": {a: {**(k["candidate"] or {}), "launch": k["launch"]}}, "summary": k["summary"]}
            _put_launch(r, launch, [a], sides=("candidate",))
            # the first launch's first draw stays the unbiased figure a published speed uses
            if a in first["summary"].get("first_attempt_ratios", {}):
                r["summary"].setdefault("first_attempt_ratios", {})[a] = first["summary"]["first_attempt_ratios"][a]
        _client().summarize(r)
        r["relaunch"].update({"decision": "relaunched", "flags": flags, "finished_at": _client().utcnow(),
                              "seconds": round(time.monotonic() - t0, 3),
                              "clock": {**clock, "working_left_at_finish_s": round(until - time.monotonic(), 1)}})
        r["summary"]["relaunch"] = {"decision": "relaunched", "flags": flags, "axes": {a: {
            "trigger": triggers[a], "first_launch_ratio": first_ratios[a],
            "reference_values": [(first["reference"][a] or {}).get("value"), (second["reference"][a] or {}).get("value")],
            "candidate_ratios_against_the_best_reference": {str(j): x for j, x in kept[a]["ratios"].items()},
            "kept": {"reference": best_ref[a]["launch"], "candidate": kept[a]["launch"]}} for a in axes}}
    rec = _client().update_record(record, finish)
    out = {"decision": "relaunched", "flags": flags, "axes": rec["summary"]["relaunch"]["axes"]}
    print("[efficiency] relaunch " + json.dumps(out, sort_keys=True, default=str), flush=True)
    return out


def judge(axis: str, recipe_name: str) -> tuple[bool, str, dict]:
    """A scored node with `infb.efficiency.relaunch` set: the first launch's own outcome for the axis, exactly as
    `run_axis` returned it in the measurement node, unless the relaunch node relaunched this axis; then the verdict of
    the best candidate launch against the best reference launch."""
    record = record_dir(recipe_name) / "efficiency.json"
    if not getattr(_plugin(), "_REORDERED", False):
        return False, ("infb.efficiency.relaunch is set, but the deadline plugin did not order this session's nodes "
                       "(pytest -p infb_efficiency.deadline): the scored nodes would judge before the relaunch"), {}
    if not record.is_file():
        return False, f"no efficiency record at {record}: the reference phase did not run", {}
    rec = _client().read_record(record)
    rl = rec.get("relaunch") or {}
    first = ((rl.get("launches") or [{}])[0].get("results") or {}).get(axis)
    if first is None:
        return False, f"{axis}: the measurement node ({MEASURE_NODE}) did not measure this axis", {}
    if rl.get("decision") == "running":  # cut off by its node's deadline: the first launch decides, and says so
        launch = rl["launches"][0]
        rec = _client().update_record(record, lambda r: _put_launch(r, launch, rl.get("axes") or []))
        _decide(record, "interrupted", axes_restored=rl.get("axes") or [])
        return bool(first["passed"]), first["why"], first
    if rl.get("decision") != "relaunched" or axis not in (rl.get("axes") or []):
        return bool(first["passed"]), first["why"], first
    s = rec["summary"]
    detail = (s.get("relaunch") or {}).get("axes", {}).get(axis) or {}
    info = {"recipe": recipe_name, "axis": axis, "ratio": s["ratios"].get(axis), "rho": bar_of(s, axis),
            "verdict": s["verdicts"].get(axis), "relaunch": detail}
    print("[efficiency] " + json.dumps(info, sort_keys=True, default=str), flush=True)
    if info["verdict"] == "pass":
        return True, "", info
    return False, (f"{axis}: relaunched ({', '.join(detail.get('trigger') or [])}); the best candidate launch "
                   f"measures a ratio of {info['ratio']} against the best reference launch, below rho = {info['rho']} "
                   f"(the candidate's launches against it: {detail.get('candidate_ratios_against_the_best_reference')}; "
                   f"the reference's values: {detail.get('reference_values')})"), info


def scored(base_url: str, axis: str, recipe_name: str, python: str | None = None) -> tuple[bool, str, dict]:
    """A scored node: `run_axis` itself while `infb.efficiency.relaunch` is unset; set, `judge`."""
    if relaunch_config(_config()) is None:
        return run_axis(base_url, axis, recipe_name, python)
    return judge(axis, recipe_name)


