#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The two scored efficiency nodes, as a mixin for a Day-0 hidden class.

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

The candidate's single-request node takes the best of up to `infb.efficiency.c1_attempts` attempts
(1-3, default 3), with an early exit:
- The first attempt that meets rho passes the node.
- A retry starts only after an attempt that completed below rho, while fewer than `c1_attempts` have run, and
  only if the node's remaining time holds 1.2 times that attempt's own `duration_s`, plus the client's two
  margins, plus a teardown allowance. So an attempt never meets the node's own deadline.
- Every attempt re-runs the client exactly as the first, on the same seed file and so the same draw, warm-ups
  included. The node asserts that the draw is the same.
- Throughput and the reference stay one draw.
- The record keeps every attempt, the best attempt as the axis entry and the verdict, and the first attempt's
  ratio as the unbiased value. A published speed uses the first attempt, never the best.
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
#: candidate attempts on the single-request axis, `infb.efficiency.c1_attempts` (1-3).
C1_ATTEMPTS_DEFAULT = 3
C1_ATTEMPTS_MAX = 3
# The reference may be measured more than once per axis, on the already-serving reference, and the
# median draw is kept. Odd counts only, so the kept draw is one that was measured; 1 writes a single draw.
REFERENCE_DRAWS_DEFAULT = 1
REFERENCE_DRAWS_ALLOWED = (1, 3, 5)
# The candidate's throughput may be drawn more than once on its serving server, and the median draw
# decides, as the reference keeps its median (a symmetric rule adds no bias to the ratio). decode_c1
# keeps its best-of attempts.
CANDIDATE_THROUGHPUT_DRAWS_DEFAULT = 1
CANDIDATE_THROUGHPUT_DRAWS_ALLOWED = (1, 3, 5)
# decode_c1's statistic. The default is the ratio of medians; the option is the median over prompts
# of candidate/reference speed.
DECODE_C1_STATISTIC_DEFAULT = "ratio_of_medians"
# A task may set its own bar per axis, `infb.efficiency.task_bars` ({"decode_c1": "0.9"}), as a
# decimal string like rho; an axis it does not name keeps the configured rho. The record carries the
# bars (`summary.task_bars`), so the client's summary and every verdict use them.
TASK_BAR_RANGE = (Fraction(0), Fraction(2))
#: A retry starts only if the node's remaining time holds this multiple of the attempt just measured,
#: plus the client's two margins, plus `TEARDOWN_ALLOWANCE_S`.
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


def resolved_deadlines(eff: dict, recipe: str | None = None) -> dict:
    """`eff["deadlines"]`, resolved through the verifier's budget module when it names a source."""
    block = eff.get("deadlines") or {}
    if "source" not in block:
        return {**block, **((eff.get("recipes") or {}).get(recipe or "", {}).get("deadlines") or {})}
    root = str(HERE.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    from infb_verify import budgets  # noqa: PLC0415
    return budgets.deadlines(recipe)


def record_dir(name: str) -> Path:
    """Where one recipe's core record, sidecar and reference-server log live."""
    return RECORD_DIR / name


def _deadline(eff: dict, recipe: str, node: str) -> float:
    nodes = resolved_deadlines(eff).get("nodes") or {}
    for key in (f"{recipe}.{node}", node):
        if key in nodes:
            return float(nodes[key])
    raise KeyError(f"config.json infb.efficiency.deadlines.nodes names no deadline for {recipe}.{node}")


def _clean_env() -> dict:
    # LD_LIBRARY_PATH is kept for the environment snapshot's nvidia-smi, not for the client, which loads
    # nothing through it: the host driver's libraries, NVML among them, are bind-mounted where only that
    # path finds them, and nvidia-smi without it prints an NVML error instead of listing the GPUs.
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
    """The bar an axis is judged against: the task's own (`summary.task_bars`) or else the configured rho."""
    return (summary.get("task_bars") or {}).get(axis, summary.get("rho"))


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
    stat = decode_c1_statistic(eff) if axis == "decode_c1" else None
    if stat and stat != DECODE_C1_STATISTIC_DEFAULT:
        _client().update_record(record, lambda rec: rec["summary"].__setitem__("decode_c1_statistic", stat))
    bars = task_bars(eff)
    if bars:
        _client().update_record(record, lambda rec: rec["summary"].__setitem__("task_bars", bars))
    t0 = time.monotonic()
    attempts: list[dict] = []
    entries: list[dict] = []
    draws: list = []
    stopped = "disabled"
    while True:
        rows_before = _sidecar_rows(record, json.loads(record.read_text(encoding="utf-8")))
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
        need = ATTEMPT_HEADROOM * float(last["duration_s"] or 0.0) + 2 * CLIENT_MARGIN_S + TEARDOWN_ALLOWANCE_S
        if limit - (time.monotonic() - t0) < need:
            stopped = "deadline"
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
