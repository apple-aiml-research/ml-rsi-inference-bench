#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The slow-host rule: a host fault, not a verdict.

A candidate launch that ran out of `launch_s`, on a host whose own reference launch in the same trial took more than
`launch_s` / 3, is a host fault: the trial writes no reward (the -1 sentinel) and is run again. It is never scored.

Why. `launch_s` is 3 x the slowest launch measured on the grading hosts: 7,000 s, from a reference at 2,330.3 s. The
tail of launch times is one phenomenon, the rank-skew signature: one rank loads the checkpoint in about 200-240 s,
and the other three take 5-11 times as long, on an overlay filesystem. The worst reference recorded is 3,570.4 s. A
host slow enough to keep the candidate's server from health for 7,000 s shows it on the reference first: the
reference phase serves the same checkpoint on the same GPUs, earlier in the same trial. A reference launch past
`launch_s` / 3 is past every launch recorded, so this rule cannot void a trial on a host as fast as those.

What is read. Only the verifier's own records, after the session:
- session-results.jsonl, the deadline plugin's row per node phase. The candidate's launch is the session fixture's
  set-up. A set-up the plugin armed from `launch_s` that failed by its alarm ("exceeded its deadline of <launch_s> s")
  is a launch that ran out of `launch_s`.
- efficiency.json, the efficiency record. Its first reference phase is the reference's launch in this trial: every
  attempt it made, a crash's retry included, summed.
A candidate that crashes or fails any other way is not covered: the rule is for the clock, not for the tree.

    python3 slow_host.py --results <session-results.jsonl> --record <efficiency.json> --config /tests/config.json \
        [--out <slow-host.json>]

Exit 3 is a host fault, and test.sh then exits 3 with no reward written. Exit 0 covers every other case: the rule does
not apply, or it applies and the reference was fast (a verdict). The one-line INFB_SLOW_HOST record says which. It is
stdlib only and reads no candidate code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HOST_FAULT = 3
DIVISOR = 3


def launch_timeout(rows: list[dict], launch_s: float) -> dict | None:
    """The first session set-up the plugin armed from `launch_s` that failed by that alarm, or None."""
    needle = f"exceeded its deadline of {float(launch_s):g} s"
    for row in rows:
        if (row.get("when") == "setup" and row.get("deadline_from") == "launch_s" and row.get("outcome") == "failed"
                and needle in str(row.get("message") or "")):
            return row
    return None


def reference_launch_s(record: dict) -> float | None:
    """The reference's first launch in this trial, in seconds: every attempt of the first reference phase summed
    (a crash and its retry included), or its time to healthy when the attempts carry no seconds."""
    for phase in record.get("phases") or []:
        if phase.get("side") != "reference" or "relaunch" in phase:
            continue
        launch = phase.get("launch") or {}
        seconds = [a.get("seconds") for a in launch.get("attempts") or [] if isinstance(a.get("seconds"), (int, float))]
        if seconds:
            return float(sum(seconds))
        t = launch.get("time_to_healthy_s")
        return float(t) if isinstance(t, (int, float)) else None
    return None


def decide(rows: list[dict], record: dict | None, launch_s: float) -> dict:
    """The rule's reading of one trial: `host_fault` true only for a candidate launch that ran out of `launch_s` on a
    host whose reference launch took more than `launch_s` / 3."""
    threshold = float(launch_s) / DIVISOR
    timed_out = launch_timeout(rows, launch_s)
    out = {"launch_s": float(launch_s), "threshold_s": round(threshold, 3), "host_fault": False,
           "candidate_launch": None, "reference_launch_s": None}
    if timed_out is None:
        out["why"] = f"no session set-up ran out of launch_s ({float(launch_s):g} s): the rule does not apply"
        return out
    out["candidate_launch"] = {k: timed_out.get(k) for k in ("nodeid", "when", "duration", "deadline_s", "deadline_from")}
    ref = reference_launch_s(record or {})
    out["reference_launch_s"] = ref
    if ref is None:
        out["why"] = ("the candidate's launch ran out of launch_s, and no reference launch is recorded to judge the "
                      "host by: a verdict, as before the rule")
        return out
    if ref > threshold:
        out["host_fault"] = True
        out["why"] = (f"the candidate's launch ran out of launch_s ({float(launch_s):g} s) on a host whose own reference "
                      f"launch took {ref:.1f} s, more than launch_s / {DIVISOR} ({threshold:.1f} s): a host fault, not a "
                      "verdict; no reward is written and the trial is run again")
    else:
        out["why"] = (f"the candidate's launch ran out of launch_s ({float(launch_s):g} s) on a host whose reference "
                      f"launch took {ref:.1f} s, within launch_s / {DIVISOR} ({threshold:.1f} s): the launch is the "
                      "tree's, a verdict")
    return out


def _rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--record", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    launch_s = json.loads(args.config.read_text())["infb"]["efficiency"]["deadlines"]["launch_s"]
    rows = _rows(args.results) if args.results.is_file() else []
    record = json.loads(args.record.read_text()) if args.record.is_file() else None
    out = decide(rows, record, launch_s)
    print("INFB_SLOW_HOST " + json.dumps(out, sort_keys=True), flush=True)
    if args.out:
        args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    return HOST_FAULT if out["host_fault"] else 0


if __name__ == "__main__":
    sys.exit(main())
