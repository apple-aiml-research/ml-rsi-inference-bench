#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The Rust crates' tests as JUnit: `cargo test --offline` for `vllm-reasoning-parser`, `vllm-tool-parser`
and `vllm-chat` in the tree's Rust workspace, each test's result one testcase.

Test ids are `rust.<crate>.<target>.<test path>`, `<target>` being `lib`, an integration-test file's stem or `doc`
(from cargo's "Running ..." / "Doc-tests ..." section headers). A crate that does not build, or does not finish within
its limit (one crate's tests: 1,800 s), reports no tests; the grader counts an absent id as failed. The raw output of
every crate is kept beside the report.

  python -I rust_junit.py --config /tests/config.json --budget-s <s> --junit /logs/verifier/junit-rust.xml
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

APP_DIR = Path(os.environ.get("APP_DIR", "/app"))
TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
LOGS = Path(os.environ.get("VERIFIER_DIR", "/logs/verifier"))
CRATES = ("vllm-reasoning-parser", "vllm-tool-parser", "vllm-chat")
RESULT = re.compile(r"^test (.+?) \.\.\. (ok|FAILED|ignored)(?:,.*)?$")
RUNNING = re.compile(r"^\s*Running (?:unittests )?(\S+)")
DOCTESTS = re.compile(r"^\s*Doc-tests (\S+)")


def target_of(path: str) -> str:
    stem = Path(path).stem
    return "lib" if stem in ("lib", "main") else stem


def parse(text: str) -> list[dict]:
    cases, target = [], "lib"
    for line in text.splitlines():
        if m := RUNNING.match(line):
            target = target_of(m[1])
        elif DOCTESTS.match(line):
            target = "doc"
        elif m := RESULT.match(line.strip()):
            cases.append({"target": target, "name": m[1], "outcome": {"ok": "passed", "FAILED": "failed",
                                                                        "ignored": "skipped"}[m[2]]})
    return cases


def run_crate(crate: str, limit_s: float) -> tuple[list[dict], dict]:
    t0 = time.monotonic()
    cmd = ["cargo", "test", "--offline", "--manifest-path", str(APP_DIR / "rust/Cargo.toml"), "-p", crate,
           "--no-fail-fast"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=limit_s, cwd=str(APP_DIR / "rust"),
                           env=dict(os.environ, CARGO_NET_OFFLINE="true", CARGO_TERM_COLOR="never"))
        out, rc, timed_out = p.stdout + "\n" + p.stderr, p.returncode, False
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        rc, timed_out = None, True
    (LOGS / f"rust-{crate}.log").write_text(out)
    cases = [] if timed_out else parse(out)
    return cases, {"crate": crate, "rc": rc, "timed_out": timed_out, "tests": len(cases),
                   "seconds": round(time.monotonic() - t0, 1)}


def write_junit(path: Path, rows: list[dict]) -> None:
    fails = sum(r["outcome"] == "failed" for r in rows)
    skips = sum(r["outcome"] == "skipped" for r in rows)
    lines = ['<?xml version="1.0" encoding="utf-8"?>', "<testsuites>",
             f'<testsuite name="rust" tests="{len(rows)}" failures="{fails}" errors="0" skipped="{skips}">']
    for r in rows:
        classname = "rust.%s.%s" % (r["crate"], r["target"])
        head = f"<testcase classname={quoteattr(classname)} name={quoteattr(r['name'])}"
        if r["outcome"] == "passed":
            lines.append(head + "/>")
        elif r["outcome"] == "skipped":
            lines.append(head + '><skipped message="ignored"/></testcase>')
        else:
            lines.append(head + f'><failure message="failed">{escape(r.get("detail", ""))}</failure></testcase>')
    lines += ["</testsuite>", "</testsuites>", ""]
    path.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(TESTS_DIR / "config.json"))
    ap.add_argument("--budget-s", type=float, required=True)
    ap.add_argument("--junit", type=Path, required=True)
    a = ap.parse_args()
    limits = json.loads((TESTS_DIR / "eval/minimax_m3/limits.json").read_text())
    per_crate = float(limits["test_file_s"])
    t0 = time.monotonic()
    rows, summary = [], []
    for crate in CRATES:
        remaining = a.budget_s - (time.monotonic() - t0)
        cases, info = run_crate(crate, max(1.0, min(per_crate, remaining)))
        summary.append(info)
        rows += [{"crate": crate, **c} for c in cases]
    write_junit(a.junit, rows)
    (LOGS / "rust-summary.json").write_text(json.dumps(summary, indent=1))
    print("[rust] " + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
