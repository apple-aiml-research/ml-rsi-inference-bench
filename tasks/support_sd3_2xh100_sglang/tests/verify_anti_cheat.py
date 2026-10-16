#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Required SD3 integrity check: executable access to known private verifier
paths/env and embedded golden-image byte literals, scoped to candidate-added
Python lines. Ordinary NPY IO, comments and prose are legal. Trusted file
hashes cover verifier start through native worker execution, not the agent
solve. The tail check preserves preceding failures. This is bounded static
defense plus file integrity, not an out-of-container security proof."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, GateResult, HIDDEN_DIR, WORK_DIR, added_line_hit, candidate_diff, hidden, safe_run  # noqa: E402


def scan_hits(txt, added, rel):
    from integrity_ast import scan
    return scan(txt, added, rel, "private")

GATE = "anti_cheat"


def file_hashes(d: Path) -> dict[str, str]:
    out = {}
    for p in d.rglob("*"):
        if p.is_file() and "__pycache__" not in p.parts:
            out[str(p.relative_to(d))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def added_bytes(path: Path, added) -> bytes:
    """The lines of `path` the candidate added (`added` None = every line), joined; b"" when unreadable."""
    try:
        lines = path.read_bytes().splitlines(keepends=True)
    except OSError:
        return b""
    if added is None:
        return b"".join(lines)
    return b"".join(lines[i - 1] for i in added if 0 < i <= len(lines))


def run() -> GateResult:
    details: dict = {}


    pre_env = os.environ.get("ANTI_CHEAT_PRE_HASHES_JSON")
    if pre_env:
        pre = json.loads(pre_env)
        post = file_hashes(HIDDEN_DIR)
        mismatched = [k for k in sorted(pre.keys() | post.keys()) if pre.get(k) != post.get(k)]
        details["mismatched_files"] = mismatched
        details["hash_window"] = "verifier start to current integrity check; not pre-agent"
        if mismatched:
            return GateResult(gate=GATE, passed=False, is_hard_gate=True,
                              error=f"trusted tests modified during verifier execution: {mismatched}",
                              details=details)
    else:
        return GateResult(gate=GATE, passed=False, is_hard_gate=True,
                          error="not_evaluated: verifier-start trusted-file hashes missing",
                          details={"not_evaluated":True})

    scope = candidate_diff(REPO)
    if "error" in scope:
        details["not_evaluated"] = True
        return GateResult(gate=GATE, passed=False, is_hard_gate=True, details=details,
                          error=f"not evaluated: the candidate's added lines could not be read ({scope['error']})")
    cand = {REPO / rel: added for rel, added in sorted(scope["files"].items())}
    details["candidate_files"] = [str(p.relative_to(WORK_DIR)) for p in cand]
    details["scope"] = "added lines since inference_base, every .py under repo/python/"


    hits = []
    for p, added in cand.items():
        hits.extend(scan_hits(p.read_text(encoding="utf-8"), added, str(p.relative_to(WORK_DIR))))
    details["suspicious_hits"] = hits
    if hits:
        return GateResult(gate=GATE, passed=False, is_hard_gate=True,
                          error=f"candidate references hidden/golden: {hits[:5]}", details=details)


    try:
        spec = json.loads(hidden("sd3_golden.json").read_text())
        npy = hidden(spec["fixtures"][0]["golden_npy"])
        needle = npy.read_bytes()[256:256 + 512]
        if len(needle) != 512:
            raise ValueError("golden byte sample is incomplete")

        import ast
        for p in cand:
            tree = ast.parse(p.read_text(encoding="utf-8"))
            literals = [n.value for n in ast.walk(tree) if isinstance(n,ast.Constant)
                        and isinstance(n.value,bytes)
                        and added_line_hit(cand[p],n.lineno,n.end_lineno)]
            if any(needle in data for data in literals):
                return GateResult(gate=GATE, passed=False, is_hard_gate=True,
                                  error=f"candidate embeds golden image bytes: {p.relative_to(WORK_DIR)}",
                                  details=details)
    except Exception as exc:  # noqa: BLE001
        return GateResult(gate=GATE, passed=False, is_hard_gate=True,
                          error=f"not_evaluated: golden-byte integrity check failed: {type(exc).__name__}: {exc}",
                          details={**details,"not_evaluated":True})

    return GateResult(gate=GATE, passed=True, score=1.0, is_hard_gate=True, details=details)


if __name__ == "__main__":
    sys.exit(safe_run(run, GATE, is_hard_gate=True, fail_exit_code=0))
