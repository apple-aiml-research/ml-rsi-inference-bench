#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Shared result, runtime and image helpers for the SD3 verifier."""
from __future__ import annotations

import json
import os
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

WORKSPACE = Path(os.environ.get("INFB_WORKSPACE", "/app"))
WORK_DIR = WORKSPACE
REPO = WORKSPACE / "repo"
TESTS_DIR = Path(os.environ.get("INFB_TESTS_DIR", "/tests"))
HIDDEN_DIR = TESTS_DIR
RESULTS_DIR = Path(os.environ.get("INFB_VERIFIER_DIR", "/logs/verifier")) / "gates"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class GateResult:
    gate: str
    passed: bool
    score: float = 0.0
    is_hard_gate: bool = False
    level: str | None = None
    error: str | None = None
    details: dict = field(default_factory=dict)


def write_gate(r: GateResult) -> None:
    data = asdict(r)


    cfg = json.loads((Path(__file__).with_name("config.json")).read_text())
    scored = r.gate in cfg["f2p_node_ids"] or r.gate in cfg["p2p_node_ids"]
    data["details"].update(scored=scored, diagnostic=not scored)
    (RESULTS_DIR / f"{r.gate}.gate.json").write_text(
        json.dumps(data, default=str, indent=2)
    )


def engine_port_kwargs() -> dict:
    """Explicit, non-overlapping engine ports for one verifier queue.

    test.sh sets SD3_ENGINE_BASE_PORT per queue (21000 for the GPU-0 queue, 22000 for the GPU-1 one) and nothing on the
    serial schedule, where this returns {} and every engine picks its own ports as before. The block keeps two engines
    started at the same moment apart: ServerArgs probes a port and binds it later (settle_port), so two engines
    racing through the probe could both pick the default master port 30005. Passed as DiffGenerator.from_pretrained
    keyword arguments (ServerArgs fields port / scheduler_port / master_port / nccl_port). Both blocks stay below
    the default Linux ephemeral range (32768-60999)."""
    raw = os.environ.get("SD3_ENGINE_BASE_PORT") or ""
    if not raw.strip():
        return {}
    base = int(raw)
    return {"port": base, "scheduler_port": base + 100, "master_port": base + 200, "nccl_port": base + 300}


def hidden(name: str) -> Path:
    return HIDDEN_DIR / name


def save_stderr_log(tag: str, text: str) -> None:
    art = Path(os.environ.get("INFB_ARTIFACTS_DIR", "/logs/artifacts")) / "raw"
    art.mkdir(parents=True, exist_ok=True)
    (art / f"{tag}.log").write_text(text)


def safe_run(run_fn, gate: str, is_hard_gate: bool = False, level: str | None = None,
             *, fail_exit_code: int = 1) -> int:
    """Persist a verdict; callers supervised by the clock use 0 for completed FAIL.

    A negative verdict is a completed measurement, distinct from a crashed
    verifier. Exceptions still return 1 and receive no credit.
    """
    try:
        r = run_fn()
        if not isinstance(r, GateResult) or r.gate != gate or type(r.passed) is not bool:
            raise TypeError("gate function must return the named GateResult with a boolean verdict")
        write_gate(r)
        return 0 if r.passed else fail_exit_code
    except Exception as exc:  # noqa: BLE001
        write_gate(GateResult(
            gate=gate, passed=False, is_hard_gate=is_hard_gate, level=level,
            error=f"{type(exc).__name__}: {exc}",
            details={"not_evaluated": True, "trace": traceback.format_exc()[-1500:]},
        ))
        return 1


class NoImageProduced(RuntimeError):
    """The candidate ran but produced no image.

    A missing output has no pixel distance. Keep it distinct from an image
    whose measured quality fails a threshold.
    """


def to_image_array(obj):
    """sglang generate() output -> uint8 (H, W, C) ndarray. The single source of truth."""
    import numpy as np

    for attr in ("frames", "images", "videos", "output", "samples"):
        if hasattr(obj, attr):
            obj = getattr(obj, attr)
            break
    depth = 0
    while isinstance(obj, (list, tuple)) and obj and depth < 4:
        obj = obj[0]
        depth += 1
    try:
        from PIL import Image
        if isinstance(obj, Image.Image):
            return np.asarray(obj.convert("RGB"))
    except Exception:
        pass
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            obj = obj.detach().to("cpu").float().numpy()
    except Exception:
        pass


    if obj is None:
        raise NoImageProduced("generate() returned no image (None)")
    a = np.asarray(obj)
    if a.dtype == object or a.size == 0:
        raise NoImageProduced(
            f"generate() returned no usable image (dtype={a.dtype}, size={a.size})")
    while a.ndim > 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim == 3 and a.shape[0] in (1, 3, 4) and a.shape[-1] not in (1, 3, 4):
        a = np.transpose(a, (1, 2, 0))
    if a.dtype.kind == "f":
        if a.min() < 0:
            a = (a + 1.0) / 2.0
        if a.max() <= 1.0 + 1e-3:
            a = a * 255.0
    a = np.clip(a, 0, 255).astype(np.uint8)
    return a[..., :3] if a.ndim == 3 and a.shape[-1] > 3 else a


BASELINE_TAG = "inference_base"
DIFF_ROOTS = ("python",)


def candidate_diff(repo=None, base: str = BASELINE_TAG, roots=DIFF_ROOTS) -> dict:
    """{"files": {repo-relative path: sorted ADDED line numbers, or None = every line (a new, untracked file)}} for
    every .py file under `roots` that differs from `base`; {"error": why} when that cannot be read. Never raises.

    Paths come from `git diff --name-only -z` (NUL-separated, so no file name can forge a record) and line numbers
    from each file's own `-U0` hunk headers. Untracked files are listed WITHOUT .gitignore (`ls-files --others` with
    no --exclude-standard): the candidate writes .gitignore, so honouring it would let a file hide itself."""
    import re
    import subprocess

    repo = Path(repo) if repo is not None else REPO
    git = ["git", "-c", "safe.directory=*", "-c", "core.quotePath=false", "-C", str(repo)]

    def run(*args):
        return subprocess.run(git + list(args), capture_output=True, text=True, timeout=300)

    try:
        if run("rev-parse", "--verify", "-q", f"{base}^{{commit}}").returncode != 0:
            return {"error": f"no `{base}` commit in {repo}: the baseline the diff is taken against is missing"}
        changed = run("diff", "--name-only", "-z", "--no-renames", "--diff-filter=d", base, "--", *roots)
        untracked = run("ls-files", "--others", "-z", "--", *roots)
        if changed.returncode != 0 or untracked.returncode != 0:
            return {"error": f"git diff failed: {(changed.stderr or untracked.stderr).strip()[:200]}"}
        files: dict = {}
        for rel in (r for r in changed.stdout.split("\0") if r):
            if not rel.endswith(".py") or "__pycache__" in rel.split("/") or not (repo / rel).is_file():
                continue
            hunks = run("diff", "--no-color", "--no-ext-diff", "-U0", base, "--", rel)
            if hunks.returncode != 0:
                return {"error": f"git diff failed on {rel}: {hunks.stderr.strip()[:200]}"}
            added = set()
            for m in re.finditer(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", hunks.stdout, re.M):
                start, n = int(m.group(1)), int(m.group(2)) if m.group(2) is not None else 1
                added.update(range(start, start + n))
            files[rel] = sorted(added)
        for rel in (r for r in untracked.stdout.split("\0") if r):
            if rel.endswith(".py") and "__pycache__" not in rel.split("/") and (repo / rel).is_file():
                files[rel] = None
        return {"files": files}
    except Exception as exc:  # noqa: BLE001 - a scope that cannot be read is reported, never guessed
        return {"error": f"{type(exc).__name__}: {exc}"[:300]}


def added_line_hit(added, start_line: int, end_line: int) -> bool:
    """Does a match spanning [start_line, end_line] touch a line the candidate added? `added` None = every line."""
    return added is None or any(start_line <= ln <= end_line for ln in added)
