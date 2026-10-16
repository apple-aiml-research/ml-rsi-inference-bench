#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Whole-serving efficiency, per deployment on its fixed configuration.

The reference's draws come from the reference phase on the same allocation (`reference_phase()`, run
before any candidate code is imported).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from infb_verify import common

RHO = 0.8
OUT_TOKENS = 1024
IN_TOKENS = 4096
REFERENCE_RECORD = Path(os.environ.get("INFB_EFFICIENCY_REFERENCE", "/logs/verifier/efficiency/reference.json"))


def _client():
    import importlib.util

    path = Path(os.environ.get("INFB_EFFICIENCY_CLIENT", str(common.TESTS / "public/efficiency/client.py")))
    spec = importlib.util.spec_from_file_location("infb_public_efficiency_client", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _prompts(n: int, seed: int) -> list[list[int]]:
    pool = common.hidden("prose-pool.json.xz")["ids"]
    return [pool[600_000 + 5_000 * ((seed * 131 + k) % 30):][:IN_TOKENS] for k in range(n)]


def c1(srv) -> dict:
    cl = _client()
    return cl.c1(srv.url, [_prompts(1, 100 + a)[0] for a in range(cl.C1_ATTEMPTS)])


def c4(srv) -> dict:
    cl = _client()
    return cl.c4(srv.url, [_prompts(cl.C4_REQUESTS, 200 + d) for d in range(cl.C4_DRAWS)])


def reference_phase(deployments=("tp1", "tp2")) -> dict:
    """Serve the reference tree at each deployment's fixed configuration and record its draws."""
    from eval.diffusion_gemma.serving.server import Server
    from infb_verify import budgets

    tree = os.environ.get("INFB_REFERENCE_TREE", "/opt/infb/reference/app")
    out = {}
    for dep in deployments:
        recipe = "tp1_fixed" if dep == "tp1" else "tp2_fixed"
        # The same launch as the candidate's session: arm-private caches, the idle profiler, the launch cap.
        with Server(recipe, port=common.PORT + 50, arm_root=Path("/tmp/infb-reference-arm") / recipe,
                    gencfg_root=common.GENCFG_ROOT, log_path=Path("/logs/verifier") / f"reference-{recipe}.log",
                    max_wait_s=budgets.seconds("server_launch"),
                    profile_dir=common.SESSIONS.profile_root(f"reference-{recipe}"),
                    cwd=tree, env={"PYTHONPATH": tree}) as srv:
            out[dep] = {"ready_s": srv.ready_s, "c1": c1(srv), "c4": c4(srv)}
    REFERENCE_RECORD.parent.mkdir(parents=True, exist_ok=True)
    REFERENCE_RECORD.write_text(json.dumps(out, indent=1))
    return out


def node(srv, deployment: str, stat: str) -> dict:
    cand = c1(srv) if stat == "c1" else c4(srv)
    ref = json.loads(REFERENCE_RECORD.read_text())[deployment][stat] if REFERENCE_RECORD.exists() else None
    ratio = cand["value"] / ref["value"] if ref else None
    return {"candidate": cand, "reference": ref, "ratio": ratio, "rho": RHO,
            "pass": ref is None and common.CALIBRATION or (ratio is not None and ratio >= RHO)}
