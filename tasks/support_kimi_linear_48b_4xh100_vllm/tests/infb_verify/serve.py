#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Launch the served model at the recipe.

The recipe is the model card's and the vLLM recipes repository's own 4-GPU command:

    vllm serve moonshotai/Kimi-Linear-48B-A3B-Instruct \
      --tensor-parallel-size 4 --max-model-len 1048576 --trust-remote-code

launched through `tests/utils.RemoteOpenAIServer`, upstream's own test-server wrapper, which appends
`--port <free>` and `--seed 0`. The recipe's arguments and the launch environment are the evaluation
tree's own `launch.py` (the grading copy, `/tests/eval/kimi_linear/launch.py`, byte-identical to the
supplied `/app/eval/kimi_linear/launch.py`), so the graded launch and the agent's cannot drift:

* the free port, because two servers must be able to live in one verifier container, and
  because nothing in the task may depend on 8000 being free;
* `--seed 0`, because it is what upstream's wrapper does for every one of its own tests, and because
  the determinism probe should measure the engine rather than luck;
* `--worker-extension-cls worker.KDAWorker`, the recipe's own argument. Through the evaluation tree's
  launch seam `launch.environment(kda_dir=...)`, this wrapper puts the private package
  (`/tests/infb_kda_private/`) first on the server's PYTHONPATH and the trusted copy of the supplied
  `kda/` after it, so `worker.KDAWorker` is the private extension of the supplied worker: the supplied
  inspection, sample and timing RPCs plus the private correctness and binding RPCs. The recipe and the
  candidate's code are unchanged by it. Binding instrumentation is installed only in the separate
  evidence launch.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from collections.abc import Iterator
from pathlib import Path

TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
APP_DIR = Path(os.environ.get("APP_DIR", "/app"))
X = json.loads((TESTS_DIR / "config.json").read_text())["infb"]


def _load_launch():
    spec = importlib.util.spec_from_file_location("infb_kimi_launch", TESTS_DIR / "eval/kimi_linear/launch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LAUNCH = _load_launch()
PRIVATE_KDA = TESTS_DIR / "infb_kda_private"
TRUSTED_KDA = TESTS_DIR / "eval/kimi_linear/kda"
#: The recipe, minus what RemoteOpenAIServer appends itself. `--long-prefill-token-threshold 2048`
#: is in it: without it, prompt_logprobs for one 8,192-token chunk over this model's 163,840-token
#: vocabulary needs ~12 GiB the startup profiler never reserved, and EngineCore dies on OOM; at 2,048 the tensors fit inside memory already reserved.
SERVER_ARGS = list(LAUNCH.SERVER_ARGS)
assert LAUNCH.MODEL == X["model_id"] and LAUNCH.MAX_MODEL_LEN == X["max_model_len"], (
    f"launch.py serves {LAUNCH.MODEL} at {LAUNCH.MAX_MODEL_LEN}; config.json names {X['model_id']} at "
    f"{X['max_model_len']}")


def _upstream_server():
    """upstream's `tests.utils.RemoteOpenAIServer`, from the restored /app/tests package itself.

    The served session runs with `python -P`, so no candidate-added top-level module in /app shadows
    a module this process imports; the upstream test package is loaded by an explicit spec instead.
    """
    if "tests" not in sys.modules:
        import importlib.machinery  # noqa: PLC0415

        spec = importlib.machinery.PathFinder.find_spec("tests", [str(APP_DIR)])
        assert spec is not None, f"no upstream test package under {APP_DIR}"
        package = importlib.util.module_from_spec(spec)
        sys.modules["tests"] = package
        if spec.loader is not None:
            spec.loader.exec_module(package)
    loaded = getattr(sys.modules["tests"], "__path__", None) or []
    assert [str(APP_DIR / "tests")] == list(loaded), f"the upstream test package resolved to {list(loaded)}"
    from tests.utils import RemoteOpenAIServer  # noqa: PLC0415  (the restored upstream tree)
    return RemoteOpenAIServer
#: The published startup limit (eval/kimi_linear/operation-limits.json, via launch.py): spawn to a ready health
#: response, including model load and startup compilation.
STARTUP_MAX_WAIT_SECONDS = int(os.environ.get("INFB_LAUNCH_BOUND_SEC", str(LAUNCH.LAUNCH_BOUND_S)))
#: The graded server's own stdout and stderr, teed here as well as to the verifier's log:
#: the graph-mode node reads the engine's configuration and its compilation from this file, which only
#: this server writes. run.log also carries the upstream initialization test's engine, so it cannot say
#: which engine a line came from.
SERVER_LOG = Path(os.environ.get("INFB_GRADED_SERVER_LOG", "/logs/verifier/graded-server.log"))


def launch(*, binding: bool = False, deployment: str = "tp4", lifecycle: str | None = None) -> Iterator:
    """Yield a healthy RemoteOpenAIServer at `deployment` (launch.py's `tp4` recipe by default); shut it down afterwards.

    `deployment` selects one of launch.py's stated deployments, its arguments and its GPUs (`CUDA_VISIBLE_DEVICES`
    from `launch.environment(deployment=...)`: the first four devices for a TP4 deployment, all eight for `dp2`).
    `lifecycle` names a directory: the private state-lifecycle observer is installed (INFB_KDA_LIFECYCLE=1) and its
    per-worker records go there. Upstream's wrapper hands `sys.stdout` and `sys.stderr` to the server's Popen, so
    they are pointed at a `tee` for the moment the server is spawned and restored once it is healthy. The server's
    children inherit the same pipe; nothing about the server's own launch changes.
    """
    import io
    import subprocess
    import sys

    RemoteOpenAIServer = _upstream_server()

    if binding:
        server_log = SERVER_LOG.with_name("binding-server.log")
    elif deployment != "tp4" or lifecycle is not None:
        server_log = SERVER_LOG.with_name(f"{deployment}{'-lifecycle' if lifecycle else ''}-server.log")
    else:
        server_log = SERVER_LOG
    server_log.parent.mkdir(parents=True, exist_ok=True)
    tee = subprocess.Popen(["tee", "-a", str(server_log)], stdin=subprocess.PIPE, stdout=sys.stdout)
    sink = io.TextIOWrapper(tee.stdin, line_buffering=True, errors="replace")
    saved = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = sink
    kernel_root = Path(os.environ.get("INFB_KDA_DIR", "/logs/verifier/kda"))
    cache = kernel_root / "jit" / ("binding" if binding else "candidate")
    pythonpath = os.pathsep.join(p for p in (str(TRUSTED_KDA), os.environ.get("PYTHONPATH", "")) if p)
    environment = LAUNCH.environment(kda_dir=PRIVATE_KDA, cache_root=cache, pythonpath=pythonpath,
                                     deployment=deployment)
    if lifecycle is not None:
        Path(lifecycle).mkdir(parents=True, exist_ok=True)
        environment.update({"INFB_KDA_LIFECYCLE": "1", "INFB_KDA_LIFECYCLE_DIR": str(lifecycle),
                            "INFB_KDA_LIFECYCLE_DEPLOYMENT": deployment})
    if binding:
        trace_dir = kernel_root / "traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        # Explicit native markers and loaded-source identities already locate the checked entry; Python
        # stacks multiply trace size without adding binding evidence.
        environment.update({"INFB_KDA_BINDING": "1", "VLLM_TORCH_PROFILER_DIR": str(trace_dir),
                            "VLLM_TORCH_PROFILER_WITH_STACK": "0"})
    try:
        remote = RemoteOpenAIServer(X["model_id"], LAUNCH.server_args(deployment), env_dict=environment,
                                   max_wait_seconds=STARTUP_MAX_WAIT_SECONDS)
    except BaseException:
        sys.stdout, sys.stderr = saved
        _close_tee(sink, tee)
        raise
    sys.stdout, sys.stderr = saved
    try:
        with remote as server:
            yield server
    finally:
        _close_tee(sink, tee)


def _close_tee(sink, tee) -> None:
    """Close the pipe and let `tee` drain; a child still holding the pipe cannot keep the session open."""
    try:
        sink.close()
    except Exception:  # noqa: BLE001
        pass
    try:
        tee.wait(timeout=30)
    except Exception:  # noqa: BLE001
        tee.kill()


_METRIC = re.compile(r"^(?P<name>[a-zA-Z_:][\w:]*)(?P<labels>\{[^}]*\})?\s+(?P<value>[-+0-9.eE]+)\s*$")


def counter_total(metrics_text: str, name: str) -> float:
    """Sum a Prometheus counter over every label set.

    Label-aware on purpose: vLLM emits these per engine and per model, so a bare-name lookup finds
    nothing and reads as "no hits". Both the bare
    name and the `_total` spelling are accepted, because which one appears depends on the exporter.
    """
    wanted = {name, f"{name}_total"}
    total = 0.0
    seen = False
    for line in metrics_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _METRIC.match(line)
        if m and m.group("name") in wanted:
            total += float(m.group("value"))
            seen = True
    return total if seen else float("nan")
