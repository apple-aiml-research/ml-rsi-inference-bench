#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Worker extension and RPC surface for native-region measurement (development and grading).

Load with `--worker-extension-cls qwen38.measurement.worker.MeasurementWorker` and
`VLLM_SERVER_DEV_MODE=1`; drive it with `measurement.client`. Every request carries the protocol
identity (a digest of this package's source) and a request ID; every TP rank answers with its
rank, recipe, candidate source identity and a status. A failure on one rank is returned, never
raised, so no rank's answer is stranded in the executor.

Actions: `identity` (resolved regions), `capture` (run a case schedule and keep its calls),
`draws` (one matrix entry of the timing protocol for a captured case), `use` (the region's output
is consumed: a perturbed run changes the next decoder layer's input beyond the variation of repeated
unperturbed runs), `memory` (allocation evidence for a captured case), `release`.

This public worker validates each draw against the case's own captured native outputs at the
public allowances. Grading subclasses supply independent expected outputs instead (`expected`).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from pathlib import Path

ALLOWANCES = {"qsa": {"atol": 0.02, "rtol": 0.016}, "ple": {"atol": 0.01, "rtol": 0.016}}


def protocol_identity() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def within(actual, expected, allowance) -> dict:
    """Every element within atol + rtol·|expected| (the public logical allowance)."""
    import torch
    if actual.shape != expected.shape:
        raise AssertionError(f"output shape {tuple(actual.shape)} differs from {tuple(expected.shape)}")
    a, e = actual.float(), expected.float()
    if not bool(torch.isfinite(a).all()):
        raise AssertionError("nonfinite native output")
    diff = (a - e).abs()
    bad = int((diff > allowance["atol"] + allowance["rtol"] * e.abs()).sum())
    if bad:
        raise AssertionError(f"{bad} of {diff.numel()} output elements exceed the allowance "
                             f"(max abs difference {float(diff.max())})")
    return {"max_abs": float(diff.max()) if diff.numel() else 0.0, "elements": diff.numel()}


class MeasurementWorker:
    """Never accepts a caller-supplied algorithm, timer, expected value or verdict."""

    def expected(self, result: dict, capture, case: dict):
        """Independent expected outputs for the measured and probe calls, or None (public)."""
        return None

    def qwen38_measurement(self, serialized: str) -> dict:
        import torch
        import torch.distributed as dist
        from qwen38.serving.mtp import source_identity
        rank = dist.get_rank()
        identity = {"rank": rank, "protocol": protocol_identity()}
        try:
            request = json.loads(serialized)
            identity["request_id"] = request.get("request_id")
            if request.get("protocol") != identity["protocol"]:
                raise ValueError("client and worker measurement protocols differ")
            identity["source"] = source_identity(os.environ.get("QWEN38_ENGINE_ROOT", "/app"))
            identity["recipe"] = "mtp3" if (self.vllm_config.num_speculative_tokens or 0) == 3 else "ordinary"
            action = request.get("action")
            handler = {"identity": self._identity, "capture": self._capture, "draws": self._draws,
                       "use": self._use, "memory": self._memory, "release": self._release}.get(action)
            if handler is None:
                raise ValueError(f"unknown measurement action {action!r}")
            with torch.inference_mode():
                value = handler(request)
            # Plain builtin types only: the engine's serializer rejects str subclasses and tensors,
            # and a rejected RPC result stops the engine core's output thread.
            value = json.loads(json.dumps(value, allow_nan=False))
            return {**identity, "status": "complete", "value": value}
        except Exception as exc:
            return {**identity, "status": "failed", "error": repr(exc), "traceback": traceback.format_exc()[-6000:]}

    def _cases(self) -> dict:
        return self.__dict__.setdefault("_qwen38_cases", {})

    def _regions(self, component: str, layer: int, enclosing=None):
        from . import native
        model = self.model_runner.get_model()
        region = native.effective(next(r for r in native.declaration() if r["component"] == component), enclosing)
        entries = native.resolve(model, region, self.vllm_config.compilation_config.static_forward_context)
        if not 0 <= layer < len(entries):
            raise ValueError(f"{component}: no loaded region at index {layer} ({len(entries)} loaded)")
        return entries[layer], region

    def _identity(self, request) -> dict:
        from . import native
        model = self.model_runner.get_model()
        out = {}
        for region in native.declaration():
            out[region["component"]] = {"region": {k: v for k, v in region.items()},
                                        "loaded": [native.identity(e, region) for e in native.resolve(
                                            model, region, self.vllm_config.compilation_config.static_forward_context)]}
        return out

    def _capture(self, request) -> dict:
        import torch
        from . import native, schedule
        case = request["case"]
        entry, region = self._regions(case["component"], case.get("layer", 0),
                                      (request.get("enclosing") or {}).get(case["component"]))
        started = time.perf_counter()
        result = schedule.run(self, case, request["prompts"], request["streams"], [(entry, region)],
                              vocabulary=request["vocabulary"])
        seconds = time.perf_counter() - started
        capture = result["captures"][0]
        full = [c for c in capture.calls if c["phase"] == "full"]
        steps = [s for s in result["layout"] if s["phase"] != "history"]
        if len(capture.calls) != len(result["layout"]) or len(full) != len(steps):
            raise AssertionError(f"the declared region ran {len(capture.calls)} times in "
                                 f"{len(result['layout'])} runner steps")
        expected = self.expected(result, capture, case)
        allowance = ALLOWANCES[region["component"]]
        checks = []
        if expected is not None:
            for call, row in zip(full, expected):
                searches = row.get("selection_searches") or []
                checks.append({"phase": row["phase"], **within(call["output"], row["expected"], allowance),
                               "selection_searches": searches})
        # Keep only what replay and validation need.
        for call in capture.calls:
            if call["phase"] == "history":
                call.pop("hidden", None)
        stored = {"case": case, "entry": entry, "region": region, "result": result, "full": full,
                  "inputs": [native.Inputs(c) for c in full],
                  "expected": [row["expected"] for row in expected] if expected is not None else None,
                  "measured": sum(1 for s in steps if s["phase"] == "measured")}
        self._cases()[request["id"]] = stored
        torch.cuda.synchronize()
        return {"id": request["id"], "identity": native.identity(entry, region), "capture_s": seconds,
                "steps": [{k: (v if k != "input_ids" else len(v)) for k, v in s.items()} for s in result["layout"]],
                "executed": result["executed"], "accepted": result["accepted"],
                "blocks": len(result["blocks"]),
                "state_rows_bytes": sum(r.nbytes() for rows in result["snapshots"]["pre"] for r in rows),
                "outputs": [{"shape": list(c["output"].shape), "dtype": str(c["output"].dtype),
                             "sum": float(c["output"].float().sum())} for c in full],
                "checks": checks,
                "expected_source": "independent" if expected is not None else "captured native outputs"}

    def _sequence(self, stored):
        from . import native
        entry, region = stored["entry"], stored["region"]
        calls = stored["full"][:stored["measured"]]

        def sequence():
            return [native.replay(self.vllm_config, entry, region, call) for call in calls]
        return sequence

    def _restore(self, stored):
        rows = stored["result"]["snapshots"]["pre"][0]
        inputs = stored["inputs"][:stored["measured"]]

        def restore():
            for r in rows:
                r.restore()
            for i in inputs:
                i.restore()
        return restore

    def _validate(self, stored):
        from . import native
        entry, region = stored["entry"], stored["region"]
        full, measured = stored["full"], stored["measured"]
        reference = stored["expected"] or [c["output"] for c in full]
        allowance = ALLOWANCES[region["component"]]
        probe_pre = stored["result"]["snapshots"]["probe_pre"][0]
        post = stored["result"]["snapshots"]["post"][0]

        def validate(outputs):
            rows = [within(o, reference[i], allowance) for i, o in enumerate(outputs)]
            # Evidence, not a verdict: whether this draw left exactly the captured post-state.
            same = post.equal(post.current()) if post is not None else None
            for r in probe_pre:
                r.restore()
            stored["inputs"][measured].restore()
            probe = native.replay(self.vllm_config, entry, region, full[measured])
            rows.append({"probe": True, **within(probe, reference[measured], allowance)})
            return {"max_abs": max(r["max_abs"] for r in rows), "post_state_bitwise_equal": same}
        return validate

    def _draws(self, request) -> dict:
        from . import timing
        stored = self._cases()[request["id"]]
        mode = stored["case"]["mode"]
        record = timing.draws(self._sequence(stored), self._restore(stored), self._validate(stored),
                              execution="eager" if mode == "prefill" else "cuda_graph",
                              warmups=request.get("warmups", timing.WARMUPS),
                              samples=request.get("samples", timing.DRAWS))
        record["id"], record["matrix"] = request["id"], request["matrix"]
        return record

    def _use(self, request) -> dict:
        """Raw evidence that the region's output is consumed: the next decoder layer's input at the
        measured steps for two unperturbed runs and one run whose region output is perturbed in place."""
        import torch
        from . import schedule
        stored = self._cases()[request["id"]]
        case = stored["case"]
        amplitude = float(request.get("amplitude", 1.0))

        def perturb(value):
            pattern = torch.arange(value.shape[-1], device=value.device, dtype=torch.float32).remainder_(7).sub_(3)
            value.add_((pattern * amplitude).to(value.dtype))

        acceptance = case.get("acceptance", "native")
        downstream = self._downstream(stored["entry"])

        def once(intervene, tag):
            # The same request IDs for every run, so batch order and request slots repeat.
            prompts = {f"{k}-use": v for k, v in request["prompts"].items()}
            streams = {f"{k}-use": v for k, v in request["streams"].items()}
            planned = acceptance if acceptance == "native" else {f"{k}-use": v for k, v in acceptance.items()}
            run = schedule.run(self, dict(case, acceptance=planned), prompts, streams,
                               [(stored["entry"], stored["region"])], vocabulary=request["vocabulary"],
                               intervene=intervene, record_model_output=True, downstream=downstream)
            return run["downstream_inputs"]

        # Every run starts from the same cache contents in every layer's rows of the case's blocks,
        # so a deterministic engine repeats exactly and any difference is the intervention's.
        from . import native
        everything = native.bound_state(self.model_runner.get_model(),
                                        self.vllm_config.compilation_config.static_forward_context)
        baseline = native.StateRows(everything, stored["result"]["blocks"])

        def fresh(intervene, tag):
            baseline.restore()
            return once(intervene, tag)
        normal, repeat, perturbed = fresh(None, "normal"), fresh(None, "repeat"), fresh(perturb, "perturbed")
        del baseline
        tol = request.get("tolerance") or {"rtol": 0.0, "atol": 0.0}
        rows = []
        for a, b, c in zip(normal, repeat, perturbed):
            a, b, c = a.float(), b.float(), c.float()
            rows.append({"elements": a.numel(),
                         "repeat_max_abs": float((a - b).abs().max()),
                         "perturbed_max_abs": float((a - c).abs().max()),
                         "perturbed_changed": int(((a - c).abs() > 0).sum()),
                         "tolerance": tol,
                         "repeat_beyond_tolerance": int(((a - b).abs() > tol["atol"] + tol["rtol"] * b.abs()).sum()),
                         "perturbed_beyond_tolerance": int(((a - c).abs() > tol["atol"] + tol["rtol"] * c.abs()).sum())})
        state = self.__dict__.setdefault("_qwen38_use", {})
        state[request["id"]] = {"normal": normal, "repeat": repeat, "perturbed": perturbed}
        return {"id": request["id"], "amplitude": amplitude, "steps": rows}

    def _downstream(self, entry):
        """The next decoder layer after the region's checkpoint layer (its input consumes the region's
        layer output), or the model when the region is in the last layer."""
        import re
        model = self.model_runner.get_model()
        name = re.sub(r"\.layers\.\d+\..*$", f".layers.{entry['layer'] + 1}", entry["name"])
        modules = dict(model.named_modules())
        return modules.get(name, model)

    def _memory(self, request) -> dict:
        from . import memory
        stored = self._cases()[request["id"]]
        return memory.account(self, stored, request)

    def _release(self, request) -> dict:
        import torch
        self._cases().pop(request["id"], None)
        torch.cuda.empty_cache()
        return {"released": request["id"]}
