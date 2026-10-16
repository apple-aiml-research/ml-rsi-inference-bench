#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Identify served SD3 component implementations from records made inside the worker."""


from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

RECORD_ENV = "INFB_SD3_COMPONENT_RECORD_FILE"
TAP_ENV = "INFB_SD3_COMPONENT_TAP_DIR"
RECORDER_DIR = Path(__file__).resolve().parent / "component_recorder"
MODEL_COMPONENTS = ("text_encoder", "text_encoder_2", "text_encoder_3", "vae", "transformer")


SCORED_GATES = {
    "native_text_encoders": ("text_encoder", "text_encoder_2", "text_encoder_3"),
    "native_vae": ("vae",),

    "native_transformer": ("transformer",),
}


WHOLE_MODEL_GATES = ("native_transformer",)
INFORMATIONAL: dict = {}
FOREIGN_PREFIXES = ("transformers.", "diffusers.")
NATIVE_PREFIXES = ("sglang.",)


WHOLE_MODEL_MODULES = ("diffusers.models.transformers", "diffusers.models.autoencoders", "diffusers.models.unets",
                       "diffusers.pipelines", "transformers")


DENOISER_MODEL_MODULES = ("diffusers.models.transformers", "diffusers.models.unets", "diffusers.pipelines")

HOOK_TARGETS = {
    "loader": "sglang.multimodal_gen.runtime.loader.component_loaders.component_loader"
              ".PipelineComponentLoader.load_component",
    "pipeline": "sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base.ComposedPipelineBase.__init__",
    "transformers": "transformers.modeling_utils.PreTrainedModel.__init__",
    "diffusers": "diffusers.models.modeling_utils.ModelMixin.__init__",
}


ATTN_TARGET = "sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend.AttentionImpl"
STALE_SLACK_S = 5.0
_PROBLEM_EVENTS = ("patch_skipped", "patch_error", "record_error", "truncated")


def arm(environ, record_path, recorder_dir=RECORDER_DIR, clock=time.time) -> dict:
    """Point the NEXT spawned interpreters at the recorder: prepend its dir to PYTHONPATH, name the record file.

    Never raises. Returns the token disarm() needs to restore both variables EXACTLY (unset stays unset, an
    empty PYTHONPATH stays empty -- a trailing ':' would put the CWD on the worker's sys.path). Any record file
    already at the path is removed first, so every line the probe reads was written after this call.
    """
    token = {"armed": False, "record": str(record_path), "recorder_dir": str(recorder_dir), "prev": None}
    try:
        if not (Path(recorder_dir) / "sitecustomize.py").is_file():
            token["error"] = f"recorder missing: {Path(recorder_dir) / 'sitecustomize.py'}"
            return token
        rp = Path(record_path)
        rp.parent.mkdir(parents=True, exist_ok=True)
        if rp.is_symlink() or rp.exists():
            token["removed_preexisting"] = True
            rp.unlink()
        tap = rp.parent / "sd3_component_tap"
        tap.mkdir(parents=True, exist_ok=True)
        for stale in list(tap.iterdir()):
            if stale.is_file() or stale.is_symlink():
                stale.unlink()
        token["tap_dir"] = str(tap)
        token["prev"] = {k: environ.get(k) for k in (RECORD_ENV, TAP_ENV, "PYTHONPATH")}
        old = environ.get("PYTHONPATH")
        environ["PYTHONPATH"] = str(recorder_dir) + (os.pathsep + old if old else "")
        environ[RECORD_ENV] = str(rp)
        environ[TAP_ENV] = str(tap)
        token["armed_at"] = clock()
        token["armed"] = True
    except Exception as exc:  # noqa: BLE001 - arming must never stop the engine build
        token["error"] = f"{type(exc).__name__}: {exc}"[:300]
        disarm(environ, token)
    return token


def disarm(environ, token) -> None:
    """Restore PYTHONPATH and the record variable to exactly what arm() found. Never raises."""
    try:
        prev = (token or {}).get("prev")
        if not prev:
            return
        for key, value in prev.items():
            if value is None:
                environ.pop(key, None)
            else:
                environ[key] = value
        token["disarmed"] = True
    except Exception:  # noqa: BLE001
        pass


def read_records(path) -> tuple[list, int]:
    """(events, unparsable_line_count). A missing file is ([], 0): the recorder never wrote."""
    p = Path(path)
    if not p.is_file():
        return [], 0
    events, bad = [], 0
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            bad += 1
            continue
        if isinstance(rec, dict):
            events.append(rec)
        else:
            bad += 1
    return events, bad


def _pair(x):
    if isinstance(x, (list, tuple)) and len(x) == 2 and all(isinstance(s, str) for s in x):
        return [x[0], x[1]]
    return None


def _is_fsdp_base(entry) -> bool:
    return bool(entry) and entry[1] == "FSDPModule" and entry[0].startswith("torch.distributed.fsdp")


def impl_of(mro):
    """[module, qualname] of the implementing class; None when the MRO is unreadable. Mirrors the recorder's
    `_impl`: skip torch's classes by module, and skip FSDP2's dynamic wrapper by POSITION -- it is the entry
    directly before FSDPModule. Its module is not torch's when the wrapped class derives from ABC: the class
    is then built inside ABCMeta.__new__ and can appear as `abc.FSDP<Name>`. A module-only rule cannot
    distinguish that wrapper from the implementing class."""
    entries = [e for e in (_pair(x) for x in (mro or [])) if e is not None]
    for i, (mod, qual) in enumerate(entries):
        if mod == "builtins":
            break
        if mod == "torch" or mod.startswith("torch."):
            continue
        if i + 1 < len(entries) and _is_fsdp_base(entries[i + 1]):
            continue
        return [mod, qual]
    return entries[0] if entries else None


def kind_of(module) -> str:
    if not isinstance(module, str) or not module:
        return "unrecorded"
    if module.startswith(NATIVE_PREFIXES):
        return "native"
    if module.startswith(FOREIGN_PREFIXES):
        return "delegated"
    return "other"


def classify(info) -> dict:
    """{impl, kind} for one recorded object ({raw, mro, ...} as the recorder writes it)."""
    if not isinstance(info, dict):
        return {"impl": None, "kind": "unrecorded"}
    impl = impl_of(info.get("mro")) or _pair(info.get("raw"))
    return {"impl": impl, "kind": kind_of(impl[0]) if impl else "unrecorded"}


def _combine(kinds: list) -> str:
    if not kinds:
        return "unrecorded"
    if "delegated" in kinds:
        return "delegated"
    if all(k == "native" for k in kinds):
        return "native"
    if "other" in kinds:
        return "other"
    return "unrecorded"


def _under(module, prefixes) -> bool:
    return isinstance(module, str) and any(module == p or module.startswith(p + ".") for p in prefixes)


def is_whole_model_module(module) -> bool:
    return _under(module, WHOLE_MODEL_MODULES)


def _whole_in_mro(mro, prefixes=WHOLE_MODEL_MODULES) -> list:
    """`module.qualname` of every MRO entry defined in one of `prefixes` (whole-model modules by default)."""
    return [".".join(e) for e in (_pair(x) for x in (mro or [])) if e and _under(e[0], prefixes)]


def whole_model_evidence(served_infos, window_events, outside_events) -> dict:
    """Pure: the DiT whole-model rule on one component's record. Every part must be
    evaluable from the record, or it is a reason in `not_evaluable`, never a pass.

    What the worker SERVES (each `served` record of the component):
      mro      no class of its hierarchy is defined in a WHOLE_MODEL_MODULES module (a thin SGLang subclass of
               diffusers' SD3Transformer2DModel is `native` by impl_of, and this is what sees it);
      forward  the class that defines `forward` is not a transformers/diffusers class, nor is the function itself
               (`method_fn`: a class attribute `forward = <diffusers' forward>` is owned by
               the SGLang class but defined by diffusers);
      tree     no submodule is, or derives from, a whole-model-module class. Read from the tree's `whole` field
               (every such class, untruncated); when only `foreign` is available, it holds each
               submodule's implementing class, so it cannot see an SGLang subclass of a whole model, and keeps
               only the 16 most frequent kinds, so it is evaluable only when nothing was cut.
    What the worker BUILT (the `instantiated` events, fired by every transformers/diffusers model __init__):
      in its own loader window, a model whose MRO holds a whole-model class although its implementing class is
      SGLang's (a model whose implementing class is foreign is already every native gate's `foreign_built_in_
      window` reason, so it is not repeated here); outside every loader window, any model whose MRO holds a
      diffusers denoiser or pipeline class.
    """
    reasons, not_evaluable = [], []
    detail = {"mro_whole": [], "forward": None, "forward_fn": None, "tree_whole": [], "tree_source": None,
              "built_in_window": [], "built_outside": []}
    infos = [i for i in (served_infos or []) if isinstance(i, dict)]
    for info in infos:
        bad = _whole_in_mro(info.get("mro"))
        detail["mro_whole"] += bad
        if bad:
            reasons.append(f"its class hierarchy holds {bad}")
        owners = info.get("method_owner") if isinstance(info.get("method_owner"), dict) else {}
        owner = _pair(owners.get("forward"))
        detail["forward"] = owner
        if owner is None:
            not_evaluable.append("the class that defines its `forward` was not recorded")
        elif kind_of(owner[0]) == "delegated":
            reasons.append(f"its forward is {'.'.join(owner)}.forward")
        fns = info.get("method_fn") if isinstance(info.get("method_fn"), dict) else {}
        fn = _pair(fns.get("forward"))
        detail["forward_fn"] = fn
        if fn is not None and kind_of(fn[0]) == "delegated":
            reasons.append(f"its forward function is {'.'.join(fn)}, defined in {fn[0]}")
        tree = info.get("tree") if isinstance(info.get("tree"), dict) else None
        if tree is None or info.get("tree_error"):
            not_evaluable.append(("its module tree walk failed" if tree is not None else
                                  "its module tree was not recorded")
                                 + (f" ({info['tree_error']})" if info.get("tree_error") else ""))
            continue
        if tree.get("truncated"):
            not_evaluable.append(f"its module tree walk was cut at {tree.get('n_modules')} modules")
            continue
        if isinstance(tree.get("whole"), dict):
            detail["tree_source"] = "whole"
            found = sorted(tree["whole"])
            if tree.get("whole_kinds") and not found:
                found = [f"{tree['whole_kinds']} whole-model kind(s)"]
        else:
            detail["tree_source"] = "foreign"
            foreign = tree.get("foreign") if isinstance(tree.get("foreign"), dict) else {}
            found = sorted(k for k in foreign if any(k.startswith(p + ".") for p in WHOLE_MODEL_MODULES))
            kinds = tree.get("foreign_kinds")
            if not found and (not isinstance(kinds, int) or kinds > len(foreign)):
                not_evaluable.append(f"this record's tree keeps {len(foreign)} of {kinds} foreign kinds and predates "
                                     "the untruncated whole-model field")
        detail["tree_whole"] += found
        if found:
            via = tree.get("whole_via") if isinstance(tree.get("whole_via"), dict) else {}
            reasons.append("its module tree holds " + ", ".join(
                k + (f" (as {via[k]})" if via.get(k) and via[k] != k else "") for k in found[:8]))
    for e in window_events or []:
        impl = impl_of(e.get("mro")) or _pair(e.get("cls"))
        whole = ([e["whole"]] if isinstance(e.get("whole"), str) else []) or _whole_in_mro(e.get("mro"))
        if whole and not (impl and kind_of(impl[0]) == "delegated"):
            name = ".".join(impl or ["?"])
            detail["built_in_window"].append(name)
            reasons.append(f"built {name}, a {whole[0]}, while loading it")
    for e in outside_events or []:
        whole = ([e["whole"]] if isinstance(e.get("whole"), str) and any(
            e["whole"].startswith(p + ".") for p in DENOISER_MODEL_MODULES) else []) or _whole_in_mro(
            e.get("mro"), DENOISER_MODEL_MODULES)
        if whole:
            name = ".".join(impl_of(e.get("mro")) or _pair(e.get("cls")) or ["?"])
            detail["built_outside"].append(name)
            reasons.append(f"built {name} outside every loader window" + (f" (a {whole[0]})" if whole[0] != name
                                                                        else ""))
    if not infos:
        not_evaluable.append("no served record of it")
    return {"reasons": reasons, "not_evaluable": not_evaluable, **detail}


def summarize(events, armed_at=None, loader_lines=None, parent_pid=None) -> dict:
    """Pure: the probe's whole reading of one record. No I/O, no clock."""
    fresh, stale = [], 0
    for e in events or []:
        if not isinstance(e, dict):
            continue
        t = e.get("t")
        if armed_at is not None and isinstance(t, (int, float)) and t < armed_at - STALE_SLACK_S:
            stale += 1
            continue
        fresh.append(e)

    pids: dict = {}
    for e in fresh:
        rec = pids.setdefault(e.get("pid"), {"installed": False, "patched": [], "events": 0, "argv": None})
        rec["events"] += 1
        if e.get("event") == "installed":
            rec["installed"] = True
            rec["argv"] = e.get("argv")
        elif e.get("event") == "patched" and isinstance(e.get("target"), str):
            rec["patched"].append(e["target"])
    by_target = {k: sorted(p for p, r in pids.items() if v in r["patched"]) for k, v in HOOK_TARGETS.items()}
    workers = sorted({e.get("pid") for e in fresh if e.get("event") in ("load_begin", "load_end", "served")},
                     key=str)
    served = [e for e in fresh if e.get("event") == "served" and isinstance(e.get("modules"), dict)]
    ran = any(pids.get(p, {}).get("installed") and pids.get(p, {}).get("patched") for p in workers)

    missing = []
    if ran:
        if not served:
            missing.append("no `served` line: what the pipeline holds was never recorded "
                           "(ComposedPipelineBase.__init__ not reached, or its hook not installed)")
        for hook in ("loader", "pipeline"):
            if not any(p in by_target[hook] for p in workers):
                missing.append(f"the {hook} hook ({HOOK_TARGETS[hook]}) was not patched in the worker")
        for e in fresh:
            if e.get("pid") in workers and (e.get("event") == "patch_error" or (
                    e.get("event") == "patch_skipped" and e.get("why") != "already wrapped")):
                missing.append(f"{e.get('event')} on {e.get('target')}: {e.get('why') or e.get('error')}")
            if e.get("pid") in workers and e.get("event") == "truncated":
                missing.append("the worker's record was truncated")
    complete = bool(ran and not missing)

    loader_lines = loader_lines if isinstance(loader_lines, dict) else {}
    built_outside = [e for e in fresh if e.get("event") == "instantiated" and not e.get("component")]
    components = {}
    for comp in MODEL_COMPONENTS:
        served_cls = [classify(e["modules"].get(comp)) for e in served if comp in e["modules"]]
        loaded_cls = [classify(e.get("obj")) for e in fresh
                      if e.get("event") == "load_end" and e.get("component") == comp and e.get("ok")]
        in_window = [e for e in fresh if e.get("event") == "instantiated" and e.get("component") == comp]
        built_in_window = []
        for e in in_window:
            impl = impl_of(e.get("mro")) or _pair(e.get("cls"))
            if impl and kind_of(impl[0]) == "delegated":
                built_in_window.append(impl)
        source, chosen = (("served", served_cls) if served_cls else ("loaded", loaded_cls) if loaded_cls
                          else (None, []))
        kind = _combine([c["kind"] for c in chosen])
        impl = next((c["impl"] for c in chosen if c["kind"] == kind), chosen[0]["impl"] if chosen else None)
        swapped = bool(served_cls and loaded_cls
                       and _combine([c["kind"] for c in served_cls]) != _combine([c["kind"] for c in loaded_cls]))
        tree, owners = {}, {}
        for e in served:
            info = e["modules"].get(comp)
            if isinstance(info, dict) and isinstance(info.get("tree"), dict):
                tree = info["tree"]
            if isinstance(info, dict) and isinstance(info.get("method_owner"), dict):
                owners = {m: _pair(o) for m, o in info["method_owner"].items() if _pair(o)}
        line = loader_lines.get(comp) if isinstance(loader_lines.get(comp), dict) else None
        says_native = (line.get("impl") == "sgl-diffusion") if line else None
        agrees = (says_native == (kind == "native")) if (line and kind != "unrecorded") else None
        components[comp] = {
            "kind": kind, "impl": impl, "source": source,
            "served": [c["impl"] for c in served_cls], "loaded": [c["impl"] for c in loaded_cls],
            "swapped_after_load": swapped,
            "foreign_built_in_window": built_in_window,
            "tree_foreign_total": tree.get("foreign_total"), "tree_foreign": tree.get("foreign"),


            "method_owner": owners,
            "foreign_method_owner": sorted(m for m, o in owners.items() if kind_of(o[0]) == "delegated"),
            "loader_line": line, "loader_line_says_native": says_native, "agrees_with_loader_line": agrees,

            "whole_model": whole_model_evidence(
                [e["modules"].get(comp) for e in served if comp in e["modules"]], in_window, built_outside),
        }

    outside = []
    for e in built_outside:
        impl = impl_of(e.get("mro")) or _pair(e.get("cls"))
        if impl and kind_of(impl[0]) == "delegated":
            outside.append({"impl": impl, "via": e.get("via"), "pid": e.get("pid")})

    def _verdict(names, whole=False) -> dict:
        if not ran:
            return {"verdict": "FAIL", "not_evaluated": True,
                    "reasons": ["the recorder did not demonstrably run in the worker (no process that loaded "
                                "components reported `installed` + `patched`) -- fail-closed"]}
        if not complete:
            return {"verdict": "FAIL", "not_evaluated": True,
                    "reasons": ["the worker's component record is incomplete -- fail-closed: "
                                + "; ".join(missing)]}
        reasons, unevaluable = [], []
        for c in names:
            rec = components[c]
            if rec["kind"] == "unrecorded":
                reasons.append(f"{c}: no served or loaded record")
            elif rec["source"] != "served":
                reasons.append(f"{c}: the pipeline's record does not hold it (only the loader's return value, "
                               f"{'.'.join(rec['impl'] or ['?', '?'])}, was seen)")
            elif rec["kind"] == "delegated":
                reasons.append(f"{c}: implemented by {'.'.join(rec['impl'])}")
            elif rec["kind"] == "other":
                reasons.append(f"{c}: implementing class {'.'.join(rec['impl'] or ['?', '?'])} is not SGLang's own")
            if rec["foreign_built_in_window"]:
                reasons.append(f"{c}: built {['.'.join(x) for x in rec['foreign_built_in_window']]} while loading it")
            if whole and rec["source"] == "served":
                wm = rec["whole_model"]
                reasons += [f"{c}: {r}" for r in wm["reasons"]]
                unevaluable += [f"{c}: {r} -- not evaluable, fail-closed" for r in wm["not_evaluable"]]
        reasons += unevaluable
        return {"verdict": "FAIL" if reasons else "PASS",
                "not_evaluated": bool(unevaluable) and len(reasons) == len(unevaluable), "reasons": reasons}

    problems = [{k: e.get(k) for k in ("event", "pid", "target", "why", "where", "error") if k in e}
                for e in fresh if e.get("event") in _PROBLEM_EVENTS]
    dit_calls = [[e.get("pid"), e.get("t")] for e in fresh
                 if e.get("event") == "dit_call" and isinstance(e.get("t"), (int, float))]
    dit_counter = any(e.get("event") == "dit_counter_installed" for e in fresh)

    attention = {
        "window_installed": any(e.get("event") == "dit_counter_installed" and e.get("attn_window") for e in fresh),
        "impl_wrapped_pids": sorted((p for p, r in pids.items() if ATTN_TARGET in r["patched"]), key=str),
        "dit_attn_calls": [[e.get("pid"), e.get("t"), e.get("attn")] for e in fresh
                           if e.get("event") == "dit_call" and isinstance(e.get("t"), (int, float))
                           and isinstance(e.get("attn"), dict)][:20000],
        "probes": [{k: e.get(k) for k in ("pid", "t", "n", "impls", "delta", "repeat_rel", "repeat_calls",
                                          "per_call_rel", "reruns", "rerun_s", "error") if k in e}
                   for e in fresh if e.get("event") == "attn_probe"][:8],
        "captures": [{k: e.get(k) for k in ("pid", "t", "call", "file", "impl", "q_shape", "k_shape", "out_shape",
                                            "softmax_scale", "causal", "rows", "heads") if k in e}
                     for e in fresh if e.get("event") == "attn_capture"][:64],
        "errors": [{k: e.get(k) for k in ("pid", "where", "error") if k in e}
                   for e in fresh if e.get("event") == "attn_probe_error"][:20],
    }
    return {
        "recorder": {
            "ran_in_worker": bool(ran), "evidence_complete": complete, "missing": missing,
            "worker_pids": workers,
            "installed_pids": sorted((p for p, r in pids.items() if r["installed"]), key=str),
            "patched": by_target,
            "per_pid": {str(p): r for p, r in pids.items()},
            "parent_pid_events": (pids.get(parent_pid, {}).get("events", 0) if parent_pid is not None else None),
            "problems": problems[:50], "events": len(fresh), "stale_events": stale,
        },
        "components": components,
        "foreign_models_built_outside_any_loader_window": outside[:50],
        "dit_counter_installed": dit_counter,
        "dit_calls": dit_calls[:20000],
        "native_execution": {
            "installed": any(e.get("event") == "batch_observer_installed" and e.get("pid") in workers for e in fresh),
            "installations": [e for e in fresh if e.get("event") == "batch_observer_installed"],
            "forwards": [e for e in fresh if e.get("event") == "batch_dit"][:20000],
            "errors": [e for e in fresh if e.get("event") == "batch_observer_error"][:100],
        },
        "attention": attention,

        "gate_verdicts": {g: _verdict(names, whole=g in WHOLE_MODEL_GATES) for g, names in SCORED_GATES.items()},
        "informational": {
            **{g: _verdict(names) for g, names in INFORMATIONAL.items()},
            "no_unattributed_foreign_models": {
                "verdict": ("PASS" if not outside else "FAIL") if ran else "FAIL",
                "not_evaluated": not ran, "count": len(outside)},
        },
        "disagreements_with_loader_line": [c for c in MODEL_COMPONENTS
                                           if components[c]["agrees_with_loader_line"] is False],
    }


NATIVE_FALLBACK_REASON = "scored images came from the diffusers fallback, not the candidate's port"


def native_pipeline_evidence(summary) -> dict:
    """Pure: did the worker serve SGLang's own SD3 pipeline, and build no transformers/diffusers model outside the
    loader? A DiffusersExecutionStage fallback produces reference-pipeline images, which cannot establish
    correctness of a native SGLang implementation."""
    s = summary if isinstance(summary, dict) else {}
    rec = s.get("recorder") or {}
    dit = (s.get("components") or {}).get("transformer") or {}
    outside = s.get("foreign_models_built_outside_any_loader_window") or []
    reasons = []
    if not rec.get("ran_in_worker"):
        reasons.append("no worker record of a loaded pipeline (the recorder saw no process load components)")
    elif not rec.get("evidence_complete"):
        reasons.append("the worker's component record is incomplete: " + "; ".join(rec.get("missing") or []))
    if dit.get("source") != "served" or dit.get("kind") != "native":
        reasons.append("the pipeline the worker served holds no SGLang DiT (transformer: %s, %s)"
                       % (dit.get("kind"), ".".join(dit.get("impl") or ["?"])))
    if outside:
        names = sorted({".".join(o.get("impl") or ["?"]) for o in outside})
        reasons.append(f"{len(outside)} transformers/diffusers model(s) built outside every loader window: "
                       + ", ".join(names)[:300])
    return {"ok": not reasons, "reasons": reasons, "foreign_outside": len(outside),
            "not_evaluated": bool(reasons) and not outside and (
                not rec.get("ran_in_worker") or not rec.get("evidence_complete"))}


def dit_calls_between(summary, t0, t1) -> "int | None":
    'Pure: how many forwards of the served DiT the worker recorded inside [t0, t1]; None when this record carries'

    s = summary if isinstance(summary, dict) else {}
    if not s.get("dit_counter_installed") or not isinstance(t0, (int, float)) or not isinstance(t1, (int, float)):
        return None
    workers = set((s.get("recorder") or {}).get("worker_pids") or [])
    return sum(1 for pid, t in (s.get("dit_calls") or []) if (not workers or pid in workers) and t0 <= t <= t1)


def read_summary(token, loader_lines, parent_pid=None) -> dict:
    """Read the armed engine's record and summarize the evidence used by native component gates.

    Never raises. A token that was never armed, or a record that cannot be read, yields a summary whose
    gate_verdicts FAIL with the reason (the recorder did not run), never a pass.
    """
    token = token if isinstance(token, dict) else {}
    try:
        events, bad = read_records(token.get("record") or "")
        out = summarize(events, armed_at=token.get("armed_at"), loader_lines=loader_lines, parent_pid=parent_pid)
        out["recorder"]["unparsable_lines"] = bad
    except Exception as exc:  # noqa: BLE001
        why = f"the component record could not be read: {type(exc).__name__}: {exc}"[:300]
        out = {"error": why, "gate_verdicts": {g: {"verdict": "FAIL", "not_evaluated": True,
                                                   "reasons": [why + " -- fail-closed"]}
                                               for g in SCORED_GATES}}
    if not token.get("armed"):
        why = f"the recorder was never armed ({token.get('error') or 'no token'})"
        out.setdefault("recorder", {})["armed_error"] = why
        for v in (out.get("gate_verdicts") or {}).values():
            if v.get("not_evaluated"):
                v["reasons"] = [why] + list(v.get("reasons") or [])
    out["armed"] = {k: token.get(k) for k in ("armed", "armed_at", "record", "error", "removed_preexisting")}
    return out


def probe(token, loader_lines, out_path, load_error=None, parent_pid=None) -> dict:
    """End of run: re-read the record (now also holding anything built lazily) and write the diagnostic JSON.
    Never raises; nothing reads the file it writes."""
    out = read_summary(token, loader_lines, parent_pid=parent_pid)
    if load_error:
        out["engine_load_error"] = str(load_error)[:300]
    try:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, p)
    except Exception as exc:  # noqa: BLE001
        out["write_error"] = f"{type(exc).__name__}: {exc}"[:300]
    return out


def one_line(summary) -> str:
    """The stdout summary the verifier prints (test-stdout.txt is collected on every backend)."""
    s = summary if isinstance(summary, dict) else {}
    r = s.get("recorder") or {}
    comps = s.get("components") or {}
    parts = [f"{c}={(comps.get(c) or {}).get('kind')}"
             f"({'.'.join((comps.get(c) or {}).get('impl') or ['?'])})" for c in MODEL_COMPONENTS]
    would = s.get("gate_verdicts") or {}
    return (f"[component-probe] post-generation re-read: ran_in_worker={r.get('ran_in_worker')} "
            f"worker_pids={r.get('worker_pids')} " + " ".join(parts)
            + " | gates: " + " ".join(f"{g}={(v or {}).get('verdict')}" for g, v in would.items())
            + f" | disagree_with_loader_line={s.get('disagreements_with_loader_line')}")


LOADER_LINE_RE = re.compile(r"Loaded (\w+): (\S+) \(([^)]+) version\)")


def parse_loader_lines(text) -> dict:
    """{component: {"cls", "impl"}} from any text holding the loader's lines (the verifier's test-stdout.txt)."""
    return {comp: {"cls": cls, "impl": ver.strip()} for comp, cls, ver in LOADER_LINE_RE.findall(text or "")}


def main(argv=None) -> int:
    """Offline re-derivation from collected artifacts:

        python3 component_probe.py RECORD.jsonl [ARMED_AT] [--stdout test-stdout.txt] [--inrun probe.json]

    ARMED_AT drops lines written before the verifier armed the record (the in-run probe JSON's armed.armed_at;
    --inrun supplies it too). Loader lines, for the disagreement column, come from --stdout (the verifier's own
    stdout, the independent copy) or else from --inrun's components[*].loader_line.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    opts = {}
    for flag in ("--stdout", "--inrun"):
        if flag in args:
            i = args.index(flag)
            if i + 1 >= len(args):
                print(main.__doc__, file=sys.stderr)
                return 2
            opts[flag] = args[i + 1]
            del args[i:i + 2]
    if not args:
        print(main.__doc__, file=sys.stderr)
        return 2
    armed_at, lines = (float(args[1]) if len(args) > 1 else None), None
    if "--inrun" in opts:
        inrun = json.loads(Path(opts["--inrun"]).read_text(encoding="utf-8"))
        if armed_at is None:
            armed_at = (inrun.get("armed") or {}).get("armed_at")
        lines = {c: r.get("loader_line") for c, r in (inrun.get("components") or {}).items() if r.get("loader_line")}
    if "--stdout" in opts:
        lines = parse_loader_lines(Path(opts["--stdout"]).read_text(encoding="utf-8", errors="replace"))
    events, bad = read_records(args[0])
    out = summarize(events, armed_at=armed_at, loader_lines=lines)
    out["recorder"]["unparsable_lines"] = bad
    out["offline"] = {"record": args[0], "armed_at": armed_at, "loader_lines_from":
                      "--stdout" if "--stdout" in opts else "--inrun" if lines is not None else None}
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
