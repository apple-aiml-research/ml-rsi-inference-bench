#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The bounded speculative-transition observer, loaded into the candidate's server processes
by site's `sitecustomize` hook (test_hidden.py puts this directory first on the server's PYTHONPATH and names the
record directory in INFB_SPEC_OBSERVER_DIR; nothing here runs without that variable).

It wraps the engine's generic verification boundaries -- `EagleVerifyInputV2Mixin.sample` and the legacy
`EagleVerifyInput.verify` -- after their modules load, and ONLY while `<dir>/on` exists (the generated-history nodes
create it for their own untimed schedules and remove it after) records per verification step, for every request in
the batch: its identity, the pre-verify length, the verified positions, the accepted indices, the returned token count
(the V2 count, which includes the bonus token, cross-checked against the accepted indices), the accepted drafts
(returned minus one), and the next committed length. It reads small
logical metadata after the step's own computation, outside graph capture; it records no compressor buffer and changes
no value the engine computes. At most MAX_RECORDS steps per process are written (`<dir>/spec-<pid>.jsonl`).

Any other `sitecustomize` further along sys.path (the distribution's) is executed after this one, unchanged.
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import sys
import time

MAX_RECORDS = 4096
_DIR = os.environ.get("INFB_SPEC_OBSERVER_DIR")
_HERE = os.path.dirname(os.path.abspath(__file__))
_count = 0


def _active() -> bool:
    return bool(_DIR) and os.path.exists(os.path.join(_DIR, "on"))


def _write(rec: dict) -> None:
    global _count
    if _count >= MAX_RECORDS:
        return
    _count += 1
    try:
        with open(os.path.join(_DIR, f"spec-{os.getpid()}.jsonl"), "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    except OSError:
        pass


def _ids(batch) -> list:
    reqs = getattr(batch, "reqs", None)
    if reqs:
        return [str(getattr(r, "rid", "")) for r in reqs]
    rpi = getattr(batch, "req_pool_indices", None)
    return [f"slot:{int(x)}" for x in rpi.tolist()] if rpi is not None else []


def _record(kind: str, self, batch, accepted_drafts, accept_index) -> None:
    try:
        n = int(getattr(self, "draft_token_num", 0))
        lens = batch.seq_lens_cpu.tolist() if getattr(batch, "seq_lens_cpu", None) is not None \
            else batch.seq_lens.tolist()
        acc = accepted_drafts.tolist() if accepted_drafts is not None else []
        idx = accept_index.tolist() if accept_index is not None else []
        ids = _ids(batch)
        rows = []
        for i, pre in enumerate(lens):
            count = int(acc[i]) if i < len(acc) else None
            ai = [int(x) for x in idx[i]] if i < len(idx) else []
            by_index = sum(1 for x in ai if x >= 0) if ai else None
            returned = count if count is not None else by_index
            rows.append({"request": ids[i] if i < len(ids) else None, "pre_len": int(pre),
                         "verified": [int(pre), int(pre) + n], "returned": returned,
                         "returned_by_index": by_index, "consistent": by_index is None or by_index == returned,
                         "accepted_drafts": None if returned is None else int(returned) - 1, "accept_index": ai,
                         "next_len": None if returned is None else int(pre) + int(returned)})
        _write({"t": time.time(), "boundary": kind, "draft_tokens": n, "rows": rows})
    except Exception as exc:  # noqa: BLE001 - an observation problem is recorded, never raised into the engine
        _write({"t": time.time(), "boundary": kind, "observer_error": f"{type(exc).__name__}: {exc}"[:300]})


def _patch_v2(mod) -> None:
    cls = getattr(mod, "EagleVerifyInputV2Mixin", None)
    if cls is None or getattr(cls.sample, "_infb_observed", False):
        return
    orig = cls.sample

    def sample(self, batch, *args, **kwargs):
        out = orig(self, batch, *args, **kwargs)
        if _active() and not batch.forward_mode.is_idle():
            predict, accepted, accept_index = out
            _record("EagleVerifyInputV2Mixin.sample", self, batch, accepted, accept_index)
        return out

    sample._infb_observed = True
    cls.sample = sample


def _patch_v1(mod) -> None:
    cls = getattr(mod, "EagleVerifyInput", None)
    if cls is None or not hasattr(cls, "verify") or getattr(cls.verify, "_infb_observed", False):
        return
    orig = cls.verify

    def verify(self, batch, *args, **kwargs):
        out = orig(self, batch, *args, **kwargs)
        if _active():
            accepted = getattr(out, "num_accepted_drafts", None) or getattr(out, "accept_length_per_req_cpu", None)
            if isinstance(accepted, list):
                import torch  # noqa: PLC0415

                accepted = torch.tensor(accepted)
            _record("EagleVerifyInput.verify", self, batch, accepted, getattr(out, "accepted_indices", None))
        return out

    verify._infb_observed = True
    cls.verify = verify


_TARGETS = {"sglang.srt.speculative.eagle_info_v2": _patch_v2, "sglang.srt.speculative.eagle_info": _patch_v1}


class _AfterImport(importlib.abc.MetaPathFinder):
    """Patch a target module right after it executes; every other import is untouched."""

    def find_spec(self, name, path, target=None):
        if name not in _TARGETS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return None
        loader = spec.loader
        orig = loader.exec_module

        def exec_module(module, _orig=orig, _name=name):
            _orig(module)
            try:
                _TARGETS[_name](module)
            except Exception as exc:  # noqa: BLE001
                _write({"t": time.time(), "patch_error": f"{_name}: {type(exc).__name__}: {exc}"[:300]})

        loader.exec_module = exec_module
        return spec


def _chain() -> None:
    """Run the next `sitecustomize` on sys.path (skipping this directory), as site would have."""
    for entry in sys.path:
        if not entry or os.path.abspath(entry) == _HERE:
            continue
        cand = os.path.join(entry, "sitecustomize.py")
        if os.path.isfile(cand):
            spec = importlib.util.spec_from_file_location("_infb_chained_sitecustomize", cand)
            if spec and spec.loader:
                try:
                    spec.loader.exec_module(importlib.util.module_from_spec(spec))
                except Exception:  # noqa: BLE001 - the distribution's hook keeps its own failure behaviour
                    pass
            return


if _DIR:
    sys.meta_path.insert(0, _AfterImport())
_chain()
