#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Unscored common-input DiT readout, appended to the frozen in-run reference generator.

common_input_readout.py runs generate_goldens_inrun._gen_script() unchanged with an empty fixture spec, so `pipe` is
the independent diffusers pipeline under the frozen mixed recipe, then this text. Every snapshot is read with
torch.load(weights_only=True), and only after its file sha256 matches the manifest. The diffusers transformer is
replayed on exactly the captured inputs under the mixed convention (bf16 autocast). One JSON line is appended per
manifest item as it is decided; no gate, bar, verdict or reward reads any of it.
"""

_SD3_READOUT_NAMES = ("hidden_states", "encoder_hidden_states", "pooled_projections", "timestep")


def _sd3_readout(pipe, manifest, out, deadline):
    import hashlib
    import json
    import os
    import time
    import torch

    device = next(iter(pipe.transformer.parameters())).device

    limits = manifest.get("limits") or {}
    reserve = float(limits.get("item_reserve_s", 5.0))
    save_max = int(limits.get("save_reference_bytes_max", 0))
    log = open(os.path.join(out, "readout.jsonl"), "a", encoding="utf-8")

    def emit(rec):
        log.write(json.dumps(rec, sort_keys=True) + "\n")
        log.flush()
        os.fsync(log.fileno())

    def file_sha(path):
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 22), b""):
                h.update(block)
        return h.hexdigest()

    def tensor_sha(t):

        dense = torch.empty(tuple(t.shape), dtype=t.dtype)
        dense.copy_(t.detach().cpu())
        return hashlib.sha256(dense.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()

    def describe(t):
        return {"dtype": str(t.dtype).replace("torch.", ""), "shape": list(t.shape), "sha256": tensor_sha(t)}

    def delta(a, b):
        if a is None or b is None:
            return None
        a, b = a.detach().to(device).float(), b.detach().to(device).float()
        if a.shape != b.shape or not a.numel() or not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
            return {"valid": False, "a_shape": list(a.shape), "b_shape": list(b.shape)}
        d = a - b
        rows = [float(d[i].norm() / b[i].norm().clamp_min(1e-30)) for i in range(a.shape[0])] if a.ndim > 1 else []
        return {"valid": True, "bit_identical": bool(torch.equal(a, b)), "max_abs": float(d.abs().max()),
                "relative_l2": float(d.norm() / b.norm().clamp_min(1e-30)), "relative_l2_per_row": rows}

    def load(path, sha):
        if file_sha(path) != sha:
            return None, "hash_mismatch"
        try:
            cap = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:  # noqa: BLE001 - never retried with weights_only=False
            return None, "unreadable_weights_only: " + f"{type(exc).__name__}: {exc}"[:300]
        if not isinstance(cap, dict) or any(type(cap.get(k)) is not torch.Tensor
                                            for k in _SD3_READOUT_NAMES + ("output",)):
            return None, "malformed_snapshot"
        return cap, None

    in_channels = int(pipe.transformer.config.in_channels)
    emit({"event": "start", "monotonic": time.monotonic(), "deadline": deadline, "items": len(manifest["items"]),
          "transformer": type(pipe.transformer).__module__ + "." + type(pipe.transformer).__name__,
          "transformer_dtypes": sorted({str(p.dtype) for p in pipe.transformer.parameters()})})
    cache, saved, slowest = {}, 0, 0.0
    for item in manifest["items"]:
        t0 = time.monotonic()
        rec = {"id": item["id"], "kind": item["kind"]}
        if t0 + max(2.0 * slowest, reserve) > deadline:
            emit({**rec, "outcome": "time_budget_exhausted"})
            continue
        try:
            cap, problem = load(item["path"], item["sha256"])
            if problem:
                rec["outcome"] = problem.split(":")[0]
                rec["error"] = problem
                continue
            inputs = {k: cap[k] for k in _SD3_READOUT_NAMES}
            described = {k: describe(v) for k, v in inputs.items()}
            rec.update(meta={k: cap.get(k) for k in ("phase", "forward_n", "rank", "mode", "request_key")},
                       inputs=described, output=describe(cap["output"]))
            rec["input_key"] = hashlib.sha256(json.dumps([[k, described[k]] for k in _SD3_READOUT_NAMES],
                                                         sort_keys=True).encode()).hexdigest()
            if item["kind"] == "rank_agreement":
                if not item.get("compare_path"):
                    rec["outcome"] = "compare_missing"
                    continue
                other, problem = load(item["compare_path"], item["compare_sha256"])
                if problem:
                    rec.update(outcome="compare_" + problem.split(":")[0], error=problem)
                    continue
                rec.update(outcome="compared", compare_to=item["compare_to"],
                           inputs_equal={k: bool(torch.equal(inputs[k], other[k])) for k in _SD3_READOUT_NAMES},
                           output_vs_compare=delta(cap["output"], other["output"]))
                continue
            x = inputs["hidden_states"]
            want = item.get("expect_latent_hw")
            if x.ndim != 4 or int(x.shape[1]) != in_channels or (want is not None and list(x.shape[2:]) != list(want)):
                rec.update(outcome="not_replayable_shape", expect_latent_hw=want)
                continue
            rows = int(x.shape[0])
            ref = cache.get(rec["input_key"])
            if ref is not None:
                rec["dedup_of"] = ref["unit"]
            else:
                values = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
                    batch = pipe.transformer(**values, return_dict=False)[0]
                    solo = None
                    if rows > 1:
                        solo = torch.cat([pipe.transformer(**{k: (v[i:i + 1] if v.ndim and v.shape[0] == rows else v)
                                                              for k, v in values.items()}, return_dict=False)[0]
                                          for i in range(rows)])
                ref = {"unit": item["id"], "batch": batch.detach().cpu(),
                       "solo": solo.detach().cpu() if solo is not None else None}
                nbytes = sum(t.numel() * t.element_size() for t in (ref["batch"], ref["solo"]) if t is not None)
                if saved + nbytes <= save_max:
                    path = os.path.join(out, "reference_" + rec["input_key"][:24] + ".pt")
                    torch.save({"batch": ref["batch"], "solo": ref["solo"]}, path)
                    saved += nbytes
                    ref.update(file=os.path.basename(path), file_sha256=file_sha(path))
                cache[rec["input_key"]] = ref
            native = cap["output"]
            rec.update(outcome="replayed", rows=rows, reference_file=ref.get("file"),
                       reference_file_sha256=ref.get("file_sha256"), reference_output=describe(ref["batch"]),
                       native_vs_reference_batch=delta(native, ref["batch"]),
                       native_vs_reference_solo=delta(native, ref["solo"]),
                       reference_batch_vs_solo=delta(ref["batch"], ref["solo"]))
        except Exception as exc:  # noqa: BLE001 - one item's failure is recorded, the rest continue
            rec.update(outcome="error", error=f"{type(exc).__name__}: {exc}"[:500])
        finally:
            if "outcome" in rec:
                rec["elapsed_s"] = time.monotonic() - t0
                slowest = max(slowest, rec["elapsed_s"])
                emit(rec)
    emit({"event": "end", "monotonic": time.monotonic(), "reference_bytes_saved": saved})
    log.close()
    cache.clear()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _sd3_readout_entry():
    import json as _json
    import os as _os
    with open(_os.environ["SD3_READOUT_MANIFEST"], encoding="utf-8") as fh:
        manifest = _json.load(fh)
    _sd3_readout(pipe, manifest, out, float(_os.environ["SD3_READOUT_DEADLINE"]))  # noqa: F821 - generator globals
