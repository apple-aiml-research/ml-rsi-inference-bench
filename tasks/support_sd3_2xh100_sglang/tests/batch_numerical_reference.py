#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Appended inside the isolated diffusers reference process for readouts only.

Uses the same already loaded `pipe`, `spec`, `out` and torch. No gate or bar is
defined here. Common-input DiT replay separates encoder/input divergence from
DiT batch-shape rounding, without running candidate code in the reference.
"""
from pathlib import Path as _Path
import hashlib as _hashlib
import time as _time


def _pixel_delta(a, b):
    import numpy as np
    from skimage.metrics import structural_similarity

    a, b = np.asarray(a), np.asarray(b)
    if (a.shape != b.shape or a.ndim != 3 or a.shape[-1] != 3 or min(a.shape[:2]) < 7
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or a.min() < 0 or b.min() < 0 or a.max() > 255 or b.max() > 255):
        return {'valid': False, 'a_shape': list(a.shape), 'b_shape': list(b.shape)}
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return {'valid': True, 'bit_identical': bool(np.array_equal(a, b)),
            'psnr_db': 100.0 if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse)),
            'ssim': float(np.mean([structural_similarity(a[:, :, c], b[:, :, c], data_range=255)
                                   for c in range(3)]))}


def _mixed_batch_images(pipeline, fixtures, device='cuda', negative_prompt=' ', cfg_layout='concat'):
    """Independent diffusers batch under the exact existing mixed recipe, all output rows retained.

    Fixtures are already expanded with per-request seed+i. A list of independent
    generators is required: one shared generator would give the wrong row noises.
    This does not call the native engine or consume its output/conditioning.
    """
    import numpy as np
    import hashlib

    if not fixtures or len({(f['cfg'], f['steps'], f['h'], f['w']) for f in fixtures}) != 1:
        raise ValueError('A reference group must have one CFG, step count and size')
    f = fixtures[0]
    pe, npe, ppe, nppe = pipeline.encode_prompt(
        prompt=[x['prompt'] for x in fixtures], prompt_2=None, prompt_3=None,
        negative_prompt=negative_prompt if f['cfg'] > 1 else None,
        do_classifier_free_guidance=f['cfg'] > 1, device=device, max_sequence_length=256)
    generators = [torch.Generator(device=device).manual_seed(x['seed']) for x in fixtures]
    calls, first_inputs = [], []

    def fingerprint_rows(value):
        if value is None:
            return None
        value = value.detach().cpu().contiguous()
        if value.ndim == 0:
            value = value.reshape(1)
        return [dict(shape=list(row.shape), dtype=str(row.dtype),
                     sha256=hashlib.sha256(row.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
                for row in value]

    def witness(_module, args, kwargs):
        hidden = kwargs.get('hidden_states', args[0] if args else None)
        if hidden is not None:
            calls.append(int(hidden.shape[0]))
            first_calls = 2 if f['cfg'] > 1 and cfg_layout.startswith('split_') else 1
            if len(first_inputs) < first_calls:
                first_inputs.append({key: fingerprint_rows(kwargs.get(key)) for key in
                                     ('hidden_states', 'encoder_hidden_states', 'pooled_projections', 'timestep')})

    hook = pipeline.transformer.register_forward_pre_hook(witness, with_kwargs=True)
    try:
        with reference_cfg_layout(pipeline, f['cfg'], cfg_layout), torch.no_grad(), \
                torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            latents = pipeline(prompt_embeds=pe, negative_prompt_embeds=npe,
                               pooled_prompt_embeds=ppe, negative_pooled_prompt_embeds=nppe,
                               num_inference_steps=f['steps'], guidance_scale=f['cfg'],
                               height=f['h'], width=f['w'], generator=generators, output_type='latent').images
        latents = latents.float() / pipeline.vae.config.scaling_factor + pipeline.vae.config.shift_factor
        with torch.no_grad():
            decoded = pipeline.vae.decode(latents, return_dict=False)[0]
        images = [np.asarray(x) for x in pipeline.image_processor.postprocess(decoded, output_type='pil')]
    finally:
        hook.remove()
    if len(images) != len(fixtures):
        raise ValueError('Independent reference returned an incomplete batch')
    return images, {'cfg': f['cfg'], 'seeds': [x['seed'] for x in fixtures], 'dit_rows': calls,
                    'cfg_layout': 'diffusers concatenates negative rows before positive rows',
                    'execution_layout': cfg_layout, 'first_step_inputs': first_inputs,
                    'prompt_dtype': str(pe.dtype), 'pooled_dtype': str(ppe.dtype)}


def _numeric_delta(a, b):
    a, b = a.detach().float(), b.detach().float()
    if a.shape != b.shape or not a.numel() or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        return {"valid": False, "a_shape": list(a.shape), "b_shape": list(b.shape)}
    d = a - b
    return {"valid": True, "bit_identical": bool(torch.equal(a, b)),
            "max_abs": float(d.abs().max()),
            "relative_l2": float(d.norm() / b.norm().clamp_min(1e-30))}


def _batch_reference_readout():
    import numpy as np

    start = _time.monotonic()
    root = _Path(os.environ["SD3_BATCH_CAPTURE_DIR"])
    files = sorted(root.glob("rank*_n*.pt"))
    report = {"scored": False, "reference": "independent diffusers, same frozen mixed recipe",
              "files_present": len(files), "common_input_dit": [], "encoder_batch_vs_solo": [],
              "final_image_batch_vs_solo": [], "timing_s": {},
              "limits": "Independent numerical and image measurements; this report does not assign gate credit."}


    selected = [p for p in files if p.name.startswith("rank0_")]
    if not selected:
        report["error"] = "No rank0 common-input snapshots; diagnostic unavailable"
        return report
    if len(selected) > 32:
        report["error"] = "Unexpected snapshot count exceeds frozen bound"
        return report
    with torch.no_grad():
        for path in selected:
            try:
                cap = torch.load(path, map_location="cpu", weights_only=True)
                values = {k: cap[k].to("cuda") for k in
                          ("hidden_states", "encoder_hidden_states", "pooled_projections", "timestep")}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    merged = pipe.transformer(**values, return_dict=False)[0]
                    if values['hidden_states'].shape[0] == 1:

                        split = merged
                    else:
                        single = []
                        for i in range(values["hidden_states"].shape[0]):
                            row = {k: v[i:i+1] if v.ndim and v.shape[0] == values["hidden_states"].shape[0] else v
                                   for k, v in values.items()}
                            single.append(pipe.transformer(**row, return_dict=False)[0])
                        split = torch.cat(single)
                report["common_input_dit"].append({
                    "file": str(path), "sha256": _hashlib.sha256(path.read_bytes()).hexdigest(),
                    "phase": cap["phase"], "n": cap["forward_n"],
                    "rows": values["hidden_states"].shape[0],
                    "native_vs_reference_batch": _numeric_delta(cap["output"].to("cuda"), merged),
                    "native_vs_reference_solo": _numeric_delta(cap["output"].to("cuda"), split),
                    "reference_batch_vs_solo": _numeric_delta(merged, split)})
            except Exception as exc:
                report["common_input_dit"].append({"file": str(path), "error": f"{type(exc).__name__}: {exc}"[:500]})
        report['timing_s']['common_input_dit'] = _time.monotonic() - start
        for cfg in (1, 7):
            group_start = _time.monotonic()
            fixtures = [f for f in spec["fixtures"] if f["cfg"] == cfg]
            prompts = [f["prompt"] for f in fixtures]
            kwargs = dict(prompt_2=None, prompt_3=None, negative_prompt=" " if cfg > 1 else None,
                          do_classifier_free_guidance=cfg > 1, device="cuda", max_sequence_length=256)
            batch = pipe.encode_prompt(prompt=prompts, **kwargs)
            singles = [pipe.encode_prompt(prompt=p, **kwargs) for p in prompts]
            for k, name in enumerate(("positive", "negative", "pooled_positive", "pooled_negative")):
                if batch[k] is None:
                    continue
                serial = torch.cat([s[k] for s in singles])
                report["encoder_batch_vs_solo"].append({"cfg": cfg, "field": name,
                                                        "metric": _numeric_delta(batch[k], serial)})


            heldout = [{**f, 'id': f'heldout_cfg{cfg}_{i}', 'seed': seed}
                       for i, (f, seed) in enumerate(zip(fixtures[:2], (314159, 271828)))]
            for split, group in [('original', fixtures), ('heldout_seed', heldout)]:
                if split == 'heldout_seed':
                    for fixture in group:
                        solo_images, _ = _mixed_batch_images(pipe, [fixture])
                        np.save(_Path(out) / f"{fixture['id']}.npy", solo_images[0])
                images, witness = _mixed_batch_images(pipe, group)
                rows = []
                for i, (fixture, img) in enumerate(zip(group, images)):
                    path = _Path(out) / f"independent_batch_{fixture['id']}.npy"
                    np.save(path, img)
                    anchor = _Path(out) / f"{fixture['id']}.npy"


                    pairs = {f['id']: _pixel_delta(img, np.load(_Path(out) / f"{f['id']}.npy")) for f in group}
                    rows.append({'fixture': fixture['id'], 'row': i, 'seed': fixture['seed'],
                                 'prompt': fixture['prompt'], 'image': str(path),
                                 'sha256': _hashlib.sha256(path.read_bytes()).hexdigest(),
                                 'solo_sha256': _hashlib.sha256(anchor.read_bytes()).hexdigest(),
                                 'own_solo': pairs[fixture['id']], 'all_solo': pairs})
                report['final_image_batch_vs_solo'].append({'cfg': cfg, 'split': split,
                                                           'witness': witness, 'rows': rows})
            report['timing_s'][f'encoder_and_images_cfg{cfg}'] = _time.monotonic() - group_start
    report['timing_s']['total'] = _time.monotonic() - start
    return report


try:
    _batch_numeric_report = _batch_reference_readout()
except Exception as _exc:
    _batch_numeric_report = {"scored": False, "error": f"{type(_exc).__name__}: {_exc}"[:600]}
json.dump(_batch_numeric_report, open(os.path.join(out, "batch_numerical_reference.json"), "w"), indent=2)
