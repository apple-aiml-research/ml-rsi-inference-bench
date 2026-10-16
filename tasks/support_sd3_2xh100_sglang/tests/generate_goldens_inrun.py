#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
'Generate the similarity reference IN-RUN, from HF diffusers.'


from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


SPEC_NAME = "sd3_golden.json"


REF_SETS = {"bf16": ("bfloat16", "", ""), "fp32": ("float32", "fp32", "golden_fp32"), "mixed": ("mixed", "mixed", "golden_mixed")}


DEFAULT_REF_SETS = "bf16,fp32,mixed"


def log(*a):
    print("[golden-inrun]", *a, flush=True)


def _spec(tests_dir: Path) -> dict:
    return json.loads((tests_dir / SPEC_NAME).read_text())


def _gen_script() -> str:
    'Generate the reference with DIFFUSERS, in its own process.'


    return r'''
import json, os, sys
import numpy as np, torch

# The candidate repo is deliberately NOT on sys.path (see main()): a candidate can create
# repo/python/diffusers/__init__.py and would otherwise get to define the reference it is scored
# against. Verify that what we imported is the real installed package, outside the workspace.
import diffusers
_ws = os.environ.get("INFB_WORKSPACE", "/app")
_loc = os.path.dirname(os.path.abspath(diffusers.__file__))
if _loc.startswith(os.path.abspath(_ws)):
    print(f"[golden-inrun] FATAL: diffusers resolved INSIDE the workspace ({_loc}); a candidate "
          f"cannot be allowed to supply the reference implementation", flush=True)
    raise SystemExit(9)
print(f"[golden-inrun] diffusers {diffusers.__version__} from {_loc}", flush=True)

# Record the package versions used to produce the reference. In particular, an absent
# `accelerate` can make transformers fall back to `low_cpu_mem_usage=False`, loading in
# the default dtype and casting afterwards rather than loading directly in bf16.
import importlib.metadata as _md
_vers = []
for _p in ("torch", "transformers", "accelerate", "kernels", "safetensors", "numpy", "pillow"):
    try:
        _vers.append(f"{_p}={_md.version(_p)}")
    except Exception:  # noqa: BLE001 - an absent package is exactly what this line has to report
        _vers.append(f"{_p}=ABSENT")
print("[golden-inrun] env: " + "  ".join(_vers), flush=True)

from diffusers import StableDiffusion3Pipeline
spec = json.load(open(os.environ["SD3_SPEC"]))
out = os.environ["SD3_OUT"]
model = os.environ["SD3_MODEL"]
os.makedirs(out, exist_ok=True)

# REFERENCE PRECISION. The bf16 profile uses bfloat16; fp32 loads every component at fp32 and turns TF32 off for
# both matmuls and cuDNN convolutions (torch leaves cuDNN's on by default), so no op silently computes below fp32.
_dt_name = os.environ.get("SD3_REF_DTYPE", "bfloat16")
_mixed = _dt_name == "mixed"
_dt = {"bfloat16": torch.bfloat16, "float32": torch.float32, "mixed": torch.float32}[_dt_name]
if _dt is torch.float32 and not _mixed:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
pipe = StableDiffusion3Pipeline.from_pretrained(model, torch_dtype=_dt)
if _mixed:
    # the mixed profile: CLIP-L / CLIP-G fp16, T5 fp32 (as loaded), DiT bf16, VAE fp32 (as loaded)
    pipe.text_encoder.to(torch.float16)
    pipe.text_encoder_2.to(torch.float16)
    pipe.transformer.to(torch.bfloat16)
pipe = pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)


def _run_mixed(fx, g, neg):
    """The mixed convention (tests/sd3_precision_profiles.py "mixed"): the prompts are encoded OUTSIDE autocast (CLIP
    fp16, T5 fp32; their cat promotes to fp32), the latents are drawn at that fp32, the denoising loop runs under
    torch.autocast(bfloat16) on the bf16 transformer, and the fp32 VAE decodes outside autocast."""
    do_cfg = fx["cfg"] > 1.0
    pe, npe, ppe, nppe = pipe.encode_prompt(prompt=fx["prompt"], prompt_2=None, prompt_3=None,
                                            negative_prompt=neg if do_cfg else None, do_classifier_free_guidance=do_cfg,
                                            device="cuda", max_sequence_length=256)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lat = pipe(prompt_embeds=pe, negative_prompt_embeds=npe, pooled_prompt_embeds=ppe,
                   negative_pooled_prompt_embeds=nppe, num_inference_steps=fx["steps"], guidance_scale=fx["cfg"],
                   height=fx["h"], width=fx["w"], generator=g, output_type="latent").images
    lat = (lat.to(torch.float32) / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    with torch.no_grad():
        image = pipe.vae.decode(lat, return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")[0], {
        "prompt_embeds": str(pe.dtype), "pooled": str(ppe.dtype)}


def _noise_fingerprint(fx):
    """The initial noise a fixture's seed gives at this set's latent dtype, drawn from a SEPARATE identically seeded
    generator the way diffusers' prepare_latents draws it (the same seed is not the same noise across dtypes)."""
    import hashlib
    from diffusers.utils.torch_utils import randn_tensor
    lat_dtype = torch.float32 if (_mixed or _dt is torch.float32) else torch.bfloat16
    shape = (1, pipe.transformer.config.in_channels, fx["h"] // pipe.vae_scale_factor, fx["w"] // pipe.vae_scale_factor)
    n = randn_tensor(shape, generator=torch.Generator(device="cuda").manual_seed(fx["seed"]), device=torch.device("cuda"),
                     dtype=lat_dtype)
    return {"dtype": str(lat_dtype), "shape": list(shape),
            "sha256": hashlib.sha256(n.detach().float().cpu().numpy().tobytes()).hexdigest()}
# What every component really holds, recorded with the set: a module kept at another precision (transformers'
# _keep_in_fp32_modules, for one) would otherwise hide inside a set named for one dtype.
_components = {}
for _name in ("text_encoder", "text_encoder_2", "text_encoder_3", "transformer", "vae"):
    _m = getattr(pipe, _name, None)
    _components[_name] = sorted({str(q.dtype) for q in _m.parameters()}) if _m is not None else ["absent"]
_precision = {"set_dtype": _dt_name, "components": _components,
              "tf32": {"matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                       "cudnn": bool(torch.backends.cudnn.allow_tf32)},
              "convention": ("mixed: encode outside autocast, denoise under bf16 autocast, fp32 VAE decode outside it"
                             if _mixed else "every component at the set dtype, no autocast")}
print("[golden-inrun] precision: " + json.dumps(_precision, sort_keys=True), flush=True)
fixtures = []
for fx in spec["fixtures"]:
    g = torch.Generator(device="cuda").manual_seed(fx["seed"])
    extra = {}
    if not _mixed:
        o = pipe(prompt=fx["prompt"], negative_prompt=os.environ.get("SD3_NEG", " "),
                 num_inference_steps=fx["steps"], guidance_scale=fx["cfg"],
                 height=fx["h"], width=fx["w"], generator=g, output_type="pil")
        img = np.asarray(o.images[0])
    else:
        pil, extra = _run_mixed(fx, g, os.environ.get("SD3_NEG", " "))
        img = np.asarray(pil)
    npy = f"{fx['id']}.npy"
    np.save(os.path.join(out, npy), img)
    try:
        extra["initial_noise"] = _noise_fingerprint(fx)
    except Exception as exc:  # noqa: BLE001 - a fingerprint failure is recorded, the reference image stands
        extra["initial_noise"] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
    fixtures.append({**fx, "golden_npy": npy, "shape": list(img.shape), **({"reference": extra} if extra else {})})
    print(f"[golden-inrun]   {fx['id']}: {img.shape} {img.dtype}", flush=True)
json.dump({**{k: v for k, v in spec.items() if k != "fixtures"}, "fixtures": fixtures, "precision": _precision,
           "verifier_run_id": os.environ.get("SD3_VERIFIER_RUN_ID")},
          open(os.path.join(out, "sd3_golden.json"), "w"), indent=2)
print("[golden-inrun] wrote " + os.path.join(out, "sd3_golden.json"), flush=True)
'''


def main() -> int:
    tests_dir = Path(os.environ.get("TESTS_DIR", Path(__file__).parent)).resolve()
    out = Path(os.environ.get("SD3_GOLDEN_OUT", "")).resolve()
    python_bin = os.environ.get("SD3_PYTHON", sys.executable)
    model = os.environ.get("SD3_MODEL", "")
    repo = Path(os.environ.get("SD3_CANDIDATE_REPO", "")).resolve()

    if not (tests_dir / SPEC_NAME).is_file():
        log(f"FATAL: fixture spec not found: {tests_dir / SPEC_NAME}")
        return 2
    if not model:


        log("FATAL: SD3_MODEL unset — refusing to generate the reference off an unknown checkpoint")
        return 2

    spec = _spec(tests_dir)
    out.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)


    keep = [q for q in (env.get("PYTHONPATH", "") or "").split(":")
            if q and (not repo.exists() or not q.startswith(str(repo)))]
    env["PYTHONPATH"] = ":".join(keep)
    env["SD3_SPEC"] = str(tests_dir / SPEC_NAME)
    env["SD3_MODEL"] = model


    outs = {}
    wanted = [x.strip() for x in os.environ.get("SD3_REF_SETS", DEFAULT_REF_SETS).split(",") if x.strip()]
    unknown = [x for x in wanted if x not in REF_SETS]
    if unknown:
        log(f"FATAL: SD3_REF_SETS names unknown set(s) {unknown}; known: {sorted(REF_SETS)}")
        return 2
    bootstrap = {x.strip() for x in os.environ.get("SD3_GOLDEN_BOOTSTRAP", "").split(",") if x.strip()}
    def generate(name, gpu=None):
        dtype, sub, _anchors = REF_SETS[name]
        o = out / sub if sub else out
        o.mkdir(parents=True, exist_ok=True)
        log(f"generating the diffusers reference, set {name} (every component {dtype}; "
            f"{len(spec['fixtures'])} fixtures); PYTHONPATH stripped of {repo}")
        child_env = {**env, "SD3_OUT": str(o), "SD3_REF_DTYPE": dtype}
        if gpu is not None:
            child_env['CUDA_VISIBLE_DEVICES'] = gpu
        started = time.monotonic()
        r = subprocess.run([python_bin, "-c", _gen_script()], env=child_env,
                           capture_output=True, text=True)
        return name, o, r, {'set': name, 'gpu': gpu, 'elapsed_s': time.monotonic() - started,
                            'returncode': r.returncode}


    if os.environ.get('SD3_VERIFIER_SCHEDULE') == 'two_queues':
        visible = [v.strip() for v in env.get('CUDA_VISIBLE_DEVICES', '0,1').split(',') if v.strip()]
        if len(visible) < 2 or len(set(visible[:2])) != 2:
            log('FATAL: two_queues reference generation requires two distinct visible GPUs')
            return 2
        queues = [[n for n in ('bf16', 'mixed') if n in wanted], [n for n in ('fp32',) if n in wanted]]
        def run_queue(index):
            return [generate(n, visible[index]) for n in queues[index]]
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = [r for rows in pool.map(run_queue, range(2)) for r in rows]
    else:
        results = [generate(n) for n in REF_SETS if n in wanted]
    (out / 'reference_generation_timing.json').write_text(json.dumps([r[3] for r in results], indent=2))
    failed = False
    for name, o, r, _timing in results:
        sys.stdout.write(r.stdout[-6000:])
        if r.returncode != 0 or not (o / SPEC_NAME).is_file():


            log(f"FATAL: in-run reference generation failed (set {name})")
            sys.stderr.write(r.stdout[-4000:] + r.stderr[-4000:])
            failed = True
            continue
        n = len(json.loads((o / SPEC_NAME).read_text())["fixtures"])
        log(f"in-run diffusers reference ready: set {name}, {n} fixtures in {o}")
        outs[name] = o
    if failed:
        return 4
    for name, (_dtype, _sub, anchors) in REF_SETS.items():
        if name not in outs:
            continue
        adir = tests_dir / anchors if anchors else tests_dir
        if name in bootstrap and not adir.is_dir():


            log(f"BOOTSTRAP: set {name} has no committed anchors in {adir}; it is UNVERIFIED until they are committed")
            (outs[name] / "UNVERIFIED_BOOTSTRAP").write_text(f"no committed anchors in {adir}\n")
            continue
        rc = _agree_with_committed(outs[name], adir, label=name)
        if rc:
            return rc
    return 0


def missing_committed_goldens(fixtures: list, tests_dir: Path) -> list:
    'Ids of scored fixtures whose committed, externally verified `.npy` is absent. Pure; no numpy.'


    return [str(fx.get("id")) for fx in fixtures
            if not fx.get("golden_npy") or not (Path(tests_dir) / str(fx["golden_npy"])).is_file()]


def _agree_with_committed(out: Path, tests_dir: Path, label: str = "bf16") -> int:
    """Cross-check the freshly generated reference against the committed, externally verified one.

    Independently produced references must agree within the configured PSNR floor. This
    detects reference drift before candidate scoring while allowing bounded numerical differences.

    EVERY scored fixture must have its committed anchor: one missing `.npy` is FATAL (exit 5, the trial is
    invalid), not a fixture to skip. A skipped fixture is scored against a reference nothing verified.

    Called once per reference set (`label`), with that set's committed anchors: tests/ for bf16,
    tests/golden_fp32/ for fp32, and tests/golden_mixed/ for mixed. Any set failing its check fails the run --
    the verifier must never score against an unverified set.
    """
    import numpy as np

    floor = float(os.environ.get("SD3_GOLDEN_AGREE_MIN_PSNR", "40.0"))
    spec = json.loads((out / SPEC_NAME).read_text())
    fixtures = spec.get("fixtures") or []
    missing = missing_committed_goldens(fixtures, tests_dir)
    if not fixtures or missing:
        log(f"FATAL: set {label}: {len(missing)} of {len(fixtures)} scored fixture(s) have no committed verified golden "
            f"in {tests_dir}: {missing}. Every fixture the gate scores must be checked against its "
            f"committed tests/<id>.npy; refusing to score against an unverified reference.")
        return 5
    worst, worst_id, checked, exact = 1e9, None, 0, 0
    per_fixture = []
    for fx in fixtures:
        cm = tests_dir / fx["golden_npy"]
        a = np.load(out / fx["golden_npy"]).astype(np.float64)
        b = np.load(cm).astype(np.float64)
        if a.shape != b.shape:
            log(f"FATAL: set {label}: shape mismatch vs committed golden for {fx['id']}: {a.shape} != {b.shape}")
            return 5
        mse = float(np.mean((a - b) ** 2))
        p = 100.0 if mse <= 1e-9 else 20 * np.log10(255.0) - 10 * np.log10(mse)
        checked += 1
        exact += mse <= 1e-9
        per_fixture.append(
            (fx["id"], p, float(np.mean(np.abs(a - b))), float(np.max(np.abs(a - b))))
        )
        if p < worst:
            worst, worst_id = p, fx["id"]


    for fid, p, mad, mx in sorted(per_fixture, key=lambda r: r[1]):
        log(f"  [{label}] vs committed {fid:<14} psnr {p:7.3f} dB   mean|d| {mad:6.3f}   max|d| {mx:5.0f}")
    log(f"set {label}: agreement with the committed verified golden: {exact}/{checked} bit-identical, "
        f"worst psnr {worst:.3f} ({worst_id}), floor {floor}")
    if worst < floor:
        log(f"FATAL: set {label}: the in-run reference disagrees with the committed verified golden "
            f"({worst:.3f} dB < {floor} on {worst_id}). One of the two is wrong; refusing to score "
            f"against an unverified reference. If the committed golden is the stale one, regenerate "
            f"the independent reference with the configured checkpoint and generation settings.")
        return 5
    return 0


if __name__ == "__main__":
    sys.exit(main())
