#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"The deterministic FP8 asset recipe for SD3-medium's DiT (recipe_v2.json), and its content-identity check."
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_RECIPE = HERE / "recipe_v2.json"
WEIGHTS = "diffusion_pytorch_model.safetensors"
FP8_MAX = 448.0
BLOCK = 128


def _log(msg: str) -> None:
    print(f"[sd3-fp8-recipe] {msg}", flush=True)


def sha256_file(path: Path, bufsize: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_recipe(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        r = json.load(f)
    for key in ("recipe_id", "source", "format", "quantized_modules"):
        if key not in r:
            raise SystemExit(f"recipe {path} lacks {key!r}")
    return r


NORMATIVE = (("recipe_id",), ("source", "sha256"), ("source", "file"), ("source", "n_tensors"),
             ("format", "quantization_config"), ("format", "weight_dtype"), ("format", "scale_tensor_suffix"),
             ("format", "scale_dtype"), ("format", "scale_arithmetic"), ("quantized_modules", "include_regex"),
             ("quantized_modules", "expected_count"))


def recipe_fingerprint(recipe: dict) -> str:
    """sha256 over the recipe's NORMATIVE fields only (what changes the bytes), not its prose or its measured
    `expected` block -- so a documentation edit does not orphan a measurement, and a semantic edit always does."""
    body = {}
    for path in NORMATIVE:
        v = recipe
        for k in path:
            v = v[k]
        body[".".join(path)] = v
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def select_quantized(keys, recipe: dict) -> list[str]:
    """Module names (key minus `.weight`) the recipe quantizes, sorted."""
    pats = [re.compile(p) for p in recipe["quantized_modules"]["include_regex"]]
    mods = sorted(k[: -len(".weight")] for k in keys
                  if k.endswith(".weight") and any(p.match(k[: -len(".weight")]) for p in pats))
    return mods


def modules_not_converted(keys, quantized: list[str]) -> list[str]:
    q = set(quantized)
    return sorted(k[: -len(".weight")] for k in keys if k.endswith(".weight") and k[: -len(".weight")] not in q)


def block_fp8(w, torch):
    'Block-128 E4M3 quantisation, for weights whose dims are multiples of 128 (asserted by the caller).'


    n, k = w.shape
    t = w.reshape(n // BLOCK, BLOCK, k // BLOCK, BLOCK)
    amax = t.abs().amax(dim=(1, 3))
    zero = int((amax == 0).sum().item())
    if zero:
        raise ValueError(f"{zero} all-zero block(s)")
    amax32 = amax.to(torch.float32)
    scale = amax32 / torch.full_like(amax32, FP8_MAX)
    q = t / scale[:, None, :, None]
    q = q.clamp(min=-FP8_MAX, max=FP8_MAX).reshape(n, k).to(torch.float8_e4m3fn)
    return q.contiguous(), scale.contiguous()


def tensor_sha256(t, torch) -> str:
    c = t.detach().contiguous().cpu()
    return hashlib.sha256(c.view(torch.uint8).numpy().tobytes()).hexdigest()


def content_identity(entries: list[dict]) -> str:
    lines = sorted(f"{e['name']}\t{e['dtype']}\t{list(e['shape'])}\t{e['sha256']}\n" for e in entries)
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def _versions() -> dict:
    out = {}
    for mod in ("torch", "safetensors", "numpy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception as e:
            out[mod] = f"unavailable: {type(e).__name__}"
    return out


def generate(args) -> int:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    recipe = load_recipe(Path(args.recipe))
    src_dir = Path(args.src)
    src = src_dir / WEIGHTS if src_dir.is_dir() else src_dir
    cfg_path = (src_dir if src_dir.is_dir() else src_dir.parent) / "config.json"
    if not src.is_file() or not cfg_path.is_file():
        _log(f"FATAL: need {WEIGHTS} and config.json under {src_dir}")
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    src_sha = sha256_file(src)
    want = recipe["source"]["sha256"]
    if src_sha != want:
        _log(f"source sha256 {src_sha} != the pinned {want}")
        if not args.allow_source_mismatch:
            _log("FATAL: refusing to build the asset from a checkpoint the recipe does not name (exit 3)")
            return 3
        _log("WARNING: --allow-source-mismatch; the manifest will NOT match the frozen expectation")
    t_hash = time.time() - t0

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        _log("FATAL: --device cuda but no CUDA device")
        return 2

    tensors = {}
    entries = []
    n_cross_checked = 0
    t1 = time.time()
    with safe_open(str(src), framework="pt", device="cpu") as f:
        keys = sorted(f.keys())
        if len(keys) != recipe["source"]["n_tensors"]:
            _log(f"FATAL: source holds {len(keys)} tensors, recipe expects {recipe['source']['n_tensors']}")
            return 5
        quantized = select_quantized(keys, recipe)
        exp = recipe["quantized_modules"]["expected_count"]
        if len(quantized) != exp:
            _log(f"FATAL: recipe selects {len(quantized)} modules, expected {exp}")
            return 5
        qset = set(quantized)
        for key in keys:
            w = f.get_tensor(key)
            mod = key[: -len(".weight")] if key.endswith(".weight") else None
            if mod in qset:
                if w.dim() != 2 or w.shape[0] % BLOCK or w.shape[1] % BLOCK:
                    _log(f"FATAL: {key} shape {tuple(w.shape)} is not a 2-D multiple of {BLOCK}")
                    return 5
                if w.dtype != torch.float16:
                    _log(f"FATAL: {key} is {w.dtype}, the recipe is defined on the float16 source")
                    return 5
                try:
                    q, s = block_fp8(w.to(device), torch)
                    q, s = q.cpu(), s.cpu()
                    if device != "cpu" and not args.no_cpu_cross_check:


                        qc, sc = block_fp8(w, torch)
                        if not (torch.equal(q.view(torch.uint8), qc.view(torch.uint8)) and torch.equal(s, sc)):
                            nd = int((q.view(torch.uint8) != qc.view(torch.uint8)).sum()) + int((s != sc).sum())
                            _log(f"FATAL: {key}: the {device} result differs from the CPU result in {nd} element(s) (exit 6)")
                            return 6
                        n_cross_checked += 1
                except ValueError as e:
                    _log(f"FATAL: {key}: {e}")
                    return 5
                tensors[key] = q
                tensors[mod + ".weight_scale_inv"] = s
            else:
                tensors[key] = w.contiguous()
    t_quant = time.time() - t1

    for name in sorted(tensors):
        t = tensors[name]
        entries.append({"name": name, "dtype": str(t.dtype).replace("torch.", ""), "shape": list(t.shape),
                        "sha256": tensor_sha256(t, torch)})

    t2 = time.time()
    save_file(tensors, str(out / WEIGHTS), metadata={"format": "pt"})
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    qc = dict(recipe["format"]["quantization_config"])
    qc["modules_to_not_convert"] = modules_not_converted(keys, quantized)
    cfg["quantization_config"] = qc
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    t_write = time.time() - t2

    fp8_bytes = sum(t.numel() for n, t in tensors.items() if t.dtype == torch.float8_e4m3fn)
    scale_bytes = sum(t.numel() * 4 for n, t in tensors.items() if n.endswith(".weight_scale_inv"))
    other_bytes = sum(t.numel() * t.element_size() for n, t in tensors.items()
                      if t.dtype != torch.float8_e4m3fn and not n.endswith(".weight_scale_inv"))
    manifest = {
        "recipe_id": recipe["recipe_id"],
        "recipe_fingerprint": recipe_fingerprint(recipe),
        "source_sha256": src_sha,
        "source_matches_recipe": src_sha == want,
        "device": device,
        "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "cpu_cross_checked_modules": n_cross_checked,
        "versions": _versions(),
        "n_tensors": len(entries),
        "n_quantized_modules": len(quantized),
        "quantized_modules": quantized,
        "bytes": {"fp8_weights": fp8_bytes, "scales": scale_bytes, "other": other_bytes,
                  "total": fp8_bytes + scale_bytes + other_bytes},
        "tensor_set_sha256": content_identity(entries),
        "file_sha256": {WEIGHTS: sha256_file(out / WEIGHTS), "config.json": sha256_file(out / "config.json")},
        "file_size": {WEIGHTS: (out / WEIGHTS).stat().st_size},
        "seconds": {"source_hash": round(t_hash, 2), "quantize": round(t_quant, 2), "write": round(t_write, 2)},
        "tensors": entries,
    }
    with open(out / "fp8_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
    _log(f"wrote {out}: {len(quantized)} quantized modules, {len(entries)} tensors, "
         f"{manifest['bytes']['total'] / 1e9:.3f} GB of tensors; tensor_set_sha256 {manifest['tensor_set_sha256']}")
    return 0


def asset_entries(asset: Path) -> list[dict]:
    import torch
    from safetensors import safe_open

    entries = []
    with safe_open(str(asset / WEIGHTS), framework="pt", device="cpu") as f:
        for name in sorted(f.keys()):
            t = f.get_tensor(name)
            entries.append({"name": name, "dtype": str(t.dtype).replace("torch.", ""), "shape": list(t.shape),
                            "sha256": tensor_sha256(t, torch)})
    return entries


def verify(args) -> int:
    asset = Path(args.asset)
    if args.expect:
        with open(args.expect, encoding="utf-8") as f:
            exp = json.load(f)
        want = exp["tensor_set_sha256"]
    else:
        recipe = load_recipe(Path(args.recipe))
        if not recipe.get("expected"):
            _log("FATAL: the recipe carries no measured `expected` block yet; pass --expect <manifest>")
            return 2
        want = recipe["expected"]["tensor_set_sha256"]
    entries = asset_entries(asset)
    have = content_identity(entries)
    if have != want:
        _log(f"MISMATCH: asset content {have} != expected {want}")
        if args.expect:
            by = {e["name"]: e for e in exp.get("tensors", [])}
            diff = [e["name"] for e in entries if by.get(e["name"], {}).get("sha256") != e["sha256"]]
            missing = sorted(set(by) - {e["name"] for e in entries})
            _log(f"  {len(diff)} tensor(s) differ (first: {diff[:5]}), {len(missing)} missing (first: {missing[:5]})")
        return 4
    _log(f"OK: {asset} content identity {have} ({len(entries)} tensors)")
    return 0


def hash_cmd(args) -> int:
    entries = asset_entries(Path(args.asset))
    print(json.dumps({"tensor_set_sha256": content_identity(entries), "n_tensors": len(entries)}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--src", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    g.add_argument("--recipe", default=str(DEFAULT_RECIPE))
    g.add_argument("--allow-source-mismatch", action="store_true")
    g.add_argument("--no-cpu-cross-check", action="store_true",
                   help="skip re-deriving every tensor on CPU after a GPU build (the default refuses a mismatch)")
    v = sub.add_parser("verify")
    v.add_argument("--asset", required=True)
    v.add_argument("--recipe", default=str(DEFAULT_RECIPE))
    v.add_argument("--expect", default=None)
    h = sub.add_parser("hash")
    h.add_argument("--asset", required=True)
    args = ap.parse_args(argv)
    return {"generate": generate, "verify": verify, "hash": hash_cmd}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
