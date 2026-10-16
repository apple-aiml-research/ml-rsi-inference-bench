#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Generate the synthetic SD3 LoRA adapters of lora_recipe_v3.json, deterministically, with numpy only.

WHAT IT WRITES under --out:

    A/pytorch_lora_weights.safetensors, B/pytorch_lora_weights.safetensors
        diffusers' SD3 transformer-LoRA layout: transformer.<module>.lora_A.weight [r, in] and
        transformer.<module>.lora_B.weight [out, r] in BF16, with `lora_adapter_metadata` carrying
        transformer.r / transformer.lora_alpha / transformer.target_modules
    lora_manifest.json
        per adapter: file sha256, content identity, r, lora_alpha, targets, and per module the achieved
        RMS of (lora_alpha / r) * B @ A next to the recipe's target

HOW THE VALUES ARE MADE (lora_recipe_v3.json "values"): a splitmix64 stream keyed by sha256 of
(rng_namespace, adapter, module, matrix, seed) gives integers in [-127, 127]; A = v * 2^-a_exp and B = v * 2^-b_exp with
b_exp chosen per module so the RMS of the scaled delta lands within sqrt(2) of 2^delta_rms_log2. Every entry is a
7-bit integer times a power of two, so BF16 stores it exactly and the file means the same numbers to every reader.
Nothing is sampled from numpy's Generator (whose streams may change between numpy versions): the integers are
pure uint64 arithmetic.

USAGE
    sd3_lora_synth.py generate --out <dir> [--recipe lora_recipe_v3.json] [--shapes sd3m_dit_linear_shapes.json]
    sd3_lora_synth.py verify   --out <dir> [--recipe ...]     # regenerate in memory, compare with the files
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import safetensors_np as stn  # noqa: E402

DEFAULT_RECIPE = HERE / "lora_recipe_v3.json"
DEFAULT_SHAPES = HERE / "sd3m_dit_linear_shapes.json"
WEIGHT_NAME = "pytorch_lora_weights.safetensors"
PREFIX = "transformer."
METADATA_KEY = "lora_adapter_metadata"

_M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def _key64(*parts) -> np.uint64:
    return np.uint64(int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little"))


def splitmix64(key: np.uint64, n: int) -> np.ndarray:
    """n successive splitmix64 outputs of the stream whose state starts at `key` (uint64 wraparound arithmetic)."""
    with np.errstate(over="ignore"):
        idx = np.arange(1, n + 1, dtype=np.uint64)
        z = key + idx * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return z


def int_matrix(key: np.uint64, shape) -> np.ndarray:
    n = int(np.prod(shape))
    u = splitmix64(key, n)
    v = ((u >> np.uint64(32)) % np.uint64(255)).astype(np.int64) - 127
    return v.reshape(shape)


def resolve_targets(rules, shapes: dict) -> list[str]:
    mods = []
    for rule in rules:
        lo, hi = rule["blocks"]
        for i in range(lo, hi + 1):
            for kind in rule["kinds"]:
                name = f"transformer_blocks.{i}.{kind}"
                if name in shapes:
                    mods.append(name)
                elif rule.get("absent") == "skip":
                    continue
                else:
                    raise ValueError(f"target {name} is not a module of the checkpoint (rule absent={rule.get('absent')})")
    if len(set(mods)) != len(mods):
        raise ValueError("a module is targeted twice")
    return sorted(mods)


def pattern_value(pattern: dict, mod: str):
    """peft's rank_pattern / alpha_pattern lookup (lora_reference._pattern_value): the first key K, in order, with
    re.match(rf"(.*\\.)?({K})$", mod)."""
    import re
    for key, val in (pattern or {}).items():
        if re.match(rf"(.*\.)?({key})$", mod):
            return val
    return None


def make_adapter(recipe: dict, name: str, shapes: dict):
    """-> (tensors {key: np.float32}, metadata {str: str}, stats {module: {...}}).

    v4 fields of an adapter spec (all optional; a spec without them makes the v3 bytes exactly):
      use_rslora     the delta is scaled by lora_alpha / sqrt(r) (peft use_rslora), and B is sized for that scale
      alpha_pattern  {module-suffix pattern: alpha}, peft's per-module alpha override, and B sized for each module's alpha
      variant        "dora": a DoRA file (a lora_magnitude_vector per module, use_dora in the metadata);
                     "lora_bias": a LoRA with a bias on B (a lora_B.bias per module, lora_bias in the metadata)
                     -- adapters the engine must refuse, not serve as plain LoRA"""
    spec = recipe["adapters"][name]
    r, alpha, a_exp = int(spec["r"]), float(spec["lora_alpha"]), int(spec["a_exp"])
    use_rslora = bool(spec.get("use_rslora", False))
    alpha_pattern = dict(spec.get("alpha_pattern") or {})
    variant = spec.get("variant")
    if variant not in (None, "dora", "lora_bias"):
        raise ValueError(f"{name}: unknown variant {variant!r}")
    target_rms = 2.0 ** int(spec["delta_rms_log2"])
    mods = resolve_targets(spec["targets"], shapes)
    tensors, stats = {}, {}


    ns = recipe.get("rng_namespace", recipe["recipe_id"])
    for m in mods:
        out_f, in_f = shapes[m]
        ai = int_matrix(_key64(ns, name, m, "A", spec["seed"]), (r, in_f))
        bi = int_matrix(_key64(ns, name, m, "B", spec["seed"]), (out_f, r))
        if not ai.any() or not bi.any():
            raise ValueError(f"{name}:{m}: an all-zero factor")
        a = ai.astype(np.float64) * 2.0 ** -a_exp
        b0 = bi.astype(np.float64)


        fro2 = float(np.trace((b0.T @ b0) @ (a @ a.T)))
        pa = pattern_value(alpha_pattern, m)
        a_mod = float(pa) if pa is not None else alpha
        scaling = a_mod / (float(np.sqrt(r)) if use_rslora else r)
        rms_unit = scaling * float(np.sqrt(fro2 / (out_f * in_f)))
        b_exp = int(np.round(np.log2(rms_unit / target_rms)))
        b = bi.astype(np.float64) * 2.0 ** -b_exp
        rms = rms_unit * 2.0 ** -b_exp
        a32, b32 = a.astype(np.float32), b.astype(np.float32)
        if not (stn.is_bf16_exact(a32) and stn.is_bf16_exact(b32)):
            raise ValueError(f"{name}:{m}: values not exactly representable in bf16")
        tensors[f"{PREFIX}{m}.lora_A.weight"] = a32
        tensors[f"{PREFIX}{m}.lora_B.weight"] = b32
        if variant == "dora":
            mi = int_matrix(_key64(ns, name, m, "M", spec["seed"]), (out_f,))
            tensors[f"{PREFIX}{m}.lora_magnitude_vector"] = (1.0 + mi * 2.0 ** -7).astype(np.float32)
        elif variant == "lora_bias":
            ci = int_matrix(_key64(ns, name, m, "C", spec["seed"]), (out_f,))
            tensors[f"{PREFIX}{m}.lora_B.bias"] = (ci * 2.0 ** -10).astype(np.float32)
        stats[m] = {"shape": [out_f, in_f], "b_exp": b_exp, "delta_rms": rms,
                    "delta_rms_over_target": rms / target_rms,
                    **({"alpha": a_mod} if pa is not None else {}), **({"scaling": scaling} if use_rslora else {})}


    meta_obj = {f"{PREFIX}r": r, f"{PREFIX}lora_alpha": alpha, f"{PREFIX}target_modules": mods,
                f"{PREFIX}rank_pattern": {}, f"{PREFIX}alpha_pattern": alpha_pattern,
                f"{PREFIX}use_dora": variant == "dora", f"{PREFIX}lora_bias": variant == "lora_bias"}
    if use_rslora:
        meta_obj[f"{PREFIX}use_rslora"] = True
    metadata = {"format": "pt", METADATA_KEY: json.dumps(meta_obj, indent=2, sort_keys=True)}
    return tensors, metadata, stats


def serialize(tensors: dict, metadata: dict) -> bytes:
    return stn.dumps({k: stn.bf16(v) for k, v in tensors.items()}, metadata)


def content_identity(tensors: dict) -> str:
    lines = []
    for k in sorted(tensors):
        v = np.ascontiguousarray(tensors[k], dtype=np.float32)
        lines.append(f"{k}\tbf16\t{list(v.shape)}\t{hashlib.sha256(stn.to_bf16_bits(v).tobytes()).hexdigest()}\n")
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def build(recipe: dict, shapes: dict) -> dict:
    out = {}
    for name in sorted(recipe["adapters"]):
        tensors, metadata, stats = make_adapter(recipe, name, shapes)
        blob = serialize(tensors, metadata)
        out[name] = {"tensors": tensors, "metadata": metadata, "stats": stats, "blob": blob}
    return out


def manifest_of(recipe: dict, built: dict) -> dict:
    m = {"recipe_id": recipe["recipe_id"], "adapters": {}}
    for name, b in built.items():
        spec = recipe["adapters"][name]
        ratios = [s["delta_rms_over_target"] for s in b["stats"].values()]
        m["adapters"][name] = {
            "file": f"{name}/{WEIGHT_NAME}",
            "file_sha256": hashlib.sha256(b["blob"]).hexdigest(),
            "file_size": len(b["blob"]),
            "tensor_set_sha256": content_identity(b["tensors"]),
            "r": spec["r"], "lora_alpha": spec["lora_alpha"], "scale_alpha_over_r": spec["lora_alpha"] / spec["r"],
            "n_modules": len(b["stats"]),
            "n_params": int(sum(v.size for v in b["tensors"].values())),
            "delta_rms_over_target": {"min": min(ratios), "max": max(ratios)},
            "modules": b["stats"],
        }
    return m


def _load_inputs(args):
    recipe = json.loads(Path(args.recipe).read_text(encoding="utf-8"))
    shapes = {k: tuple(v) for k, v in json.loads(Path(args.shapes).read_text(encoding="utf-8"))["linear_shapes"].items()}
    return recipe, shapes


def cmd_generate(args) -> int:
    recipe, shapes = _load_inputs(args)
    built = build(recipe, shapes)
    out = Path(args.out)
    for name, b in built.items():
        (out / name).mkdir(parents=True, exist_ok=True)
        (out / name / WEIGHT_NAME).write_bytes(b["blob"])
    man = manifest_of(recipe, built)
    (out / "lora_manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    for name, a in man["adapters"].items():
        print(f"[sd3-lora-synth] {name}: {a['n_modules']} modules, {a['file_size']} B, r={a['r']} alpha={a['lora_alpha']},"
              f" rms/target in [{a['delta_rms_over_target']['min']:.3f}, {a['delta_rms_over_target']['max']:.3f}],"
              f" file {a['file_sha256']}")
    return 0


def cmd_verify(args) -> int:
    recipe, shapes = _load_inputs(args)
    built = build(recipe, shapes)
    bad = 0
    for name, b in built.items():
        p = Path(args.out) / name / WEIGHT_NAME
        if not p.is_file() or p.read_bytes() != b["blob"]:
            print(f"[sd3-lora-synth] MISMATCH {p}")
            bad += 1
    exp = recipe.get("expected") or {}
    for name, b in built.items():
        want = (exp.get(name) or {}).get("file_sha256")
        have = hashlib.sha256(b["blob"]).hexdigest()
        if want and want != have:
            print(f"[sd3-lora-synth] {name}: regenerated {have} != recipe expected {want}")
            bad += 1
    print(f"[sd3-lora-synth] verify: {'OK' if not bad else f'{bad} mismatch(es)'}")
    return 0 if not bad else 4


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("generate", "verify"):
        p = sub.add_parser(c)
        p.add_argument("--out", required=True)
        p.add_argument("--recipe", default=str(DEFAULT_RECIPE))
        p.add_argument("--shapes", default=str(DEFAULT_SHAPES))
    args = ap.parse_args(argv)
    return {"generate": cmd_generate, "verify": cmd_verify}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
