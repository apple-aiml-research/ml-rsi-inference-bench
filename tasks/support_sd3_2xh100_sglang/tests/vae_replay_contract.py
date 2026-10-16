#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Independent VAE replay settings, derived from the verifier's decode tap.

The checkpoint supplies architecture/weights. Only observed precision and tile
geometry may vary; inconsistent metadata refuses the replay, never earns credit.
"""
from __future__ import annotations

import math

CONFIG_KEYS = ("in_channels", "out_channels", "latent_channels", "block_out_channels",
               "down_block_types", "up_block_types", "layers_per_block", "act_fn",
               "norm_num_groups", "sample_size", "scaling_factor", "shift_factor",
               "force_upcast", "use_quant_conv", "use_post_quant_conv")


def dtype_name(value):
    name = str(value).removeprefix("torch.")
    if name not in ("float32", "float16", "bfloat16"):
        raise ValueError(f"unsupported or missing observed VAE dtype: {name}")
    return name


def replay_settings(recorded, captured_dtype):
    observed = dtype_name(recorded.get("in_dtype"))
    if observed != dtype_name(captured_dtype):
        raise ValueError("VAE observer dtype disagrees with captured latent dtype")
    geometry = {}
    for key in ("tile_latent", "tile_sample"):
        value = recorded.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"invalid observed {key}: {value}")
        geometry[key] = value
    overlap = recorded.get("overlap")
    if isinstance(overlap, bool) or not isinstance(overlap, (int, float)) or not math.isfinite(overlap) or not 0 <= overlap < 1:
        raise ValueError(f"invalid observed overlap: {overlap}")
    geometry["overlap"] = float(overlap)
    return observed, geometry


def decode_reference(model, device, z, recorded, loader, dtypes):
    """Load once, decode untiled/default-tiled/observed-tiled at matched precision.

    `loader` is the trusted AutoencoderKL.from_pretrained; dependency injection
    lets CPU tests exercise these exact calls without serving a candidate.
    """
    name, geometry = replay_settings(recorded, z.dtype)
    dtype = dtypes[name]
    vae = loader(model, subfolder="vae", torch_dtype=dtype).to(device, dtype=dtype)
    cfg = getattr(vae, "config", {})
    observed_config = recorded.get("vae_config") or {}
    for key, value in observed_config.items():
        if key not in CONFIG_KEYS:
            raise ValueError(f"unrecognized observed VAE config field: {key}")
        expected = cfg.get(key) if hasattr(cfg, "get") else getattr(cfg, key, None)
        if isinstance(value, (list, tuple)) and isinstance(expected, (list, tuple)):
            value, expected = list(value), list(expected)
        if value != expected:
            raise ValueError(f"served VAE config differs from checkpoint: {key}")
    blocks = cfg.get("block_out_channels") if hasattr(cfg, "get") else getattr(cfg, "block_out_channels", None)
    if blocks and geometry["tile_sample"] != geometry["tile_latent"] * 2 ** (len(blocks) - 1):
        raise ValueError("observed VAE tile sizes disagree with checkpoint scale factor")
    defaults = {"tile_sample": int(vae.tile_sample_min_size),
                "tile_latent": int(vae.tile_latent_min_size),
                "overlap": float(vae.tile_overlap_factor)}
    latent = z.to(device, dtype=dtype)
    vae.disable_tiling()
    untiled = vae.decode(latent, return_dict=False)[0]
    vae.enable_tiling()
    default_tiled = vae.decode(latent, return_dict=False)[0]
    try:
        vae.tile_overlap_factor = geometry["overlap"]
        vae.tile_latent_min_size = geometry["tile_latent"]
        vae.tile_sample_min_size = geometry["tile_sample"]
        recorded_tiled = default_tiled if geometry == defaults else vae.decode(latent, return_dict=False)[0]
    finally:
        vae.tile_overlap_factor = defaults["overlap"]
        vae.tile_latent_min_size = defaults["tile_latent"]
        vae.tile_sample_min_size = defaults["tile_sample"]
    return untiled, default_tiled, recorded_tiled, {
        "replay_dtype": name, "recorded_tiling_replayed": geometry,
        "reference_tiles": defaults, "observed_config_checked": sorted(observed_config)}
