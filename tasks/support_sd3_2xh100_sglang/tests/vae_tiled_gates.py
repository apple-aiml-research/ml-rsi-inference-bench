#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
'SD3 vae tiled gates utilities.'


from __future__ import annotations

import math

POLICY = {

    "MIN_TILES": 2,


    "SEAM_MARGIN_DB": 4.0,


    "SEAM_EXCESS_MAX_U8": 1.0,
    "SEAM_EXCESS_RATIO": 4.0,


    "REF_PSNR_DB": 45.0,

    "REPEAT_PSNR_DB": 45.0,
}


POLICY_STATE = "specified"

POLICY_BASIS = {'MIN_TILES': 'Required bound for MIN_TILES.', 'SEAM_MARGIN_DB': 'Required bound for SEAM_MARGIN_DB.', 'SEAM_EXCESS_MAX_U8': 'Required bound for SEAM_EXCESS_MAX_U8.', 'SEAM_EXCESS_RATIO': 'Required bound for SEAM_EXCESS_RATIO.', 'REF_PSNR_DB': 'Required bound for REF_PSNR_DB.', 'REPEAT_PSNR_DB': 'Required bound for REPEAT_PSNR_DB.'}

VERDICTS = ("pass", "fail", "not_evaluated")


def psnr(a, b) -> float | None:
    """The verifier's image psnr (uint8 HWC, mse over every element, 100 for identical); None if either is missing or
    the shapes differ. numpy is imported only here, so the judge stays stdlib."""
    if a is None or b is None:
        return None
    import numpy as np

    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return None
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 100.0 if mse <= 1e-9 else 10.0 * math.log10(255.0 ** 2 / mse)


SEAM_BORDER = 16


def seam_excess(x, u, border: int = SEAM_BORDER) -> dict | None:
    """The seam-line statistic (uint8 HWC images, x = tiled, u = untiled): per row r, j_x(r) = mean |x[r] - x[r-1]|
    over columns and channels, j_u(r) the same on u; the row excess is max_r (j_x(r) - j_u(r)), the column excess the
    same along the other axis. The `border` outermost rows and columns are left out: a tile strip whose contrast the
    per-tile normalisation changed can scale the image's own edges there. A seam of this tile grid does not sit within 16 px
    of the image edge (the seams are at multiples of tile - blend, 768 px for SD3's VAE). Returns {"row": ..., "row_at":
    r, "col": ..., "col_at": c, "max": ..., "border": border}; None if a side is missing or the shapes differ."""
    if x is None or u is None:
        return None
    import numpy as np

    x, u = np.asarray(x).astype(np.float64), np.asarray(u).astype(np.float64)
    if x.shape != u.shape or x.ndim != 3 or x.shape[0] < 2 or x.shape[1] < 2:
        return None
    b = border if min(x.shape[0], x.shape[1]) > 2 * border + 2 else 0
    er = np.abs(x[1:] - x[:-1]).mean(axis=(1, 2)) - np.abs(u[1:] - u[:-1]).mean(axis=(1, 2))
    ec = np.abs(x[:, 1:] - x[:, :-1]).mean(axis=(0, 2)) - np.abs(u[:, 1:] - u[:, :-1]).mean(axis=(0, 2))
    er_in, ec_in = er[b:len(er) - b], ec[b:len(ec) - b]
    out = {"row": float(er_in.max()), "row_at": int(er_in.argmax()) + b + 1, "col": float(ec_in.max()),
           "col_at": int(ec_in.argmax()) + b + 1, "border": b}
    out["max"] = max(out["row"], out["col"])
    return out


def _result(verdict: str, reasons: list, details: dict) -> dict:
    return {"verdict": verdict, "reasons": reasons, "details": details}


def _fmt(v) -> str:
    return "None" if v is None else f"{v:.2f} dB"


def recorded_tiling(line: dict | None, policy: dict = POLICY) -> tuple:
    """(held, why) for the recorded clause on one `vae_decode` line: None when the line is missing or carries no tile
    count (the recorder found no `decoder` submodule to count -- the instrument, not a verdict)."""
    if not line:
        return None, "the recorder saw no VAE decode for the tiled request"
    tiles, tiled = line.get("tiles"), line.get("tiled")
    if tiles is None or tiled is None:
        return None, ("the recorder's vae_decode line carries no tile count (tiles=None): the served VAE has no "
                      "`decoder` submodule the recorder could count, so the executed tiling is unobserved")
    hw, tl = line.get("latent_hw"), line.get("tile_latent")
    held = bool(tiled) and int(tiles) >= int(policy["MIN_TILES"])
    why = (f"latent {hw} on a {tl} x {tl} latent tile: tiled={bool(tiled)}, {int(tiles)} decoder pass(es), "
           f"{line.get('blends')} seam blend(s) of extent {line.get('blend_extent')}")
    if not held:
        why += f" -- the --vae-tiling switch did not tile the decode (bar: tiled, >= {policy['MIN_TILES']} tiles)"
    return held, why


def vae_tiled_decode_is_real(recorded: dict | None, readings: dict | None, policy: dict = POLICY,
                             extra: dict | None = None) -> dict:
    """`vae_tiled_decode_is_real`. `recorded` = the tiled engine's `vae_decode` line of the fixed-latent
    request (recorded_tiling); `readings` = the replay's numbers on that latent:
        tiled_vs_untiled          psnr(engine tiled, U)          untiled_source  "engine" | "reference"
        ref_tiled_vs_ref_untiled  psnr(diffusers tiled, diffusers untiled): the reference's own tiling cost
        seam_excess               seam_excess(engine tiled, U)   (the dict, or None)
        seam_excess_ref_tiled_vs_ref_untiled   the same statistic for the reference's own tiling (its ramp)
        tiled_vs_ref_tiled_at_recorded   psnr(engine tiled, diffusers tiled at the recorded tiling)
        recorded_tiling_replayed  {overlap, tile_latent, tile_sample} the reference was set to, or None
    A clause that cannot be read is listed under `missing`; the verdict is `fail` when any clause that was read did
    not hold (a measured defect stands whatever else is missing), `not_evaluated` when nothing failed and a clause is
    missing, `pass` only when every applicable clause was read and held."""
    p = dict(readings or {})
    reasons, missing, held_notes = [], [], []


    vdt = str((recorded or {}).get("in_dtype") or "").replace("torch.", "") or None
    rdt = str(p.get("replay_dtype") or "float32").replace("torch.", "")
    cross = (f"the served VAE decoded in {vdt}, the reference replay in {rdt}: the two tiling costs are not comparable "
             "(precision-matched replay required)") if vdt is not None and vdt != rdt else None
    held, why = recorded_tiling(recorded, policy)
    tiled_ok = held is True
    if held is None:
        missing.append(f"recorded: {why}")
    elif not held:
        reasons.append(f"recorded: {why}")
    else:
        held_notes.append(f"recorded: {why}")
    seam, src = p.get("tiled_vs_untiled"), p.get("untiled_source")
    ref_cost = p.get("ref_tiled_vs_ref_untiled")
    margin = float(policy["SEAM_MARGIN_DB"])
    rel_ok = None
    if cross:
        missing.append(f"seam_psnr: {cross}")
    elif seam is None:
        missing.append("seam_psnr: no untiled decode of the fixed latent to compare the tiled one with"
                       + (f" ({p['untiled_missing']})" if p.get("untiled_missing") else ""))
    elif ref_cost is None:
        missing.append("seam_psnr: the reference's own tiling cost on this latent (diffusers tiled vs untiled) is "
                       "missing" + (f" ({p['reference_missing']})" if p.get("reference_missing") else ""))
    else:
        rel_ok = seam >= ref_cost - margin
        line = (f"seam_psnr: tiled vs untiled ({src}) {_fmt(seam)} against the reference's own tiling cost "
                f"{_fmt(ref_cost)} on this latent (margin {margin:.1f} dB)")
        (held_notes if rel_ok else reasons).append(line + ("" if rel_ok else
                                                           " -- the tiles reassemble worse than the reference's do"))
    ex = p.get("seam_excess")
    ex_max = ex.get("max") if isinstance(ex, dict) else None
    ref_ex = p.get("seam_excess_ref_tiled_vs_ref_untiled")
    ref_ex_max = ref_ex.get("max") if isinstance(ref_ex, dict) else None
    ex_ok = None
    if ex_max is None:
        missing.append("seam_line: the seam-line statistic could not be read"
                       + (f" ({p['untiled_missing']})" if p.get("untiled_missing") else ""))
    else:
        bar = max(float(policy["SEAM_EXCESS_MAX_U8"]),
                  float(policy["SEAM_EXCESS_RATIO"]) * float(ref_ex_max or 0.0))
        ex_ok = ex_max <= bar
        line = (f"seam_line: largest excess line jump {ex_max:.3f} uint8 levels (row {ex['row']:.3f} at {ex['row_at']}, "
                f"col {ex['col']:.3f} at {ex['col_at']}; bar {bar:.2f} = max({policy['SEAM_EXCESS_MAX_U8']:.2f}, "
                f"{policy['SEAM_EXCESS_RATIO']:.0f} x the reference's own {ref_ex_max if ref_ex_max is None else round(ref_ex_max, 3)}))")
        (held_notes if ex_ok else reasons).append(line + ("" if ex_ok else " -- a hard seam the untiled image lacks"))
    if tiled_ok:
        ref = p.get("tiled_vs_ref_tiled_at_recorded")
        at = p.get("recorded_tiling_replayed")
        if cross:
            missing.append(f"reference: {cross}")
        elif ref is None:
            missing.append("reference: the diffusers tiled decode of the fixed latent at the recorded tiling is "
                           "missing" + (f" ({p['reference_missing']})" if p.get("reference_missing") else ""))
        elif ref >= float(policy["REF_PSNR_DB"]):
            held_notes.append(f"reference: tiled vs diffusers tiled at the recorded tiling {at} {_fmt(ref)} >= "
                              f"{policy['REF_PSNR_DB']:.1f}")
        elif rel_ok and ex_ok and seam is not None and ref_cost is not None and seam >= ref_cost:
            held_notes.append(f"reference: tiled vs diffusers tiled at the recorded tiling {at} {_fmt(ref)} < "
                              f"{policy['REF_PSNR_DB']:.1f}, but another blending that is at least as good as the "
                              f"reference's on this latent ({_fmt(seam)} >= {_fmt(ref_cost)}) and seamless")
        else:
            reasons.append(f"reference: tiled vs diffusers tiled at the recorded tiling {at} {_fmt(ref)} < "
                           f"{policy['REF_PSNR_DB']:.1f} dB, and not as good as the reference's own tiling either")
    details = {"recorded": recorded, "readings": p, "missing": missing, "held": held_notes,
               "policy_state": POLICY_STATE,
               "bars": {"min_tiles": policy["MIN_TILES"], "seam_margin_db": margin,
                        "seam_excess_max_u8": policy["SEAM_EXCESS_MAX_U8"],
                        "seam_excess_ratio": policy["SEAM_EXCESS_RATIO"], "ref_db": policy["REF_PSNR_DB"],
                        "repeat_db": policy["REPEAT_PSNR_DB"]}}
    if extra:
        details.update(extra)
    if reasons:
        return _result("fail", reasons + [f"(unread) {m}" for m in missing], details)
    if missing:
        return _result("not_evaluated", missing, details)
    return _result("pass", held_notes, details)


def lines_in_window(events: list, t0: float, t1: float, event: str = "vae_decode") -> list:
    """The recorder's `event` lines stamped inside [t0, t1] (one request's window; requests run one at a time)."""
    return [e for e in events if e.get("event") == event and t0 <= float(e.get("t") or 0.0) <= t1]


def peak_ratio(tiled_line: dict | None, untiled_line: dict | None) -> float | None:
    """peak_mem_bytes(tiled) / peak_mem_bytes(untiled): the memory the tiling buys (reported, no bar)."""
    a = (tiled_line or {}).get("peak_mem_bytes")
    b = (untiled_line or {}).get("peak_mem_bytes")
    return None if not a or not b else float(a) / float(b)


def peak_delta_ratio(tiled_line: dict | None, untiled_line: dict | None) -> float | None:
    """The decodes' own increments over what was allocated when each started (peak_delta_bytes; the recorder reads the
    allocator, never resets it): the memory the tiling saves, without the resident weights (reported, no bar)."""
    a = (tiled_line or {}).get("peak_delta_bytes")
    b = (untiled_line or {}).get("peak_delta_bytes")
    return None if not a or not b else float(a) / float(b)
