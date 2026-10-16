#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
'SD3 precision profiles: ONE implementation,'


from __future__ import annotations

import math


SWITCHES = ("dit_precision", "vae_precision", "text_encoder_precisions", "disable_autocast")
COMPONENTS = ("text_encoder", "text_encoder_2", "text_encoder_3", "transformer", "vae")
DTYPES = ("bf16", "fp16", "fp32")
_TORCH = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


def _enc(w: str, c: str | None = None) -> dict:
    return {"weights": w, "compute": c or w}


PROFILES = {
    "bf16": {
        "engine": {"dit_precision": "bf16", "vae_precision": "bf16", "text_encoder_precisions": ["bf16", "bf16", "bf16"]},
        "env": {},
        "definition": {
            "text_encoder": _enc("bf16"), "text_encoder_2": _enc("bf16"), "text_encoder_3": _enc("bf16"),
            "merged_text_embeds": "bf16",
            "initial_noise": "bf16",
            "transformer": _enc("bf16"),
            "vae": {"weights": "bf16", "compute": "bf16", "upcast": False},
            "tf32": "default",
        },
        "reference": ("diffusers StableDiffusion3Pipeline.from_pretrained(torch_dtype=bfloat16): every component bf16, no "
                      "autocast; noise drawn at the merged embeds' dtype (bf16); the VAE decodes bf16 latents (the SD3 "
                      "pipeline never upcasts); cuda generator per fixture; negative prompt \" \". The recipe of the "
                      "committed tests/<id>.npy"),
    },
    "fp32": {
        "engine": {"dit_precision": "fp32", "vae_precision": "fp32", "text_encoder_precisions": ["fp32", "fp32", "fp32"]},
        "env": {"NVIDIA_TF32_OVERRIDE": "0"},
        "definition": {
            "text_encoder": _enc("fp32"), "text_encoder_2": _enc("fp32"), "text_encoder_3": _enc("fp32"),
            "merged_text_embeds": "fp32",
            "initial_noise": "fp32",
            "transformer": _enc("fp32"),
            "vae": {"weights": "fp32", "compute": "fp32", "upcast": False},
            "tf32": "off",
        },
        "reference": ("diffusers torch_dtype=float32 for every component, no autocast, both allow_tf32 flags False; noise "
                      "fp32. The recipe of the committed tests/golden_fp32/<id>.npy"),
    },
    "mixed": {


        "engine": {"dit_precision": "bf16", "vae_precision": "fp32", "text_encoder_precisions": ["fp16", "fp16", "fp32"]},
        "env": {},
        "definition": {
            "text_encoder": _enc("fp16"), "text_encoder_2": _enc("fp16"), "text_encoder_3": _enc("fp32"),
            "merged_text_embeds": "fp32",
            "initial_noise": "fp32",
            "transformer": _enc("bf16"),
            "vae": {"weights": "fp32", "compute": "fp32", "upcast": False},
            "tf32": "default",
        },
        "reference": ("diffusers: text_encoder / text_encoder_2 at float16, text_encoder_3 at float32, encode_prompt "
                      "outside autocast (the CLIP/T5 cat promotes to float32); the denoising loop under "
                      "torch.autocast(bfloat16) on the bfloat16 transformer, latents (noise) float32; the float32 VAE "
                      "decodes outside autocast"),
    },
}


def engine_kwargs(profile: str) -> dict:
    """The switches the verifier passes to DiffGenerator.from_pretrained for `profile` (a copy)."""
    e = PROFILES[profile]["engine"]
    return {k: (list(v) if isinstance(v, list) else v) for k, v in e.items()}


def cli_args(profile: str) -> list:
    """The same switches as `sglang serve` flags."""
    e = PROFILES[profile]["engine"]
    return ["--dit-precision", e["dit_precision"], "--vae-precision", e["vae_precision"],
            "--text-encoder-precisions", *e["text_encoder_precisions"]]


def torch_name(d: str) -> str:
    return _TORCH[d]


def short(torch_dtype_name: str | None) -> str | None:
    """'torch.bfloat16' / 'bfloat16' -> 'bf16'; anything else unchanged (int64 ids, None)."""
    if torch_dtype_name is None:
        return None
    t = str(torch_dtype_name).replace("torch.", "")
    return {v: k for k, v in _TORCH.items()}.get(t, t)


def _majority(census: dict) -> str | None:
    """The dtype holding the most parameter elements ({dtype: numel}); None for an empty census."""
    if not census:
        return None
    return short(max(census.items(), key=lambda kv: kv[1])[0])


MAIN_KINDS = ("gemm", "conv", "attn", "attn_impl")


HIGHER_PRECISION_PATHS: tuple = ()


CONSUMED_MIN_EFFECT = 0.0
CONSUMED_NOISE_FACTOR = 10.0


PROBE_PLAN = {"k": 8, "delta": 0.5}


ARCHITECTURALLY_UNCONSUMED = (
    ("transformer", "transformer_blocks.{last:transformer_blocks}.attn.add_q_proj",
     "the last joint block is context_pre_only: its context-query rows of the joint attention feed only the context "
     "stream, which that block drops"),
)


def parse_key(key: str) -> dict | None:
    """A witness signature 'gemm:addmm:bf16/bf16>fp32' -> {kind, op, operands, out}; None for 'overflow' / malformed."""
    try:
        kind, op, rest = str(key).split(":", 2)
        ops, out = rest.split(">", 1)
    except ValueError:
        return None
    return {"kind": kind, "op": op, "operands": [x for x in ops.split("/") if x], "out": out}


def _registered(profile: str, comp: str, sig: dict) -> str | None:
    """The reason a pre-registered higher-precision path allows this op, else None."""
    for prof, c, prefix, dtype, reason in HIGHER_PRECISION_PATHS:
        if prof == profile and c == comp and f"{sig['kind']}:{sig['op']}".startswith(prefix) \
                and set(sig["operands"]) == {dtype}:
            return reason
    return None


def _witness(events: list) -> dict:
    """The operand witness per component: served calls, calls with no main op it could see, failed calls, witness
    errors, every main-op signature with its calls and MACs, and the smallest share of the weights read in a call."""
    out = {c: {"calls": 0, "no_main": 0, "failed": 0, "errors": 0, "ops": {}, "weights_read_min": None}
           for c in COMPONENTS}
    for e in events:
        if e.get("event") != "precision_ops" or e.get("component") not in out:
            continue
        o = out[e["component"]]
        o["calls"] += 1
        o["no_main"] += 0 if e.get("n_main") else 1
        o["failed"] += 1 if e.get("failed") else 0
        o["errors"] += int(e.get("errors") or 0)
        for field in ("main", "impl"):
            for key, val in (e.get(field) or {}).items():
                n, macs = (list(val) + [0, 0])[:2] if isinstance(val, (list, tuple)) else (0, 0)
                ent = o["ops"].setdefault(str(key), [0, 0])
                ent[0] += int(n or 0)
                ent[1] += int(macs or 0)
        share = (e.get("weights_read") or {}).get("numel_share")
        if isinstance(share, (int, float)):
            o["weights_read_min"] = share if o["weights_read_min"] is None else min(o["weights_read_min"], share)
    return out


def _declared_dead(comp: str, module: str | None, indexed: dict) -> str | None:
    """The reason a sampled op's module is declared dead compute (ARCHITECTURALLY_UNCONSUMED), else None."""
    import re

    if not module:
        return None
    for c, pattern, reason in ARCHITECTURALLY_UNCONSUMED:
        if c != comp:
            continue
        ok = True

        def last(m):
            nonlocal ok
            if m.group(1) not in (indexed or {}):
                ok = False
                return "?"
            return str(indexed[m.group(1)])

        path = re.sub(r"\{last:([^}]+)\}", last, pattern)
        if ok and module == path:
            return reason
    return None


def consumption(probe: dict | None, comp: str = "") -> dict:
    """One `precision_probe` line reduced: each sampled op consumed (its perturbation moved the output past the floor),
    not consumed, declared dead (ARCHITECTURALLY_UNCONSUMED, by its module) or unevaluated (not perturbable -- an out= /
    in-place overload -- or no finite reading)."""
    if probe is None:
        return {"ran": False, "why": "no probe line"}
    if probe.get("error"):
        return {"ran": False, "why": f"the probe failed: {probe['error']}"}
    rep = probe.get("repeat_rel")
    ok_rep = isinstance(rep, (int, float)) and math.isfinite(rep)
    floor = max(CONSUMED_MIN_EFFECT, CONSUMED_NOISE_FACTOR * rep) if ok_rep else None
    targets = []
    for t in probe.get("targets") or []:
        r = t.get("rel")
        state = "unevaluated"
        if floor is not None and t.get("perturbed") and isinstance(r, (int, float)) and math.isfinite(r):
            state = "consumed" if r > floor else "not_consumed"
        dead = _declared_dead(comp, t.get("module"), probe.get("indexed") or {})
        if dead and state == "not_consumed":
            state = "declared_dead"
        targets.append({"i": t.get("i"), "key": t.get("key"), "module": t.get("module"), "macs": t.get("macs"),
                        "rel": r, "state": state, **({"why": dead} if state == "declared_dead" else {})})
    return {"ran": True, "n_main": probe.get("n_main"), "repeat_rel": rep, "floor": floor, "targets": targets,
            "why": None if floor is not None else f"no finite unchanged re-run ({rep}), so there is no noise floor"}


def observed(events: list) -> dict:
    """The record reduced per component: `compute` (the operand dtypes of every witnessed main op -- the judged
    reading), the witness detail, the consumption probe, the parameter census (`weights`), and the OUTER state as a
    diagnostic (autocast, input / output dtypes of the `precision` lines and the DiT's `dit_call` lines); plus the
    initial noise (the latents' dtype at each request's first scheduler step), the processes' TF32 state and the number
    of DiT forwards (`dit_call` lines)."""
    wit = _witness(events)
    outer: dict = {c: {"weights": None, "autocast": set(), "inputs": set(), "outputs": set(), "calls": 0}
                   for c in COMPONENTS}

    def note(o, ac, x_in, x_out):
        o["calls"] += 1
        ac = ac or {}
        o["autocast"].add("unknown" if ac.get("unknown") else (short(ac.get("dtype")) or "off"))
        if x_in:
            o["inputs"].add(short(x_in))
        if x_out:
            o["outputs"].add(short(x_out))

    dit_calls = 0
    probes: dict = {}
    for e in events:
        ev = e.get("event")
        if ev == "precision" and e.get("component") in outer:
            o = outer[e["component"]]
            if e.get("params"):
                o["weights"] = _majority(e["params"])
            if "autocast" in e:
                note(o, e.get("autocast"), e.get("in"), e.get("out"))
        elif ev == "dit_call":
            dit_calls += 1
            if isinstance(e.get("prec"), dict):
                p = e["prec"]
                note(outer["transformer"], p.get("autocast"), p.get("in"), p.get("out"))
        elif ev == "precision_probe" and e.get("component") in outer:
            probes.setdefault(e["component"], e)
    comps = {}
    for c in COMPONENTS:
        w = wit[c]
        compute = sorted({d for key in w["ops"] for d in ((parse_key(key) or {}).get("operands") or ["?"])})
        comps[c] = {"weights": outer[c]["weights"], "compute": compute, "calls": w["calls"],
                    "witness": {k: w[k] for k in ("calls", "no_main", "failed", "errors", "weights_read_min")}
                    | {"ops": dict(sorted(w["ops"].items()))},
                    "consumption": consumption(probes.get(c), c),
                    "outer": {"autocast": sorted(outer[c]["autocast"]), "inputs": sorted(outer[c]["inputs"]),
                              "outputs": sorted(outer[c]["outputs"]), "calls": outer[c]["calls"],
                              "diagnostic": True}}
    noise = {short(e.get("sample")) for e in events if e.get("event") == "precision_latents" and e.get("sample")}
    envs = [e for e in events if e.get("event") == "precision_env"]
    return {"components": comps, "initial_noise": sorted(noise), "dit_calls": dit_calls,
            "witness_errors": [e for e in events if e.get("event") == "precision_witness_error"][:8],
            "tf32": [{"tf32": e.get("tf32"), "env": e.get("env")} for e in envs]}


def _tf32_off(env_line: dict) -> bool:
    """TF32 is off in a process when NVIDIA_TF32_OVERRIDE=0 forces it off at the library level, or both flags say so."""
    env, flags = env_line.get("env") or {}, env_line.get("tf32") or {}
    return str(env.get("NVIDIA_TF32_OVERRIDE")) == "0" or (flags.get("matmul") is False and flags.get("cudnn") is False)


def _fmt_ops(rows: list, total: int) -> str:
    return ", ".join(f"{k} x{n} ({(100.0 * m / total) if total else 0.0:.1f}% of its main MACs)" for k, n, m in rows[:4])


def canonical_ids_sha(ids) -> tuple:
    """The canonical identity of one tokenized prompt: (sha256 of its token ids without padding (0) and the end token
    (1), core length). The recorder's copy is component_recorder/sitecustomize._joint_ids_sha; the two must agree."""
    import hashlib
    core = [int(i) for i in ids]
    while core and core[-1] in (0, 1):
        core.pop()
    while core and core[0] == 0:
        core.pop(0)
    return hashlib.sha256(",".join(str(i) for i in core).encode()).hexdigest(), len(core)


WINDOW_SLACK_S = 0.05


def joint_conditioning(profile: str, events: list, ledger: list | None = None, tokenize=None) -> dict:
    """"Joint conditioning uses dtype promotion" (instruction), judged where the DiT consumes it, for every request
    and every row. The recorder hashes every T5 output row with the canonical token ids that produced it, the T5 part
    of every distinct DiT conditioning, and one line per DiT forward naming the conditioning it consumed.

    Row identity: a consumed row whose T5 part, cast to the DiT's compute precision, equals a recorded T5 output cast
    the same way carries the promoted merge -- handed over at the promoted dtype (autocast casts it at the first GEMM)
    or already cast to the compute precision at the DiT's entry: the same consumed values, so both code styles are
    honoured. A row that matches only through an fp16 detour (or, when the DiT computes fp32, a bf16 one) is a
    demotion: a mismatch. A row that matches no recorded T5 output is a coverage gap, never skipped.

    Request coverage (the verifier's ledger: label, prompt, cfg, its window [t0, t1], the prompt's canonical ids or a
    tokenizer to compute them): within the request's window, the DiT must consume a row carrying the T5 output of the
    request's own prompt tokens (promoted), and for cfg > 1 a row carrying the T5 output of another prompt (the
    negative). Consumed through a detour: a mismatch. Not observed, or no T5 output recorded for the prompt's tokens:
    a coverage gap. A request that raised is reported, not required.

    Never a pass: a guard that dropped records (precision_joint_overflow), a truncated record, a joint witness error,
    a DiT forward without a call line or naming an unrecorded conditioning, a T5 output whose token ids could not be
    read, a record without these fields (an older recorder), no ledger. Pooled projections are not joint conditioning."""
    cd = _TORCH[PROFILES[profile]["definition"]["transformer"]["compute"]]
    det_paths = [f"float16>{cd}"] + (["bfloat16>float32"] if cd == "float32" else [])
    t5 = [e for e in events if e.get("event") == "precision_joint_t5"]
    dits = [e for e in events if e.get("event") == "precision_joint_dit"]
    calls = [e for e in events if e.get("event") == "precision_joint_call"]
    out = {"compute": cd, "t5_calls": len(t5), "t5_rows": sum(len(e.get("rows") or []) for e in t5),
           "dit_calls": len(dits), "call_lines": len(calls), "dit_dtypes": sorted({str(e.get("dtype")) for e in dits}),
           "mismatches": [], "unmeasured": [], "rows": [], "requests": [], "gaps": {}}

    def gap(kind, text):
        out["gaps"][kind] = out["gaps"].get(kind, 0) + 1
        if out["gaps"][kind] <= 6:
            out["unmeasured"].append("joint conditioning: " + text)
        elif out["gaps"][kind] == 7:
            out["unmeasured"].append(f"joint conditioning: more {kind} gaps (counted in observed.joint_conditioning.gaps)")

    if not t5 or not dits:
        gap("no_reading", "no " + ("T5 output" if not t5 else "DiT conditioning")
            + " was recorded, so the dtype promotion the DiT consumes is unmeasured")
        return out
    legacy = (any("status" not in e for e in t5) or any("w" not in e for e in dits) or not calls)
    if legacy:
        gap("legacy_record", "the record has no per-call lines or token identities (a recorder older than the "
                             "request/row coverage), so no request can be proven covered")
        return out
    for e in events:
        k = e.get("event")
        if k == "precision_joint_overflow":
            gap("overflow", f"the recorder's guard dropped {e.get('dropped')} {e.get('kind')} line(s) (max {e.get('max')})")
        elif k == "truncated":
            gap("truncated", f"the record was truncated at {e.get('max_events')} lines")
        elif k == "precision_witness_error" and e.get("where") == "joint":
            gap("witness_error", f"the {e.get('component')} joint hook failed: {e.get('error')}")

    promoted, detour = {}, {}
    for e in t5:
        if e.get("status") != "ok":
            gap("t5_unreadable", "a T5 output could not be read, so the rows it produced cannot be identified")
            continue
        L = str((e.get("shape") or [0, 0])[-2])
        for r in e.get("rows") or []:
            if r.get("ids") is None:
                gap("t5_ids", "a T5 forward's token ids could not be read, so its rows name no prompt")
            promoted.setdefault((L, r.get(cd)), set()).add(r.get("ids"))
            for d in det_paths:
                if r.get(d) is not None and r.get(d) != r.get(cd):
                    detour.setdefault((L, r.get(d)), set()).add((r.get("ids"), d))

    ident = {}
    for e in dits:
        rows = []
        for i, r in enumerate(e.get("rows") or []):
            found = None
            for L, h in r.items():
                x = (h or {}).get(cd)
                if (L, x) in promoted:
                    found = {"state": "promoted", "ids": sorted(i_ for i_ in promoted[(L, x)] if i_), "len": L}
                    break
            if found is None:
                for L, h in r.items():
                    x = (h or {}).get(cd)
                    if (L, x) in detour:
                        found = {"state": "detour", "ids": sorted(a for a, _ in detour[(L, x)] if a),
                                 "via": sorted({d for _, d in detour[(L, x)]}), "len": L}
                        break
            if found is None:
                found = {"state": "unknown"}
            rows.append(found)
            row = {"conditioning": e.get("n"), "w": e.get("w"), "row": i, "dtype": e.get("dtype"), **found}
            out["rows"].append(row)
            if found["state"] == "detour":
                out["mismatches"].append(f"joint conditioning: DiT conditioning {e.get('n')} row {i} "
                                         f"({short(e.get('dtype'))}) carries the T5 output only through a "
                                         f"{short(found['via'][0].split('>')[0])} detour; the promoted merge reaches the "
                                         f"DiT's {short(cd)} compute without it")
            elif found["state"] == "unknown":
                gap("unknown_row", f"DiT conditioning {e.get('n')} row {i} matches no recorded T5 output at "
                                   f"{short(cd)} or through a detour, so its dtype path is unidentified")
        ident[e.get("w")] = (e, rows)

    for c in calls:
        if c.get("status") != "ok":
            gap("call_" + str(c.get("status")), f"DiT forward {c.get('c')} consumed a conditioning the recorder could "
                                                f"not hash ({c.get('status')})")
        elif c.get("w") not in ident:
            gap("call_unrecorded", f"DiT forward {c.get('c')} names conditioning {c.get('w')} that was never recorded")
    forwards = sum(1 for e in events if e.get("event") == "dit_call")
    if forwards > len(calls):
        gap("call_missing", f"{forwards - len(calls)} of {forwards} DiT forward(s) wrote no joint call line")

    if ledger is None:
        gap("no_ledger", "no request ledger was handed to the judge, so request coverage is unproven")
        return out
    for q in ledger:
        ids_p = q.get("ids_sha")
        if ids_p is None and tokenize is not None:
            try:
                ids_p = canonical_ids_sha(tokenize(q.get("prompt")))[0]
            except Exception as exc:  # noqa: BLE001
                ids_p = None
                q = dict(q, tokenize_error=f"{type(exc).__name__}: {exc}"[:120])
        t0, t1 = q.get("t0"), q.get("t1")
        rec = {"label": q.get("label"), "prompt": q.get("prompt"), "cfg": q.get("cfg"), "ok": q.get("ok", True),
               "ids_sha": ids_p, "calls": 0, "prompt_row": None, "negative_row": None}
        if t0 is None or t1 is None:
            rec["prompt_row"] = "no window"
        else:
            win = [c for c in calls if c.get("status") == "ok" and c.get("w") in ident
                   and t0 - WINDOW_SLACK_S <= float(c.get("t") or 0) <= t1 + WINDOW_SLACK_S]
            rec["calls"] = len(win)
            consumed = [r for c in win for r in ident[c["w"]][1]]
            if ids_p is None:
                rec["prompt_row"] = "prompt tokens unknown"
            elif any(r["state"] == "promoted" and ids_p in r["ids"] for r in consumed):
                rec["prompt_row"] = "promoted"
            elif any(r["state"] == "detour" and ids_p in r["ids"] for r in consumed):
                rec["prompt_row"] = "detour"
            elif not any(ids_p in v for v in promoted.values()):
                rec["prompt_row"] = "no T5 output for its tokens"
            else:
                rec["prompt_row"] = "not consumed in its window"
            if float(q.get("cfg") or 1) > 1:
                others = [r for r in consumed if r["state"] in ("promoted", "detour") and r["ids"] and ids_p not in r["ids"]]
                rec["negative_row"] = ("promoted" if any(r["state"] == "promoted" for r in others) else
                                       "detour" if others else "not consumed in its window")
        out["requests"].append(rec)
        req = f"request {q.get('label')} ({q.get('prompt')!r:.48})"
        if not rec["ok"]:
            continue
        for part in ("prompt_row", "negative_row"):
            v = rec[part]
            if v is None or v == "promoted":
                continue
            if v == "detour":
                out["mismatches"].append(f"joint conditioning: {req}: its {part.split('_')[0]} row reached the DiT only "
                                         f"through a detour")
            else:
                gap("request_" + part, f"{req}: {part.replace('_', ' ')} {v}")
    return out


def precision_verdict(profile: str, events: list, require_consumption: bool = True, ledger: list | None = None,
                      tokenize=None) -> dict:
    """Every way the recorded execution differs from the profile's definition, in two lists: `mismatches` (measured and
    wrong: the candidate) and `unmeasured` (no execution reading: never a pass; the instrument or undetermined). Also
    `measured` and `attribution` (candidate / instrument / undetermined / None), recorded separately."""
    d = PROFILES[profile]["definition"]
    obs = observed(events)
    bad, unmeasured = [], []
    healthy = not obs["witness_errors"] and any(obs["components"][c]["witness"]["calls"] for c in COMPONENTS)
    for comp in COMPONENTS:
        want, got = d[comp], obs["components"][comp]
        w = got["witness"]
        if got["weights"] is None:
            if w["calls"]:
                unmeasured.append(f"{comp}: no parameter census was recorded")
        elif got["weights"] != want["weights"]:
            bad.append(f"{comp}: weights {got['weights']}, the profile says {want['weights']}")
        if not w["calls"]:
            if healthy:
                bad.append(f"{comp}: never ran while the profile was served (no witnessed call)")
            else:
                unmeasured.append(f"{comp}: no operand-witness line, so its compute precision is unmeasured (compute "
                                  f"is never inferred from the weights or the autocast state)")
            continue
        if w["errors"]:
            unmeasured.append(f"{comp}: the operand witness failed on {w['errors']} op(s)")
        if w["no_main"]:
            unmeasured.append(f"{comp}: {w['no_main']} of {w['calls']} call(s) ran no main op the witness can see "
                              f"(a kernel outside the dispatcher): compute unmeasured")
        if w["failed"]:
            bad.append(f"{comp}: {w['failed']} served call(s) raised")
        total = sum(m for _, m in w["ops"].values())
        off = []
        for key, (n, m) in w["ops"].items():
            sig = parse_key(key)
            if sig is None:
                unmeasured.append(f"{comp}: an unreadable witness signature {key!r}")
                continue
            if sig["kind"] not in MAIN_KINDS:
                continue
            if set(sig["operands"]) != {want["compute"]} and not _registered(profile, comp, sig):
                off.append((key, n, m))
        if off:
            off.sort(key=lambda r: -r[2])
            bad.append(f"{comp}: main ops computed off the profile's {want['compute']}: "
                       + _fmt_ops(off, total))
        if require_consumption:
            cons = got["consumption"]
            if not cons["ran"] or cons.get("floor") is None:
                unmeasured.append(f"{comp}: consumption unmeasured ({cons.get('why')})")
            else:
                states = [t["state"] for t in cons["targets"]]
                if not any(s == "consumed" for s in states):
                    unmeasured.append(f"{comp}: the consumption probe found no sampled op consumed or evaluated "
                                      f"({states or 'no target'})")
                dead = [t for t in cons["targets"] if t["state"] == "not_consumed"]
                if dead:
                    bad.append(f"{comp}: sampled main op(s) whose result is not consumed (perturbing it moved the output "
                               f"by <= {cons['floor']:.3g}): " + ", ".join(
                                   f"#{t['i']} {t['key']} ({t.get('module')}) rel={t['rel']:.3g}" for t in dead[:4]))
    if obs["components"]["transformer"]["witness"]["calls"] < obs["dit_calls"]:
        unmeasured.append(f"transformer: {obs['dit_calls'] - obs['components']['transformer']['witness']['calls']} "
                          f"of {obs['dit_calls']} DiT forward(s) were not witnessed")
    joint = joint_conditioning(profile, events, ledger=ledger, tokenize=tokenize)
    bad += joint["mismatches"]
    unmeasured += joint["unmeasured"]
    obs["joint_conditioning"] = joint
    if not obs["initial_noise"]:
        unmeasured.append("initial noise: no request's first scheduler step was recorded")
    elif obs["initial_noise"] != [d["initial_noise"]]:
        bad.append(f"initial noise: {obs['initial_noise']}, the profile says {d['initial_noise']}")
    if d["tf32"] == "off":
        if not obs["tf32"]:
            unmeasured.append("tf32: no process recorded its TF32 state")
        elif not all(_tf32_off(x) for x in obs["tf32"]):
            bad.append(f"tf32: on in a process ({obs['tf32']}), the profile says off")
    attribution = ("candidate" if bad else None) if not unmeasured else (
        "candidate" if bad else ("instrument" if not healthy or obs["witness_errors"] else "undetermined"))
    return {"mismatches": bad, "unmeasured": unmeasured, "measured": not unmeasured, "attribution": attribution,
            "healthy_witness": healthy, "observed": obs}


def precision_mismatches(profile: str, events: list, require_consumption: bool = True) -> list:
    """Every reason the profile is NOT certified by the record: the measured mismatches, then the unmeasured clauses.
    Empty = honoured, with every clause measured. Unmeasured is never honoured."""
    v = precision_verdict(profile, events, require_consumption)
    return v["mismatches"] + v["unmeasured"]
