#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The fidelity gate -- the served model against the model author's own implementation of it.

This is the node that a server which does not compute this architecture cannot pass, however fluent it
is and however well it scores on a public benchmark. `/app/eval/deepseek_v41_flash/fidelity/` gives the agent the
method and its code (`check.py`, loaded here from the grading copy); this module runs it on the hidden sequences
against each band's bar.

**The reference.** The model-author implementation that ships inside the checkpoint
(`inference/model.py`, `engram.py`, `kernel.py`, revision dba1be0a), converted with its own
`convert.py --model-parallel 4 --expert-dtype fp4` and run unmodified at MP 4 on four B200s, teacher-
forced on the same token ids with the output head returning every position. It is independent of vLLM
by construction: its own TileLang kernels, its own Engram hashing and tables, its own two-level
indexer. Its kernels ran under TileLang 0.1.12 rather than the 0.1.8 its requirements name, because
0.1.8's GEMMs return corrupt values on B200 (measured: non-finite at random rows on every run, and
finite-but-wrong values even on 512-token prefills; `provenance` in the data says what ran). What ships
is its OUTPUT, as data.

**The statistic**, as the public README declares it:

    lr = logp_reference - logp_candidate      at the sequence's own next token
    k3 = exp(lr) - 1 - lr                     the low-variance KL estimator; +inf when exp(lr) overflows
    per sequence: the mean of k3 over positions
    per band:     the median over that band's sequences
    the gate:     EVERY band's median at most that band's own T_b

**Structure first, one parser, every attempt kept**. The recorded data is validated
before any request (`check_reference`). Each reading is parsed by the grading copy's one typed parser: exactly one
choice, exactly one prompt entry per token, the documented null first, then an entry for the sequence's own token
with a finite numeric log-probability at every position. A structurally invalid reading fails its sequence and the
node at once: it is never re-sent, never outvoted by a later reading, and no band median is taken over a draw that
holds one. Every attempted reading is kept (`attempts[{ordinal, structural_status, errors, positions,
token_alignment, k3}]`) with the selected one, and the whole record is written to
`/logs/verifier/fidelity/test_fidelity_kl.json` before any verdict.

**Measured twice before it fails**. A structurally valid sequence whose mean
k3 is over its band's T_b is re-sent exactly once and keeps the lower of its two valid readings, so it counts as
over only if both were; a band whose median is over its T_b is then drawn once more whole (every sequence again,
under the same rule) and fails only if both draws' medians are over. No other draw, seed or retry exists.

**Why every band on its own bar.** The four bands -- 512, 2,048, 8,192 and 20,480 tokens -- put different
mechanisms in play: at 512 every compressed position is visible and sparse selection is a no-op; at 2,048
and 8,192 the top-512 selection is active on both compressor ratios and the longer band spans prefill
chunks; at 20,480 the two-level indexer's candidate pool (2,048 blocks of 8 at layer 20) is smaller than
the stream, so candidate filtering is active. A fault confined to one mechanism moves one band, and a
median over the whole population is blind to that; a bar per band is not, and it is set from that band's own
oracle readings, so a band the oracle reads low is not held to the bar of the band it reads highest.

**Teacher-forced, by token ids, on the fresh path.** The prompt is sent as the recorded token ids,
because a retokenization that differs by one token compares two different sequences.
`SamplingParams.__post_init__` sets `skip_reading_prefix_cache = prompt_logprobs is not None`, so these
requests never read the prefix cache.
"""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

import pytest

from infb_verify import _trusted, serve

TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
X = json.loads((TESTS_DIR / "config.json").read_text())["infb"]
BARS = X.get("bars", {})
T_BY_BAND = {int(b): float(v) for b, v in (BARS.get("fidelity_T_per_band") or {}).items()}
DATA = json.loads((TESTS_DIR / "hidden" / "fidelity.json").read_text())
SEQUENCES = DATA["sequences"]


_FIDELITY = _trusted.load("fidelity/check.py")
RECORD = Path(os.environ.get("INFB_FIDELITY_RECORD_DIR", "/logs/verifier/fidelity")) / "test_fidelity_kl.json"


def _fetch(server):
    url = server.url_for("v1/completions")
    return lambda ids: _FIDELITY.request_prompt_logprobs(url, X["model_id"], ids)


def _persist(record: dict) -> None:
    """Write the whole record before any verdict is raised."""
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(record, indent=1, default=str))


def _status(rows: list[dict]) -> str:
    """Any structurally invalid reading is the candidate's. The server answered when this node started (`serve.down`),
    so a draw whose every failure is a transport failure is where the server stopped answering: the primary cause, a
    candidate failure, which `_primary_cause` names; later served nodes are blocked."""
    bad = [a for r in rows for a in r["attempts"] if a["structural_status"] != "valid"]
    return "candidate_fail" if bad else "pass"


def _primary_cause(rows: list[dict]) -> str | None:
    bad = [a for r in rows for a in r["attempts"] if a["structural_status"] != "valid"]
    return "the server stopped answering during this node" if bad and all(a["transport_failed"] for a in bad) else None


def test_fidelity_kl(server):
    """Every length band's median k3 against the independent reference is at most that band's T_b; every attempted
    reading structurally valid; each sequence re-sent at most once and each band drawn at most twice."""
    record: dict = {"node": "test_fidelity_kl", "reference": DATA["provenance"], "bars": T_BY_BAND}
    try:
        _FIDELITY.check_reference(SEQUENCES)
    except _FIDELITY.ReferenceFault as exc:
        record.update(status="reference_fault", error=str(exc))
        _persist(record)
        pytest.fail(f"reference_fault: {exc}")
    bands = sorted({s["band"] for s in SEQUENCES})
    assert T_BY_BAND and set(bands) <= set(T_BY_BAND), f"no T_b for bands {sorted(set(bands) - set(T_BY_BAND))}"
    if why := serve.down(server):
        record.update(status="blocked_by:server_unreachable", blocked=why)
        _persist(record)
        pytest.fail(serve.blocked_message(why))
    fetch = _fetch(server)

    first = {b: _FIDELITY.read_band(fetch, [s for s in SEQUENCES if s["band"] == b], T_BY_BAND[b]) for b in bands}
    per_seq = [row for b in bands for row in first[b][0]]
    record.update(first_draw=per_seq, status=_status(per_seq))
    if cause := _primary_cause(per_seq):
        record["primary_cause"] = cause
    _persist(record)
    print("INFB_FIDELITY " + json.dumps({"reference": DATA["provenance"], "sequences": per_seq}))
    _FIDELITY.assert_structural(per_seq)
    by_band = {b: first[b][1] for b in bands}
    print("INFB_FIDELITY_BANDS " + json.dumps(by_band))

    verdict: dict[int, dict] = {}
    for b in bands:
        t_b, median = T_BY_BAND[b], by_band[b]
        entry = {"T": t_b, "median": median, "pass": median <= t_b}  # the passing direction: NaN fails
        if not entry["pass"]:
            rows, median2 = _FIDELITY.read_band(fetch, [s for s in SEQUENCES if s["band"] == b], t_b)
            record.setdefault("redraws", {})[str(b)] = rows
            record["status"] = _status(rows) if _status(rows) != "pass" else record["status"]
            if cause := _primary_cause(rows):
                record["primary_cause"] = cause
            _persist(record)
            print("INFB_FIDELITY_REDRAW " + json.dumps({"band": b, "sequences": rows, "median": median2}))
            _FIDELITY.assert_structural(rows)
            entry.update({"median_redrawn": median2, "pass": median2 <= t_b})
        verdict[b] = entry
    failing = [b for b in bands if not verdict[b]["pass"]]
    record.update(verdict=verdict, status="candidate_fail" if failing else "pass")
    _persist(record)
    print("INFB_FIDELITY_VERDICT " + json.dumps(verdict))

    worst_seq = sorted(per_seq, key=lambda c: c["k3_mean"], reverse=True)[:3]
    assert not failing, (
        "the median k3 against the reference implementation is over its band's T_b on both draws of "
        + ", ".join(
            f"the {b}-token band ({verdict[b]['median']:.6g}, then {verdict[b]['median_redrawn']:.6g}, "
            f"over T_b {verdict[b]['T']:.6g})"
            for b in failing
        )
        + ". Every band: "
        + ", ".join(f"{b}:{by_band[b]:.6g} (T_b {T_BY_BAND[b]:.6g})" for b in bands)
        + "; the three worst sequences of the first draw are "
        + ", ".join(f"{c['id']} {c['k3_mean']:.6g}" for c in worst_seq)
    )
