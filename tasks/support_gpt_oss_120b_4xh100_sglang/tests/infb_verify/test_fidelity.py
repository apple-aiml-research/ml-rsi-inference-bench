#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The scored fidelity node: does the candidate's served model agree with the model author's own implementation?"""
from __future__ import annotations

import json
import math
import os
import pathlib
import statistics
import unittest

import requests

FIDELITY_DIR = pathlib.Path(os.environ.get("INFB_FIDELITY_DIR", "/tests/hidden/fidelity"))
CONFIG = pathlib.Path(os.environ.get("INFB_CONFIG", "/tests/config.json"))
SKIP_POSITIONS = 32
LONG_WINDOW = 1024


def _config() -> dict:
    return json.loads(CONFIG.read_text())


def _sequences(kind: str) -> list[dict]:
    out = []
    for p in sorted(FIDELITY_DIR.glob("*.json")):
        if p.name == "index.json":
            continue
        d = json.loads(p.read_text())
        if d["kind"] == kind:
            assert d.get("reference"), f"{d['id']} has no reference log-probabilities"
            out.append(d)
    return out


# The image keeps this byte-identical to the editable public protocol.
import sys
sys.path.insert(0, "/tests/public_eval")
from gpt_oss.fidelity import (
    candidate_logprobs, candidate_logprobs_windowed, abs_r_median, k3,
)


class TestFidelity(unittest.TestCase):
    """Launches the candidate's server once and scores every hidden sequence against the reference."""

    @classmethod
    def setUpClass(cls):
        from sglang.srt.utils import kill_process_tree
        from sglang.test.test_utils import (
            DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            DEFAULT_URL_FOR_TEST,
            popen_launch_server,
        )

        cls._kill = kill_process_tree
        cls.base_url = DEFAULT_URL_FOR_TEST
        cfg = _config()["infb"]
        cls.process = popen_launch_server(
            cfg.get("model_repo", "openai/gpt-oss-120b"),
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--tp", "4", "--cuda-graph-max-bs", "200", "--mem-fraction-static", "0.93"],
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            cls._kill(cls.process.pid)

    @classmethod
    def _sequences_by_band(cls) -> dict:
        """The hidden set, by band, in the order of the files; the bands and their sizes are the config's."""
        if getattr(cls, "_by_band", None) is None:
            bands = _config()["infb"]["bars"]["fidelity_bands"]
            hidden = _sequences("hidden")
            by_band = {b: [s for s in hidden if s["id"].split("_")[0] == b] for b in bands}
            for b, spec in bands.items():
                assert len(by_band[b]) == spec["sequences"], \
                    f"the {b} band is {spec['sequences']} sequences, found {len(by_band[b])}"
            assert sum(len(v) for v in by_band.values()) == len(hidden), "a hidden sequence is in no band"
            cls._by_band = by_band
        return cls._by_band

    @classmethod
    def _reading(cls, seq: dict, band: str, k: int) -> list:
        """The k-th reading of one sequence's served log-probabilities (0 the first), each fetched once and shared by
        both statistics: readings 1 and 3 are the per-sequence re-sends, reading 2 the band's re-draw."""
        cache = cls.__dict__.get("_readings")
        if cache is None:
            cache = cls._readings = {}
        got = cache.setdefault(seq["id"], [])
        while len(got) <= k:
            skip = _config()["infb"]["bars"]["fidelity_bands"][band]["skip"]
            if band == "long":
                got.append(candidate_logprobs_windowed(cls.base_url, seq["token_ids"], skip, seq_id=seq["id"]))
            else:
                got.append(candidate_logprobs(cls.base_url, seq["token_ids"], seq_id=seq["id"]))
        return got[k]

    def _draw(self, band: str, stat, T: float, first: int) -> tuple[float, dict, dict]:
        """One draw of a band: every sequence's statistic from reading `first`, and a sequence at or over T re-sent
        once (reading `first` + 1), counting as over only if both readings are. Returns the median, the values
        used and the re-sent pairs."""
        skip = _config()["infb"]["bars"]["fidelity_bands"][band]["skip"]
        values, resent = {}, {}
        for seq in self._sequences_by_band()[band]:
            ref = seq["reference"]["logprobs"]
            v = stat(self._reading(seq, band, first), ref, skip)
            if not (v < T):
                v2 = stat(self._reading(seq, band, first + 1), ref, skip)
                resent[seq["id"]] = (v, v2)
                v = min(v, v2)
            values[seq["id"]] = v
        assert all(math.isfinite(v) or v == math.inf for v in values.values()), \
            f"a per-sequence statistic is NaN, so the median has no defined order: {values}"
        return statistics.median(values.values()), values, resent

    def _grade(self, name: str, stat, bar: str) -> None:
        """Every band's median against that band's own bar, re-drawn once whole before a band fails; the verdict is
        the conjunction over bands."""
        bands = _config()["infb"]["bars"]["fidelity_bands"]
        failed = {}
        for band, spec in bands.items():
            T = spec[bar]
            median, values, resent = self._draw(band, stat, T, 0)
            draws = [median]
            if not (median < T):
                median2, values2, resent2 = self._draw(band, stat, T, 2)
                draws.append(median2)
                resent.update({f"{k} (draw 2)": v for k, v in resent2.items()})
                if not (median2 < T):
                    failed[band] = draws
                values = values2 if median2 < T else values
            worst = max(values.items(), key=lambda kv: kv[1])
            print(f"[INFB] {name}[{band}]={draws[-1]:.6g} T={T:.6g} ratio={draws[-1] / T:.4g} n={len(values)} "
                  f"draws={[round(d, 6) for d in draws]} worst={worst[0]}:{worst[1]:.6g} "
                  f"resent={json.dumps({k: [round(x, 6) for x in v] for k, v in resent.items()})}")
            print(f"[INFB] {name}_per_sequence[{band}]={json.dumps({k: round(v, 9) for k, v in values.items()})}")
        self.assertEqual(failed, {}, f"{name}: the band median is not below its bar in every band, in both draws: "
                                     f"{failed}; the bars are {json.dumps({b: s[bar] for b, s in bands.items()})}")

    def test_fidelity_abs_r_median(self):
        self._grade("fidelity_abs_r_median", abs_r_median, "T_r")

    def test_fidelity_kl_median(self):
        self._grade("fidelity_kl_median", k3, "T")


if __name__ == "__main__":
    unittest.main()
