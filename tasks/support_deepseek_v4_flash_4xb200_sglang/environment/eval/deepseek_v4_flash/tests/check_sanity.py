#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Upstream's twelve server-sanity checks (`ServerSanityMixin` in `server_sanity_kit.py`: health, generation,
short correctness, streaming, concurrency, long prompt, determinism and the rest), run unchanged against a server that
is already serving -- the one `serve.py` started -- instead of launching another. The acceptance test runs the same
methods after its own launch; this lets them run in sequence with the other checks on one server."""
from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import serve  # noqa: E402 - the acceptance class, with the adapter applied


def sanity_methods(cls) -> list[str]:
    import sglang.test.kits.server_sanity_kit as kit  # noqa: PLC0415 - supplied by the adapter until the tree has it

    return sorted(n for n in dir(kit.ServerSanityMixin) if n.startswith("test_") and hasattr(cls, n))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="upstream's server-sanity checks against a running server")
    ap.add_argument("--base-url", required=True)
    args = ap.parse_args(argv)
    base = serve.acceptance_class()
    import requests  # noqa: PLC0415
    import sglang.test.kits.server_sanity_kit as kit  # noqa: PLC0415 - supplied by the adapter until the tree has it

    limit = serve.LIMITS["http_request"]["limit_s"]
    kit._REQUEST_TIMEOUT = limit  # noqa: SLF001 - the kit's module constant

    class OnRunningServer(base):
        @classmethod
        def setUpClass(cls):
            from sglang.test.test_utils import try_cached_model  # noqa: PLC0415 - as the class's own setUpClass

            cls.base_url = args.base_url
            cls.model = try_cached_model(sys.modules["deepseek_v4_flash_eval_acceptance"].MODEL)

        @classmethod
        def tearDownClass(cls):
            pass

        def test_health(self):
            self.assertEqual(requests.get(self.base_url + "/health", timeout=limit).status_code, 200)

        def test_health_generate(self):
            self.assertEqual(requests.get(self.base_url + "/health_generate", timeout=limit).status_code, 200)

    names = sanity_methods(base)
    suite = unittest.TestSuite(OnRunningServer(n) for n in names)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(f"server sanity: {result.testsRun - len(result.failures) - len(result.errors)} of {len(names)} passed")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
