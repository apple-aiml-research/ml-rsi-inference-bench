#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The retained MTP3 native production/state gate, on its own diagnostic launch.

Its native interventions target the owners the MTP3 component record bound; without a complete
record there is nothing to target, and the node reports itself blocked rather than failed.
"""
import json

import pytest

from infb_verify import expanded, serve


@pytest.fixture(scope="module", params=["mtp3"])
def production_server(request):
    recipe = request.param
    directory = expanded.LOGS / recipe
    try:
        mapping_rows = expanded.production_mapping(recipe)
    except Exception as exc:  # noqa: BLE001 - the prerequisite record is missing or incomplete
        pytest.fail(f"blocked_by:infb_verify.test_mtp component record: {exc!r}"[:2000])
    mapping = directory / "native-mapping.json"
    expanded.write(mapping, mapping_rows)
    log = directory / "production-launch.log"
    for server in serve.launch(extra_args=serve.MTP_ARGS, launch_log=log, native_mapping=mapping):
        yield recipe, server, log


def test_production_and_state(production_server):
    recipe, server, log = production_server
    configuration = serve.served_configuration(log.read_text(), recipe)
    assert not configuration["departures"], configuration
    components = json.loads((expanded.LOGS / recipe / "components.json").read_text())
    result = expanded.production(server, recipe, components, log)
    assert result["status"] == "complete", result
