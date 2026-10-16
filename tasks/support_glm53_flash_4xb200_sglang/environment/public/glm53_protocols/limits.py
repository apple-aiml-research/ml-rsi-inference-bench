#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The task's operation limits: the resource contract every check and measurement runs under (instruction, README)."""
from __future__ import annotations

STARTUP_S = 1750
HTTP_S = 900
INVOCATION_S = 1800
UNIT_FILE_S = 1800
ACTION_S = 300
TEARDOWN_S = 180
TEARDOWN_KILL_AFTER_S = 10

LIMITS = {"startup_s": STARTUP_S, "http_s": HTTP_S, "invocation_s": INVOCATION_S, "unit_file_s": UNIT_FILE_S,
          "action_s": ACTION_S, "teardown_s": TEARDOWN_S, "teardown_kill_after_s": TEARDOWN_KILL_AFTER_S}
