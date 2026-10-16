#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# One prepared Git root captures both engine and kernel source. A capture error
# is loud, and cannot leave an old successful artifact in place.
set -euo pipefail
exec /usr/bin/python3 -I -S /opt/infb/capture.py
