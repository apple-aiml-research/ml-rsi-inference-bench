#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2018-2020 Philippe Tillet
# SPDX-FileCopyrightText: Copyright 2020-2022 OpenAI
# Vendored from https://github.com/triton-lang/triton/tree/8f9f695ea8fde23a0c7c88e4ab256634ca27789f/python/triton_kernels.
# proton options

import os

_launch_metadata_allow_sync = None


def launch_metadata_allow_sync():
    global _launch_metadata_allow_sync
    if _launch_metadata_allow_sync is None:
        _launch_metadata_allow_sync = not (os.getenv("PROTON_LAUNCH_METADATA_NOSYNC") == "1")
    return _launch_metadata_allow_sync


def set_launch_metadata_allow_sync(allow_sync: bool):
    global _launch_metadata_allow_sync
    _launch_metadata_allow_sync = allow_sync
