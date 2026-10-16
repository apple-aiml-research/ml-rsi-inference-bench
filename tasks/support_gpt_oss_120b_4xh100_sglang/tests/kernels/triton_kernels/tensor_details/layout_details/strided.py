#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2018-2020 Philippe Tillet
# SPDX-FileCopyrightText: Copyright 2020-2022 OpenAI
# Vendored from https://github.com/triton-lang/triton/tree/8f9f695ea8fde23a0c7c88e4ab256634ca27789f/python/triton_kernels.
from .base import Layout


class StridedLayout(Layout):
    name: str = None

    def __init__(self, shape) -> None:
        super().__init__(shape)

    def swizzle_data(self, data):
        return data

    def unswizzle_data(self, data):
        return data

    def swizzle_block_shape(self, block_shape):
        return block_shape
