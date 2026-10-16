#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2018-2020 Philippe Tillet
# SPDX-FileCopyrightText: Copyright 2020-2022 OpenAI
# Adapted from https://github.com/triton-lang/triton/tree/8f9f695ea8fde23a0c7c88e4ab256634ca27789f/python/triton_kernels.
# Modified by Apple Inc. in 2026.
import torch
import triton
import triton.language as tl
from .base import Layout

class HopperMXScaleLayout(Layout):
    name: str = 'HOPPER_SCALE'

    def __init__(self, shape, mx_axis, num_warps=8) -> None:
        assert num_warps & num_warps - 1 == 0, 'warps_n must be a power of 2'
        super().__init__(shape)
        self.mx_axis = mx_axis
        self.num_warps = num_warps
        *self.leading_shape, _, _ = shape

    def _maybe_mT(self, data):
        if self.mx_axis == len(self.leading_shape):
            return data.contiguous().mT
        return data

    def swizzle_data(self, data):
        raise NotImplementedError('packed FP4 operation is not implemented')

    def unswizzle_data(self, data):
        raise NotImplementedError('packed FP4 operation is not implemented')

    def swizzle_block_shape(self, block_shape):
        return block_shape

@triton.jit
def unswizzle_mxfp4_scale_hopper(x, mx_axis: tl.constexpr, num_warps: tl.constexpr):
    tl.static_assert(False, 'packed FP4 operation is not implemented')
