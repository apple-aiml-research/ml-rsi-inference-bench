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

def right_shift_unsigned(x, shift):
    raise NotImplementedError('packed FP4 operation is not implemented')

def _compress_fp4(x):
    raise NotImplementedError('packed FP4 operation is not implemented')

def _compress_fourth(x):
    raise NotImplementedError('packed FP4 operation is not implemented')

def _pack_bits(x: torch.Tensor, mx_axis: int):
    raise NotImplementedError('packed FP4 operation is not implemented')

def _bf16_to_fp4e2m1(x):
    raise NotImplementedError('packed FP4 operation is not implemented')

def _bf16x2_to_fp4e2m1x2(x):
    raise NotImplementedError('packed FP4 operation is not implemented')

def _unpack_bits(x, mx_axis: int):
    raise NotImplementedError('packed FP4 operation is not implemented')

class HopperMXValueLayout(Layout):
    name: str = 'HOPPER_VALUE'

    def __init__(self, shape, mx_axis, mma_version=3):
        super().__init__(shape)
        assert mx_axis in range(len(shape))
        self.mx_axis = mx_axis
        self.mma_version = mma_version
        *self.leading_shape, self.K, self.N = shape

    def _maybe_mT(self, data):
        if self.mx_axis == len(self.leading_shape):
            return data.mT
        return data

    def swizzle_data(self, data):
        raise NotImplementedError('packed FP4 operation is not implemented')

    def unswizzle_data(self, data):
        raise NotImplementedError('packed FP4 operation is not implemented')

    def swizzle_block_shape(self, block_shape):
        return block_shape

@triton.jit
def _unshuffle_triton(x, mma_version: tl.constexpr):
    tl.static_assert(False, 'packed FP4 operation is not implemented')

@triton.jit
def _unpack_fp4_to_bf16_triton(x):
    tl.static_assert(False, 'packed FP4 operation is not implemented')

@triton.jit
def mxfp4_to_bf16_triton(x, scale, mx_axis: tl.constexpr):
    tl.static_assert(False, 'packed FP4 operation is not implemented')
