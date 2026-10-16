#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2018-2020 Philippe Tillet
# SPDX-FileCopyrightText: Copyright 2020-2022 OpenAI
# Vendored from https://github.com/triton-lang/triton/tree/8f9f695ea8fde23a0c7c88e4ab256634ca27789f/python/triton_kernels.
import torch
from dataclasses import dataclass

MAX_FINITE_FLOAT8E5 = 57344.0
MAX_FINITE_FLOAT8E4NV = 448.0
MAX_FINITE_FLOAT8E4B8 = 240.0


@dataclass(frozen=True)
class BaseFlexData:
    dtype: torch.dtype | None = None

    def view(self, x: torch.Tensor):
        if self.dtype is None:
            return x
        return x.view(self.dtype)

    def reinterpret(self, x):
        if self.dtype is None or x.dtype.itemsize > 1:
            return x
        return x.view(self.dtype)


@dataclass(frozen=True)
class InFlexData(BaseFlexData):
    scale: torch.Tensor | None = None

    @property
    def is_per_batch(self):
        return False if self.scale is None else len(self.scale) > 1


@dataclass(frozen=True)
class OutFlexData(BaseFlexData):
    expected_scale: torch.Tensor | None = None
    actual_scale: torch.Tensor | None = None
    checksum_scale: torch.Tensor | None = None

    def __iter__(self):
        yield self.expected_scale
        yield self.actual_scale
        yield self.checksum_scale
