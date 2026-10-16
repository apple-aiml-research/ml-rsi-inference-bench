#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Runtime setup for standalone native-operator checks, without loading an engine."""


def initialize_triton_allocator():
    """Supply ordinary aligned scratch for kernels outside the server's compiler setup."""
    import torch
    import triton

    def allocate(size, alignment, stream):
        # Allocation and launch must use the same stream. The returned view retains its
        # backing tensor; over-allocation also honors alignments above PyTorch's minimum.
        with torch.cuda.stream(torch.cuda.ExternalStream(stream)):
            backing = torch.empty(size + alignment, dtype=torch.uint8, device='cuda')
            offset = (-backing.data_ptr()) % alignment
            return backing[offset:offset + size]

    triton.set_allocator(allocate)
