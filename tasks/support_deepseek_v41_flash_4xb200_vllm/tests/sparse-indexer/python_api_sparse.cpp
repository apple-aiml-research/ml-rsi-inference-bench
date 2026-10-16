//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
// Adapted from https://github.com/deepseek-ai/DeepGEMM.
// Modified by Apple Inc. in 2026.
// The narrow extension of the supplied MXFP4 sparse MQA-logits runtime (vllm.third_party.deep_gemm_v41_sparse).
//
// It registers DeepJIT's module-local runtime, the runtime's configuration APIs and exactly four attention
// operations, with the keyword names and defaults of csrc/apis/attention.hpp's own registrations. Nothing else
// in this source tree is registered here: its GEMM, MoE, hyper-connection and layout APIs stay unregistered,
// and the engine's ordinary DeepGEMM provider (vllm.third_party.deep_gemm) is a separate build.
#include <pybind11/pybind11.h>
#include <torch/python.h>

#include <deep_jit/backend/cuda/backend.hpp>
#include <deep_jit/python_api.hpp>

#include "apis/config.hpp"
#include "apis/attention.hpp"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME _C
#endif

// ReSharper disable once CppParameterMayBeConstPtrOrRef
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "MXFP4 sparse MQA-logits runtime (four attention operations)";

    // Register JIT objects
    deep_jit::register_python_api(m, deep_gemm::jit);

    // Register config APIs
    deep_gemm::config::register_apis(m);

    // The four sparse MQA-logits operations
    m.def("get_sparse_mqa_logits_metadata", &deep_gemm::attention::get_sparse_mqa_logits_metadata,
          py::arg("cu_seq_len_k_start"), py::arg("cu_seq_len_k_end"), py::arg("num_kv_tokens"),
          py::arg("sparse_kv_block_indices"), py::arg("qk_dtype"), py::arg("sparse_block_kv"),
          py::arg("use_unaligned_ks") = false);
    m.def("get_paged_sparse_mqa_logits_metadata", &deep_gemm::attention::get_paged_sparse_mqa_logits_metadata,
          py::arg("context_lens"), py::arg("block_table"), py::arg("indices"), py::arg("page_kv"),
          py::arg("sparse_kv_block_indices"), py::arg("qk_dtype"), py::arg("sparse_block_kv"));
    m.def("fp8_fp4_sparse_mqa_logits", &deep_gemm::attention::fp8_fp4_sparse_mqa_logits,
          py::arg("q"), py::arg("kv"), py::arg("weights"), py::arg("metadata"),
          py::arg("num_max_sparse_blocks"), py::arg("sparse_block_kv"),
          py::arg("use_unaligned_ks") = false);
    m.def("fp8_fp4_paged_sparse_mqa_logits", &deep_gemm::attention::fp8_fp4_paged_sparse_mqa_logits,
          py::arg("q"), py::arg("kv_cache"), py::arg("weights"), py::arg("metadata"),
          py::arg("num_max_sparse_blocks"), py::arg("sparse_block_kv"));
}
