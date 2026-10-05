// src/kernels/sycl/native_moe.cpp — SYCL port of src/kernels/cuda/native_moe.cu
// (weighted expert-output reduction + optional shared expert add).
//
// Arithmetic adapted from the MIT-licensed pinned ggml CUDA
// moe-weighted-reduction.cu at 3cf03257f219afbe7334045ff7c6a06ac68c627d.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "strata/kernels/native_moe.hpp"

#include <cuda_runtime.h>  // sycl_compat shim

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strata::kernels {
namespace {
std::atomic<bool> enabled{false};

inline sycl::queue& Q(void* stream) { return strata::sycl_compat::q_for(stream); }

void combine_launch(const float* parts, const float* weights, const float* shared, float* output, int64_t n_embd,
                    int k, int n_tok, void* stream) {
    Q(stream).parallel_for(sycl::range<2>((size_t) n_tok, (size_t) n_embd), [=](sycl::id<2> id) {
        const size_t tk = id[0];
        const size_t col = id[1];
        const float* p = parts + tk * (size_t) k * n_embd;
        const float* w = weights + tk * (size_t) k;
        const float* sh = shared ? shared + tk * (size_t) n_embd : nullptr;
        float sum = p[col] * w[0];
        for (int expert = 1; expert < k; ++expert) {
            sum += p[(size_t) expert * n_embd + col] * w[expert];
        }
        if (sh) sum += sh[col];
        output[tk * (size_t) n_embd + col] = sum;
    });
}

bool valid_span(const void* p, size_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(p);
    return p && address % alignof(float) == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const auto ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}
}  // namespace

void native_moe_combine_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_moe_combine_enabled() { return enabled.load(std::memory_order_relaxed); }

void native_moe_combine(const float* parts, const float* weights, const float* shared, float* output, int64_t n_embd,
                        int64_t k, void* stream) {
    if (!stream || n_embd <= 0 || n_embd > std::numeric_limits<int>::max() || k < 1 || k > 15)
        throw std::invalid_argument("native MoE combine requires a stream, positive width and 1..15 experts");
    const size_t row_bytes = size_t(n_embd) * sizeof(float);
    const size_t part_bytes = row_bytes * size_t(k), weight_bytes = size_t(k) * sizeof(float);
    if (!valid_span(parts, part_bytes) || !valid_span(weights, weight_bytes) || !valid_span(output, row_bytes) ||
        (shared && !valid_span(shared, row_bytes)) || overlap(output, row_bytes, parts, part_bytes) ||
        overlap(output, row_bytes, weights, weight_bytes) || (shared && overlap(output, row_bytes, shared, row_bytes)))
        throw std::invalid_argument("native MoE combine requires aligned spans and disjoint output");
    combine_launch(parts, weights, shared, output, n_embd, (int) k, 1, stream);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

void native_moe_combine_multi(const float* parts, const float* weights, const float* shared, float* output,
                              int64_t n_embd, int64_t k, int n_tok, void* stream) {
    if (!stream || n_embd <= 0 || k < 1 || k > 15 || n_tok < 1)
        throw std::invalid_argument("native MoE combine (multi) requires a stream, width, 1..15 experts, tokens");
    combine_launch(parts, weights, shared, output, n_embd, (int) k, n_tok, stream);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

}  // namespace strata::kernels
