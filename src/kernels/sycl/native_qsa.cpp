// src/kernels/sycl/native_qsa.cpp — SYCL port of src/kernels/cuda/native_qsa.cu
// (llama.cpp-derived rms norm + output gate for the QSA attention path).
//
// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{norm.cu,common.cuh,unary.cu}.
//
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
#include "strata/kernels/native_qsa.hpp"

#include <cuda_runtime.h>  // sycl_compat shim

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {
std::atomic<bool> enabled{false};

using sycl::local_accessor;
using sycl::nd_item;

inline sycl::queue& Q(void* stream) { return strata::sycl_compat::q_for(stream); }

// the CUDA norm's two-level reduction: strided partials, warp xor-sum,
// sums[32], then a second warp xor-sum over the per-warp partials. The
// butterflies become per-32-lane local trees with the identical pairing.
template <int BlockSize>
void norm_launch(const float* input, const float* gamma, float* output, int n_cols, int n_rows, float eps,
                 void* stream) {
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<float, 1> sums(sycl::range<1>(32), hnd);
        local_accessor<float, 1> red(sycl::range<1>(BlockSize), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) n_rows * BlockSize),
                                           sycl::range<1>(BlockSize)),
                         [=](nd_item<1> it) {
                             const int tid = (int) it.get_local_id(0);
                             const int lane = tid & 31;
                             const size_t row_offset = (size_t) it.get_group(0) * n_cols;
                             const float* in = input + row_offset;
                             float* out = output + row_offset;
                             float partial = 0.0f;
                             for (int col = tid; col < n_cols; col += BlockSize) {
                                 const float x = in[col];
                                 partial += x * x;
                             }
                             red[tid] = partial;
                             it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
                             for (int o = 16; o > 0; o >>= 1) {
                                 const float add = red[tid ^ o];
                                 it.barrier(sycl::access::fence_space::local_space);
                                 partial += add;
                                 red[tid] = partial;
                                 it.barrier(sycl::access::fence_space::local_space);
                             }
                             if (lane == 0) sums[tid / 32] = partial;
                             it.barrier(sycl::access::fence_space::local_space);
                             partial = lane < BlockSize / 32 ? sums[lane] : 0.0f;
                             red[tid] = partial;
                             it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
                             for (int o = 16; o > 0; o >>= 1) {
                                 const float add = red[tid ^ o];
                                 it.barrier(sycl::access::fence_space::local_space);
                                 partial += add;
                                 red[tid] = partial;
                                 it.barrier(sycl::access::fence_space::local_space);
                             }
                             const float mean = partial / n_cols;
                             const float scale = sycl::rsqrt(mean + eps);
                             for (int col = tid; col < n_cols; col += BlockSize)
                                 out[col] = scale * in[col] * gamma[col];
                         });
    });
}

void gate_launch(const float* attn, const float* q_full, float* output, int n_head, int head_dim, void* stream) {
    const size_t count = (size_t) n_head * head_dim;
    Q(stream).parallel_for(count, [=](size_t i) {
        const size_t head = i / head_dim, channel = i % head_dim;
        const float raw = q_full[head * 2 * head_dim + head_dim + channel];
        const float sigmoid = 1.0f / (1.0f + sycl::exp(-raw));
        output[i] = attn[i] * sigmoid;
    });
}

std::size_t elements(int cols, int rows) {
    if (cols <= 0 || rows <= 0 || std::uint64_t(cols) * rows > std::uint64_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("native QSA requires positive bounded dimensions");
    return std::size_t(cols) * rows;
}
bool valid(const void* ptr, std::size_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(ptr);
    return ptr && address % 4 == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, std::size_t an, const void* b, std::size_t bn) {
    const auto ap = reinterpret_cast<std::uintptr_t>(a), bp = reinterpret_cast<std::uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}
void buffers(const float* input, std::size_t in_bytes, const float* weight, std::size_t weight_bytes, float* output,
             void* stream) {
    if (!stream || !valid(input, in_bytes) || !valid(weight, weight_bytes) || !valid(output, in_bytes) ||
        overlap(input, in_bytes, weight, weight_bytes) || overlap(output, in_bytes, weight, weight_bytes) ||
        (input != output && overlap(input, in_bytes, output, in_bytes)))
        throw std::invalid_argument(
            "native QSA requires a stream, aligned spans, and disjoint buffers or exact input/output alias");
}
void check_launch() {
    const auto result = cudaGetLastError();
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("native QSA launch: ") + cudaGetErrorString(result));
}
}  // namespace

void native_qsa_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_qsa_enabled() { return enabled.load(std::memory_order_relaxed); }

void native_qsa_rms_norm_weighted(const float* input, const float* gamma, float* output, int n_cols, int n_rows,
                                  float epsilon, void* stream) {
    const auto count = elements(n_cols, n_rows);
    (void) count;
    if (!std::isfinite(epsilon) || epsilon < 0.0f)
        throw std::invalid_argument("native QSA requires finite nonnegative epsilon");
    buffers(input, count * 4, gamma, std::size_t(n_cols) * 4, output, stream);
    if (n_cols < 1024)
        norm_launch<256>(input, gamma, output, n_cols, n_rows, epsilon, stream);
    else
        norm_launch<1024>(input, gamma, output, n_cols, n_rows, epsilon, stream);
    check_launch();
}

void native_qsa_gate_apply(const float* attn, const float* q_full, float* output, int n_head, int head_dim,
                           void* stream) {
    const auto count = elements(head_dim, n_head);
    buffers(attn, count * 4, q_full, count * 8, output, stream);
    gate_launch(attn, q_full, output, n_head, head_dim, stream);
    check_launch();
}

}  // namespace strata::kernels
