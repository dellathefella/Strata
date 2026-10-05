// src/kernels/sycl/native_router.cpp — SYCL port of src/kernels/cuda/native_router.cu
// (512-expert top-10 softmax router, llama.cpp-derived, pinned geometry).
//
// Adapted from topk-moe.cu/common.cuh in llama.cpp
// 3cf03257f219afbe7334045ff7c6a06ac68c627d; finite F32, 512-expert/10-output path.
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
#include "strata/kernels/native_router.hpp"

#include <cuda_runtime.h>  // sycl_compat shim

#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace strata::kernels {
namespace {
std::atomic<bool> enabled{false};

using sycl::local_accessor;
using sycl::nd_item;

inline sycl::queue& Q(void* stream) { return strata::sycl_compat::q_for(stream); }

// one work-group per token, 32 lanes (the CUDA block was dim3(32,8) with only
// row zero active; the SYCL group is the 32 active lanes). The selection
// butterflies carry (best, expert) PAIRS: both halves exchange through local
// memory with the identical xor pairing and tie-break (other > best ||
// (other == best && other_id < expert)).
void route_launch(const float* logits, int32_t* ids, float* weights, int n_tok, void* stream) {
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<float, 1> redB(sycl::range<1>(32), hnd);
        local_accessor<int, 1> redE(sycl::range<1>(32), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) n_tok * 32), sycl::range<1>(32)),
                         [=](nd_item<1> it) {
                             const int tk = (int) it.get_group(0);
                             const int lane = (int) it.get_local_id(0);
                             const float* lg = logits + (size_t) tk * 512;
                             int32_t* my_ids = ids + (size_t) tk * 10;
                             float* my_w = weights + (size_t) tk * 10;
                             float values[16];
#pragma unroll
                             for (int i = 0; i < 16; ++i) values[i] = lg[lane + i * 32];
                             float maximum = -INFINITY;
#pragma unroll
                             for (int i = 0; i < 16; ++i) maximum = sycl::fmax(maximum, values[i]);
                             redB[lane] = maximum;
                             it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
                             for (int o = 16; o > 0; o >>= 1) {
                                 const float other = redB[lane ^ o];
                                 it.barrier(sycl::access::fence_space::local_space);
                                 maximum = sycl::fmax(maximum, other);
                                 redB[lane] = maximum;
                                 it.barrier(sycl::access::fence_space::local_space);
                             }
                             float sum = 0.0f;
#pragma unroll
                             for (int i = 0; i < 16; ++i) {
                                 values[i] = sycl::exp(values[i] - maximum);
                                 sum += values[i];
                             }
                             redB[lane] = sum;
                             it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
                             for (int o = 16; o > 0; o >>= 1) {
                                 const float other = redB[lane ^ o];
                                 it.barrier(sycl::access::fence_space::local_space);
                                 sum += other;
                                 redB[lane] = sum;
                                 it.barrier(sycl::access::fence_space::local_space);
                             }
                             const float reciprocal = 1.0f / sum;
#pragma unroll
                             for (int i = 0; i < 16; ++i) {
                                 values[i] *= reciprocal;
                                 if (sycl::isnan(values[i])) values[i] = -FLT_MAX;
                             }
                             float selected = 0.0f, selected_sum = 0.0f;
                             for (int rank = 0; rank < 10; ++rank) {
                                 float best = values[0];
                                 int expert = lane;
#pragma unroll
                                 for (int i = 1; i < 16; ++i) {
                                     if (values[i] > best) {
                                         best = values[i];
                                         expert = lane + i * 32;
                                     }
                                 }
#pragma unroll
                                 for (int o = 16; o > 0; o >>= 1) {
                                     redB[lane] = best;
                                     redE[lane] = expert;
                                     it.barrier(sycl::access::fence_space::local_space);
                                     const float other = redB[lane ^ o];
                                     const int other_id = redE[lane ^ o];
                                     it.barrier(sycl::access::fence_space::local_space);
                                     if (other > best || (other == best && other_id < expert)) {
                                         best = other;
                                         expert = other_id;
                                     }
                                 }
                                 if ((expert & 31) == lane) {
                                     values[expert / 32] = -INFINITY;
                                     my_ids[rank] = expert;
                                     // Deliberately accumulate by WINNING EXPERT lane, not output rank.
                                     // Multiple selected experts in one lane add in selection order.
                                     selected_sum += best;
                                 }
                                 if (rank == lane) selected = best;
                             }
                             redB[lane] = selected_sum;
                             it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
                             for (int o = 16; o > 0; o >>= 1) {
                                 const float other = redB[lane ^ o];
                                 it.barrier(sycl::access::fence_space::local_space);
                                 selected_sum += other;
                                 redB[lane] = selected_sum;
                                 it.barrier(sycl::access::fence_space::local_space);
                             }
                             selected_sum = sycl::fmax(selected_sum, 6.103515625e-5f);
                             const float inverse_selected_sum = 1.0f / selected_sum;
                             if (lane < 10) my_w[lane] = selected * inverse_selected_sum;
                         });
    });
}

bool valid(const void* p, size_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(p);
    return p && address % 4 == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const auto ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}
}  // namespace

void native_router_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_router_enabled() { return enabled.load(std::memory_order_relaxed); }

void native_router_top10(const float* logits, int32_t* ids, float* weights, void* stream) {
    if (!stream || !valid(logits, 512 * 4) || !valid(ids, 10 * 4) || !valid(weights, 10 * 4) ||
        overlap(logits, 512 * 4, ids, 10 * 4) || overlap(logits, 512 * 4, weights, 10 * 4) ||
        overlap(ids, 10 * 4, weights, 10 * 4))
        throw std::invalid_argument("native router requires a stream, aligned spans, and disjoint outputs");
    route_launch(logits, ids, weights, 1, stream);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

void native_router_top10_multi(const float* logits, int32_t* ids, float* weights, int n_tok, void* stream) {
    if (!stream || n_tok < 1 || !valid(logits, (size_t) n_tok * 512 * 4) || !valid(ids, (size_t) n_tok * 10 * 4) ||
        !valid(weights, (size_t) n_tok * 10 * 4))
        throw std::invalid_argument("native router (multi) requires a stream and aligned [n,512]/[n,10] buffers");
    route_launch(logits, ids, weights, n_tok, stream);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

}  // namespace strata::kernels
