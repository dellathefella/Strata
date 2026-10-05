// src/kernels/sycl/native_gr_norm.cpp — SYCL port of src/kernels/cuda/native_gr_norm.cu.
// Pinned block_reduce<SUM> structure: per-chunk xor butterfly, sums[] handoff,
// then ONE xor tree over the chunk sums whose result every thread reads (the
// CUDA version has every warp repeat that tree; all repetitions are bit-wise
// identical, so a single shared tree + broadcast is the same numbers).
#include "strata/kernels/native_gr_norm.hpp"

#include <cuda_runtime.h>  // sycl_compat shim

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

template <int BlockSize>
void launch_weighted_rms_norm(sycl::queue& q, const float* input, const float* gamma, float* output,
                              int n_cols, float epsilon, int n_rows) {
    constexpr int NW = BlockSize / 32;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> red(sycl::range<1>(BlockSize), h);
        sycl::local_accessor<float, 1> sums(sycl::range<1>(32), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t)n_rows * BlockSize),
                                         sycl::range<1>(BlockSize)),
                       [=](sycl::nd_item<1> it) {
            const int tid = (int)it.get_local_id(0);
            const int chunk = tid / 32, lane = tid % 32;
            const size_t row_offset = (size_t)it.get_group(0) * n_cols;
            const float* in = input + row_offset;
            const float* g = gamma + row_offset;
            float* out = output + row_offset;

            float partial = 0.0f;
            for (int col = tid; col < n_cols; col += BlockSize) {
                const float value = in[col];
                partial += value * value;
            }
            // per-chunk xor butterfly (same pairing as norm_warp_sum)
            red[tid] = partial;
            for (int off = 16; off > 0; off >>= 1) {
                it.barrier();
                if ((lane & off) == 0) red[chunk * 32 + lane] += red[chunk * 32 + lane + off];
            }
            if (lane == 0) sums[chunk] = red[chunk * 32];
            it.barrier();
            // one tree over sums[0..NW-1]; every thread reads the result
            if (chunk == 0) red[lane] = (lane < NW) ? sums[lane] : 0.0f;
            for (int off = 16; off > 0; off >>= 1) {
                it.barrier();
                if (chunk == 0 && (lane & off) == 0) red[lane] += red[lane + off];
            }
            it.barrier();
            const float total = red[0];

            const float mean = total / n_cols;
            const float scale = sycl::rsqrt(mean + epsilon);
            for (int col = tid; col < n_cols; col += BlockSize)
                out[col] = scale * in[col] * g[col];
        });
    });
}

void check_pointer(const void* p) {
    if (!p || reinterpret_cast<std::uintptr_t>(p) % alignof(float) != 0)
        throw std::invalid_argument(
            "native GR RMSNorm requires non-null four-byte aligned pointers");
}

}  // namespace

void native_gr_rms_norm_weighted(const float* input, const float* gamma, float* output, int n_cols,
                                 int n_rows, float epsilon, void* stream) {
    if (n_cols <= 0 || n_rows <= 0 || !std::isfinite(epsilon) || epsilon < 0.0f)
        throw std::invalid_argument(
            "native GR RMSNorm requires positive dimensions and finite nonnegative epsilon");
    check_pointer(input);
    check_pointer(gamma);
    check_pointer(output);
    auto& q = strata::sycl_compat::q_for(stream);
    try {
        if (n_cols < 1024)
            launch_weighted_rms_norm<256>(q, input, gamma, output, n_cols, epsilon, n_rows);
        else
            launch_weighted_rms_norm<1024>(q, input, gamma, output, n_cols, epsilon, n_rows);
    } catch (const sycl::exception& e) {
        throw std::runtime_error(std::string("native GR RMSNorm launch: ") + e.what());
    }
}

}  // namespace strata::kernels
