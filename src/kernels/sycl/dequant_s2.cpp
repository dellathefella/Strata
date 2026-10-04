// src/kernels/sycl/dequant_s2.cpp — SYCL port of src/kernels/cuda/dequant_s2.cu.
// The S2 (Q2_0) scalar dequantizer: (code - 1) in the integer domain, then ONE
// multiply by the group scale — transcribed 1:1.
#include "strata/kernels/dequant_s2.hpp"

#include <cuda_runtime.h>  // sycl_compat shim

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {
constexpr int QK = 64;  // S2 group = Q2_0 block = 64 elements
constexpr int CODES_PER_BYTE = 4;
}  // namespace

void dequant_s2(const uint8_t* codes, const float* scales, float* out, int64_t n_blocks) {
    if (n_blocks <= 0) return;
    auto& q = strata::sycl_compat::default_queue();
    try {
        q.parallel_for(sycl::range<1>((size_t)n_blocks), [=](sycl::id<1> bid) {
            const long long b = bid[0];
            const float d = scales[b];
            const uint8_t* c = codes + b * (QK / CODES_PER_BYTE);
            float* y = out + b * QK;
            for (int j = 0; j < QK; ++j) {
                const int code = (c[j / CODES_PER_BYTE] >> ((j % CODES_PER_BYTE) * 2)) & 0x03;
                y[j] = (float)(code - 1) * d;
            }
        }).wait_and_throw();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "dequant_s2: %s\n", e.what());
        std::exit(1);
    }
}

}  // namespace strata::kernels
