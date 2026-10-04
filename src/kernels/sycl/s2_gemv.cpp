// src/kernels/sycl/s2_gemv.cpp — SYCL port of src/kernels/cuda/s2_gemv.cu.
// Naive S2 GEMV, one work-item per output row; arithmetic transcribed 1:1
// (code-1 in the integer domain, then group scale, then activation).
#include "strata/kernels/s2_gemv.hpp"

#include <cuda_fp16.h>     // sycl_compat shim
#include <cuda_runtime.h>  // sycl_compat shim

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {
constexpr int QK = 64;
}

void s2_gemv(const uint16_t* x, const uint8_t* codes, const float* scales, float* y, int64_t n_in,
             int64_t n_out) {
    if (n_in <= 0 || n_out <= 0) return;
    if (n_in % QK != 0) {
        std::fprintf(stderr, "s2_gemv: n_in %lld is not a multiple of %d\n", (long long)n_in, QK);
        std::exit(1);
    }
    auto& q = strata::sycl_compat::default_queue();
    try {
        q.parallel_for(sycl::range<1>((size_t)n_out), [=](sycl::id<1> oid) {
            const long long o = oid[0];
            const long long nb = n_in / QK;
            const uint8_t* c = codes + o * nb * (QK / 4);
            const float* s = scales + o * nb;

            float acc = 0.0f;
            for (long long b = 0; b < nb; ++b) {
                const float d = s[b];
                const uint8_t* cb = c + b * (QK / 4);
                const uint16_t* xb = x + b * QK;
                for (int j = 0; j < QK; ++j) {
                    const int code = (cb[j >> 2] >> ((j & 3) * 2)) & 0x03;
                    acc += (float)(code - 1) * d * __half2float(__ushort_as_half(xb[j]));
                }
            }
            y[o] = acc;
        }).wait_and_throw();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "s2_gemv: %s\n", e.what());
        std::exit(1);
    }
}

}  // namespace strata::kernels
