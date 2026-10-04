// src/kernels/sycl/s2_gemv_quads.cpp — SYCL port of src/kernels/cuda/s2_gemv_quads.cu.
// S2 GEMV with one code-byte load and one 64-bit activation load per quad of
// four elements; four accumulators; deterministic shared-memory tree reduce
// (same pairing as CUDA). uint2/__half2 loads -> direct sycl::half2 loads
// (same values, alignment guaranteed by the format).
#include "strata/kernels/s_gemv.hpp"

#include <cuda_fp16.h>     // sycl_compat shim
#include <cuda_runtime.h>  // sycl_compat shim

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {
constexpr int QK_S2 = 64;
}

void s2_gemv_quads(const uint16_t* x, const uint8_t* codes, const float* scales, float* y,
                   int64_t n_in, int64_t n_out, int threads_per_row) {
    if (n_in <= 0 || n_out <= 0) return;
    if (n_in % 4 != 0) {
        std::fprintf(stderr, "s2_gemv_quads: n_in %lld is not a multiple of 4\n", (long long)n_in);
        std::exit(1);
    }
    const int tpr = threads_per_row;
    auto& q = strata::sycl_compat::default_queue();
    try {
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> partial(sycl::range<1>((size_t)tpr), h);
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t)n_out * tpr),
                                             sycl::range<1>((size_t)tpr)),
                           [=](sycl::nd_item<1> it) {
                               const long long o = (long long)it.get_group(0);
                               if (o >= n_out) return;
                               const int tid = (int)it.get_local_id(0);

                               const long long n_quads = n_in / 4;
                               const uint8_t* c = codes + o * n_quads;
                               const float* s = scales + o * (n_in / QK_S2);

                               float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                               for (long long qi = tid; qi < n_quads; qi += threads_per_row) {
                                   const uint8_t byte = c[qi];  // ONE load for four codes
                                   const float d = s[qi >> 4];  // (q*4) >> 6
                                   const sycl::half2 h01 =
                                       *reinterpret_cast<const sycl::half2*>(x + qi * 4);
                                   const sycl::half2 h23 =
                                       *reinterpret_cast<const sycl::half2*>(x + qi * 4 + 2);
                                   const float w0 = (float)((int)(byte & 3) - 1) * d;
                                   const float w1 = (float)((int)((byte >> 2) & 3) - 1) * d;
                                   const float w2 = (float)((int)((byte >> 4) & 3) - 1) * d;
                                   const float w3 = (float)((int)((byte >> 6) & 3) - 1) * d;
                                   a0 += w0 * __low2float(h01);
                                   a1 += w1 * __high2float(h01);
                                   a2 += w2 * __low2float(h23);
                                   a3 += w3 * __high2float(h23);
                               }
                               partial[tid] = (a0 + a1) + (a2 + a3);
                               it.barrier();
                               for (int step = threads_per_row / 2; step > 0; step >>= 1) {
                                   if (tid < step) partial[tid] += partial[tid + step];
                                   it.barrier();
                               }
                               if (tid == 0) y[o] = partial[0];
                           });
        }).wait_and_throw();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "s2_gemv_quads: %s\n", e.what());
        std::exit(1);
    }
}

}  // namespace strata::kernels
