// src/kernels/sycl/s2_gemv_fast.cpp — SYCL port of src/kernels/cuda/s2_gemv_fast.cu.
// S2 GEMV with the 256x4 code LUT (CUDA __constant__ -> a one-time device USM
// buffer, single-device stage: the shim reports device 0 only) and optional
// shared-memory staging of x. Accumulation order and the tree reduce are the
// CUDA kernel's exactly.
#include "strata/kernels/s_gemv.hpp"

#include <cuda_fp16.h>     // sycl_compat shim
#include <cuda_runtime.h>  // sycl_compat shim

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int QK_S2 = 64;
constexpr int MAX_SHARED_HALVES = 4096;  // 8 KB of local memory for x staging

float* g_lut = nullptr;  // device USM [256][4]

void ensure_lut(sycl::queue& q) {
    if (g_lut) return;
    float host[256][4];
    for (int b = 0; b < 256; ++b)
        for (int k = 0; k < 4; ++k) host[b][k] = (float)(((b >> (2 * k)) & 3) - 1);
    g_lut = sycl::malloc_device<float>(256 * 4, q);
    if (!g_lut) {
        std::fprintf(stderr, "s2_gemv_fast: LUT allocation failed\n");
        std::exit(1);
    }
    q.memcpy(g_lut, host, sizeof(host)).wait_and_throw();
}

template <bool STAGE_X>
void launch(sycl::queue& q, const float* lut, const uint16_t* x, const uint8_t* codes,
            const float* scales, float* y, long long n_in, long long n_out, int tpr) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<sycl::half, 1> sx(sycl::range<1>(STAGE_X ? MAX_SHARED_HALVES : 1), h);
        sycl::local_accessor<float, 1> partial(sycl::range<1>((size_t)tpr), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t)n_out * tpr),
                                         sycl::range<1>((size_t)tpr)),
                       [=](sycl::nd_item<1> it) {
                           const long long o = (long long)it.get_group(0);
                           if (o >= n_out) return;
                           const int tid = (int)it.get_local_id(0);

                           if (STAGE_X) {
                               const sycl::half* xh = reinterpret_cast<const sycl::half*>(x);
                               for (long long i = tid; i < n_in; i += tpr) sx[i] = xh[i];
                           }
                           it.barrier();

                           const long long n_quads = n_in / 4;
                           const uint8_t* c = codes + o * n_quads;
                           const float* s = scales + o * (n_in / QK_S2);

                           float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                           for (long long qi = tid; qi < n_quads; qi += tpr) {
                               const uint8_t byte = c[qi];  // ONE load for four codes
                               const float d = s[qi >> 4];  // (q*4) >> 6
                               const sycl::float4 cv =
                                   *reinterpret_cast<const sycl::float4*>(lut + (size_t)byte * 4);
                               const sycl::half2 h01 =
                                   STAGE_X ? sycl::half2(sx[qi * 4], sx[qi * 4 + 1])
                                           : *reinterpret_cast<const sycl::half2*>(x + qi * 4);
                               const sycl::half2 h23 =
                                   STAGE_X ? sycl::half2(sx[qi * 4 + 2], sx[qi * 4 + 3])
                                           : *reinterpret_cast<const sycl::half2*>(x + qi * 4 + 2);
                               a0 += cv.x() * d * __low2float(h01);
                               a1 += cv.y() * d * __high2float(h01);
                               a2 += cv.z() * d * __low2float(h23);
                               a3 += cv.w() * d * __high2float(h23);
                           }
                           partial[tid] = (a0 + a1) + (a2 + a3);
                           it.barrier();
                           for (int step = tpr / 2; step > 0; step >>= 1) {
                               if (tid < step) partial[tid] += partial[tid + step];
                               it.barrier();
                           }
                           if (tid == 0) y[o] = partial[0];
                       });
    });
}

}  // namespace

void s2_gemv_fast(const uint16_t* x, const uint8_t* codes, const float* scales, float* y,
                  int64_t n_in, int64_t n_out, int threads_per_row, bool stage_x) {
    if (n_in <= 0 || n_out <= 0) return;
    if (n_in % 4 != 0 || n_in % QK_S2 != 0) {
        std::fprintf(stderr, "s2_gemv_fast: n_in %lld must be a multiple of %d\n", (long long)n_in,
                     QK_S2);
        std::exit(1);
    }
    auto& q = strata::sycl_compat::default_queue();
    try {
        ensure_lut(q);
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "s2_gemv_fast: LUT upload failed: %s\n", e.what());
        std::exit(1);
    }
    if (stage_x && n_in > MAX_SHARED_HALVES) {
        std::fprintf(stderr,
                     "s2_gemv_fast: n_in %lld exceeds the %d-half shared staging limit\n",
                     (long long)n_in, MAX_SHARED_HALVES);
        std::exit(1);
    }
    try {
        if (stage_x)
            launch<true>(q, g_lut, x, codes, scales, y, n_in, n_out, threads_per_row);
        else
            launch<false>(q, g_lut, x, codes, scales, y, n_in, n_out, threads_per_row);
        q.wait_and_throw();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "s2_gemv_fast: %s\n", e.what());
        std::exit(1);
    }
}

}  // namespace strata::kernels
