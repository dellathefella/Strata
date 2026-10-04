// src/kernels/sycl/s2_gemv_q8.cpp — SYCL port of src/kernels/cuda/s2_gemv_q8.cu.
//
// The S2 GEMV over ggml Q8_0-quantized activations (the activation contract for
// Q2_0 experts). Arithmetic is transcribed 1:1 from the CUDA source so the
// parity gate (s2_gemv_q8_parity) measures the port, not a rewrite:
//   * one work-group per output row, threads_per_row work-items
//   * extern __shared__ float partial[] -> sycl::local_accessor
//   * same deterministic shared-memory tree reduction (order == CUDA's)
#include "strata/kernels/s2_gemv_q8.hpp"

#include <cuda_fp16.h>     // sycl_compat shim
#include <cuda_runtime.h>  // sycl_compat shim

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int QK_S2 = 64;
constexpr int QK8_0 = 32;

}  // namespace

void s2_gemv_q8(const uint8_t* act, const uint8_t* codes, const float* scales, float* y, int64_t n_in,
                int64_t n_out, int threads_per_row, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    if (n_in % QK8_0 != 0 || n_in % QK_S2 != 0) {
        std::fprintf(stderr, "s2_gemv_q8: n_in %lld must be a multiple of %d\n", (long long)n_in, QK_S2);
        std::exit(1);
    }
    auto& q = strata::sycl_compat::q_for(stream);
    const int tpr = threads_per_row;
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
                               const uint8_t* c = codes + o * n_quads;  // one code byte per quad
                               const float* s = scales + o * (n_in / QK_S2);

                               float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                               for (long long qi = tid; qi < n_quads; qi += threads_per_row) {
                                   const uint8_t byte = c[qi];
                                   const float d = s[qi >> 4];  // (q*4) >> 6, the S2 group index
                                   const long long ablk = (qi * 4) / QK8_0;
                                   const uint8_t* blk = act + ablk * 34;
                                   const uint16_t dbits = (uint16_t)(blk[0] | (blk[1] << 8));
                                   const float dx = __half2float(__ushort_as_half(dbits));
                                   const int8_t* xq = reinterpret_cast<const int8_t*>(blk + 2);
                                   const int off = (int)((qi * 4) % QK8_0);

                                   const float w0 = (float)((int)(byte & 3) - 1) * d;
                                   const float w1 = (float)((int)((byte >> 2) & 3) - 1) * d;
                                   const float w2 = (float)((int)((byte >> 4) & 3) - 1) * d;
                                   const float w3 = (float)((int)((byte >> 6) & 3) - 1) * d;
                                   a0 += w0 * ((float)xq[off + 0] * dx);
                                   a1 += w1 * ((float)xq[off + 1] * dx);
                                   a2 += w2 * ((float)xq[off + 2] * dx);
                                   a3 += w3 * ((float)xq[off + 3] * dx);
                               }
                               partial[tid] = (a0 + a1) + (a2 + a3);
                               it.barrier();
                               for (int step = threads_per_row / 2; step > 0; step >>= 1) {
                                   if (tid < step) partial[tid] += partial[tid + step];
                                   it.barrier();
                               }
                               if (tid == 0) y[o] = partial[0];
                           });
        });
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "s2_gemv_q8 launch: %s\n", e.what());
        std::exit(1);
    }
    if (stream == nullptr) cudaDeviceSynchronize();
}

}  // namespace strata::kernels
