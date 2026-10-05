// src/kernels/sycl/native_ple_postops.cpp — SYCL port of
// src/kernels/cuda/native_ple_postops.cu: the fused PLE post-projection
// (rms norms through the ported native_gr_rms_norm_weighted, the 4-stream
// gate dot, the value broadcast, the dilated conv + residual). The gate's
// 8-partial-lane sum and the two-level reduction keep the pinned order;
// __fmul_rn/__fadd_rn stay plain * and + under -ffp-contract=off.
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/native_gr_norm.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/ngram.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>  // sycl_compat shim

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {
constexpr int N = 2560, H = 4, D = N * H, HISTORY = 9;

using sycl::local_accessor;
using sycl::nd_item;

inline sycl::queue& Q(void* stream) { return strata::sycl_compat::q_for(stream); }

void launch_check(void* stream) {
    if (stream != nullptr) return;
    const auto error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("native PLE postops launch: ") + cudaGetErrorString(error));
}

// gate: one work-group per stream (512 threads), 8 strided partials per lane,
// xor tree, 16 warp partials, xor tree again; thread 0 writes the sigmoid of
// the signed sqrt of the clamped scaled sum.
void gate_launch(const float* key, const float* query, float* gate, float scale, int rows, void* stream) {
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<float, 1> partials(sycl::range<1>(32), hnd);
        local_accessor<float, 1> red(sycl::range<1>(512), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) rows * 512), sycl::range<1>(512)),
                         [=](nd_item<1> it) {
                             const int row = (int) it.get_group(0);
                             const int tx = (int) it.get_local_id(0);
                             float sums = 0.0f;
                             for (int j = 0; j < 8; ++j) {
                                 const int d = tx + j * 512;
                                 sums += d < N ? key[(size_t) row * N + d] * query[(size_t) row * N + d] : 0.0f;
                             }
                             float sum = sums;
                             const int lane = tx % 32;
                             red[tx] = sum;
                             it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
                             for (int o = 16; o > 0; o >>= 1) {
                                 const float other = red[tx ^ o];
                                 it.barrier(sycl::access::fence_space::local_space);
                                 sum += other;
                                 red[tx] = sum;
                                 it.barrier(sycl::access::fence_space::local_space);
                             }
                             if (!lane) partials[tx / 32] = sum;
                             it.barrier(sycl::access::fence_space::local_space);
                             sum = lane < 16 ? partials[lane] : 0.0f;
                             red[tx] = sum;
                             it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
                             for (int o = 16; o > 0; o >>= 1) {
                                 const float other = red[tx ^ o];
                                 it.barrier(sycl::access::fence_space::local_space);
                                 sum += other;
                                 red[tx] = sum;
                                 it.barrier(sycl::access::fence_space::local_space);
                             }
                             if (tx == 0) {
                                 const float s = scale * sum;
                                 const float mag = sycl::sqrt(sycl::fmax(sycl::fabs(s), 1e-6f));
                                 const float sign = (float) ((s > 0.0f) - (s < 0.0f));
                                 gate[row] = 1.0f / (1.0f + sycl::exp(-(sign * mag)));
                             }
                         });
    });
}

void broadcast_launch(const float* value, const float* gate, float* gated, int T, void* stream) {
    const size_t total = (size_t) T * D;
    Q(stream).parallel_for(total, [=](size_t i) {
        const size_t t = i / D, d = i % D;
        gated[i] = value[t * N + d % N] * gate[t * H + d / N];
    });
}

void conv_residual_launch(const float* history, const float* normalized, const uint16_t* weights,
                          const float* hidden, const float* gated, float* conv, float* result, int T, void* stream) {
    const size_t total = (size_t) T * D;
    Q(stream).parallel_for(total, [=](size_t i) {
        const int t = (int) (i / D), c = (int) (i % D);
        float sum = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int p = t - HISTORY + 3 * k;
            const float x = p >= 0 ? normalized[(size_t) p * D + c] : history[(size_t) c * HISTORY + (HISTORY + p)];
            const float w = f32_from_f16(weights[c * 4 + k]);
            const float term = x * w;
            sum = k == 0 ? term : sum + term;
        }
        const float activation = sum / (1.0f + sycl::exp(-sum));
        if (conv != nullptr) conv[i] = activation;
        if (result != nullptr) result[i] = hidden[i] + (gated[i] + activation);
    });
}

// rms_rep: one 1024-thread work-group per (token, stream) row, the pinned
// two-level norm of rms_rep_kernel
void rms_rep_launch(const float* input, const float* gamma, float* output, int rows, void* stream) {
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<float, 1> sums(sycl::range<1>(32), hnd);
        local_accessor<float, 1> red(sycl::range<1>(1024), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) rows * 1024), sycl::range<1>(1024)),
                         [=](nd_item<1> it) {
                             const int row = (int) it.get_group(0);
                             const int tid = (int) it.get_local_id(0);
                             const float* in = input + (size_t) row * N;
                             float* out = output + (size_t) row * N;
                             const float* gm = gamma + (size_t) (row % H) * N;
                             float partial = 0.0f;
                             for (int col = tid; col < N; col += 1024) {
                                 const float value = in[col];
                                 partial += value * value;
                             }
                             const int lane = tid & 31;
                             red[tid] = partial;
                             it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
                             for (int o = 16; o > 0; o >>= 1) {
                                 const float other = red[tid ^ o];
                                 it.barrier(sycl::access::fence_space::local_space);
                                 partial += other;
                                 red[tid] = partial;
                                 it.barrier(sycl::access::fence_space::local_space);
                             }
                             if (lane == 0) sums[tid / 32] = partial;
                             it.barrier(sycl::access::fence_space::local_space);
                             partial = 0.0f;
                             if (lane < 32) partial = sums[lane];
                             red[tid] = partial;
                             it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
                             for (int o = 16; o > 0; o >>= 1) {
                                 const float other = red[tid ^ o];
                                 it.barrier(sycl::access::fence_space::local_space);
                                 partial += other;
                                 red[tid] = partial;
                                 it.barrier(sycl::access::fence_space::local_space);
                             }
                             const float mean = partial / N;
                             const float scale = sycl::rsqrt(mean + NG_RMS_EPS);
                             for (int col = tid; col < N; col += 1024) out[col] = scale * in[col] * gm[col];
                         });
    });
}

void history_batch_launch(float* history, const float* normalized, int T, void* stream) {
    Q(stream).parallel_for((size_t) D, [=](size_t c) {
        float h[HISTORY];
#pragma unroll
        for (int r = 0; r < HISTORY; ++r) {
            const int p = T - HISTORY + r;
            h[r] = p >= 0 ? normalized[(size_t) p * D + c] : history[c * HISTORY + (T + r)];
        }
#pragma unroll
        for (int r = 0; r < HISTORY; ++r) history[c * HISTORY + r] = h[r];
    });
}

}  // namespace

void native_ple_postops(const float* projected_key, const float* hidden, const float* value, const float* history,
                        const PleWeights& w, const NativePlePostopsBuffers& b, void* stream) {
    if (!stream) throw std::invalid_argument("native PLE postops require an explicit stream");
    if (!projected_key || !hidden || !value || !history || !b.key || !b.query || !b.gate || !b.gated ||
        !b.normalized || !b.conv || !b.result)
        throw std::invalid_argument("native PLE postops require nonnull spans");
    native_gr_rms_norm_weighted(projected_key, w.norm_key, b.key, N, H, NG_RMS_EPS, stream);
    native_gr_rms_norm_weighted(hidden, w.norm_query, b.query, N, H, NG_RMS_EPS, stream);
    gate_launch(b.key, b.query, b.gate, 1.0f / std::sqrt((float) N), H, stream);
    broadcast_launch(value, b.gate, b.gated, 1, stream);
    launch_check(stream);
    native_gr_rms_norm_weighted(b.gated, w.norm_conv, b.normalized, N, H, NG_RMS_EPS, stream);
    conv_residual_launch(history, b.normalized, w.conv1d_f16, hidden, b.gated, b.conv, b.result, 1, stream);
    launch_check(stream);
}

void native_ple_postops_batch(float* key, float* hidden, const float* value, float* history, const PleWeights& w,
                              float* query_norm, float* gated, float* gate, int T, void* stream) {
    if (!stream || T <= 0 || !key || !hidden || !value || !history || !query_norm || !gated || !gate)
        throw std::invalid_argument("native PLE postops batch: null input or empty batch");
    const int rows = T * H;
    const size_t blocks = (size_t) ((size_t) T * D + 255) / 256;
    (void) blocks;
    rms_rep_launch(key, w.norm_key, key, rows, stream);
    rms_rep_launch(hidden, w.norm_query, query_norm, rows, stream);
    gate_launch(key, query_norm, gate, 1.0f / std::sqrt((float) N), rows, stream);
    broadcast_launch(value, gate, gated, T, stream);
    rms_rep_launch(gated, w.norm_conv, query_norm, rows, stream);
    conv_residual_launch(history, query_norm, w.conv1d_f16, hidden, gated, nullptr, hidden, T, stream);
    history_batch_launch(history, query_norm, T, stream);
    launch_check(stream);
}

}  // namespace strata::kernels
