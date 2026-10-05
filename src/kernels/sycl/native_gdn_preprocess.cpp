// src/kernels/sycl/native_gdn_preprocess.cpp — SYCL port of
// src/kernels/cuda/native_gdn_preprocess.cu (conv+silu, rms/l2 norm, beta
// sigmoid, gate softplus, output norm). Elementwise and 128-wide row norms;
// the norm's two-level reduction (warp xor-sum, 8 warp sums, xor-sum again)
// becomes local-memory trees with the identical pairing; every warp recomputes
// the second tree redundantly so all barriers stay work-group-uniform.
#include "strata/kernels/native_gdn_preprocess.hpp"

#include <cuda_runtime.h>  // sycl_compat shim

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace strata::kernels {
namespace {
constexpr int S = 128;
constexpr int THREADS = 256;

using sycl::local_accessor;
using sycl::nd_item;

inline sycl::queue& Q(void* stream) { return strata::sycl_compat::q_for(stream); }

inline float sigmoid_f(float v) { return 1.0f / (1.0f + sycl::exp(v * -1.0f)); }

// the norm_sum helper: xor-sum within the 32-lane chunk, lane 0 of each chunk
// into sums[8], then every chunk redundantly xor-sums sums[0..7]
inline float norm_sum(float value, int tx, local_accessor<float, 1> sums, local_accessor<float, 1> red,
                      nd_item<1> it) {
    const int lane = tx & 31;
    red[tx] = value;
    it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float other = red[tx ^ o];
        it.barrier(sycl::access::fence_space::local_space);
        value += other;
        red[tx] = value;
        it.barrier(sycl::access::fence_space::local_space);
    }
    if (lane == 0) sums[tx / 32] = value;
    it.barrier(sycl::access::fence_space::local_space);
    value = lane < 8 ? sums[lane] : 0.0f;
    red[tx] = value;
    it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float other = red[tx ^ o];
        it.barrier(sycl::access::fence_space::local_space);
        value += other;
        red[tx] = value;
        it.barrier(sycl::access::fence_space::local_space);
    }
    return value;
}

}  // namespace

void native_gdn_conv_silu(float* history, const float* input, const float* weights, float* raw_output,
                          float* silu_output, int64_t channels, int64_t d_conv, void* stream) {
    if (d_conv != 4) throw std::invalid_argument("native GDN conv expects d_conv 4");
    const int64_t n = channels;
    Q(stream).parallel_for((size_t) n, [=](size_t c) {
        float values[4] = {history[c * 3], history[c * 3 + 1], history[c * 3 + 2], input[c]};
        float sum = 0.0f;
#pragma unroll
        for (int tap = 0; tap < 4; ++tap) sum += values[tap] * weights[c * 4 + tap];
        sum = sum + 0.0f;  // the native SSM kernel's zero-bias add, pinned
        raw_output[c] = sum;
        silu_output[c] = sum / (1.0f + sycl::exp(-sum));
#pragma unroll
        for (int tap = 0; tap < 3; ++tap) history[c * 3 + tap] = values[tap + 1];
    });
}

void native_gdn_l2_norm(float* input, int64_t rows, int64_t cols, float epsilon, void* stream) {
    if (cols != S) throw std::invalid_argument("native GDN l2 norm expects 128 columns");
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<float, 1> sums(sycl::range<1>(THREADS / 32), hnd);
        local_accessor<float, 1> red(sycl::range<1>(THREADS), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) rows * THREADS), sycl::range<1>(THREADS)),
                         [=](nd_item<1> it) {
                             const int col = (int) it.get_local_id(0);
                             float* row = input + (size_t) it.get_group(0) * S;
                             const float value = col < S ? row[col] : 0.0f;
                             float partial = 0.0f;
                             if (col < S) partial += value * value;
                             partial = norm_sum(partial, col, sums, red, it);
                             const float scale = sycl::rsqrt(partial / S + epsilon);
                             if (col < S) row[col] = (scale * value) * (1.0f / sycl::sqrt((float) S));
                         });
    });
}

void native_gdn_beta_gate(float* beta, int64_t heads, void* stream) {
    const int64_t n = heads;
    Q(stream).parallel_for((size_t) n, [=](size_t i) { beta[i] = sigmoid_f(beta[i]); });
}

void native_gdn_gate(const float* alpha, const float* dt, const float* ssm_a, float* gate, int64_t heads,
                     void* stream) {
    const int64_t n = heads;
    Q(stream).parallel_for((size_t) n, [=](size_t i) {
        const float value = alpha[i] + dt[i];
        const float softplus = value > 20.0f ? value : sycl::log1p(sycl::exp(value));
        gate[i] = softplus * ssm_a[i];
    });
}

void native_gdn_out_norm(const float* output, const float* z, const float* gamma, float* destination, int64_t heads,
                         int64_t cols, float epsilon, void* stream) {
    if (cols != S) throw std::invalid_argument("native GDN out norm expects 128 columns");
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<float, 1> sums(sycl::range<1>(THREADS / 32), hnd);
        local_accessor<float, 1> red(sycl::range<1>(THREADS), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) heads * THREADS), sycl::range<1>(THREADS)),
                         [=](nd_item<1> it) {
                             const int col = (int) it.get_local_id(0);
                             const size_t offset = (size_t) it.get_group(0) * S;
                             const float value = col < S ? output[offset + col] : 0.0f;
                             float partial = 0.0f;
                             if (col < S) partial += value * value;
                             partial = norm_sum(partial, col, sums, red, it);
                             const float scale = sycl::rsqrt(partial / S + epsilon);
                             if (col < S) {
                                 const float weighted = (scale * value) * gamma[col];
                                 destination[offset + col] = weighted * sigmoid_f(z[offset + col]);
                             }
                         });
    });
}

}  // namespace strata::kernels
