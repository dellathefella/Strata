// src/kernels/sycl/fused_gdn.cpp — SYCL port of src/kernels/cuda/fused_gdn.cu
// (single-token fused gated-DeltaNet: conv+L2, alpha/beta projections,
// state step + output rms norm). Same transcription rules as gdn.cpp and
// verify_kernels.cpp: xor butterflies -> per-32-lane local trees, identical
// pairing and order; __expf -> sycl::exp; fmaf -> sycl::fma.
#include "strata/kernels/fused_gdn.hpp"

#include <cuda_runtime.h>  // sycl_compat shim

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {
constexpr int S = 128;   // state size (rows = cols = 128)
constexpr int RG = 4;    // row groups
constexpr int RPG = S / RG;

using sycl::local_accessor;
using sycl::nd_item;

inline sycl::queue& Q(void* stream) { return strata::sycl_compat::q_for(stream); }

void fail(const char* what, void* stream) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
    (void) stream;
}

inline float sigmoid_f(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }

inline float xor_reduce32(float v, int lane, local_accessor<float, 1> red, nd_item<1> it) {
    red[lane] = v;
    it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float add = red[lane ^ o];
        it.barrier(sycl::access::fence_space::local_space);
        v += add;
        red[lane] = v;
        it.barrier(sycl::access::fence_space::local_space);
    }
    return v;
}

inline uint32_t bf16_pair(const uint16_t* w, int i) {
    return (uint32_t) w[i] | ((uint32_t) w[i + 1] << 16);
}

}  // namespace

void fused_gdn_conv_l2(float* history, const float* qkv, const float* conv_w, float* h, int channels, int qk_heads,
                       float eps, void* stream) {
    if (!history || !qkv || !conv_w || !h || channels % S != 0 || qk_heads < 0 || qk_heads > channels / S) {
        std::fprintf(stderr, "fused_gdn_conv_l2: invalid arguments\n");
        std::exit(1);
    }
    const size_t gx = (size_t) (channels / S);
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<float, 1> red(sycl::range<1>(S), hnd);
        local_accessor<float, 1> part(sycl::range<1>(S / 32), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>(gx * S), sycl::range<1>(S)), [=](nd_item<1> it) {
            const int bx = (int) it.get_group(0);
            const int tx = (int) it.get_local_id(0);
            const int c = bx * S + tx;
            const float v0 = history[c * 3], v1 = history[c * 3 + 1], v2 = history[c * 3 + 2];
            const float x = qkv[c];
            const float sum = v0 * conv_w[c * 4] + v1 * conv_w[c * 4 + 1] + v2 * conv_w[c * 4 + 2] +
                              x * conv_w[c * 4 + 3];
            history[c * 3] = v1;
            history[c * 3 + 1] = v2;
            history[c * 3 + 2] = x;
            float y = sum / (1.0f + sycl::exp(-sum));
            if (bx < qk_heads) {
                float sq = xor_reduce32(y * y, tx, red, it);
                if ((tx & 31) == 0) part[tx >> 5] = sq;
                it.barrier(sycl::access::fence_space::local_space);
                const float ss = part[0] + part[1] + part[2] + part[3];
                y *= sycl::rsqrt(ss + eps);
            }
            h[c] = y;
        });
    });
    fail("fused_gdn_conv_l2", stream);
}

void fused_gdn_ab(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt, const float* ssm_a,
                  float* gate, float* beta, int n_embd, int h_v, void* stream) {
    if (!x || !w_alpha || !w_beta || !dt || !ssm_a || !gate || !beta || n_embd % 8 != 0 || h_v <= 0) {
        std::fprintf(stderr, "fused_gdn_ab: invalid arguments\n");
        std::exit(1);
    }
    const int n = n_embd;
    const size_t blocks = (size_t) ((2 * h_v + 7) / 8);
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<float, 1> red(sycl::range<1>(256), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>(blocks * 256), sycl::range<1>(256)),
                         [=](nd_item<1> it) {
                             const int lid = (int) it.get_local_id(0);
                             const int row = (int) it.get_group(0) * 8 + (lid >> 5);
                             const int lane = lid & 31;
                             const bool live = row < 2 * h_v;
                             const bool is_beta = row >= h_v;
                             const int r = is_beta ? row - h_v : row;
                             const uint16_t* w = (is_beta ? w_beta : w_alpha) + (size_t) r * n;
                             float acc = 0.0f;
                             for (int j = lane; j < n / 8; j += 32) {
                                 uint32_t wv[4];
#pragma unroll
                                 for (int e = 0; e < 4; ++e) wv[e] = bf16_pair(w, j * 8 + e * 2);
                                 const float* xt = x + (size_t) j * 8;
#pragma unroll
                                 for (int e = 0; e < 4; ++e) {
                                     acc = sycl::fma(sycl::bit_cast<float>(wv[e] << 16), xt[e * 2], acc);
                                     acc = sycl::fma(sycl::bit_cast<float>(wv[e] & 0xffff0000u), xt[e * 2 + 1], acc);
                                 }
                             }
                             acc = xor_reduce32(live ? acc : 0.0f, lid, red, it);
                             if (lane != 0 || !live) return;
                             if (is_beta) {
                                 beta[r] = sigmoid_f(acc);
                             } else {
                                 const float v = acc + dt[r];
                                 const float sp = v > 20.0f ? v : sycl::log1p(sycl::exp(v));
                                 gate[r] = sp * ssm_a[r];
                             }
                         });
    });
    fail("fused_gdn_ab", stream);
}

void fused_gdn_step_norm(float* state, const float* q, const float* k, const float* v, const float* gate,
                         const float* beta, const float* z, const float* gamma, float eps, float* y, int h_k, int h_v,
                         void* stream) {
    if (!state || !q || !k || !v || !gate || !beta || !z || !gamma || !y || h_k <= 0 || h_v <= 0 || h_v % h_k) {
        std::fprintf(stderr, "fused_gdn_step_norm: invalid arguments\n");
        std::exit(1);
    }
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<float, 1> sk(sycl::range<1>(S), hnd);
        local_accessor<float, 1> sq(sycl::range<1>(S), hnd);
        local_accessor<float, 1> red(sycl::range<1>(RG * S), hnd);
        local_accessor<float, 1> wsum(sycl::range<1>(S * RG / 32), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) h_v * S * RG), sycl::range<1>(S * RG)),
                         [=](nd_item<1> it) {
                             const int head = (int) it.get_group(0);
                             const int tid = (int) it.get_local_id(0);
                             const int col = tid % S;
                             const int rg = tid / S;
                             const int qh = head % h_k;
                             if (tid < S) {
                                 sk[tid] = k[qh * S + tid];
                                 sq[tid] = q[qh * S + tid];
                             }
                             float s[RPG];
                             float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
                             const size_t row_stride = (size_t) h_v * S;
#pragma unroll
                             for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
                             it.barrier(sycl::access::fence_space::local_space);
                             const float g = sycl::exp(gate[head]);
                             float kv = 0.0f;
#pragma unroll
                             for (int r = 0; r < RPG; ++r) kv = sycl::fma(s[r], sk[rg * RPG + r], kv);
                             red[rg * S + col] = kv;
                             it.barrier(sycl::access::fence_space::local_space);
                             const float kv_col = red[col] + red[S + col] + red[2 * S + col] + red[3 * S + col];
                             const float delta = (v[head * S + col] - g * kv_col) * beta[head];
                             float o = 0.0f;
#pragma unroll
                             for (int r = 0; r < RPG; ++r) {
                                 s[r] = sycl::fma(g, s[r], sk[rg * RPG + r] * delta);
                                 o = sycl::fma(s[r], sq[rg * RPG + r], o);
                                 base[r * row_stride] = s[r];
                             }
                             it.barrier(sycl::access::fence_space::local_space);
                             red[rg * S + col] = o;
                             it.barrier(sycl::access::fence_space::local_space);
                             float oc = 0.0f, sq_part = 0.0f;
                             if (rg == 0) {
                                 oc = (red[col] + red[S + col] + red[2 * S + col] + red[3 * S + col]) *
                                      sycl::rsqrt((float) S);
                                 sq_part = oc * oc;
                             }
                             sq_part = xor_reduce32(sq_part, tid, red, it);
                             if ((tid & 31) == 0) wsum[tid >> 5] = sq_part;
                             it.barrier(sycl::access::fence_space::local_space);
                             if (rg == 0) {
                                 const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
                                 const float scale = sycl::rsqrt(ss / (float) S + eps);
                                 const float zz = z[head * S + col];
                                 y[head * S + col] = oc * scale * gamma[col] * sigmoid_f(zz);
                             }
                         });
    });
    fail("fused_gdn_step_norm", stream);
}

}  // namespace strata::kernels
