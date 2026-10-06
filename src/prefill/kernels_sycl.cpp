// src/prefill/kernels_sycl.cpp - SYCL port of src/prefill/kernels.cu (milestone 5).
// Same arithmetic, same summation orders, same thread-to-element mappings: the
// warp shuffles become per-work-group local-memory xor trees (identical
// pairing), the __syncthreads become work-group barriers, and no thread ever
// returns before a barrier.  bf()/hf() use the repo's bit helpers.
#include "strata/prefill/kernels.hpp"

#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/router_top10.hpp"

#include <cuda_runtime.h>  // sycl_compat shim

#include <sycl/sycl.hpp>

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::prefill {
namespace {

using sycl::local_accessor;
using sycl::nd_item;

constexpr int N = 2560, HC = 4, D = N * HC, LR = 320;
constexpr int S = 128, HK = 16, HV = 48, C = 10240;
constexpr int RG = 4, RPG = S / RG;
constexpr int CB = 32, NCB = S / CB;
constexpr int CONV_TILE = 64;
constexpr int GRW_PER = (N + 255) / 256;

inline sycl::queue& Q(void* stream) { return strata::sycl_compat::q_for(stream); }

void check(void* stream, const char* what) {
    if (stream != nullptr) return;
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "prefill %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

inline float sigm(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }
inline uint16_t bf(float f) { return strata::kernels::bf16_from_f32(f); }
inline float unbf(uint16_t h) { return strata::kernels::f32_from_bf16(h); }
inline uint16_t bf_lo(float f, uint16_t hi) { return bf(f - unbf(hi)); }
inline uint16_t hf(float f) { return strata::kernels::f16_from_f32(f); }
inline float unhf(uint16_t h) { return strata::kernels::f32_from_f16(h); }
inline uint16_t hf_sat(float f) {
    return hf(f == f ? sycl::fmin(sycl::fmax(f, -65504.0f), 65504.0f) : f);
}

// warp xor reduction over a per-work-group local array (lid ^ o stays inside
// the warp because o <= 16 and work-groups are multiples of 32)
inline float warp_sum_l(float v, int lid, local_accessor<float, 1> red, nd_item<1> it) {
    red[lid] = v;
    it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float add = red[lid ^ o];
        it.barrier(sycl::access::fence_space::local_space);
        v += add;
        red[lid] = v;
        it.barrier(sycl::access::fence_space::local_space);
    }
    return v;
}

inline float warp_max_l(float v, int lid, local_accessor<float, 1> red, nd_item<1> it) {
    red[lid] = v;
    it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float other = red[lid ^ o];
        it.barrier(sycl::access::fence_space::local_space);
        v = sycl::fmax(v, other);
        red[lid] = v;
        it.barrier(sycl::access::fence_space::local_space);
    }
    return v;
}

// CUDA block_sum(v, sh) for blockDim.x <= 1024: warp sums into sh[32], warp 0
// reduces them, sh[0] broadcasts.  red is a full-work-group scratch.
inline float block_sum_l(float v, local_accessor<float, 1> red, local_accessor<float, 1> sh, int lid, int wg,
                         nd_item<1> it) {
    const int lane = lid & 31, w = lid >> 5;
    v = warp_sum_l(v, lid, red, it);
    if (lane == 0) sh[w] = v;
    it.barrier(sycl::access::fence_space::local_space);
    const int nw = (wg + 31) >> 5;
    float t = (lid < nw) ? sh[lid] : 0.0f;
    red[lid] = (w == 0) ? t : 0.0f;
    it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float add = red[lid ^ o];
        it.barrier(sycl::access::fence_space::local_space);
        t += add;
        red[lid] = t;
        it.barrier(sycl::access::fence_space::local_space);
    }
    if (lid == 0) sh[0] = t;
    it.barrier(sycl::access::fence_space::local_space);
    return sh[0];
}

// ---------------------------------------------------------------- hyper-connection
void gr_norm_launch(const float* R, const float* w, float eps, float* xn, uint16_t* xn16, uint16_t* xn16_lo,
                    int64_t rows, void* stream) {
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<float, 1> red(sycl::range<1>(256), hnd);
        local_accessor<float, 1> sh(sycl::range<1>(32), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) rows * 256), sycl::range<1>(256)),
                         [=](nd_item<1> it) {
                             const int64_t row = (int64_t) it.get_group(0);
                             const int lid = (int) it.get_local_id(0);
                             const int c = (int) (row % HC);
                             const float* r = R + row * N;
                             float ss = 0.0f;
                             for (int d = lid; d < N; d += 256) ss += r[d] * r[d];
                             const float rs = sycl::rsqrt(block_sum_l(ss, red, sh, lid, 256, it) / (float) N + eps);
                             for (int d = lid; d < N; d += 256) {
                                 const float v = r[d] * rs * w[c * N + d];
                                 xn[row * N + d] = v;
                                 const uint16_t h = bf(v);
                                 xn16[row * N + d] = h;
                                 if (xn16_lo) xn16_lo[row * N + d] = bf_lo(v, h);
                             }
                         });
    });
    check(stream, "gr_norm");
}

void gr_norm_rs_launch(const float* R, const float* w, float eps, float* rs_out, uint16_t* xn16, uint16_t* xn16_lo,
                       int64_t rows, void* stream) {
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<float, 1> red(sycl::range<1>(256), hnd);
        local_accessor<float, 1> sh(sycl::range<1>(32), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) rows * 256), sycl::range<1>(256)),
                         [=](nd_item<1> it) {
                             const int64_t row = (int64_t) it.get_group(0);
                             const int lid = (int) it.get_local_id(0);
                             const int c = (int) (row % HC);
                             const float* r = R + row * N;
                             float ss = 0.0f;
                             for (int d = lid; d < N; d += 256) ss += r[d] * r[d];
                             const float rs = sycl::rsqrt(block_sum_l(ss, red, sh, lid, 256, it) / (float) N + eps);
                             if (lid == 0) rs_out[row] = rs;
                             for (int d = lid; d < N; d += 256) {
                                 const float v = r[d] * rs * w[c * N + d];
                                 const uint16_t h = bf(v);
                                 xn16[row * N + d] = h;
                                 if (xn16_lo) xn16_lo[row * N + d] = bf_lo(v, h);
                             }
                         });
    });
    check(stream, "gr_norm_rs");
}

void gr_mix_r_launch(const float* R, const float* rs, const float* w, const float* g, float* mixed, uint16_t* mixed16,
                     int64_t T, uint16_t* mixed_h, uint16_t* mixed16_lo, void* stream) {
    const int64_t n = T * N;
    const size_t blocks = (size_t) ((n + 255) / 256);
    Q(stream).parallel_for(sycl::range<1>(blocks * 256), [=](size_t i) {
        if (i >= (size_t) n) return;
        const int64_t t = (int64_t) i / N, d = (int64_t) i % N;
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) {
            const int64_t j = t * D + c * N + d;
            const float x = R[j] * rs[t * HC + c] * w[c * N + d];
            s = sycl::fma(x, sigm(g[j]), s);
        }
        s /= (float) HC;
        mixed[i] = s;
        if (mixed16) {
            const uint16_t h = bf(s);
            mixed16[i] = h;
            if (mixed16_lo) mixed16_lo[i] = bf_lo(s, h);
        }
        if (mixed_h) mixed_h[i] = hf(s);
    });
    check(stream, "gr_mix_r");
}

void gr_write_norm_rs_launch(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* w, float eps,
                             float* rs_out, uint16_t* xn16, uint16_t* xn16_lo, int64_t rows, void* stream) {
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<float, 1> red(sycl::range<1>(256), hnd);
        local_accessor<float, 1> sh(sycl::range<1>(32), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) rows * 256), sycl::range<1>(256)),
                         [=](nd_item<1> it) {
                             const int64_t row = (int64_t) it.get_group(0);
                             const int lid = (int) it.get_local_id(0);
                             const int64_t t = row / HC;
                             const int c = (int) (row % HC);
                             float* r = R + row * N;
                             const float sc = 2.0f * sigm(inj[t * inj_ld + c] / (float) HC);
                             float v[GRW_PER];
                             float ss = 0.0f;
                             int k = 0;
#pragma unroll
                             for (int d = lid; d < N; d += 256, ++k) {
                                 const float x = sycl::fma(bo[t * N + d], sc, r[d]);
                                 r[d] = x;
                                 v[k] = x;
                                 ss += x * x;
                             }
                             const float rs = sycl::rsqrt(block_sum_l(ss, red, sh, lid, 256, it) / (float) N + eps);
                             if (lid == 0) rs_out[row] = rs;
                             k = 0;
#pragma unroll
                             for (int d = lid; d < N; d += 256, ++k) {
                                 const float x = v[k] * rs * w[c * N + d];
                                 const uint16_t h = bf(x);
                                 xn16[row * N + d] = h;
                                 if (xn16_lo) xn16_lo[row * N + d] = bf_lo(x, h);
                             }
                         });
    });
    check(stream, "gr_write_norm_rs");
}

void gr_silu_launch(const float* lo, uint16_t* lo16, uint16_t* lo16_lo, int64_t n, void* stream) {
    const size_t blocks = (size_t) ((n + 255) / 256);
    Q(stream).parallel_for(sycl::range<1>(blocks * 256), [=](size_t i) {
        if (i >= (size_t) n) return;
        const float x = lo[i] / (float) HC;
        const float v = x / (1.0f + sycl::exp(-x));
        const uint16_t h = bf(v);
        lo16[i] = h;
        if (lo16_lo) lo16_lo[i] = bf_lo(v, h);
    });
    check(stream, "gr_silu");
}

void gr_mix_launch(const float* xn, const float* g, float* mixed, uint16_t* mixed16, int64_t T, uint16_t* mixed_h,
                   uint16_t* mixed16_lo, void* stream) {
    const int64_t n = T * N;
    const size_t blocks = (size_t) ((n + 255) / 256);
    Q(stream).parallel_for(sycl::range<1>(blocks * 256), [=](size_t i) {
        if (i >= (size_t) n) return;
        const int64_t t = (int64_t) i / N, d = (int64_t) i % N;
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) {
            const int64_t j = t * D + c * N + d;
            s = sycl::fma(xn[j], sigm(g[j]), s);
        }
        s /= (float) HC;
        mixed[i] = s;
        if (mixed16) {
            const uint16_t h = bf(s);
            mixed16[i] = h;
            if (mixed16_lo) mixed16_lo[i] = bf_lo(s, h);
        }
        if (mixed_h) mixed_h[i] = hf(s);
    });
    check(stream, "gr_mix");
}

void gr_write_launch(float* R, const float* bo, const float* inj, int64_t inj_ld, int64_t T, void* stream) {
    const int64_t n = T * D;
    const size_t blocks = (size_t) ((n + 255) / 256);
    Q(stream).parallel_for(sycl::range<1>(blocks * 256), [=](size_t i) {
        if (i >= (size_t) n) return;
        const int64_t t = (int64_t) i / D, c = ((int64_t) i % D) / N, d = (int64_t) i % N;
        R[i] = sycl::fma(bo[t * N + d], 2.0f * sigm(inj[t * inj_ld + c] / (float) HC), R[i]);
    });
    check(stream, "gr_write");
}

void gr_broadcast_launch(const float* e, float* R, int64_t T, void* stream) {
    const int64_t n = T * D;
    const size_t blocks = (size_t) ((n + 255) / 256);
    Q(stream).parallel_for(sycl::range<1>(blocks * 256), [=](size_t i) {
        if (i >= (size_t) n) return;
        const int64_t t = (int64_t) i / D, d = (int64_t) i % N;
        R[i] = e[t * N + d];
    });
    check(stream, "gr_broadcast");
}

// ---------------------------------------------------------------- GDN
void gdn_gates_launch(const float* ab, const float* dt, const float* ssm_a, float* gate, float* beta, int64_t T,
                      void* stream) {
    const int64_t n = T * HV;
    const size_t blocks = (size_t) ((n + 255) / 256);
    Q(stream).parallel_for(sycl::range<1>(blocks * 256), [=](size_t i) {
        if (i >= (size_t) n) return;
        const int64_t t = (int64_t) i / HV, h = (int64_t) i % HV;
        const float v = ab[t * 2 * HV + h] + dt[h];
        gate[i] = (v > 20.0f ? v : sycl::log1p(sycl::exp(v))) * ssm_a[h];
        beta[i] = sigm(ab[t * 2 * HV + HV + h]);
    });
    check(stream, "gdn_gates");
}

void gdn_conv_walk_launch(float* hist, const float* qkv, const float* w, float* h, int64_t T, void* stream) {
    Q(stream).parallel_for(sycl::range<1>((size_t) C), [=](size_t c) {
        float v0 = hist[c * 3], v1 = hist[c * 3 + 1], v2 = hist[c * 3 + 2];
        const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
        for (int64_t t = 0; t < T; ++t) {
            const float x = qkv[t * C + c];
            const float s = v0 * w0 + v1 * w1 + v2 * w2 + x * w3;
            h[t * C + c] = s / (1.0f + sycl::exp(-s));
            v0 = v1; v1 = v2; v2 = x;
        }
        hist[c * 3] = v0; hist[c * 3 + 1] = v1; hist[c * 3 + 2] = v2;
    });
}

void gdn_conv_tiled_launch(const float* hist, const float* qkv, const float* w, float* h, int64_t T, void* stream) {
    const size_t tiles = (size_t) ((T + CONV_TILE - 1) / CONV_TILE);
    Q(stream).parallel_for(sycl::range<2>((size_t) C, tiles), [=](sycl::id<2> id) {
        const size_t c = id[0];
        const int64_t t0 = (int64_t) id[1] * CONV_TILE;
        const int64_t t1 = t0 + CONV_TILE < T ? t0 + CONV_TILE : T;
        auto input = [&](int64_t t) -> float { return t >= 0 ? qkv[t * C + c] : hist[c * 3 + (int) (t + 3)]; };
        float v0 = input(t0 - 3), v1 = input(t0 - 2), v2 = input(t0 - 1);
        const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
        for (int64_t t = t0; t < t1; ++t) {
            const float x = qkv[t * C + c];
            const float s = v0 * w0 + v1 * w1 + v2 * w2 + x * w3;
            h[t * C + c] = s / (1.0f + sycl::exp(-s));
            v0 = v1; v1 = v2; v2 = x;
        }
    });
}

void gdn_conv_hist_launch(float* hist, const float* qkv, int64_t T, void* stream) {
    Q(stream).parallel_for(sycl::range<1>((size_t) C), [=](size_t c) {
        float v[3];
        for (int k = 0; k < 3; ++k) {
            const int64_t t = T - 3 + k;
            v[k] = t >= 0 ? qkv[t * C + c] : hist[c * 3 + (int) (t + 3)];
        }
        hist[c * 3] = v[0]; hist[c * 3 + 1] = v[1]; hist[c * 3 + 2] = v[2];
    });
}

void gdn_l2_launch(float* h, float eps, int64_t T, void* stream) {
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<float, 1> red(sycl::range<1>(128), hnd);
        local_accessor<float, 1> part(sycl::range<1>(4), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) T * 2 * HK * 128), sycl::range<1>(128)),
                         [=](nd_item<1> it) {
                             const int64_t blk = (int64_t) it.get_group(0);
                             const int64_t t = blk / (2 * HK);        // dim3(2*HK, T): x fastest
                             const int head = (int) (blk % (2 * HK));
                             const int lid = (int) it.get_local_id(0);
                             float* x = h + t * C + head * S;
                             const float v = x[lid];
                             float sq = warp_sum_l(v * v, lid, red, it);
                             const int lane = lid & 31, w = lid >> 5;
                             if (lane == 0) part[w] = sq;
                             it.barrier(sycl::access::fence_space::local_space);
                             const float ss = part[0] + part[1] + part[2] + part[3];
                             x[lid] = v * sycl::rsqrt(ss + eps);
                         });
    });
}

// the serial one-block-per-head recurrence (STRATA_GDN_REC_HEADS / T <= 0)
void gdn_rec_launch(float* state, const float* h, const float* gate, const float* beta, const float* z,
                    const float* gamma, float eps, float* y, uint16_t* y16, int64_t T, void* stream) {
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<float, 1> sk(sycl::range<1>(S), hnd);
        local_accessor<float, 1> sq(sycl::range<1>(S), hnd);
        local_accessor<float, 1> red(sycl::range<1>(RG * S), hnd);
        local_accessor<float, 1> red2(sycl::range<1>(S * RG), hnd);
        local_accessor<float, 1> wsum(sycl::range<1>(16), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) HV * S * RG), sycl::range<1>(S * RG)),
                         [=](nd_item<1> it) {
                             const int head = (int) it.get_group(0);
                             const int col = (int) (it.get_local_id(0) % S);
                             const int rg = (int) (it.get_local_id(0) / S);
                             const int tid = (int) it.get_local_id(0);
                             const int qh = head % HK;
                             float s[RPG];
                             float* base = state + ((size_t) (rg * RPG) * HV + head) * S + col;
                             const size_t rs = (size_t) HV * S;
#pragma unroll
                             for (int r = 0; r < RPG; ++r) s[r] = base[r * rs];
                             const float g_col = gamma[col];
                             for (int64_t t = 0; t < T; ++t) {
                                 const float* ht = h + t * C;
                                 it.barrier(sycl::access::fence_space::local_space);
                                 if (tid < S) {
                                     sq[tid] = ht[qh * S + tid];
                                     sk[tid] = ht[HK * S + qh * S + tid];
                                 }
                                 it.barrier(sycl::access::fence_space::local_space);
                                 const float g = sycl::exp(gate[t * HV + head]);
                                 float kv = 0.0f;
#pragma unroll
                                 for (int r = 0; r < RPG; ++r) kv = sycl::fma(s[r], sk[rg * RPG + r], kv);
                                 red[rg * S + col] = kv;
                                 it.barrier(sycl::access::fence_space::local_space);
                                 const float kv_col = red[col] + red[S + col] + red[2 * S + col] + red[3 * S + col];
                                 const float delta = (ht[2 * HK * S + head * S + col] - g * kv_col) * beta[t * HV + head];
                                 float o = 0.0f;
#pragma unroll
                                 for (int r = 0; r < RPG; ++r) {
                                     s[r] = sycl::fma(g, s[r], sk[rg * RPG + r] * delta);
                                     o = sycl::fma(s[r], sq[rg * RPG + r], o);
                                 }
                                 it.barrier(sycl::access::fence_space::local_space);
                                 red[rg * S + col] = o;
                                 it.barrier(sycl::access::fence_space::local_space);
                                 float oc = 0.0f, sp = 0.0f;
                                 if (rg == 0) {
                                     oc = (red[col] + red[S + col] + red[2 * S + col] + red[3 * S + col]) *
                                          sycl::rsqrt((float) S);
                                     sp = oc * oc;
                                 }
                                 sp = warp_sum_l(sp, tid, red2, it);
                                 if ((tid & 31) == 0) wsum[tid >> 5] = sp;
                                 it.barrier(sycl::access::fence_space::local_space);
                                 if (rg == 0) {
                                     const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
                                     const float v = oc * sycl::rsqrt(ss / (float) S + eps) * g_col *
                                                     sigm(z[t * HV * S + head * S + col]);
                                     y[t * HV * S + head * S + col] = v;
                                     y16[t * HV * S + head * S + col] = hf(v);
                                 }
                             }
#pragma unroll
                             for (int r = 0; r < RPG; ++r) base[r * rs] = s[r];
                         });
    });
    check(stream, "gdn_rec");
}

void gdn_rec_cols_launch(float* state, const float* h, const float* gate, const float* beta, float* oc_out, int64_t T,
                         bool pipelined, void* stream) {
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<float, 1> sk(sycl::range<1>(S), hnd);
        local_accessor<float, 1> sq(sycl::range<1>(S), hnd);
        local_accessor<float, 1> red(sycl::range<1>(RG * CB), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) HV * NCB * CB * RG), sycl::range<1>(CB * RG)),
                         [=](nd_item<1> it) {
                             const int head = (int) (it.get_group(0) / NCB), cb = (int) (it.get_group(0) % NCB);
                             const int c = (int) (it.get_local_id(0) % CB);
                             const int rg = (int) (it.get_local_id(0) / CB);
                             const int tid = (int) it.get_local_id(0);
                             const int col = cb * CB + c;
                             const int qh = head % HK;
                             float s[RPG];
                             float* base = state + ((size_t) (rg * RPG) * HV + head) * S + col;
                             const size_t rs = (size_t) HV * S;
#pragma unroll
                             for (int r = 0; r < RPG; ++r) s[r] = base[r * rs];
                             constexpr int NT = CB * RG, LPT = S / NT;   // LPT == 1
                             float nq[LPT], nk[LPT], nv = 0.0f, ng = 0.0f, nb = 0.0f;
                             auto fetch = [&](int64_t t) {
                                 const float* ht = h + t * C;
#pragma unroll
                                 for (int u = 0; u < LPT; ++u) {
                                     nq[u] = ht[qh * S + tid + u * NT];
                                     nk[u] = ht[HK * S + qh * S + tid + u * NT];
                                 }
                                 nv = ht[2 * HK * S + head * S + col];
                                 ng = gate[t * HV + head];
                                 nb = beta[t * HV + head];
                             };
                             if (pipelined && T > 0) fetch(0);
                             for (int64_t t = 0; t < T; ++t) {
                                 float cq[LPT], ck[LPT];
                                 float cv, cg, cbt;
                                 if (pipelined) {
#pragma unroll
                                     for (int u = 0; u < LPT; ++u) { cq[u] = nq[u]; ck[u] = nk[u]; }
                                     cv = nv; cg = ng; cbt = nb;
                                 }
                                 it.barrier(sycl::access::fence_space::local_space);
                                 if (pipelined) {
#pragma unroll
                                     for (int u = 0; u < LPT; ++u) {
                                         sq[tid + u * NT] = cq[u];
                                         sk[tid + u * NT] = ck[u];
                                     }
                                 } else if (tid < S) {
                                     const float* ht = h + t * C;
                                     sq[tid] = ht[qh * S + tid];
                                     sk[tid] = ht[HK * S + qh * S + tid];
                                 }
                                 it.barrier(sycl::access::fence_space::local_space);
                                 if (pipelined && t + 1 < T) fetch(t + 1);
                                 const float* ht = h + t * C;
                                 const float g = sycl::exp(pipelined ? cg : gate[t * HV + head]);
                                 float kv = 0.0f;
#pragma unroll
                                 for (int r = 0; r < RPG; ++r) kv = sycl::fma(s[r], sk[rg * RPG + r], kv);
                                 red[rg * CB + c] = kv;
                                 it.barrier(sycl::access::fence_space::local_space);
                                 const float kv_col = red[c] + red[CB + c] + red[2 * CB + c] + red[3 * CB + c];
                                 const float vv = pipelined ? cv : ht[2 * HK * S + head * S + col];
                                 const float bt = pipelined ? cbt : beta[t * HV + head];
                                 const float delta = (vv - g * kv_col) * bt;
                                 float o = 0.0f;
#pragma unroll
                                 for (int r = 0; r < RPG; ++r) {
                                     s[r] = sycl::fma(g, s[r], sk[rg * RPG + r] * delta);
                                     o = sycl::fma(s[r], sq[rg * RPG + r], o);
                                 }
                                 it.barrier(sycl::access::fence_space::local_space);
                                 red[rg * CB + c] = o;
                                 it.barrier(sycl::access::fence_space::local_space);
                                 if (rg == 0)
                                     oc_out[t * HV * S + head * S + col] =
                                         (red[c] + red[CB + c] + red[2 * CB + c] + red[3 * CB + c]) *
                                         sycl::rsqrt((float) S);
                             }
#pragma unroll
                             for (int r = 0; r < RPG; ++r) base[r * rs] = s[r];
                         });
    });
    check(stream, "gdn_rec_cols");
}

void gdn_out_norm_launch(const float* z, const float* gamma, float eps, float* y, uint16_t* y16, int64_t T,
                         void* stream) {
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<float, 1> red(sycl::range<1>(128), hnd);
        local_accessor<float, 1> wsum(sycl::range<1>(4), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) T * HV * 128), sycl::range<1>(128)),
                         [=](nd_item<1> it) {
                             const int64_t blk = (int64_t) it.get_group(0);
                             const int64_t t = blk % T;              // dim3(T, HV): x fastest
                             const int head = (int) (blk / T);
                             const int col = (int) it.get_local_id(0);
                             const size_t at = (size_t) t * HV * S + (size_t) head * S + col;
                             const float oc = y[at];
                             float sp = warp_sum_l(oc * oc, col, red, it);
                             if ((col & 31) == 0) wsum[col >> 5] = sp;
                             it.barrier(sycl::access::fence_space::local_space);
                             const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
                             const float v = oc * sycl::rsqrt(ss / (float) S + eps) * gamma[col] *
                                             sigm(z[t * HV * S + head * S + col]);
                             y[at] = v;
                             y16[at] = hf(v);
                         });
    });
    check(stream, "gdn_out_norm");
}

// ---------------------------------------------------------------- MoE
template <int REG>
void route_launch(const float* logits, int32_t* ids, float* wout, int64_t T, void* stream) {
    const size_t blocks = (size_t) ((T + 7) / 8);
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<float, 1> redF(sycl::range<1>(256), hnd);
        local_accessor<int32_t, 1> redI(sycl::range<1>(256), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>(blocks * 256), sycl::range<1>(256)),
                         [=](nd_item<1> it) {
                             const int lid = (int) it.get_local_id(0);
                             const int64_t t = (int64_t) it.get_group(0) * 8 + (lid >> 5);
                             const int lane = lid & 31;
                             const bool live = t < T;
                             const float* lg = logits + (live ? t : 0) * (REG * 32);
                             float v[REG];
#pragma unroll
                             for (int i = 0; i < REG; ++i) v[i] = live ? lg[lane + i * 32] : 0.0f;
                             float mx = -INFINITY;
#pragma unroll
                             for (int i = 0; i < REG; ++i) mx = sycl::fmax(mx, v[i]);
                             mx = warp_max_l(mx, lid, redF, it);
                             float sum = 0.0f;
#pragma unroll
                             for (int i = 0; i < REG; ++i) {
                                 v[i] = sycl::exp(v[i] - mx);
                                 sum += v[i];
                             }
                             const float rcp = 1.0f / warp_sum_l(sum, lid, redF, it);
#pragma unroll
                             for (int i = 0; i < REG; ++i) {
                                 v[i] *= rcp;
                                 if (v[i] != v[i]) v[i] = -FLT_MAX;
                             }
                             float selected = 0.0f, selected_sum = 0.0f;
                             for (int rank = 0; rank < 10; ++rank) {
                                 float best = v[0];
                                 int ex = lane;
#pragma unroll
                                 for (int i = 1; i < REG; ++i)
                                     if (v[i] > best) { best = v[i]; ex = lane + i * 32; }
#pragma unroll
                                 for (int m = 16; m; m >>= 1) {
                                     redF[lid] = best;
                                     redI[lid] = ex;
                                     it.barrier(sycl::access::fence_space::local_space);
                                     const float ob = redF[lid ^ m];
                                     const int oi = redI[lid ^ m];
                                     it.barrier(sycl::access::fence_space::local_space);
                                     if (ob > best || (ob == best && oi < ex)) { best = ob; ex = oi; }
                                 }
                                 if (live) {
                                     if ((ex & 31) == lane) { v[ex / 32] = -INFINITY; selected_sum += best; }
                                     if (lane == 0) ids[t * 10 + rank] = ex;
                                     if (rank == lane) selected = best;
                                 }
                             }
                             selected_sum = sycl::fmax(warp_sum_l(selected_sum, lid, redF, it), 6.103515625e-5f);
                             if (live && lane < 10) wout[t * 10 + lane] = selected / selected_sum;
                         });
    });
    check(stream, "route");
}

template <bool HALF>
void blob_dequant_launch(const uint8_t* blob, uint16_t* gu16, uint16_t* d16, void* stream) {
    constexpr size_t O_D_CODES = (size_t) 1280 * 640, O_GU_SC = O_D_CODES + (size_t) 2560 * 160,
                     O_D_SC = O_GU_SC + (size_t) 1280 * 40 * 2;
    const int64_t n_gu = 1280LL * 640, n_d = 2560LL * 160;
    const size_t blocks = (size_t) ((n_gu + n_d + 255) / 256);
    Q(stream).parallel_for(sycl::range<1>(blocks * 256), [=](size_t ii) {
        const int64_t i = (int64_t) ii;
        if (i < n_gu) {
            const int64_t row = i / 640, byte = i % 640;
            const uint8_t ccode = blob[row * 640 + byte];
            const uint8_t* sp = blob + O_GU_SC + (size_t) (row * 40 + (byte * 4) / 64) * 2;
            const float d = unhf((uint16_t) (sp[0] | (sp[1] << 8)));
            uint16_t* o = gu16 + row * 2560 + byte * 4;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float v = (float) (((ccode >> (2 * k)) & 3) - 1) * d;
                o[k] = HALF ? hf(v) : bf(v);
            }
        } else if (i < n_gu + n_d) {
            const int64_t j = i - n_gu, row = j / 160, byte = j % 160;
            const uint8_t ccode = blob[O_D_CODES + row * 160 + byte];
            const uint8_t* sp = blob + O_D_SC + (size_t) (row * 10 + (byte * 4) / 64) * 2;
            const float d = unhf((uint16_t) (sp[0] | (sp[1] << 8)));
            uint16_t* o = d16 + row * 640 + byte * 4;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float v = (float) (((ccode >> (2 * k)) & 3) - 1) * d;
                o[k] = HALF ? hf(v) : bf(v);
            }
        }
    });
}

void swiglu_il_launch(const float* gu, uint16_t* h16, int64_t n, void* stream) {
    const int64_t total = n * 640;
    const size_t blocks = (size_t) ((total + 255) / 256);
    Q(stream).parallel_for(sycl::range<1>(blocks * 256), [=](size_t ii) {
        const int64_t i = (int64_t) ii;
        if (i >= total) return;
        const int64_t r = i / 640, k = i % 640;
        const float g = gu[r * 1280 + 2 * k], u = gu[r * 1280 + 2 * k + 1];
        h16[i] = hf_sat(g / (1.0f + sycl::exp(-g)) * u);
    });
    check(stream, "swiglu_interleaved");
}

void swiglu_pair_launch(const float* g, const float* u, uint16_t* h16, int64_t n, void* stream) {
    const int64_t total = n * 640;
    const size_t blocks = (size_t) ((total + 255) / 256);
    Q(stream).parallel_for(sycl::range<1>(blocks * 256), [=](size_t i) {
        if (i >= (size_t) total) return;
        const float a = g[i];
        h16[i] = hf_sat(a / (1.0f + sycl::exp(-a)) * u[i]);
    });
    check(stream, "swiglu_pair");
}

void copy_i32_launch(int32_t* dst, const int32_t* src, int64_t n, void* stream) {
    const int64_t b = (n + 255) / 256;
    const size_t blocks = (size_t) (b < 256 ? b : 256);
    Q(stream).parallel_for(sycl::range<1>(blocks * 256), [=](size_t i) {
        for (int64_t j = (int64_t) i; j < n; j += (int64_t) blocks * 256) dst[j] = src[j];
    });
    check(stream, "copy_i32");
}

void gather_rows16_launch(const uint16_t* x, const int32_t* src, uint16_t* dst, int64_t n, int64_t width,
                          void* stream) {
    const int64_t per = width / 8;
    const int64_t total = n * per;
    const size_t blocks = (size_t) ((total + 255) / 256);
    using V4 = sycl::vec<uint32_t, 4>;
    Q(stream).parallel_for(sycl::range<1>(blocks * 256), [=](size_t ii) {
        const int64_t i = (int64_t) ii;
        if (i >= total) return;
        const int64_t r = i / per, j = i % per;
        reinterpret_cast<V4*>(dst)[r * per + j] = reinterpret_cast<const V4*>(x)[(int64_t) src[r] * per + j];
    });
    check(stream, "gather_rows16");
}

void moe_combine_launch(const float* Dm, const int32_t* slot, const float* w, const float* shared, const float* sg,
                        float* bo, int64_t T, void* stream) {
    const int64_t total = T * N;
    const size_t blocks = (size_t) ((total + 255) / 256);
    Q(stream).parallel_for(sycl::range<1>(blocks * 256), [=](size_t ii) {
        const int64_t i = (int64_t) ii;
        if (i >= total) return;
        const int64_t t = i / N, d = i % N;
        float s = 0.0f;
#pragma unroll
        for (int k = 0; k < 10; ++k) s = sycl::fma(w[t * 10 + k], Dm[(int64_t) slot[t * 10 + k] * N + d], s);
        bo[i] = s + shared[i] * sigm(sg[t]);
    });
    check(stream, "moe_combine");
}

// ---------------------------------------------------------------- QSA helpers
void rms_rows_launch(float* x, const float* w, int64_t rows, int64_t cols, int64_t ld, float eps, void* stream) {
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<float, 1> red(sycl::range<1>(256), hnd);
        local_accessor<float, 1> sh(sycl::range<1>(32), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) rows * 256), sycl::range<1>(256)),
                         [=](nd_item<1> it) {
                             const int64_t row = (int64_t) it.get_group(0);
                             const int lid = (int) it.get_local_id(0);
                             float* r = x + row * ld;
                             float ss = 0.0f;
                             for (int64_t c = lid; c < cols; c += 256) ss += r[c] * r[c];
                             const float s = sycl::rsqrt(block_sum_l(ss, red, sh, lid, 256, it) / (float) cols + eps);
                             it.barrier(sycl::access::fence_space::local_space);
                             for (int64_t c = lid; c < cols; c += 256) r[c] = s * r[c] * w[c];
                         });
    });
    check(stream, "rms_rows");
}

template <bool TAB>
void rope_launch(float* x, int64_t rows, int64_t heads, int64_t dim, int64_t ld, int64_t pos0, float theta_scale,
                 float freq_scale, float corr_low, float corr_high, float ext_factor, float mscale,
                 const int32_t* mtab, strata::kernels::RopeTab rt, void* stream) {
    Q(stream).submit([&](sycl::handler& hnd) {
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) rows * 32), sycl::range<1>(32)),
                         [=](nd_item<1> it) {
                             const int64_t row = (int64_t) it.get_group(0);
                             const int pair = (int) it.get_local_id(0);
                             const int64_t t = row / heads;
                             const int64_t h = row % heads;
                             float* p = x + t * ld + h * dim;
                             float c, s;
                             if (!(TAB && strata::kernels::rope_tab_cs(
                                              rt, strata::kernels::mrope_pos(mtab, (int) (pos0 + t), pair), pair, c,
                                              s))) {
                                 const float theta_extrap =
                                     (float) strata::kernels::mrope_pos(mtab, (int) (pos0 + t), pair) *
                                     sycl::pow(theta_scale, (float) pair);
                                 strata::kernels::rope_scaled_angle(theta_extrap, freq_scale, corr_low, corr_high,
                                                                    ext_factor, mscale, pair, c, s);
                             }
                             const float a = p[pair], b = p[pair + 32];
                             p[pair] = a * c - b * s;
                             p[pair + 32] = a * s + b * c;
                         });
    });
    check(stream, "rope");
}

void split_q_launch(const float* qf, float* q, int64_t T, void* stream) {
    const int64_t total = T * 24 * 256;
    const size_t blocks = (size_t) ((total + 255) / 256);
    Q(stream).parallel_for(sycl::range<1>(blocks * 256), [=](size_t ii) {
        const int64_t i = (int64_t) ii;
        if (i >= total) return;
        const int64_t t = i / (24 * 256), h = (i / 256) % 24, d = i % 256;
        q[i] = qf[t * 24 * 512 + h * 512 + d];
    });
    check(stream, "split_q");
}

void gate_attn_launch(const float* a, const float* qf, uint16_t* o16, int64_t T, void* stream) {
    const int64_t total = T * 24 * 256;
    const size_t blocks = (size_t) ((total + 255) / 256);
    Q(stream).parallel_for(sycl::range<1>(blocks * 256), [=](size_t ii) {
        const int64_t i = (int64_t) ii;
        if (i >= total) return;
        const int64_t t = i / (24 * 256), h = (i / 256) % 24, d = i % 256;
        o16[i] = hf(a[i] * (1.0f / (1.0f + sycl::exp(-qf[t * 24 * 512 + h * 512 + 256 + d]))));
    });
    check(stream, "gate_attn");
}

void kv_append_launch(const float* K, const float* V, int64_t pos0, const int32_t* table, int64_t page_size,
                      uint16_t* k_pool, uint16_t* v_pool, int8_t* k_q, int8_t* v_q, uint16_t* k_scale,
                      uint16_t* v_scale, strata::kernels::KvHostPools host, strata::kernels::KvHostPools stage,
                      int64_t T, void* stream) {
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<float, 1> red(sycl::range<1>(64), hnd);
        local_accessor<float, 1> wm(sycl::range<1>(2), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) T * 2 * 8 * 64), sycl::range<1>(64)),
                         [=](nd_item<1> it) {
                             const int64_t blk = (int64_t) it.get_group(0);   // dim3(T,2,8), x fastest
                             const int64_t t = blk % T;
                             const int kvh = (int) ((blk / T) % 2);
                             const int z = (int) (blk / (T * 2));
                             const int g = z >> 1;
                             const bool is_v = (z & 1) != 0;
                             const int lid = (int) it.get_local_id(0);
                             const int d = g * 64 + lid;
                             const float x = (is_v ? V : K)[t * 512 + kvh * 256 + d];
                             const int64_t pos = pos0 + t;
                             const int64_t page = (int64_t) table[pos / page_size];
                             const int64_t row = (page * 2 + kvh) * page_size + pos % page_size;
                             const int64_t row_id = ((pos / page_size) * 2 + kvh) * page_size + pos % page_size;
                             if (k_pool != nullptr) {
                                 const uint16_t hh = hf(x);
                                 if (page >= 0) (is_v ? v_pool : k_pool)[row * 256 + d] = hh;
                                 if (host.k_pool != nullptr) (is_v ? host.v_pool : host.k_pool)[row_id * 256 + d] = hh;
                                 if (stage.k_pool != nullptr) (is_v ? stage.v_pool : stage.k_pool)[row_id * 256 + d] = hh;
                                 return;
                             }
                             float a = sycl::fabs(x);
                             a = warp_max_l(a, lid, red, it);
                             if ((lid & 31) == 0) wm[lid >> 5] = a;
                             it.barrier(sycl::access::fence_space::local_space);
                             const float amax = sycl::fmax(wm[0], wm[1]);
                             const uint16_t sb = hf(amax / 127.0f);
                             const float sf = unhf(sb);
                             int q = 0;
                             if (sf > 0.0f) {
                                 q = (int) sycl::rint(x / sf);
                                 q = q < -127 ? -127 : (q > 127 ? 127 : q);
                             }
                             if (page >= 0) {
                                 (is_v ? v_q : k_q)[row * 256 + d] = (int8_t) q;
                                 if (lid == 0) (is_v ? v_scale : k_scale)[row * 4 + g] = sb;
                             }
                             if (host.k_q != nullptr) {
                                 (is_v ? host.v_q : host.k_q)[row_id * 256 + d] = (int8_t) q;
                                 if (lid == 0) (is_v ? host.v_scale : host.k_scale)[row_id * 4 + g] = sb;
                             }
                             if (stage.k_q != nullptr) {
                                 (is_v ? stage.v_q : stage.k_q)[row_id * 256 + d] = (int8_t) q;
                                 if (lid == 0) (is_v ? stage.v_scale : stage.k_scale)[row_id * 4 + g] = sb;
                             }
                         });
    });
    check(stream, "kv_append");
}

void to_f16_launch(const float* x, uint16_t* y, int64_t n, void* stream) {
    const int64_t b = (n + 255) / 256;
    const size_t blocks = (size_t) (b < 4096 ? b : 4096);
    Q(stream).parallel_for(sycl::range<1>(blocks * 256), [=](size_t i) {
        for (int64_t j = (int64_t) i; j < n; j += (int64_t) blocks * 256) y[j] = hf(x[j]);
    });
    check(stream, "to_f16");
}

void round_f16_launch(const float* x, float* y, int64_t n, void* stream) {
    const int64_t b = (n + 255) / 256;
    const size_t blocks = (size_t) (b < 4096 ? b : 4096);
    Q(stream).parallel_for(sycl::range<1>(blocks * 256), [=](size_t i) {
        for (int64_t j = (int64_t) i; j < n; j += (int64_t) blocks * 256) y[j] = unhf(hf(x[j]));
    });
    check(stream, "round_f16");
}

void to_bf16_launch(const float* x, uint16_t* y, uint16_t* ylo, int64_t n, void* stream) {
    const int64_t b = (n + 255) / 256;
    const size_t blocks = (size_t) (b < 4096 ? b : 4096);
    Q(stream).parallel_for(sycl::range<1>(blocks * 256), [=](size_t i) {
        for (int64_t j = (int64_t) i; j < n; j += (int64_t) blocks * 256) {
            const uint16_t h = bf(x[j]);
            y[j] = h;
            if (ylo) ylo[j] = bf_lo(x[j], h);
        }
    });
    check(stream, "to_bf16");
}

}  // namespace

// ---------------------------------------------------------------- host entry points
void kv_append(const float* K, const float* V, int64_t T, int64_t pos0, const int32_t* page_table, int64_t page_size,
               uint16_t* k_pool, uint16_t* v_pool, int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale,
               void* stream, const strata::kernels::KvHostPools* host, const strata::kernels::KvHostPools* stage) {
    if (T <= 0) return;
    kv_append_launch(K, V, pos0, page_table, page_size, k_pool, v_pool, k_q, v_q, k_scale, v_scale,
                     host ? *host : strata::kernels::KvHostPools{}, stage ? *stage : strata::kernels::KvHostPools{},
                     T, stream);
}
void to_f16(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    to_f16_launch(x, y, n, stream);
}
void round_f16(const float* x, float* y, int64_t n, void* stream) {
    if (n <= 0) return;
    round_f16_launch(x, y, n, stream);
}
void to_bf16(const float* x, uint16_t* y, int64_t n, void* stream, uint16_t* ylo) {
    if (n <= 0) return;
    to_bf16_launch(x, y, ylo, n, stream);
}

void gr_norm(const float* R, const float* w_norm, float eps, float* xn, uint16_t* xn16, int64_t T, void* stream,
             uint16_t* xn16_lo) {
    gr_norm_launch(R, w_norm, eps, xn, xn16, xn16_lo, T * HC, stream);
}
void gr_norm_rs(const float* R, const float* w_norm, float eps, float* rs, uint16_t* xn16, int64_t T, void* stream,
                uint16_t* xn16_lo) {
    gr_norm_rs_launch(R, w_norm, eps, rs, xn16, xn16_lo, T * HC, stream);
}
void gr_mix_r(const float* R, const float* rs, const float* w_norm, const float* gated, float* mixed, uint16_t* mixed16,
              int64_t T, void* stream, uint16_t* mixed_h, uint16_t* mixed16_lo) {
    gr_mix_r_launch(R, rs, w_norm, gated, mixed, mixed16, T, mixed_h, mixed16_lo, stream);
}
void gr_write_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* w_norm_next, float eps,
                      float* rs, uint16_t* xn16, int64_t T, void* stream, uint16_t* xn16_lo) {
    gr_write_norm_rs_launch(R, bo, inj, inj_ld, w_norm_next, eps, rs, xn16, xn16_lo, T * HC, stream);
}
void gr_silu(const float* lo, uint16_t* lo16, int64_t T, void* stream, uint16_t* lo16_lo) {
    gr_silu_launch(lo, lo16, lo16_lo, T * LR, stream);
}
void gr_mix(const float* xn, const float* gated, float* mixed, uint16_t* mixed16, int64_t T, void* stream,
            uint16_t* mixed_h, uint16_t* mixed16_lo) {
    gr_mix_launch(xn, gated, mixed, mixed16, T, mixed_h, mixed16_lo, stream);
}
void gr_write(float* R, const float* bo, const float* inj, int64_t inj_ld, int64_t T, void* stream) {
    gr_write_launch(R, bo, inj, inj_ld, T, stream);
}
void gr_broadcast(const float* e, float* R, int64_t T, void* stream) {
    gr_broadcast_launch(e, R, T, stream);
}
void gdn_gates(const float* ab, const float* dt, const float* ssm_a, float* gate, float* beta, int64_t T, void* stream) {
    gdn_gates_launch(ab, dt, ssm_a, gate, beta, T, stream);
}
void gdn_conv(float* history, const float* qkv, const float* conv_w, float* h, int64_t T, float eps, void* stream) {
    static const bool serial = std::getenv("STRATA_GDN_CONV_SERIAL") != nullptr;   // the old walk (A/B)
    if (serial || T <= CONV_TILE) {
        gdn_conv_walk_launch(history, qkv, conv_w, h, T, stream);
    } else {
        gdn_conv_tiled_launch(history, qkv, conv_w, h, T, stream);
        gdn_conv_hist_launch(history, qkv, T, stream);
    }
    gdn_l2_launch(h, eps, T, stream);
    check(stream, "gdn_conv");
}
void gdn_recurrence(float* state, const float* h, const float* gate, const float* beta, const float* z,
                    const float* gamma, float eps, float* y, uint16_t* y16, int64_t T, void* stream) {
    static const bool serial = std::getenv("STRATA_GDN_REC_HEADS") != nullptr;   // one-block-per-head kernel (A/B)
    if (serial || T <= 0) {
        gdn_rec_launch(state, h, gate, beta, z, gamma, eps, y, y16, T, stream);
    } else {
        static const bool pipe = [] {
            const char* v = std::getenv("STRATA_GDN_PIPELINE");
            return v == nullptr || std::atoi(v) != 0;
        }();
        gdn_rec_cols_launch(state, h, gate, beta, y, T, pipe, stream);
        gdn_out_norm_launch(z, gamma, eps, y, y16, T, stream);
    }
}
void route(const float* logits, int32_t* ids, float* weights, int64_t T, int64_t n_expert, void* stream) {
    if (n_expert == 512)
        route_launch<16>(logits, ids, weights, T, stream);
    else if (n_expert == 256)
        route_launch<8>(logits, ids, weights, T, stream);
    else
        strata::kernels::router_top10(logits, (int) T, (int) n_expert, 10, ids, weights, stream);
}
void blob_dequant(const uint8_t* blob, uint16_t* gu16, uint16_t* down16, void* stream) {
    blob_dequant_launch<false>(blob, gu16, down16, stream);
    check(stream, "blob_dequant");
}
void blob_dequant_f16(const uint8_t* blob, uint16_t* gu16, uint16_t* down16, void* stream) {
    blob_dequant_launch<true>(blob, gu16, down16, stream);
    check(stream, "blob_dequant_f16");
}
void swiglu_interleaved(const float* gu, uint16_t* h16, int64_t n, void* stream) {
    if (n <= 0) return;
    swiglu_il_launch(gu, h16, n, stream);
}
void swiglu_pair(const float* g, const float* u, uint16_t* h16, int64_t n, void* stream) {
    if (n <= 0) return;
    swiglu_pair_launch(g, u, h16, n, stream);
}
void copy_i32(int32_t* dst, const int32_t* src, int64_t n, void* stream) {
    if (n <= 0) return;
    copy_i32_launch(dst, src, n, stream);
}
void gather_rows16(const uint16_t* x16, const int32_t* src, uint16_t* dst16, int64_t n, int64_t width, void* stream) {
    if (n <= 0) return;
    gather_rows16_launch(x16, src, dst16, n, width, stream);
}
void moe_combine(const float* Dm, const int32_t* slot, const float* w, const float* shared, const float* sg, float* bo,
                 int64_t T, void* stream) {
    moe_combine_launch(Dm, slot, w, shared, sg, bo, T, stream);
}
void rms_rows(float* x, const float* w, int64_t rows, int64_t cols, int64_t ld, float eps, void* stream) {
    if (rows <= 0) return;
    rms_rows_launch(x, w, rows, cols, ld, eps, stream);
}
void rope(float* x, int64_t T, int64_t heads, int64_t dim, int64_t ld, int64_t pos0,
          const strata::kernels::RopeScaling& scaling, void* stream) {
    if (const char* why = strata::kernels::rope_scaling_invalid(scaling)) {
        std::fprintf(stderr, "prefill rope: invalid rope scaling: %s\n", why);
        std::exit(1);
    }
    const float theta_scale = std::pow((float) scaling.freq_base, -2.0f / 64.0f);
    const strata::kernels::RopeKernelArgs k = scaling.kernel_args(64);
    const strata::kernels::RopeTab rt = strata::kernels::rope_table_for(scaling);
    if (rt.cos != nullptr)
        rope_launch<true>(x, T * heads, heads, dim, ld, pos0, theta_scale, k.freq_scale, k.corr_low, k.corr_high,
                          k.ext_factor, k.attn_factor, strata::kernels::mrope_table(), rt, stream);
    else
        rope_launch<false>(x, T * heads, heads, dim, ld, pos0, theta_scale, k.freq_scale, k.corr_low, k.corr_high,
                           k.ext_factor, k.attn_factor, strata::kernels::mrope_table(), rt, stream);
}
void split_q(const float* q_full, float* q, int64_t T, void* stream) {
    split_q_launch(q_full, q, T, stream);
}
void gate_attn(const float* attn, const float* q_full, uint16_t* out16, int64_t T, void* stream) {
    gate_attn_launch(attn, q_full, out16, T, stream);
}

}  // namespace strata::prefill
