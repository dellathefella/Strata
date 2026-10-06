// src/kernels/sycl/fused_gr.cpp — SYCL port of src/kernels/cuda/fused_gr.cu,
// the PLAIN variant only (fused_gr_check reports kHcPlain; the split/staged
// CUDA variants are summation-order alternatives, not correctness paths).
//
// Transcription rules as elsewhere in this directory: xor butterflies become
// per-32-lane local trees with identical pairing; bf16 pairs expand from
// uint32 halves with sycl::bit_cast; fmaf -> sycl::fma; __expf -> sycl::exp.
// TILEV is 1280 (not 2560): the staged tile is T*TILEV floats of local memory
// and B60 work-groups cap at 64 KiB; the CUDA source states both tile sizes
// are bitwise identical (the lane's chunk order is strictly increasing either
// way). Local-memory budgets here: gr_down single 40 KiB (xn[D]), gr_down_multi
// 40 KiB (tile[8][1280]), gr_up* < 12 KiB.
#include "strata/core/emulate.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>  // sycl_compat shim

#include <atomic>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int N = 2560;    // n_embd
constexpr int HC = 4;      // streams
constexpr int D = N * HC;  // 10240
constexpr int LR = 320;    // hc_lr
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int DOWN_BLOCKS = LR / WARPS;  // 40 blocks of 8 rows; one more for inject
constexpr int UP_COLS = 32;
constexpr int UP_BLOCKS = N / UP_COLS;  // 80
constexpr int UPM_COLS = 16;
constexpr int UPM_BLOCKS = N / UPM_COLS;  // 160
constexpr int TILEV = 1280;
constexpr int TQ = TILEV / 8 / 32;  // 5 uint4 weight chunks per lane per tile

using sycl::local_accessor;
using sycl::nd_item;

inline sycl::queue& Q(void* stream) { return strata::sycl_compat::q_for(stream); }

void check_launch(void* stream, const char* what) {
    if (stream != nullptr) return;
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

inline float sigmoidf_(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }

// 8 bf16 (packed as 4 uint32) against 8 floats, same order as CUDA's dot8.
inline float dot8(const uint32_t w[4], const float* x) {
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        acc = sycl::fma(sycl::bit_cast<float>(w[j] << 16), x[2 * j], acc);
        acc = sycl::fma(sycl::bit_cast<float>(w[j] & 0xffff0000u), x[2 * j + 1], acc);
    }
    return acc;
}

inline uint32_t bf16_pair(const uint16_t* w, int i) {
    return (uint32_t) w[i] | ((uint32_t) w[i + 1] << 16);
}

// warp_sum: xor 16..1 over the 32-lane chunk containing `lid`.
inline float warp_sum(float v, int lid, local_accessor<float, 1> red, nd_item<1> it) {
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

// ---- single-token down: norm into local xn, then one warp per output row ----
void gr_down(const FusedGrArgs& a, void* stream) {
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<float, 1> xn(sycl::range<1>(D), hnd);
        local_accessor<float, 2> part(sycl::range<2>(WARPS, HC), hnd);
        local_accessor<float, 1> s_rs(sycl::range<1>(HC), hnd);
        local_accessor<float, 1> red(sycl::range<1>(THREADS), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>((DOWN_BLOCKS + 1) * THREADS),
                                           sycl::range<1>(THREADS)),
                         [=](nd_item<1> it) {
                             const int bx = (int) it.get_group(0);
                             const int t = (int) it.get_local_id(0);
                             const int lane = t & 31, warp = t >> 5;
                             float gw[HC];
#pragma unroll
                             for (int c = 0; c < HC; ++c)
                                 gw[c] = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
                             float ss[HC] = {0.0f, 0.0f, 0.0f, 0.0f};
                             for (int i = t * 4; i < D; i += THREADS * 4) {
                                 const int c = i / N, d = i - c * N;
                                 float r[4] = {a.R[i], a.R[i + 1], a.R[i + 2], a.R[i + 3]};
                                 if (a.apply) {
#pragma unroll
                                     for (int e = 0; e < 4; ++e)
                                         r[e] = sycl::fma(a.bo_prev[d + e], gw[c], r[e]);
                                 }
                                 const float sq = r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3];
                                 ss[c] += sq;
#pragma unroll
                                 for (int e = 0; e < 4; ++e) xn[i + e] = r[e] * a.w_norm[i + e];
                             }
#pragma unroll
                             for (int c = 0; c < HC; ++c) {
                                 const float v = warp_sum(ss[c], t, red, it);
                                 if (lane == 0) part[warp][c] = v;
                             }
                             it.barrier(sycl::access::fence_space::local_space);
                             if (t < HC) {
                                 float s = 0.0f;
                                 for (int w = 0; w < WARPS; ++w) s += part[w][t];
                                 s_rs[t] = sycl::rsqrt(s / (float) N + a.eps);
                                 if (bx == 0) a.rs[t] = s_rs[t];
                             }
                             it.barrier(sycl::access::fence_space::local_space);
                             for (int i = t; i < D; i += THREADS) xn[i] *= s_rs[i / N];
                             it.barrier(sycl::access::fence_space::local_space);
                             const bool inject_block = bx == DOWN_BLOCKS;
                             const int row = inject_block ? warp : bx * WARPS + warp;
                             // predicate, do not return: warp_sum's barriers are
                             // work-group-wide (inactive chunks reduce zeros)
                             const bool active =
                                 !(inject_block && (a.w_inject == nullptr || warp >= HC));
                             const uint16_t* wrow =
                                 (inject_block ? a.w_inject : a.w_down) + (size_t) row * D;
                             float acc = 0.0f;
                             if (active) {
                                 for (int j = lane; j < D / 8; j += 32) {
                                     uint32_t wv[4];
#pragma unroll
                                     for (int e = 0; e < 4; ++e) wv[e] = bf16_pair(wrow, j * 8 + e * 2);
                                     acc += dot8(wv, &xn[j * 8]);
                                 }
                             }
                             acc = warp_sum(active ? acc : 0.0f, t, red, it);
                             if (lane != 0 || !active) return;
                             if (inject_block) {
                                 a.inject_out[row] = acc;
                             } else {
                                 const float x = acc / (float) HC;
                                 a.lo[row] = x / (1.0f + sycl::exp(-x));
                             }
                         });
    });
    check_launch(stream, "gr_down");
}

// ---- single-token up ----
void gr_up(const FusedGrArgs& a, void* stream) {
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<float, 1> lo(sycl::range<1>(LR), hnd);
        local_accessor<float, 2> g(sycl::range<2>(HC, UP_COLS), hnd);
        local_accessor<float, 1> red(sycl::range<1>(THREADS), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>(UP_BLOCKS * THREADS), sycl::range<1>(THREADS)),
                         [=](nd_item<1> it) {
                             const int bx = (int) it.get_group(0);
                             const int t = (int) it.get_local_id(0);
                             const int lane = t & 31, warp = t >> 5;
                             const int d0 = bx * UP_COLS;
                             for (int k = t; k < LR; k += THREADS) lo[k] = a.lo[k];
                             it.barrier(sycl::access::fence_space::local_space);
                             for (int r = warp; r < HC * UP_COLS; r += WARPS) {
                                 const int c = r / UP_COLS, dd = r - c * UP_COLS, i = c * N + d0 + dd;
                                 const uint16_t* wrow = a.w_up + (size_t) i * LR;
                                 uint32_t wa[4], wb[4];
#pragma unroll
                                 for (int e = 0; e < 4; ++e) wa[e] = bf16_pair(wrow, lane * 8 + e * 2);
                                 if (lane < LR / 8 - 32) {
#pragma unroll
                                     for (int e = 0; e < 4; ++e)
                                         wb[e] = bf16_pair(wrow, (32 + lane) * 8 + e * 2);
                                 }
                                 float acc = dot8(wa, &lo[lane * 8]);
                                 if (lane < LR / 8 - 32) acc += dot8(wb, &lo[(32 + lane) * 8]);
                                 acc = warp_sum(acc, t, red, it);
                                 if (lane == 0) {
                                     float rv = a.R[i];
                                     if (a.apply) {
                                         rv = sycl::fma(a.bo_prev[d0 + dd],
                                                        2.0f * sigmoidf_(a.inj_prev[c] / (float) HC), rv);
                                         a.R_out[i] = rv;
                                     }
                                     const float x = rv * a.w_norm[i] * a.rs[c];
                                     g[c][dd] = x * sigmoidf_(acc);
                                 }
                             }
                             it.barrier(sycl::access::fence_space::local_space);
                             if (t < UP_COLS) {
                                 float s = 0.0f;
#pragma unroll
                                 for (int c = 0; c < HC; ++c) s += g[c][t];
                                 a.mixed[d0 + t] = s / (float) HC;
                             }
                         });
    });
    check_launch(stream, "gr_up");
}

// ---- multi: step 1, per-token norm into the global xn scratch ----
void gr_norm_multi(const FusedGrArgs* a, int T, float* xn, void* stream) {
    FusedGrArgs av[kFusedGrMaxT];
    for (int i = 0; i < kFusedGrMaxT; ++i) av[i] = a[i < T ? i : 0];
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<float, 2> part(sycl::range<2>(WARPS, HC), hnd);
        local_accessor<float, 1> s_rs(sycl::range<1>(HC), hnd);
        local_accessor<float, 1> red(sycl::range<1>(THREADS), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) T * THREADS), sycl::range<1>(THREADS)),
                         [=](nd_item<1> it) {
                             const FusedGrArgs& aa = av[it.get_group(0)];
                             float* my_xn = xn + it.get_group(0) * (size_t) D;
                             const int t = (int) it.get_local_id(0);
                             const int lane = t & 31, warp = t >> 5;
                             float gw[HC];
#pragma unroll
                             for (int c = 0; c < HC; ++c)
                                 gw[c] = aa.apply ? 2.0f * sigmoidf_(aa.inj_prev[c] / (float) HC) : 0.0f;
                             float ss[HC] = {0.0f, 0.0f, 0.0f, 0.0f};
                             for (int i = t * 4; i < D; i += THREADS * 4) {
                                 const int c = i / N, d = i - c * N;
                                 float r[4] = {aa.R[i], aa.R[i + 1], aa.R[i + 2], aa.R[i + 3]};
                                 if (aa.apply) {
#pragma unroll
                                     for (int e = 0; e < 4; ++e)
                                         r[e] = sycl::fma(aa.bo_prev[d + e], gw[c], r[e]);
                                 }
                                 const float sq = r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3];
                                 ss[c] += sq;
#pragma unroll
                                 for (int e = 0; e < 4; ++e) my_xn[i + e] = r[e] * aa.w_norm[i + e];
                             }
#pragma unroll
                             for (int c = 0; c < HC; ++c) {
                                 const float v = warp_sum(ss[c], t, red, it);
                                 if (lane == 0) part[warp][c] = v;
                             }
                             it.barrier(sycl::access::fence_space::local_space);
                             if (t < HC) {
                                 float s = 0.0f;
                                 for (int w = 0; w < WARPS; ++w) s += part[w][t];
                                 s_rs[t] = sycl::rsqrt(s / (float) N + aa.eps);
                                 aa.rs[t] = s_rs[t];
                             }
                             it.barrier(sycl::access::fence_space::local_space);
                             for (int i = t; i < D; i += THREADS) my_xn[i] *= s_rs[i / N];
                         });
    });
    check_launch(stream, "gr_norm_multi");
}

// ---- multi: step 2, tiled down projection over all T tokens ----
void gr_down_multi(const FusedGrArgs* a, int T, const float* xn, void* stream) {
    FusedGrArgs av[kFusedGrMaxT];
    for (int i = 0; i < kFusedGrMaxT; ++i) av[i] = a[i < T ? i : 0];
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<float, 1> tile(sycl::range<1>(kFusedGrMaxT * TILEV), hnd);
        local_accessor<float, 1> red(sycl::range<1>(THREADS), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>((DOWN_BLOCKS + 1) * THREADS),
                                           sycl::range<1>(THREADS)),
                         [=](nd_item<1> it) {
                             const int bx = (int) it.get_group(0);
                             const int t = (int) it.get_local_id(0);
                             const int lane = t & 31, warp = t >> 5;
                             const bool inject_block = bx == DOWN_BLOCKS;
                             const int row = inject_block ? warp : bx * WARPS + warp;
                             const bool active =
                                 !(inject_block && (av[0].w_inject == nullptr || warp >= HC));
                             const uint16_t* wrow =
                                 (inject_block ? av[0].w_inject : av[0].w_down) +
                                 (size_t) (active ? row : 0) * D;
                             float acc[kFusedGrMaxT];
#pragma unroll
                             for (int k = 0; k < kFusedGrMaxT; ++k) acc[k] = 0.0f;
                             for (int base = 0; base < D; base += TILEV) {
                                 uint32_t wv[TQ][4];
                                 if (active) {
#pragma unroll
                                     for (int q = 0; q < TQ; ++q) {
                                         const int j = lane + 32 * q;
#pragma unroll
                                         for (int e = 0; e < 4; ++e)
                                             wv[q][e] = bf16_pair(wrow, (base + j * 8) + e * 2);
                                     }
                                 }
                                 it.barrier(sycl::access::fence_space::local_space);
                                 for (int i = t; i < T * (TILEV / 4); i += THREADS) {
                                     const int k = i / (TILEV / 4), off = i - k * (TILEV / 4);
                                     const size_t src = ((size_t) k * D + base) / 4 + off;
#pragma unroll
                                     for (int e = 0; e < 4; ++e)
                                         tile[k * TILEV + off * 4 + e] = xn[src * 4 + e];
                                 }
                                 it.barrier(sycl::access::fence_space::local_space);
                                 if (!active) continue;
#pragma unroll
                                 for (int q = 0; q < TQ; ++q) {
                                     const int j = lane + 32 * q;
#pragma unroll
                                     for (int k = 0; k < kFusedGrMaxT; ++k)
                                         if (k < T) acc[k] += dot8(wv[q], &tile[k * TILEV + j * 8]);
                                 }
                             }
                             // EVERY warp reduces on EVERY k slot: the tree carries
                             // work-group barriers, so the call count must not depend
                             // on `active` or on k < T (a mismatch here deadlocked the
                             // group; the unbounded spin then tripped the xe job
                             // timeout and wedged the card)
                             float s[kFusedGrMaxT];
#pragma unroll
                             for (int k = 0; k < kFusedGrMaxT; ++k)
                                 s[k] = warp_sum(active && k < T ? acc[k] : 0.0f, t, red, it);
#pragma unroll
                             for (int k = 0; k < kFusedGrMaxT; ++k) {
                                 if (!active || k >= T || lane != k) continue;
                                 if (inject_block) {
                                     av[k].inject_out[row] = s[k];
                                 } else {
                                     const float x = s[k] / (float) HC;
                                     av[k].lo[row] = x / (1.0f + sycl::exp(-x));
                                 }
                             }
                         });
    });
    check_launch(stream, "gr_down_multi");
}

// ---- multi: step 3, up projection with per-lane token epilogues ----
void gr_up_multi(const FusedGrArgs* a, int T, void* stream) {
    FusedGrArgs av[kFusedGrMaxT];
    for (int i = 0; i < kFusedGrMaxT; ++i) av[i] = a[i < T ? i : 0];
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<float, 2> lo(sycl::range<2>(kFusedGrMaxT, LR), hnd);
        local_accessor<float, 3> g(sycl::range<3>(kFusedGrMaxT, HC, UPM_COLS), hnd);
        local_accessor<float, 1> red(sycl::range<1>(THREADS), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>(UPM_BLOCKS * THREADS), sycl::range<1>(THREADS)),
                         [=](nd_item<1> it) {
                             const int bx = (int) it.get_group(0);
                             const int t = (int) it.get_local_id(0);
                             const int lane = t & 31, warp = t >> 5;
                             const int d0 = bx * UPM_COLS;
                             for (int i = t; i < T * LR; i += THREADS) lo[i / LR][i % LR] = av[i / LR].lo[i % LR];
                             it.barrier(sycl::access::fence_space::local_space);
                             for (int r = warp; r < HC * UPM_COLS; r += WARPS) {
                                 const int c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
                                 const uint16_t* wrow = av[0].w_up + (size_t) i * LR;
                                 uint32_t wa[4], wb[4];
#pragma unroll
                                 for (int e = 0; e < 4; ++e) wa[e] = bf16_pair(wrow, lane * 8 + e * 2);
                                 const bool has_b = lane < LR / 8 - 32;
                                 if (has_b) {
#pragma unroll
                                     for (int e = 0; e < 4; ++e)
                                         wb[e] = bf16_pair(wrow, (32 + lane) * 8 + e * 2);
                                 }
                                 float rv = 0.0f, wn = 0.0f, rsc = 0.0f, bo = 0.0f, ip = 0.0f;
                                 bool apply = false;
                                 if (lane < T) {
                                     const FusedGrArgs& aa = av[lane];
                                     rv = aa.R[i];
                                     wn = aa.w_norm[i];
                                     rsc = aa.rs[c];
                                     apply = aa.apply;
                                     if (apply) {
                                         bo = aa.bo_prev[d0 + dd];
                                         ip = aa.inj_prev[c];
                                     }
                                 }
                                 float mine = 0.0f;
#pragma unroll
                                 for (int k = 0; k < kFusedGrMaxT; ++k) {
                                     if (k >= T) break;
                                     float acc = dot8(wa, &lo[k][lane * 8]);
                                     if (has_b) acc += dot8(wb, &lo[k][(32 + lane) * 8]);
                                     acc = warp_sum(acc, t, red, it);
                                     if (lane == k) mine = acc;
                                 }
                                 if (lane < T) {
                                     if (apply) {
                                         rv = sycl::fma(bo, 2.0f * sigmoidf_(ip / (float) HC), rv);
                                         av[lane].R_out[i] = rv;
                                     }
                                     const float x = rv * wn * rsc;
                                     g[lane][c][dd] = x * sigmoidf_(mine);
                                 }
                             }
                             it.barrier(sycl::access::fence_space::local_space);
                             for (int i = t; i < T * UPM_COLS; i += THREADS) {
                                 const int k = i / UPM_COLS, col = i - k * UPM_COLS;
                                 float s = 0.0f;
#pragma unroll
                                 for (int c = 0; c < HC; ++c) s += g[k][c][col];
                                 av[k].mixed[d0 + col] = s / (float) HC;
                             }
                         });
    });
    check_launch(stream, "gr_up_multi");
}

std::atomic<int> g_variant[1] = {0};
constexpr int kHcPlain = 1;

}  // namespace

void fused_gr_read(const FusedGrArgs& a, void* stream) {
    gr_down(a, stream);
    gr_up(a, stream);
}

void fused_gr_read_multi(const FusedGrArgs* a, int n_tok, float* xn_scratch, void* stream,
                         unsigned long long* stamp_buf, int stamp_i0) {
    if (n_tok < 1 || n_tok > kFusedGrMaxT) {
        std::fprintf(stderr, "fused_gr_read_multi: n_tok %d out of range\n", n_tok);
        std::exit(1);
    }
    static const bool fdbg = std::getenv("STRATA_FGR_SYNC") != nullptr;
    auto fsync = [&](const char* n) {
        if (!fdbg) return;
        const cudaError_t e = cudaStreamSynchronize((cudaStream_t) stream);
        std::fprintf(stderr, "[dbg] fgr %-9s %s\n", n, cudaGetErrorString(e));
    };
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0, stream);
    gr_norm_multi(a, n_tok, xn_scratch, stream);
    fsync("norm");
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0 + 1, stream);
    gr_down_multi(a, n_tok, xn_scratch, stream);
    fsync("down");
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0 + 2, stream);
    gr_up_multi(a, n_tok, stream);
    fsync("up");
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0 + 3, stream);
}

bool fused_gr_supported(int64_t n_embd, int64_t hc, int64_t hc_lr) {
    return n_embd == N && hc == HC && hc_lr == LR;
}

void fused_gr_check() {
    if (g_variant[0].load() > 0) return;
    g_variant[0].store(kHcPlain);
    std::fprintf(stderr, "strata hc: SYCL: the hyper-connection read runs as the plain fused port\n");
}

int fused_gr_variant() { return g_variant[0].load() > 0 ? g_variant[0].load() : kHcPlain; }

}  // namespace strata::kernels
