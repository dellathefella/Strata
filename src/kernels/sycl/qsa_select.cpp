// src/kernels/sycl/qsa_select.cpp — SYCL port of the PORTABLE variants of
// src/kernels/cuda/qsa_select.cu: block_scores_kernel (one warp per block:
// 4-float key chunks, per-indexer-head relu'd dots) and block_topk_kernel
// (the 8-bit radix select over block scores, weights = cells per block).
// The tensor-core / cluster / register variants stay CUDA-only: on SYCL the
// wrappers always take the portable path (same ids, same order).
#include "strata/kernels/qsa_select.hpp"

#include <cuda_runtime.h>  // sycl_compat shim

#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::kernels {
namespace {
constexpr int IDX_DIM = 128, IDX_HEADS = 4, R = 4;
constexpr int SCORE_WARPS = 8;
constexpr int TOPK_T = 256;

using sycl::local_accessor;
using sycl::nd_item;
using sycl::atomic_ref;
using sycl::memory_order;
using sycl::memory_scope;

inline sycl::queue& Q(void* stream) { return strata::sycl_compat::q_for(stream); }

inline uint32_t order_key(float s) {
    const float v = s + 0.0f;
    if (!(v == v)) return 0u;
    uint32_t b;
    memcpy(&b, &v, 4);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

}  // namespace

void qsa_block_scores(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                      int64_t max_blocks, const QsaShapes& s, float* scores, void* stream, int64_t active_blocks) {
    if (nq <= 0) return;
    if (s.idx_dim != IDX_DIM || s.idx_n_head != IDX_HEADS || s.idx_block != R || nq > 65535) {
        std::fprintf(stderr, "qsa_block_scores: unsupported indexer geometry\n");
        std::exit(1);
    }
    const int64_t reach = active_blocks > 0 && active_blocks < max_blocks ? active_blocks : max_blocks;
    const size_t groups_x = (size_t) ((reach + SCORE_WARPS - 1) / SCORE_WARPS);
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<float, 1> red(sycl::range<1>(SCORE_WARPS * 32), hnd);
        hnd.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t) nq, groups_x * SCORE_WARPS * 32),
                                           sycl::range<2>(1, SCORE_WARPS * 32)),
                         [=](nd_item<2> it) {
                             const int64_t qi = (int64_t) it.get_group(0);
                             const int tx = (int) it.get_local_id(1);
                             const int32_t* st = steps + qi * kStepCount;
                             const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid];
                             const int64_t b = (int64_t) it.get_group(1) * SCORE_WARPS + (tx >> 5);
                             if (b > n_bid || b >= max_blocks) return;  // uniform per warp-chunk
                             const int lane = tx & 31;
                             const float* key = (b == n_bid) ? dead : pooled + b * IDX_DIM;
                             const float k4[4] = {key[lane * 4], key[lane * 4 + 1], key[lane * 4 + 2],
                                                  key[lane * 4 + 3]};
                             const float* q = q_idx + qi * IDX_HEADS * IDX_DIM + lane * 4;
                             float score = 0.0f;
#pragma unroll
                             for (int h = 0; h < IDX_HEADS; ++h) {
                                 const float* q4 = q + (size_t) h * IDX_DIM;
                                 float d = k4[0] * q4[0] + k4[1] * q4[1] + k4[2] * q4[2] + k4[3] * q4[3];
                                 red[tx] = d;
                                 it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
                                 for (int o = 16; o > 0; o >>= 1) {
                                     const float other = red[tx ^ o];
                                     it.barrier(sycl::access::fence_space::local_space);
                                     d += other;
                                     red[tx] = d;
                                     it.barrier(sycl::access::fence_space::local_space);
                                 }
                                 score += d > 0.0f ? d : 0.0f;
                             }
                             if (lane == 0) {
                                 if (b == n_bid && n_kv % R != 0) score += 1e9f;
                                 scores[qi * max_blocks + b] = score;
                             }
                         });
    });
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_block_scores: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

bool qsa_block_scores_tc(const float*, const float*, const float*, const int32_t*, int64_t, int64_t, const QsaShapes&,
                         float*, void*, int64_t) {
    return false;  // no tensor cores: the caller falls back to the warp kernel
}

void qsa_block_topk(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                    const QsaShapes& s, int32_t* ids, void* stream, int64_t active_blocks) {
    (void) s;
    (void) active_blocks;
    if (nq <= 0) return;
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<int, 1> hist(sycl::range<1>(256), hnd);
        local_accessor<int, 1> s_a(sycl::range<1>(TOPK_T), hnd);
        local_accessor<int, 1> s_b(sycl::range<1>(TOPK_T), hnd);
        local_accessor<int, 1> s_digit(sycl::range<1>(1), hnd);
        local_accessor<int, 1> s_above(sycl::range<1>(1), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) nq * TOPK_T), sycl::range<1>(TOPK_T)),
                         [=](nd_item<1> it) {
                             const int64_t qi = (int64_t) it.get_group(0);
                             const int t = (int) it.get_local_id(0);
                             const int32_t* st = steps + qi * kStepCount;
                             const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
                             int32_t* out = ids + qi * cap;
                             if (n_kv <= width) {
                                 for (int64_t j = t; j < n_kv; j += TOPK_T) out[j] = (int32_t) j;
                                 return;
                             }
                             const float* sc = scores + qi * max_blocks;
                             const int64_t nb = n_bid + 1;
                             const int64_t per = (nb + TOPK_T - 1) / TOPK_T;
                             const int64_t b0 = (int64_t) t * per;
                             const int64_t b1 = (b0 + per < nb) ? b0 + per : nb;
                             auto weight = [&](int64_t b) -> int {
                                 return b < n_bid ? R : (int) (n_kv - n_bid * R);
                             };
                             uint32_t prefix = 0;
                             int above = 0;
                             for (int shift = 24; shift >= 0; shift -= 8) {
                                 for (int i = t; i < 256; i += TOPK_T) hist[i] = 0;
                                 it.barrier(sycl::access::fence_space::local_space);
                                 const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
                                 for (int64_t b = b0; b < b1; ++b) {
                                     const int w = weight(b);
                                     if (w == 0) continue;
                                     const uint32_t k = order_key(sc[b]);
                                     if ((k & hi_mask) == (prefix & hi_mask)) {
                                         atomic_ref<int, memory_order::relaxed, memory_scope::work_group> h(
                                             hist[(k >> shift) & 255]);
                                         h.fetch_add(w);
                                     }
                                 }
                                 it.barrier(sycl::access::fence_space::local_space);
                                 if (t == 0) {
                                     int cum = above, d = 255;
                                     for (; d > 0; --d) {
                                         if (cum + hist[d] >= width) break;
                                         cum += hist[d];
                                     }
                                     s_digit[0] = d;
                                     s_above[0] = cum;
                                 }
                                 it.barrier(sycl::access::fence_space::local_space);
                                 prefix |= (uint32_t) s_digit[0] << shift;
                                 above = s_above[0];
                                 it.barrier(sycl::access::fence_space::local_space);
                             }
                             const uint32_t thr = prefix;
                             const int64_t eq_budget = width - above;
                             int gt = 0, eq = 0;
                             for (int64_t b = b0; b < b1; ++b) {
                                 const int w = weight(b);
                                 if (w == 0) continue;
                                 const uint32_t k = order_key(sc[b]);
                                 if (k > thr)
                                     gt += w;
                                 else if (k == thr)
                                     eq += w;
                             }
                             s_a[t] = gt;
                             s_b[t] = eq;
                             it.barrier(sycl::access::fence_space::local_space);
                             if (t == 0) {
                                 int ag = 0, ae = 0;
                                 for (int i = 0; i < TOPK_T; ++i) {
                                     const int g = s_a[i], e = s_b[i];
                                     s_a[i] = ag;
                                     s_b[i] = ae;
                                     ag += g;
                                     ae += e;
                                 }
                             }
                             it.barrier(sycl::access::fence_space::local_space);
                             const int64_t eq_before = s_b[t];
                             int64_t my_eq = eq_budget - eq_before;
                             if (my_eq < 0) my_eq = 0;
                             if (my_eq > eq) my_eq = eq;
                             const int sel = gt + (int) my_eq;
                             it.barrier(sycl::access::fence_space::local_space);
                             s_a[t] = sel;
                             it.barrier(sycl::access::fence_space::local_space);
                             if (t == 0) {
                                 int a = 0;
                                 for (int i = 0; i < TOPK_T; ++i) {
                                     const int c = s_a[i];
                                     s_a[i] = a;
                                     a += c;
                                 }
                             }
                             it.barrier(sycl::access::fence_space::local_space);
                             int64_t wpos = s_a[t];
                             int64_t eq_left = my_eq;
                             for (int64_t b = b0; b < b1; ++b) {
                                 const int w = weight(b);
                                 if (w == 0) continue;
                                 const uint32_t k = order_key(sc[b]);
                                 if (k > thr) {
                                     for (int c = 0; c < w; ++c) out[wpos++] = (int32_t) (b * R + c);
                                 } else if (k == thr) {
                                     for (int c = 0; c < w && eq_left > 0; ++c, --eq_left)
                                         out[wpos++] = (int32_t) (b * R + c);
                                 }
                             }
                         });
    });
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_block_topk: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

void qsa_block_topk_ref(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                        const QsaShapes& s, int32_t* ids, void* stream) {
    qsa_block_topk(scores, steps, nq, max_blocks, cap, s, ids, stream, 0);
}

bool qsa_block_topk_cluster(const float*, const int32_t*, int64_t, int64_t, int64_t, const QsaShapes&, int32_t*,
                            void*) {
    return false;  // no thread-block clusters on SYCL
}

}  // namespace strata::kernels
