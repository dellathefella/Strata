// src/kernels/sycl/qsa.cpp — SYCL port of src/kernels/cuda/qsa.cu
// (fp16 KV append/gather, indexer key pooling + rotation, block scoring,
// binary-lifting top-k, sparse attend, output gate).
//
// Transcription rules as elsewhere: xor/up butterflies become local-memory
// trees with identical pairing and order; the indexer's double-precision
// sums keep plain IEEE double arithmetic (the port compiles with
// -ffp-contract=off; icpx device fp64 division is documented in
// docs/SYCL_PORT.md as inexact at default flags — the qsa parity gate is
// tolerance-based, re-audit if it tightens); __ldg -> plain loads (USM).
#include "strata/core/emulate.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/rope.hpp"

#include <cuda_runtime.h>  // sycl_compat shim

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::kernels {
namespace {
constexpr int THREADS = 128;
constexpr int TOPK_THREADS = 256;

using sycl::local_accessor;
using sycl::nd_item;

inline sycl::queue& Q(void* stream) { return strata::sycl_compat::q_for(stream); }

void fail(const char* what) {
    std::fprintf(stderr, "qsa: %s\n", what);
    std::exit(1);
}
void check_launch(void* stream, const char* what) {
    if (stream != nullptr) return;
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa: %s launch: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}
void validate(const QsaShapes& s, const char* who) {
    if (s.n_head <= 0 || s.n_head_kv <= 0 || s.head_dim <= 0 || s.idx_dim <= 0 || s.idx_n_head <= 0 ||
        s.idx_block < 2 || s.page_size < 1) {
        std::fprintf(stderr, "qsa: %s: geometry is not set up\n", who);
        std::exit(1);
    }
    if (s.n_head % s.n_head_kv != 0) {
        std::fprintf(stderr, "qsa: %s: n_head %lld is not a multiple of n_head_kv %lld\n", who, (long long) s.n_head,
                     (long long) s.n_head_kv);
        std::exit(1);
    }
    if (s.head_dim % 4 != 0) {
        std::fprintf(stderr, "qsa: %s: head_dim %lld must be a multiple of 4 (the gather copies uint2)\n", who,
                     (long long) s.head_dim);
        std::exit(1);
    }
    if (s.n_rot <= 0 || s.n_rot % 2 != 0 || s.n_rot > s.head_dim || s.n_rot > s.idx_dim) {
        std::fprintf(stderr, "qsa: %s: n_rot %lld must be even and <= head_dim %lld and idx_dim %lld\n", who,
                     (long long) s.n_rot, (long long) s.head_dim, (long long) s.idx_dim);
        std::exit(1);
    }
    if (s.idx_n_head > 32) {
        std::fprintf(stderr, "qsa: %s: idx_n_head %lld > 32 (one warp per indexer head)\n", who,
                     (long long) s.idx_n_head);
        std::exit(1);
    }
}

inline float h2f(uint16_t bits) { return f32_from_f16(bits); }

inline unsigned grid_for(long long n, int threads) { return (unsigned) ((n + threads - 1) / threads); }

int32_t* step_scratch() {
    static int32_t* d_step = nullptr;
    if (d_step == nullptr) {
        if (cudaMalloc(&d_step, qsa_step_bytes()) != cudaSuccess) {
            std::fprintf(stderr, "qsa: step upload: cudaMalloc failed\n");
            std::exit(1);
        }
    }
    return d_step;
}
void step_upload_raw(const int32_t* h_step) {
    int32_t* d = step_scratch();
    if (cudaMemcpy(d, h_step, qsa_step_bytes(), cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "qsa: step upload: cudaMemcpy failed\n");
        std::exit(1);
    }
}
const int32_t* step_upload(int64_t pos, int64_t n_kv_hint, const QsaShapes& s) {
    int32_t h[kStepCount];
    qsa_step_fill(h, pos, s);
    if (n_kv_hint >= 0 && n_kv_hint != (int64_t) h[kStepNKv]) {
        std::fprintf(stderr, "qsa: step upload: n_kv %lld disagrees with pos+1 = %d\n", (long long) n_kv_hint,
                     h[kStepNKv]);
        std::exit(1);
    }
    step_upload_raw(h);
    return step_scratch();
}
const int32_t* step_upload_width(int64_t width, const QsaShapes& s) {
    int32_t h[kStepCount];
    for (int i = 0; i < kStepCount; ++i) h[i] = 0;
    h[kStepWidth] = (int32_t) width;
    (void) s;
    step_upload_raw(h);
    return step_scratch();
}

inline uint32_t order_key(float s) {
    const float v = s + 0.0f;
    if (!(v == v)) return 0u;
    uint32_t b;
    memcpy(&b, &v, 4);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// max/sum butterflies over the 32-lane chunk containing `lid`
inline float chunk_max(float v, int lid, local_accessor<float, 1> red, nd_item<1> it) {
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
inline float chunk_sum(float v, int lid, local_accessor<float, 1> red, nd_item<1> it) {
    red[lid] = v;
    it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float other = red[lid ^ o];
        it.barrier(sycl::access::fence_space::local_space);
        v += other;
        red[lid] = v;
        it.barrier(sycl::access::fence_space::local_space);
    }
    return v;
}

}  // namespace

void qsa_step_fill(int32_t* host_step, int64_t pos, const QsaShapes& s) {
    if (host_step == nullptr) return;
    if (pos < 0) fail("qsa_step_fill: pos < 0");
    const int64_t n_kv = pos + 1;
    host_step[kStepPos] = (int32_t) pos;
    host_step[kStepNKv] = (int32_t) n_kv;
    host_step[kStepNBid] = (int32_t) (n_kv / s.idx_block);
    host_step[kStepWidth] = (int32_t) qsa_selection_width(n_kv, s);
}

void kv_append_step(uint16_t* k_pool, uint16_t* v_pool, const int32_t* page_table, const int32_t* step,
                    const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                    const KvHostPools* host) {
    validate(s, "kv_append");
    const KvHostPools pools = host ? *host : KvHostPools{};
    const int kv_heads = (int) s.n_head_kv;
    const int head_dim = (int) s.head_dim;
    const int page_size = (int) s.page_size;
    const size_t n = (size_t) kv_heads * head_dim;
    Q(stream).parallel_for(n, [=](size_t i) {
        const long long pos = (long long) step[kStepPos];
        const int h = (int) (i / head_dim), d = (int) (i % head_dim);
        const long long page = (long long) page_table[pos / page_size];
        if (page >= 0) {
            const long long row = (page * kv_heads + h) * page_size + (pos % page_size);
            k_pool[row * head_dim + d] = f16_from_f32(kcur[i]);
            v_pool[row * head_dim + d] = f16_from_f32(vcur[i]);
        }
        if (pools.k_pool != nullptr) {
            const long long row = ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size);
            pools.k_pool[row * head_dim + d] = f16_from_f32(kcur[i]);
            pools.v_pool[row * head_dim + d] = f16_from_f32(vcur[i]);
        }
    });
    check_launch(stream, "kv_append");
}

void kv_append(uint16_t* k_pool, uint16_t* v_pool, const int32_t* page_table, int64_t pos, const float* kcur,
               const float* vcur, const QsaShapes& s, void* stream) {
    kv_append_step(k_pool, v_pool, page_table, step_upload(pos, -1, s), kcur, vcur, s, stream, nullptr);
}

void indexer_key_append(const float* raw, const int32_t* pos_dev, int32_t pos_base, const float* w_k_norm, float eps,
                        const QsaIndexerBuffers& b, const QsaShapes& s, const float* cos_tab, const float* sin_tab,
                        void* stream) {
    validate(s, "indexer_key_append");
    const int idx_dim = (int) s.idx_dim;
    const int r = (int) s.idx_block;
    const int n_rot = (int) s.n_rot;
    float* tail = b.tail;
    float* dead = b.dead;
    float* pooled = b.pooled;
    int32_t* block_pos = b.block_pos;
    const int32_t* mtab = mrope_table();  // host-side lookup, captured like a kernel arg
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<double, 1> s_mean(sycl::range<1>(idx_dim), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>(idx_dim), sycl::range<1>(idx_dim)), [=](nd_item<1> it) {
            const int d = (int) it.get_local_id(0);
            const int pos = pos_dev[0];
            const int slot = pos % r;
            if (slot < r - 1) tail[(size_t) slot * idx_dim + d] = raw[d];
            if (pos == 0) {
                const double p = (double) raw[d];
                double ss = 0.0;
                for (int i = 0; i < idx_dim; ++i) {
                    const double v = (double) raw[i];
                    ss += v * v;
                }
                const double inv = 1.0 / sycl::sqrt((double) (ss / (double) idx_dim + (double) eps));
                float y = (float) (p * inv * (double) w_k_norm[d]);
                if (d < n_rot) y *= cos_tab[d % (n_rot / 2)];
                dead[d] = y;
                pooled[d] = y;
            }
            if (slot != r - 1) return;
            it.barrier(sycl::access::fence_space::local_space);
            double m = 0.0;
            for (int j = 0; j < r - 1; ++j) m += (double) tail[(size_t) j * idx_dim + d];
            m += (double) raw[d];
            m = m / (double) r;
            s_mean[d] = m;
            it.barrier(sycl::access::fence_space::local_space);
            double ss = 0.0;
            for (int i = 0; i < idx_dim; ++i) ss += s_mean[i] * s_mean[i];
            const double inv = 1.0 / sycl::sqrt((double) (ss / (double) idx_dim + (double) eps));
            const int bidx = pos / r;
            pooled[(size_t) bidx * idx_dim + d] = (float) (m * inv * (double) w_k_norm[d]);
            pooled[(size_t) (bidx + 1) * idx_dim + d] = dead[d];
            if (d == 0) *block_pos = (int32_t) (pos_base + bidx * r);
            it.barrier(sycl::access::fence_space::local_space);
            const int half = n_rot / 2;
            if (d < half) {
                float* row = pooled + (size_t) bidx * idx_dim;
                const size_t toff = (size_t) mrope_pos(mtab, pos_base + bidx * r, d) * half;
                float oa, ob;
                rope_neox_pair(row[d], row[half + d], cos_tab[toff + d], sin_tab[toff + d], oa, ob);
                row[d] = oa;
                row[half + d] = ob;
            }
        });
    });
    check_launch(stream, "indexer_key_append");
}

void qsa_index_step(const float* pooled, const float* q_idx, const float* bias, const QsaShapes& s,
                    const int32_t* step, int64_t max_blocks, float* cell_scores, void* stream) {
    validate(s, "qsa_index");
    const int idx_dim = (int) s.idx_dim;
    const int idx_n_head = (int) s.idx_n_head;
    const long long r = s.idx_block;
    const size_t threads = (size_t) 32 * idx_n_head;
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<double, 1> s_dot(sycl::range<1>(32), hnd);
        local_accessor<double, 1> red(sycl::range<1>(threads), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) max_blocks * threads), sycl::range<1>(threads)),
                         [=](nd_item<1> it) {
                             const long long n_bid = (long long) step[kStepNBid];
                             const long long n_kv = (long long) step[kStepNKv];
                             const long long b = (long long) it.get_group(0);
                             const int tx = (int) it.get_local_id(0);
                             const int wid = tx >> 5, lane = tx & 31;
                             if (b > n_bid) return;  // uniform per group
                             if (wid == 0) s_dot[lane] = 0.0;
                             it.barrier(sycl::access::fence_space::local_space);
                             double acc = 0.0;
                             for (int d = lane; d < idx_dim; d += 32)
                                 acc += (double) pooled[(size_t) b * idx_dim + d] *
                                        (double) q_idx[(size_t) wid * idx_dim + d];
                             red[tx] = (float) acc;
                             it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
                             for (int o = 16; o > 0; o >>= 1) {
                                 const double other = (double) red[tx ^ o];
                                 it.barrier(sycl::access::fence_space::local_space);
                                 acc += other;
                                 red[tx] = (float) acc;
                                 it.barrier(sycl::access::fence_space::local_space);
                             }
                             if (lane == 0) s_dot[wid] = acc;
                             it.barrier(sycl::access::fence_space::local_space);
                             if (tx != 0) return;
                             double score = 0.0;
                             for (int h = 0; h < idx_n_head; ++h) score += (s_dot[h] > 0.0) ? s_dot[h] : 0.0;
                             if (bias != nullptr) score += (double) bias[b];
                             long long lo = b * r, hi = lo + r;
                             if (b == n_bid) hi = n_kv;
                             if (hi > n_kv) hi = n_kv;
                             float sc = (float) score;
                             if (b == n_bid && n_kv % r != 0) sc += 1e9f;
                             for (long long j = lo; j < hi; ++j) cell_scores[j] = sc;
                         });
    });
    check_launch(stream, "qsa_index");
}

void qsa_index(const float* pooled, int64_t n_bid, const float* q_idx, const float* bias, const QsaShapes& s,
               int64_t n_kv, float* cell_scores, void* stream) {
    int32_t h[kStepCount] = {0, 0, 0, 0};
    h[kStepNBid] = (int32_t) n_bid;
    h[kStepNKv] = (int32_t) n_kv;
    step_upload_raw(h);
    qsa_index_step(pooled, q_idx, bias, s, step_scratch(), n_bid + 1, cell_scores, stream);
}

void topk_512_step(const float* cell_scores, const QsaShapes& s, int64_t cap, const int32_t* step, int32_t* ids,
                   void* stream) {
    (void) s;
    (void) cap;
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<int, 1> s_a(sycl::range<1>(TOPK_THREADS), hnd);
        local_accessor<int, 1> s_b(sycl::range<1>(TOPK_THREADS), hnd);
        local_accessor<long long, 1> s_cgt(sycl::range<1>(1), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>(TOPK_THREADS), sycl::range<1>(TOPK_THREADS)),
                         [=](nd_item<1> it) {
                             const int t = (int) it.get_local_id(0);
                             const long long n_kv = (long long) step[kStepNKv];
                             const long long width = (long long) step[kStepWidth];
                             const long long chunk = (n_kv + TOPK_THREADS - 1) / TOPK_THREADS;
                             const long long lo = (long long) t * chunk;
                             long long hi = lo + chunk;
                             if (hi > n_kv) hi = n_kv;
                             uint32_t v = 0u;
                             for (int bit = 31; bit >= 0; --bit) {
                                 const uint32_t cand = v | (1u << bit);
                                 int c = 0;
                                 for (long long j = lo; j < hi; ++j)
                                     if (order_key(cell_scores[j]) >= cand) ++c;
                                 s_a[t] = c;
                                 it.barrier(sycl::access::fence_space::local_space);
                                 for (int s2 = TOPK_THREADS / 2; s2 > 0; s2 >>= 1) {
                                     if (t < s2) s_a[t] += s_a[t + s2];
                                     it.barrier(sycl::access::fence_space::local_space);
                                 }
                                 const int tot = s_a[0];
                                 it.barrier(sycl::access::fence_space::local_space);
                                 if (tot >= (int) width) v = cand;
                             }
                             const uint32_t thr = v;
                             int gt = 0, eq = 0;
                             for (long long j = lo; j < hi; ++j) {
                                 const uint32_t k = order_key(cell_scores[j]);
                                 if (k > thr)
                                     ++gt;
                                 else if (k == thr)
                                     ++eq;
                             }
                             s_a[t] = gt;
                             s_b[t] = eq;
                             it.barrier(sycl::access::fence_space::local_space);
                             if (t == 0) {
                                 long long ag = 0, ae = 0;
                                 for (int i = 0; i < TOPK_THREADS; ++i) {
                                     const int g = s_a[i], e = s_b[i];
                                     s_a[i] = (int) ag;
                                     s_b[i] = (int) ae;
                                     ag += g;
                                     ae += e;
                                 }
                                 s_cgt[0] = ag;
                             }
                             it.barrier(sycl::access::fence_space::local_space);
                             const long long off_eq = s_b[t];
                             const long long eq_budget = width - s_cgt[0];
                             int sel = 0;
                             {
                                 long long e = off_eq;
                                 for (long long j = lo; j < hi; ++j) {
                                     const uint32_t k = order_key(cell_scores[j]);
                                     if (k > thr)
                                         ++sel;
                                     else if (k == thr && e < eq_budget) {
                                         ++sel;
                                         ++e;
                                     }
                                 }
                             }
                             s_a[t] = sel;
                             it.barrier(sycl::access::fence_space::local_space);
                             if (t == 0) {
                                 long long a = 0;
                                 for (int i = 0; i < TOPK_THREADS; ++i) {
                                     const int c = s_a[i];
                                     s_a[i] = (int) a;
                                     a += c;
                                 }
                             }
                             it.barrier(sycl::access::fence_space::local_space);
                             long long w = s_a[t];
                             long long e = off_eq;
                             for (long long j = lo; j < hi; ++j) {
                                 const uint32_t k = order_key(cell_scores[j]);
                                 if (k > thr)
                                     ids[w++] = (int) j;
                                 else if (k == thr && e < eq_budget) {
                                     ++e;
                                     ids[w++] = (int) j;
                                 }
                             }
                         });
    });
    check_launch(stream, "topk_512");
}

void topk_512(const float* cell_scores, int64_t n_kv, const QsaShapes& s, int64_t cap, int32_t* ids, void* stream) {
    int32_t h[kStepCount] = {0, 0, 0, 0};
    h[kStepNKv] = (int32_t) n_kv;
    h[kStepWidth] = (int32_t) cap;
    step_upload_raw(h);
    topk_512_step(cell_scores, s, cap, step_scratch(), ids, stream);
}

void kv_gather_step(const uint16_t* k_pool, const uint16_t* v_pool, const int32_t* page_table, const int32_t* ids,
                    const int32_t* step, int64_t max_ids, const QsaShapes& s, uint16_t* k_scratch,
                    uint16_t* v_scratch, void* stream) {
    validate(s, "kv_gather");
    if (max_ids <= 0) return;
    const int kv_heads = (int) s.n_head_kv;
    const int head_dim = (int) s.head_dim;
    const int page_size = (int) s.page_size;
    const int per = head_dim / 4;
    const long long total = max_ids * kv_heads * (long long) per;
    Q(stream).parallel_for((size_t) total, [=](size_t i) {
        const long long n_ids = (long long) step[kStepWidth];
        if ((long long) (i / (kv_heads * (long long) per)) >= n_ids) return;
        const long long id = (long long) i / (kv_heads * (long long) per);
        const int rem = (int) (i % (kv_heads * (long long) per));
        const int h = rem / per, q = rem - h * per;
        const int cell = ids[id];
        const long long page = (long long) page_table[cell / page_size];
        const long long src = ((page * kv_heads + h) * page_size + (cell % page_size)) * (long long) per + q;
        const long long dst = (id * kv_heads + h) * (long long) per + q;
        for (int e = 0; e < 4; ++e) {
            k_scratch[dst * 4 + e] = k_pool[src * 4 + e];
            v_scratch[dst * 4 + e] = v_pool[src * 4 + e];
        }
    });
    check_launch(stream, "kv_gather");
}

void kv_gather(const uint16_t* k_pool, const uint16_t* v_pool, const int32_t* page_table, const int32_t* ids,
               int64_t n_ids, const QsaShapes& s, uint16_t* k_scratch, uint16_t* v_scratch, void* stream) {
    kv_gather_step(k_pool, v_pool, page_table, ids, step_upload_width(n_ids, s), n_ids, s, k_scratch, v_scratch,
                   stream);
}

void qsa_attend_step(const float* q, const uint16_t* k_scratch, const uint16_t* v_scratch, const int32_t* step,
                     int64_t max_ids, const QsaShapes& s, float* attn, float* weights, void* stream) {
    validate(s, "qsa_attend");
    const int n_head = (int) s.n_head;
    const int n_head_kv = (int) s.n_head_kv;
    const int head_dim = (int) s.head_dim;
    const size_t cap = (size_t) (max_ids > 0 ? max_ids : 1);
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<float, 1> s_w(sycl::range<1>(32 + cap), hnd);
        local_accessor<float, 1> red(sycl::range<1>(THREADS), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) n_head * THREADS), sycl::range<1>(THREADS)),
                         [=](nd_item<1> it) {
                             float* w = s_w.get_pointer() + 32;
                             float* redw = s_w.get_pointer();
                             const long long n_ids = (long long) step[kStepWidth];
                             const int h = (int) it.get_group(0);
                             const int d = (int) it.get_local_id(0);
                             if (n_ids == 0) {
                                 for (int i = d; i < head_dim; i += THREADS) attn[(size_t) h * head_dim + i] = 0.0f;
                                 return;
                             }
                             const int kv = h / (n_head / n_head_kv);
                             const float scale = 1.0f / sycl::sqrt((float) head_dim);
                             const int lane = d & 31, wid = d >> 5, nwarp = THREADS / 32;
                             for (long long j = d; j < n_ids; j += THREADS) {
                                 const uint16_t* krow = k_scratch + (j * n_head_kv + kv) * head_dim;
                                 float acc = 0.0f;
                                 for (int i = 0; i < head_dim; ++i)
                                     acc += h2f(krow[i]) * q[(size_t) h * head_dim + i];
                                 w[j] = acc * scale;
                             }
                             it.barrier(sycl::access::fence_space::local_space);
                             float mx = -FLT_MAX;
                             for (long long j = d; j < n_ids; j += THREADS) mx = sycl::fmax(mx, w[j]);
                             mx = chunk_max(mx, d, red, it);
                             if (lane == 0) redw[wid] = mx;
                             it.barrier(sycl::access::fence_space::local_space);
                             // EVERY warp runs the second-level tree: its
                             // barriers are work-group-wide (warp-0-only here
                             // deadlocked the group at replay)
                             mx = chunk_max(wid == 0 && lane < nwarp ? redw[lane] : -FLT_MAX, d, red, it);
                             if (d == 0) redw[0] = mx;
                             it.barrier(sycl::access::fence_space::local_space);
                             mx = redw[0];
                             it.barrier(sycl::access::fence_space::local_space);
                             float sum = 0.0f;
                             for (long long j = d; j < n_ids; j += THREADS) {
                                 const float e = sycl::exp(w[j] - mx);
                                 w[j] = e;
                                 sum += e;
                             }
                             sum = chunk_sum(sum, d, red, it);
                             if (lane == 0) redw[wid] = sum;
                             it.barrier(sycl::access::fence_space::local_space);
                             sum = chunk_sum(wid == 0 && lane < nwarp ? redw[lane] : 0.0f, d, red, it);
                             if (d == 0) redw[0] = 1.0f / sum;
                             it.barrier(sycl::access::fence_space::local_space);
                             const float inv = redw[0];
                             it.barrier(sycl::access::fence_space::local_space);
                             float acc = 0.0f;
                             for (long long j = 0; j < n_ids; ++j)
                                 acc += (w[j] * inv) * h2f(v_scratch[(j * n_head_kv + kv) * head_dim + d]);
                             attn[(size_t) h * head_dim + d] = acc;
                             if (weights != nullptr) {
                                 for (long long j = d; j < n_ids; j += THREADS)
                                     weights[(size_t) h * n_ids + j] = w[j] * inv;
                             }
                         });
    });
    check_launch(stream, "qsa_attend");
}

void qsa_attend(const float* q, const uint16_t* k_scratch, const uint16_t* v_scratch, int64_t n_ids,
                const QsaShapes& s, float* attn, float* weights, void* stream) {
    qsa_attend_step(q, k_scratch, v_scratch, step_upload_width(n_ids, s), n_ids, s, attn, weights, stream);
}

void qsa_gate_apply(const float* attn, const float* q_full, const QsaShapes& s, uint16_t* out, void* stream) {
    validate(s, "qsa_gate_apply");
    const long long n = s.n_head * s.head_dim;
    const int head_dim = (int) s.head_dim;
    Q(stream).parallel_for((size_t) n, [=](size_t i) {
        const int h = (int) (i / head_dim), d = (int) (i % head_dim);
        const double g = (double) q_full[((size_t) h * 2 * head_dim) + head_dim + d];
        const double sig = 1.0 / (1.0 + exp(-g));
        out[i] = f16_from_f32((float) ((double) attn[i] * sig));
    });
    check_launch(stream, "qsa_gate_apply");
}

void qsa_gate_apply_f32(const float* attn, const float* q_full, const QsaShapes& s, float* out, void* stream) {
    validate(s, "qsa_gate_apply_f32");
    const long long n = s.n_head * s.head_dim;
    const int head_dim = (int) s.head_dim;
    Q(stream).parallel_for((size_t) n, [=](size_t i) {
        const int h = (int) (i / head_dim), d = (int) (i % head_dim);
        const double g = (double) q_full[((size_t) h * 2 * head_dim) + head_dim + d];
        const double sig = 1.0 / (1.0 + exp(-g));
        out[i] = (float) ((double) attn[i] * sig);
    });
    check_launch(stream, "qsa_gate_apply_f32");
}

}  // namespace strata::kernels
