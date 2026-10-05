// src/kernels/sycl/qsa_decode_attn.cpp — SYCL port of src/kernels/cuda/qsa_decode_attn.cu
// (sparse decode attention over the selected cells, fp16/int8/q4 KV pools).
//
// The CUDA kernel mixes warp-strided loops with warp-scope shuffles; SYCL has
// no warp-scope barrier-free reduction, so the chunk kernel is SPLIT into
// three launches (scores / chunk max+exp-sum / value accumulate) plus the
// unchanged merge. Every reduction keeps the CUDA pairing and order:
//   scores: per-lane 8-dim dot, xor tree over the 32-lane group
//   maxsum: pair (lane, lane+32), xor tree, in-place exp into sp
//   value:  per-dim sequential over cells (no reduction at all)
// so the port is bitwise the same maths, just regrouped.
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"

#include <cuda_runtime.h>  // sycl_compat shim

#include <cfloat>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {
constexpr int HD = 256;
constexpr int G = 12;
constexpr int CHUNK = 64;
constexpr int THREADS = 256;

using sycl::local_accessor;
using sycl::nd_item;

inline sycl::queue& Q(void* stream) { return strata::sycl_compat::q_for(stream); }

inline float h2f(uint16_t bits) { return f32_from_f16(bits); }

template <int KV_MODE>
inline void load8(const QsaAttnPools& p, bool value, long long row, int d0, float* out) {
    if constexpr (KV_MODE == 0) {
        const uint16_t* base = (value ? p.v_pool : p.k_pool) + row * HD + d0;
#pragma unroll
        for (int j = 0; j < 8; ++j) out[j] = h2f(base[j]);
    } else if constexpr (KV_MODE == 1) {
        const int8_t* codes = (value ? p.v_q : p.k_q) + row * HD + d0;
        const float sc = h2f((value ? p.v_scale : p.k_scale)[row * (HD / KV_Q8_GROUP) + d0 / KV_Q8_GROUP]);
#pragma unroll
        for (int j = 0; j < 8; ++j) out[j] = (float) codes[j] * sc;
    } else if constexpr (KV_MODE == 3) {
        if (value) {
            load8<2>(p, value, row, d0, out);
        } else {
            load8<1>(p, value, row, d0, out);
        }
    } else {
        constexpr int bytes_per_head = (HD / QK4_0) * (int) sizeof(block_q4_0);
        const int b = d0 / QK4_0;
        const int rem = d0 % QK4_0;
        const block_q4_0* blk =
            reinterpret_cast<const block_q4_0*>((value ? p.v_q4 : p.k_q4) + row * bytes_per_head) + b;
        const float d = h2f(blk->d);
        const int j = (rem == 0 || rem == 16) ? 0 : 8;
        const uint8_t* bytes = blk->qs + j;
        if (rem < 16) {
#pragma unroll
            for (int k = 0; k < 8; ++k) out[k] = (float) ((int) (bytes[k] & 0x0F) - 8) * d;
        } else {
#pragma unroll
            for (int k = 0; k < 8; ++k) out[k] = (float) ((int) (bytes[k] >> 4) - 8) * d;
        }
    }
}

// xor tree over a 32-lane work-group
inline float g_sum(float v, int lane, local_accessor<float, 1> red, nd_item<1> it) {
    red[lane] = v;
    it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float other = red[lane ^ o];
        it.barrier(sycl::access::fence_space::local_space);
        v += other;
        red[lane] = v;
        it.barrier(sycl::access::fence_space::local_space);
    }
    return v;
}
inline float g_max(float v, int lane, local_accessor<float, 1> red, nd_item<1> it) {
    red[lane] = v;
    it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float other = red[lane ^ o];
        it.barrier(sycl::access::fence_space::local_space);
        v = sycl::fmax(v, other);
        red[lane] = v;
        it.barrier(sycl::access::fence_space::local_space);
    }
    return v;
}

template <int KV_MODE>
void attn_chunk_launch(const float* q, const QsaAttnPools& p, const int32_t* ids, const int32_t* steps,
                       int n_kv_heads, int page_size, float scale, float* part_acc, float* part_m, float* part_l,
                       int n_chunks, int cap, long long scratch_stride, int n_q, float* sp_glob, void* stream) {
    // ---- scores: one 32-lane group per (q, kvh, chunk, cell) ----
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<float, 1> red(sycl::range<1>(32), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) n_q * n_kv_heads * n_chunks * CHUNK * 32),
                                           sycl::range<1>(32)),
                         [=](nd_item<1> it) {
                             const size_t g = it.get_group(0);
                             const int lane = (int) it.get_local_id(0);
                             const int c = (int) (g % CHUNK);
                             const int chunk = (int) ((g / CHUNK) % n_chunks);
                             const int kvh = (int) ((g / ((size_t) CHUNK * n_chunks)) % n_kv_heads);
                             const int qz = (int) (g / ((size_t) CHUNK * n_chunks * n_kv_heads));
                             const float* my_q = q + (size_t) qz * (n_kv_heads * G) * HD;
                             const int32_t* my_ids = ids + (size_t) qz * cap;
                             const int32_t* my_step = steps + (size_t) qz * kStepCount;
                             const int n_ids = my_step[kStepWidth];
                             const int c0 = chunk * CHUNK;
                             const int n_here = sycl::min(CHUNK, n_ids - c0);
                             float* sp = sp_glob +
                                         ((size_t) qz * n_kv_heads + kvh) * (n_chunks * CHUNK * G) +
                                         (size_t) chunk * CHUNK * G;
                             if (c >= n_here) {
                                 if (lane < G) sp[lane * CHUNK + c] = -FLT_MAX;
                                 return;
                             }
                             const int cell = my_ids[c0 + c];
                             const long long page = (long long) p.page_table[cell / page_size];
                             if (page < 0) {
                                 if (lane < G) sp[lane * CHUNK + c] = -FLT_MAX;
                                 return;
                             }
                             const long long row = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
                             float k8[8];
                             load8<KV_MODE>(p, false, row, lane * 8, k8);
#pragma unroll
                             for (int h = 0; h < G; ++h) {
                                 float s = 0.0f;
#pragma unroll
                                 for (int e = 0; e < 8; ++e)
                                     s += k8[e] * my_q[(size_t) (kvh * G + h) * HD + lane * 8 + e];
                                 s = g_sum(s, lane, red, it);
                                 if (lane == 0) sp[h * CHUNK + c] = s * scale;
                             }
                         });
    });
    // ---- per-head chunk max + exp-sum over the 64 cells ----
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<float, 1> red(sycl::range<1>(32), hnd);
        hnd.parallel_for(sycl::nd_range<1>(
                             sycl::range<1>((size_t) n_q * n_kv_heads * n_chunks * G * 32), sycl::range<1>(32)),
                         [=](nd_item<1> it) {
                             const size_t g = it.get_group(0);
                             const int lane = (int) it.get_local_id(0);
                             const int h = (int) (g % G);
                             const int chunk = (int) ((g / G) % n_chunks);
                             const int kvh = (int) ((g / ((size_t) G * n_chunks)) % n_kv_heads);
                             const int qz = (int) (g / ((size_t) G * n_chunks * n_kv_heads));
                             const int32_t* my_step = steps + (size_t) qz * kStepCount;
                             const int n_ids = my_step[kStepWidth];
                             const int c0 = chunk * CHUNK;
                             const int n_here = sycl::min(CHUNK, n_ids - c0);
                             const int slot = kvh * n_chunks + chunk;
                             float* sp = sp_glob +
                                         ((size_t) qz * n_kv_heads + kvh) * (n_chunks * CHUNK * G) +
                                         (size_t) chunk * CHUNK * G;
                             const float a = sp[h * CHUNK + lane];
                             const float b = sp[h * CHUNK + lane + 32];
                             const float m = g_max(sycl::fmax(a, b), lane, red, it);
                             const bool va = lane < n_here && a > -FLT_MAX;
                             const bool vb = lane + 32 < n_here && b > -FLT_MAX;
                             const float ea = va ? sycl::exp(a - m) : 0.0f;
                             const float eb = vb ? sycl::exp(b - m) : 0.0f;
                             sp[h * CHUNK + lane] = ea;
                             sp[h * CHUNK + lane + 32] = eb;
                             const float l = g_sum(ea + eb, lane, red, it);
                             if (lane == 0) {
                                 part_m[((size_t) qz * scratch_stride) + slot * G + h] = m;
                                 part_l[((size_t) qz * scratch_stride) + slot * G + h] = l;
                             }
                         });
    });
    // ---- value accumulate: one thread per (q, kvh, chunk, head, dim) ----
    Q(stream).parallel_for(sycl::range<1>((size_t) n_q * n_kv_heads * n_chunks * G * HD), [=](size_t i) {
        const int d = (int) (i % HD);
        const int h = (int) ((i / HD) % G);
        const int chunk = (int) ((i / ((size_t) HD * G)) % n_chunks);
        const int kvh = (int) ((i / ((size_t) HD * G * n_chunks)) % n_kv_heads);
        const int qz = (int) (i / ((size_t) HD * G * n_chunks * n_kv_heads));
        const int32_t* my_step = steps + (size_t) qz * kStepCount;
        const int32_t* my_ids = ids + (size_t) qz * cap;
        const int n_ids = my_step[kStepWidth];
        const int c0 = chunk * CHUNK;
        const int n_here = sycl::min(CHUNK, n_ids - c0);
        const int slot = kvh * n_chunks + chunk;
        const float* sp = sp_glob + ((size_t) qz * n_kv_heads + kvh) * (n_chunks * CHUNK * G) +
                          (size_t) chunk * CHUNK * G;
        float acc = 0.0f;
        for (int c = 0; c < n_here; ++c) {
            const float wgt = sp[h * CHUNK + c];
            if (wgt == 0.0f) continue;
            const int cell = my_ids[c0 + c];
            const long long page = (long long) p.page_table[cell / page_size];
            if (page < 0) continue;
            const long long row = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
            float v;
            if constexpr (KV_MODE == 0) {
                v = h2f(p.v_pool[row * HD + d]);
            } else if constexpr (KV_MODE == 1) {
                const float sc = h2f(p.v_scale[row * (HD / KV_Q8_GROUP) + d / KV_Q8_GROUP]);
                v = (float) p.v_q[row * HD + d] * sc;
            } else {
                constexpr int bytes_per_head = (HD / QK4_0) * (int) sizeof(block_q4_0);
                const int b = d / QK4_0;
                const int rem = d % QK4_0;
                const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(p.v_q4 + row * bytes_per_head) + b;
                const float dd = h2f(blk->d);
                const int j = rem < 16 ? rem : (rem - 16);
                const uint8_t byte = blk->qs[j];
                const int nibble = (rem < 16) ? ((int) (byte & 0x0F) - 8) : ((int) (byte >> 4) - 8);
                v = (float) nibble * dd;
            }
            acc = sycl::fma(wgt, v, acc);
        }
        part_acc[((size_t) qz * scratch_stride + (size_t) slot * G + h) * HD + d] = acc;
    });
}

void attn_merge_launch(const float* part_acc, const float* part_m, const float* part_l, int n_chunks, float* attn,
                       long long scratch_stride, int n_head, int n_q, void* stream) {
    Q(stream).parallel_for(sycl::range<2>((size_t) n_q, (size_t) n_head * HD), [=](sycl::id<2> id) {
        const int qz = (int) id[0];
        const int h = (int) (id[1] / HD);
        const int d = (int) (id[1] % HD);
        const float* pa = part_acc + (size_t) qz * scratch_stride;
        const float* pm = part_m + (size_t) qz * scratch_stride;
        const float* pl = part_l + (size_t) qz * scratch_stride;
        float* my_attn = attn + (size_t) qz * n_head * HD;
        const int kvh = h / G, hl = h % G;
        float M = -FLT_MAX;
        for (int c = 0; c < n_chunks; ++c) M = sycl::fmax(M, pm[(kvh * n_chunks + c) * G + hl]);
        float L = 0.0f, acc = 0.0f;
        for (int c = 0; c < n_chunks; ++c) {
            const int slot = kvh * n_chunks + c;
            const float m = pm[slot * G + hl];
            if (m == -FLT_MAX) continue;
            const float w = sycl::exp(m - M);
            L = sycl::fma(pl[slot * G + hl], w, L);
            acc = sycl::fma(pa[((size_t) slot * G + hl) * HD + d], w, acc);
        }
        my_attn[(size_t) h * HD + d] = L > 0.0f ? acc / L : 0.0f;
    });
}

}  // namespace

void qsa_decode_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return;
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch || !ids || !steps ||
        !pools.page_table || n_q > 65535) {
        std::fprintf(stderr, "qsa_decode_attn_batch: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    const int kv_mode = pools.k_q4 != nullptr ? 2 : (pools.k_q != nullptr && pools.v_q4 != nullptr ? 3
                        : (pools.k_q != nullptr ? 1 : 0));
    const int n_chunks = (int) ((cap + CHUNK - 1) / CHUNK);
    const long long stride = (long long) qsa_decode_attn_scratch_floats(cap, s);
    float* part_acc = scratch;
    float* part_m = scratch + (size_t) n_chunks * s.n_head * HD;
    float* part_l = part_m + (size_t) n_chunks * s.n_head;
    // the split score/maxsum kernels need the per-chunk score tile in global
    // memory (the CUDA kernel kept it in shared): a dedicated static buffer,
    // sized for the worst case seen so far
    const size_t sp_need = (size_t) n_q * n_chunks * CHUNK * G * s.n_head;
    static float* sp_buf = nullptr;
    static size_t sp_cap = 0;
    if (sp_need > sp_cap) {
        if (sp_buf) cudaFree(sp_buf);
        if (cudaMalloc(&sp_buf, sp_need * sizeof(float)) != cudaSuccess) {
            std::fprintf(stderr, "qsa_decode_attn_batch: score tile allocation failed\n");
            std::exit(1);
        }
        sp_cap = sp_need;
    }
    float* sp_glob = sp_buf;
    const float scale = 1.0f / sycl::sqrt((float) HD);
    const int kv_heads = (int) s.n_head_kv;
    const int page_size = (int) s.page_size;
    if (kv_mode == 3)
        attn_chunk_launch<3>(q, pools, ids, steps, kv_heads, page_size, scale, part_acc, part_m, part_l, n_chunks,
                             (int) cap, stride, (int) n_q, sp_glob, stream);
    else if (kv_mode == 2)
        attn_chunk_launch<2>(q, pools, ids, steps, kv_heads, page_size, scale, part_acc, part_m, part_l, n_chunks,
                             (int) cap, stride, (int) n_q, sp_glob, stream);
    else if (kv_mode == 1)
        attn_chunk_launch<1>(q, pools, ids, steps, kv_heads, page_size, scale, part_acc, part_m, part_l, n_chunks,
                             (int) cap, stride, (int) n_q, sp_glob, stream);
    else
        attn_chunk_launch<0>(q, pools, ids, steps, kv_heads, page_size, scale, part_acc, part_m, part_l, n_chunks,
                             (int) cap, stride, (int) n_q, sp_glob, stream);
    attn_merge_launch(part_acc, part_m, part_l, n_chunks, attn, stride, (int) s.n_head, (int) n_q, stream);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_decode_attn_batch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

uint64_t qsa_decode_attn_scratch_floats(int64_t cap, const QsaShapes& s) {
    const int64_t chunks = (cap + CHUNK - 1) / CHUNK;
    return (uint64_t) chunks * (uint64_t) s.n_head * (HD + 2) + 64;
}

}  // namespace strata::kernels
