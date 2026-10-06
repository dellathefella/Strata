// src/kernels/sycl/s2_expert_grouped.cpp — SYCL port of the S2 hit path of
// src/kernels/cuda/s2_expert_grouped.cu: moe_hit_select(_multi), the grouped
// gate/up and down GEMVs over resident S2 blobs (non-"fast" kernels), swiglu,
// moe_hit_add and the scratch sizing. The 2-bit code expansion, the dp4a
// chains and the `dw * dx * (s - hx)` scale application are verbatim; the
// per-warp row reductions become per-32-lane local trees and predicated
// warps participate with zero so every barrier stays work-group-uniform.
#include "strata/kernels/dp4a.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>  // sycl_compat shim

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::kernels {
namespace {
constexpr int H = 2560;
constexpr int FF = 640;
constexpr int QK = 64;
constexpr int ROW_GU = H / 4;
constexpr int ROW_D = FF / 4;
constexpr int SC_GU = H / QK;
constexpr int SC_D = FF / QK;
constexpr size_t O_D_CODES = (size_t) 2 * FF * ROW_GU;
constexpr size_t O_GU_SCALES = O_D_CODES + (size_t) H * ROW_D;
constexpr size_t O_D_SCALES = O_GU_SCALES + (size_t) 2 * FF * SC_GU * 2;
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;

using sycl::local_accessor;
using sycl::nd_item;

inline sycl::queue& Q(void* stream) { return strata::sycl_compat::q_for(stream); }

void hit_sync(const char* what) {
    if (std::getenv("STRATA_HIT_SYNC") == nullptr) return;
    const cudaError_t e = cudaDeviceSynchronize();
    std::fprintf(stderr, "[dbg] hit %-10s %s\n", what, cudaGetErrorString(e));
}

void check(void* stream, const char* what) {
    if (stream != nullptr) return;
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "s2_expert_grouped: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

inline float f16_at(const uint8_t* p) { return f32_from_f16((uint16_t) (p[0] | (p[1] << 8))); }

inline float row_dot_s2_q8(const uint8_t* codes, const uint8_t* scales, const uint8_t* x_q8_0, int n_chunks,
                           int lane, const float* x_scales) {
    float acc = 0.0f;
    for (int c = lane; c < n_chunks; c += 32) {
        const uint8_t* cb = codes + (size_t) c * 8;
        const uint8_t* xb = x_q8_0 + (size_t) c * 34;
        const float dx = x_scales ? x_scales[c] : f16_at(xb);
        const int8_t* xq = (const int8_t*) (xb + 2);
        int s = 0;
        int hx = 0;
        const int ones = 0x01010101;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const unsigned cbyte = cb[j];
            const int cw = (int) ((cbyte & 3u) | (((cbyte >> 2) & 3u) << 8) | (((cbyte >> 4) & 3u) << 16) |
                                  (((cbyte >> 6) & 3u) << 24));
            int xw;
            memcpy(&xw, xq + 4 * j, 4);
            s = STRATA_DP4A(cw, xw, s);
            hx = STRATA_DP4A(ones, xw, hx);
        }
        const float dw = f16_at(scales + (size_t) (c >> 1) * 2);
        acc += dw * dx * (float) (s - hx);
    }
    return acc;
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

void gu_launch(const uint8_t* blob_base, const int32_t* slot_index, long long blob_bytes, const uint8_t* x_q8_0,
               const float* x_scales, float* gate_up, long long cap, const int32_t* d_count,
               const int32_t* dst_index, int tok_div, void* stream) {
    const long long rows = cap * 2LL * FF;
    const long long blocks = (rows + WARPS - 1) / WARPS;
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<float, 1> red(sycl::range<1>(THREADS), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) blocks * THREADS), sycl::range<1>(THREADS)),
                         [=](nd_item<1> it) {
                             const long long slot = (long long) it.get_group(0) * WARPS + (it.get_local_id(0) >> 5);
                             const int lane = (int) it.get_local_id(0) & 31;
                             const bool live = slot < rows && !(d_count != nullptr && (slot / (2LL * FF)) >= *d_count);
                             const int h = (int) (slot / (2LL * FF));
                             const int i = (int) (slot % (2LL * FF));
                             float acc = 0.0f;
                             if (live) {
                                 const uint8_t* blob = blob_base + (size_t) slot_index[h] * (size_t) blob_bytes;
                                 const uint8_t* xq = x_q8_0;
                                 const float* xs = x_scales;
                                 if (tok_div > 0) {
                                     const int tok = dst_index[h] / tok_div;
                                     xq += (size_t) tok * (size_t) (H / 32) * 34;
                                     if (xs != nullptr) xs += (size_t) tok * (size_t) (H / 32);
                                 }
                                 acc = row_dot_s2_q8(blob + (size_t) i * ROW_GU,
                                                     blob + O_GU_SCALES + (size_t) i * SC_GU * 2, xq, H / 32, lane, xs);
                             }
                             const float s = chunk_sum(acc, (int) it.get_local_id(0), red, it);
                             if (lane != 0 || !live) return;
                             const int r = i >> 1;
                             const size_t base = (i & 1) ? ((size_t) cap * FF + (size_t) h * FF) : ((size_t) h * FF);
                             gate_up[base + (size_t) r] = s;
                         });
    });
}

void down_launch(const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index, long long blob_bytes,
                 const uint8_t* h_q8_0, const float* h_scales, float* out, long long cap, const int32_t* d_count,
                 void* stream) {
    const long long rows = cap * (long long) H;
    const long long blocks = (rows + WARPS - 1) / WARPS;
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<float, 1> red(sycl::range<1>(THREADS), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) blocks * THREADS), sycl::range<1>(THREADS)),
                         [=](nd_item<1> it) {
                             const long long row = (long long) it.get_group(0) * WARPS + (it.get_local_id(0) >> 5);
                             const int lane = (int) it.get_local_id(0) & 31;
                             const bool live = row < rows && !(d_count != nullptr && (row / H) >= *d_count);
                             const int h = (int) (row / H);
                             const int r = (int) (row % H);
                             float acc = 0.0f;
                             if (live) {
                                 const uint8_t* blob = blob_base + (size_t) slot_index[h] * (size_t) blob_bytes;
                                 const uint8_t* xb = h_q8_0 + (size_t) h * (size_t) (FF / 32) * 34;
                                 acc = row_dot_s2_q8(blob + O_D_CODES + (size_t) r * ROW_D,
                                                     blob + O_D_SCALES + (size_t) r * SC_D * 2, xb, FF / 32, lane,
                                                     h_scales ? h_scales + (size_t) h * (size_t) (FF / 32) : nullptr);
                             }
                             const float s = chunk_sum(acc, (int) it.get_local_id(0), red, it);
                             if (lane == 0 && live) out[(size_t) dst_index[h] * H + r] = s;
                         });
    });
}

void hit_grouped(const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index, int64_t n_hits,
                 int64_t blob_bytes, const uint8_t* x_q8_0, void* scratch, float* out, void* stream,
                 const float* x_scales, const int32_t* d_count, int tok_div) {
    if (n_hits <= 0) return;
    const uint64_t gu_bytes = ((uint64_t) n_hits * (uint64_t) (2 * FF) * 4 + 15) & ~15ull;
    const uint64_t q8_bytes = ((uint64_t) n_hits * (uint64_t) (FF / 32) * 34 + 15) & ~15ull;
    float* gate_up = (float*) scratch;
    uint8_t* h_q8_0 = (uint8_t*) scratch + gu_bytes;
    float* h_scales = (float*) ((uint8_t*) scratch + gu_bytes + q8_bytes);
    gu_launch(blob_base, slot_index, blob_bytes, x_q8_0, x_scales, gate_up, n_hits, d_count, dst_index, tok_div,
              stream);
    hit_sync("gu");
    const long long pairs = n_hits * (long long) FF;
    Q(stream).parallel_for((size_t) pairs, [=](size_t i) {
        const float gv = gate_up[i];
        const float u = gate_up[pairs + i];
        gate_up[i] = (gv / (1.0f + sycl::exp(-gv))) * u;
    });
    hit_sync("swiglu");
    if (x_scales != nullptr)
        quantize_q8_0_scaled(gate_up, h_q8_0, h_scales, n_hits * (int64_t) FF, stream);
    else
        quantize_q8_0(gate_up, h_q8_0, n_hits * (int64_t) FF, stream);
    down_launch(blob_base, slot_index, dst_index, blob_bytes, h_q8_0,
                x_scales != nullptr ? h_scales : nullptr, out, n_hits, d_count, stream);
    hit_sync("down");
}

}  // namespace

uint64_t moe_hit_grouped_scratch_bytes(int64_t n_hits, int64_t n_embd, int64_t n_ff) {
    if (n_hits <= 0) return 0;
    const uint64_t gu = (uint64_t) n_hits * (uint64_t) (2 * n_ff) * 4;
    const uint64_t q8 = (uint64_t) n_hits * (uint64_t) (n_ff / 32) * 34;
    const uint64_t hs = (uint64_t) n_hits * (uint64_t) (n_ff / 32) * 4;
    const uint64_t xh = (uint64_t) (n_embd / 32) * 4;
    return ((gu + 15) & ~15ull) + ((q8 + 15) & ~15ull) + 2 * ((hs + 15) & ~15ull) + ((xh + 15) & ~15ull);
}

void moe_hit_grouped_s2(const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index, int64_t n_hits,
                        int64_t blob_bytes, const uint8_t* x_q8_0, void* scratch, float* out, void* stream,
                        const float* x_scales) {
    hit_grouped(blob_base, slot_index, dst_index, n_hits, blob_bytes, x_q8_0, scratch, out, stream, x_scales, nullptr,
                0);
}

void moe_hit_grouped_s2_dev(const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index,
                            const int32_t* d_count, int64_t cap, int64_t blob_bytes, const uint8_t* x_q8_0,
                            void* scratch, float* out, void* stream, const float* x_scales) {
    hit_grouped(blob_base, slot_index, dst_index, cap, blob_bytes, x_q8_0, scratch, out, stream, x_scales, d_count, 0);
}

void moe_hit_select(const int32_t* ids, const int32_t* res_row, int k, int n_expert, int32_t* slot, int32_t* dst,
                    int32_t* count, void* stream) {
    if (k < 1 || k > 32) {
        std::fprintf(stderr, "moe_hit_select: k must be 1..32\n");
        std::exit(1);
    }
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<unsigned, 1> m(sycl::range<1>(1), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>(32), sycl::range<1>(32)), [=](nd_item<1> it) {
            const int lane = (int) it.get_local_id(0);
            int s = -1;
            if (lane < k) {
                const int e = ids[lane];
                if (e >= 0 && e < n_expert) s = res_row[e];
            }
            if (lane == 0) m[0] = 0u;
            it.barrier(sycl::access::fence_space::local_space);
            if (s >= 0) {
                sycl::atomic_ref<unsigned, sycl::memory_order::relaxed, sycl::memory_scope::work_group> a(m[0]);
                a.fetch_or(1u << lane);
            }
            it.barrier(sycl::access::fence_space::local_space);
            const unsigned hit = m[0];
            if (s >= 0) {
                const int at = sycl::popcount(hit & ((1u << lane) - 1u));
                slot[at] = s;
                dst[at] = lane;
            }
            if (lane == 0) *count = (int32_t) sycl::popcount(hit);
        });
    });
    hit_sync("select");
}

void moe_hit_select_multi(const int32_t* ids, const int32_t* res_row, int n, int n_expert, int32_t* slot,
                          int32_t* dst, int32_t* count, void* stream) {
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<unsigned, 1> m(sycl::range<1>(4), hnd);
        local_accessor<int, 1> wc(sycl::range<1>(4), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>(128), sycl::range<1>(128)), [=](nd_item<1> it) {
            const int i = (int) it.get_local_id(0);
            const int lane = i & 31, warp = i >> 5;
            int s = -1;
            if (i < n) {
                const int e = ids[i];
                if (e >= 0 && e < n_expert) s = res_row[e];
            }
            if (lane == 0) m[warp] = 0u;
            it.barrier(sycl::access::fence_space::local_space);
            if (s >= 0) {
                sycl::atomic_ref<unsigned, sycl::memory_order::relaxed, sycl::memory_scope::work_group> a(m[warp]);
                a.fetch_or(1u << lane);
            }
            it.barrier(sycl::access::fence_space::local_space);
            const unsigned hit = m[warp];
            if (lane == 0) wc[warp] = (int) sycl::popcount(hit);
            it.barrier(sycl::access::fence_space::local_space);
            int before = 0;
            for (int w = 0; w < warp; ++w) before += wc[w];
            if (s >= 0) {
                const int at = before + (int) sycl::popcount(hit & ((1u << lane) - 1u));
                slot[at] = s;
                dst[at] = i;
            }
            if (i == 0) *count = wc[0] + wc[1] + wc[2] + wc[3];
        });
    });
    hit_sync("select_multi");
}

void moe_hit_add(float* parts, const float* hit_out, const int32_t* dst, const int32_t* count, int64_t cap,
                 int64_t n_embd, void* stream) {
    if (cap <= 0) return;
    const size_t gx = (size_t) ((n_embd + 255) / 256 < 8 ? (n_embd + 255) / 256 : 8);
    Q(stream).parallel_for(sycl::range<2>((size_t) cap, gx * 256), [=](sycl::id<2> id) {
        const size_t h = id[0];
        if ((int64_t) h >= *count) return;
        const size_t row = (size_t) dst[h] * (size_t) n_embd;
        for (int64_t i = (int64_t) id[1]; i < n_embd; i += (int64_t) gx * 256) parts[row + i] += hit_out[row + i];
    });
    hit_sync("add");
}

void moe_hit_grouped_s2_multi(const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index,
                              const int32_t* d_count, int64_t cap, int64_t blob_bytes, const uint8_t* x_q8_0,
                              const float* x_scales, int k_per_token, void* scratch, float* out, void* stream) {
    if (cap <= 0) return;
    hit_grouped(blob_base, slot_index, dst_index, cap, blob_bytes, x_q8_0, scratch, out, stream, x_scales, d_count,
                k_per_token);
}

// The SYCL port has ONE grouped implementation (the CUDA file's "new" kernels);
// the old/new switch exists so the parity harness links and its old-vs-new
// comparison degenerates to new-vs-new.
void moe_grouped_select_old(int) {}

int moe_grouped_last_path() { return 0; }

}  // namespace strata::kernels
