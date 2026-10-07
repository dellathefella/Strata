// src/kernels/sycl/iq_expert.cpp — native grouped experts for the SYCL
// backend, including the Q8_0/Q8_0 layout the official Flash-Next GGUF uses
// (upstream CUDA implements only the IQ families here; the SYCL port adds
// Q8_0 because that is the artifact lagrange holds).
//
// One work-group per entry (cap_entries groups launched; the real counts are
// read on the device so the launch is graph-capturable). Phases, in the CPU
// pool's contract order (src/kernels/cpu/expert.cpp): gate and up rows dot the
// token's Q8_1 activation, swiglu, the intermediate is quantized to Q8_1 with
// the fp32-scale contract, then the down rows dot it. Every weight dot is the
// pinned small_q8_dot Q8_0 expression (int8 x int8 DP4A over 32-element
// blocks, fp16 weight scale times the activation's fp16 scale), so a GPU hit
// equals the CPU miss for the same bytes.
//
// Scratch layout (native_expert_scratch_bytes): [gate cap*n_ff f32]
// [up cap*n_ff f32] [unused pad cap*n_ff f32] [cap*(n_ff/32) block_q8_1].
#include "strata/kernels/dp4a.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>  // sycl_compat shim

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {
constexpr int THREADS = 256;

struct Q81Blk {
    __half2 ds;
    int8_t qs[32];
};

using sycl::local_accessor;
using sycl::nd_item;

inline sycl::queue& Q(void* stream) { return strata::sycl_compat::q_for(stream); }

inline int load_int_b2(const void* ptr, int i32) {
    const auto* x = static_cast<const uint16_t*>(ptr);
    int value = x[2 * i32] << 0;
    value |= x[2 * i32 + 1] << 16;
    return value;
}

inline float wscale(const uint8_t* blk) {
    return f32_from_f16((uint16_t) blk[0] | ((uint16_t) blk[1] << 8));
}

// one Q8_0 weight row (n_blocks 32-element blocks) against one Q8_1 column;
// the work-group tree sum is part of the call so every thread participates
inline float q80_row_dot(const uint8_t* wrow, const Q81Blk* x, int n_blocks, int tid,
                         local_accessor<float, 1> red, nd_item<1> it) {
    float acc = 0.0f;
    for (int b = tid; b < n_blocks; b += THREADS) {
        const float dw = wscale(wrow + b * 34);
        const int8_t* q = reinterpret_cast<const int8_t*>(wrow + b * 34 + 2);
        const float dx = __low2float(x[b].ds);
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int v = load_int_b2(q, i);
            const int u = reinterpret_cast<const int*>(x[b].qs)[i];
            sumi = STRATA_DP4A(v, u, sumi);
        }
        acc += dw * dx * (float) sumi;
    }
    const int lid = (int) it.get_local_id(0);
    red[lid] = acc;
    it.barrier(sycl::access::fence_space::local_space);
    for (int o = THREADS / 2; o > 0; o >>= 1) {
        if (lid < o) red[lid] += red[lid + o];
        it.barrier(sycl::access::fence_space::local_space);
    }
    // EVERY thread must finish reading the result before the next call's
    // red[lid] = acc overwrites it (consecutive dots share the array)
    const float r = red[0];
    it.barrier(sycl::access::fence_space::local_space);
    return r;
}

}  // namespace

bool native_expert_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) noexcept {
    const bool gu_ok = gu_type == 8 || gu_type == 20;
    const bool d_ok = d_type == 8 || d_type == 20;
    return gu_ok && d_ok && n_embd % 32 == 0 && n_ff % 32 == 0 && n_embd % 256 == 0 && (n_ff * n_embd) % 256 == 0;
}

NativeExpertLayout native_expert_layout(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) {
    NativeExpertLayout L;
    L.gu_type = gu_type;
    L.d_type = d_type;
    L.n_embd = n_embd;
    L.n_ff = n_ff;
    L.gu_row = iq_row_bytes(gu_type, n_embd);
    L.d_row = iq_row_bytes(d_type, n_ff);
    L.up_off = (size_t) n_ff * L.gu_row;
    L.down_off = 2 * L.up_off;
    L.bytes = L.down_off + (size_t) n_embd * L.d_row;
    return L;
}

size_t native_expert_scratch_bytes(int64_t cap, int64_t n_ff) {
    const size_t f = (size_t) cap * (size_t) n_ff * sizeof(float);
    return 3 * ((f + 255) & ~(size_t) 255) + (((size_t) cap * (size_t) (n_ff / 32) * sizeof(Q81Blk) + 255) & ~(size_t) 255);
}

void native_expert_grouped(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start,
                           const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups,
                           int64_t cap_entries, const void* x_q8_1, void* scratch, float* out, void* stream) {
    if (L.gu_type != 8 || L.d_type != 8) {
        throw std::runtime_error("SYCL backend: native_expert_grouped implements Q8_0/Q8_0 only so far");
    }
    const int n_embd = (int) L.n_embd;
    const int n_ff = (int) L.n_ff;
    const int gu_blocks = n_embd / 32;
    const int d_blocks = n_ff / 32;
    const size_t gu_row = L.gu_row;
    const size_t up_off = L.up_off;
    const size_t down_off = L.down_off;
    const size_t d_row = L.d_row;
    const int row_iters_gu = (n_ff + THREADS - 1) / THREADS;
    const int row_iters_dn = (n_embd + THREADS - 1) / THREADS;
    const Q81Blk* x = static_cast<const Q81Blk*>(x_q8_1);
    float* scr = static_cast<float*>(scratch);
    const int64_t cap_e = cap_entries;
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<float, 1> red(sycl::range<1>(THREADS), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) cap_entries * THREADS), sycl::range<1>(THREADS)),
                         [=](nd_item<1> it) {
                             const int64_t e = (int64_t) it.get_group(0);
                             const int tid = (int) it.get_local_id(0);
                             const int entries = n_groups[1];
                             if (e >= entries) return;  // uniform per work-group
                             const int ng = n_groups[0];
                             int g = 0;
                             while (g + 1 < ng && e >= grp_start[g + 1]) ++g;
                             const uint8_t* blob = (const uint8_t*) (uintptr_t) grp_ptr[g];
                             const int64_t dst = ent_dst[e];
                             const int64_t tok = ent_tok[e];
                             const Q81Blk* xt = x + tok * gu_blocks;
                             float* grow = scr + (size_t) e * n_ff;
                             float* urow = scr + (size_t) cap_e * n_ff + (size_t) e * n_ff;
                             // gate/up projections (padded row loop: every
                             // thread hits the reduction barriers each iter)
                             for (int it2 = 0; it2 < row_iters_gu; ++it2) {
                                 const int r = it2 * THREADS + tid;
                                 if (r < n_ff) {
                                     grow[r] = q80_row_dot(blob + (size_t) r * gu_row, xt, gu_blocks, tid, red, it);
                                     urow[r] = q80_row_dot(blob + up_off + (size_t) r * gu_row, xt, gu_blocks, tid,
                                                           red, it);
                                 } else {
                                     q80_row_dot(blob, xt, 0, tid, red, it);
                                     q80_row_dot(blob, xt, 0, tid, red, it);
                                 }
                             }
                             it.barrier(sycl::access::fence_space::local_space);
                             // swiglu + q8_1 quantize, one 32-block per thread
                             // PER-ENTRY offset (hq + e * hb in the CUDA kernel): without it every
                            // work-group quantizes into the same blocks and the down projection
                            // reads whichever group wrote last
                            Q81Blk* hq = reinterpret_cast<Q81Blk*>(scr + 3 * (size_t) cap_e * n_ff) +
                                         (size_t) e * d_blocks;
                             for (int b = tid; b < d_blocks; b += THREADS) {
                                 float amax = 0.0f, sum = 0.0f;
                                 int8_t qs[32];
                                 float hv[32];
#pragma unroll 8
                                 for (int j = 0; j < 32; ++j) {
                                     const float gv = grow[b * 32 + j];
                                     const float uv = urow[b * 32 + j];
                                     hv[j] = gv / (1.0f + sycl::exp(-gv)) * uv;
                                     amax = sycl::fmax(amax, sycl::fabs(hv[j]));
                                     sum += hv[j];
                                 }
                                 const float d = amax / 127.0f;
                                 for (int j = 0; j < 32; ++j)
                                     qs[j] = amax == 0.0f ? 0 : (int8_t) sycl::round(hv[j] / d);
                                 hq[b].ds = __floats2half2_rn(d, sum);
#pragma unroll
                                 for (int j = 0; j < 32; ++j) hq[b].qs[j] = qs[j];
                             }
                             it.barrier(sycl::access::fence_space::local_space);
                             // down projection into out[dst]
                             float* y = out + (size_t) dst * n_embd;
                             for (int it2 = 0; it2 < row_iters_dn; ++it2) {
                                 const int r = it2 * THREADS + tid;
                                 if (r >= n_embd) continue;
                                 const uint8_t* wrow = blob + down_off + (size_t) r * d_row;
                                 float acc = 0.0f;
                                 for (int b = 0; b < d_blocks; ++b) {
                                     const float dw = wscale(wrow + b * 34);
                                     const int8_t* q = reinterpret_cast<const int8_t*>(wrow + b * 34 + 2);
                                     const float dx = __low2float(hq[b].ds);
                                     int sumi = 0;
#pragma unroll
                                     for (int i2 = 0; i2 < 8; ++i2) {
                                         const int v = load_int_b2(q, i2);
                                         const int u = reinterpret_cast<const int*>(hq[b].qs)[i2];
                                         sumi = STRATA_DP4A(v, u, sumi);
                                     }
                                     acc += dw * dx * (float) sumi;
                                 }
                                 y[r] = acc;
                             }
                         });
    });
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "native_expert_grouped: %s\n", cudaGetErrorString(err));
        std::exit(1);
    }
}

namespace {
// ggml-common.h kvalues_iq4nl as floats (artifact/dequant.hpp keeps the int8 form)
constexpr float kIq4nlF[16] = {-127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
                               1.0f,   13.0f,   25.0f,  38.0f,  53.0f,  69.0f,  89.0f,  112.0f};
}  // namespace

// The prefill MoE's f16 staging dequantizer (dq_dispatch's Q8_0 and IQ4_NL
// cases, one 256-value superblock per 32-thread group, gate/up interleaved
// rows exactly as dequant_gu_kernel lays them out).
void iq_dequant_gu_f16(int t, const void* gate, const void* up, int64_t n_ff, int64_t n_embd, uint16_t* dst,
                       void* stream) {
    if (n_embd % 256 != 0 || (t != 8 && t != 20)) {
        std::fprintf(stderr, "iq_dequant_gu_f16: type %d / %lld\n", t, (long long) n_embd);
        std::exit(1);
    }
    const int64_t per_row = n_embd / 256;
    const int64_t nblk = n_ff * per_row;
    const int blk_bytes = t == 8 ? 34 : 18;   // block_q8_0 / block_iq4_nl
    Q(stream).submit([&](sycl::handler& hnd) {
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) nblk * 2 * 32), sycl::range<1>(32)),
                         [=](nd_item<1> it) {
                             const int64_t blk = (int64_t) it.get_group(0);
                             const int parity = (int) (blk / nblk);          // dim3(nblk, 2), x fastest
                             const int64_t i = blk % nblk;
                             const int64_t r = i / per_row, c = i % per_row;
                             const int tid = (int) it.get_local_id(0);
                             const uint8_t* base = (const uint8_t*) (parity ? up : gate);
                             uint16_t* y = dst + ((2 * r + parity) * per_row + c) * 256;
                             const int ib = tid % 8, il = tid / 8;
                             const uint8_t* b = base + (i * 8 + ib) * (size_t) blk_bytes;
                             const float d = wscale(b);
                             if (t == 8) {
                                 const int8_t* qs = (const int8_t*) (b + 2);
                                 uint16_t* yy = y + 32 * ib + 8 * il;
                                 for (int j = 0; j < 8; ++j) yy[j] = f16_from_f32((float) qs[8 * il + j] * d);
                             } else {
                                 const uint8_t* q4 = b + 2 + 4 * il;
                                 uint16_t* yy = y + 32 * ib + 4 * il;
                                 for (int j = 0; j < 4; ++j) {
                                     yy[j] = f16_from_f32(d * kIq4nlF[q4[j] & 0xf]);
                                     yy[j + 16] = f16_from_f32(d * kIq4nlF[q4[j] >> 4]);
                                 }
                             }
                         });
    });
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "iq_dequant_gu_f16: %s\n", cudaGetErrorString(err));
        std::exit(1);
    }
}

}  // namespace strata::kernels
