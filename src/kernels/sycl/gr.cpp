// src/kernels/sycl/gr.cpp — SYCL port of src/kernels/cuda/gr.cu (the gated-
// recurrence read/write path). Warp-shuffle block reductions become local-
// memory trees with identical down-butterfly pairing (lane 0's accumulation
// order is bit-wise the CUDA one); FP32 sum-of-squares per the source's G5
// rule; silu/sigmoid via sycl::exp; fp32 `/` is IEEE under the backend's
// -foffload-fp32-prec-div.
#include "strata/kernels/gr.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/native_gr_norm.hpp"
#include "strata/kernels/native_gr_postops.hpp"

#include <cuda_runtime.h>  // sycl_compat shim

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::kernels {
namespace {

constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
bool fp32_activations = false;
bool native_mmvf = false;

inline float activation_f32(float x) { return x; }
inline float activation_f32(uint16_t x) { return f32_from_bf16(x); }
inline void store_activation(float* dst, int i, float x) { dst[i] = x; }
inline void store_activation(uint16_t* dst, int i, float x) { dst[i] = bf16_from_f32(x); }

inline float silu_f(float x) { return x / (1.0f + sycl::exp(-x)); }
inline float sigmoid_f(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }

// down-butterfly within one 32-lane chunk window; lane 0's value is the CUDA
// warp_sumf's bit for bit. Barriers are unconditional (all chunks participate).
inline void chunk_down_sum(const sycl::local_accessor<float, 1>& red, int base, int lane,
                           const sycl::nd_item<1>& it) {
    for (int off = 16; off > 0; off >>= 1) {
        it.barrier();
        if (lane < off) red[base + lane] += red[base + lane + off];
    }
}

void sync_if_needed(void* stream, const char* what) {
    if (stream != nullptr) return;
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// ---- gr_norm: one block per stream c, FP32 sum of squares ----
template <bool FP32_ACT>
void launch_gr_norm(sycl::queue& q, const float* R, const float* w_norm, float eps, int n_embd,
                    float* xn, uint16_t* xq, int hc) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> red(sycl::range<1>(THREADS), h);
        sycl::local_accessor<float, 1> sums(sycl::range<1>(WARPS), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t)hc * THREADS),
                                         sycl::range<1>(THREADS)),
                       [=](sycl::nd_item<1> it) {
            const int c = (int)it.get_group(0);
            const int tid = (int)it.get_local_id(0);
            const int chunk = tid / 32, lane = tid % 32;
            const float* Rc = R + (size_t)c * n_embd;
            float* xnc = xn + (size_t)c * n_embd;
            uint16_t* xqc = xq + (size_t)c * n_embd;

            float ss = 0.0f;
            for (int d = tid; d < n_embd; d += THREADS) {
                const float v = Rc[d];
                ss += v * v;
            }
            // block_sumf: leading barrier (scratch-reuse protection), per-chunk
            // tree, sums handoff, chunk-0 tree, broadcast
            it.barrier();
            red[tid] = ss;
            chunk_down_sum(red, chunk * 32, lane, it);
            if (lane == 0) sums[chunk] = red[chunk * 32];
            it.barrier();
            if (chunk == 0) red[tid] = (tid < WARPS) ? sums[tid] : 0.0f;
            for (int off = 16; off > 0; off >>= 1) {
                it.barrier();
                if (chunk == 0 && lane < off) red[lane] += red[lane + off];
            }
            it.barrier();
            const float ms = red[0] / (float)n_embd;
            const float rs = sycl::rsqrt(ms + eps);
            for (int d = tid; d < n_embd; d += THREADS) {
                const float x = Rc[d] * rs * w_norm[(size_t)c * n_embd + d];
                xnc[d] = x;
                if constexpr (!FP32_ACT) xqc[d] = bf16_from_f32(x);
            }
        });
    });
}

// ---- gr_down: one block per low-rank row, 8 warps split the reduction ----
template <typename Activation>
void launch_gr_down(sycl::queue& q, const Activation* xq, const uint16_t* w_down, int hc_dim,
                    int hc_lr, int hc, Activation* lq) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> red(sycl::range<1>(THREADS), h);
        sycl::local_accessor<float, 1> part(sycl::range<1>(WARPS), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t)hc_lr * THREADS),
                                         sycl::range<1>(THREADS)),
                       [=](sycl::nd_item<1> it) {
            const int k = (int)it.get_group(0);
            const int tid = (int)it.get_local_id(0);
            const int chunk = tid / 32, lane = tid % 32;
            const uint16_t* row = w_down + (size_t)k * hc_dim;
            float acc = 0.0f;
            for (int i = chunk * 32 + lane; i < hc_dim; i += WARPS * 32)
                acc += activation_f32(xq[i]) * f32_from_bf16(row[i]);
            red[tid] = acc;
            chunk_down_sum(red, chunk * 32, lane, it);
            if (lane == 0) part[chunk] = red[chunk * 32];
            it.barrier();
            if (chunk == 0) red[tid] = (tid < WARPS) ? part[tid] : 0.0f;
            for (int off = 16; off > 0; off >>= 1) {
                it.barrier();
                if (chunk == 0 && lane < off) red[lane] += red[lane + off];
            }
            // NOTE (from the CUDA source): this reduction order differs from the
            // one-warp version — a NUMERICS change judged by C1, not by comments.
            if (chunk == 0 && lane == 0)
                store_activation(lq, k, silu_f(red[0] / (float)hc));
        });
    });
}

// ---- gr_gate: one warp per output element ----
template <typename Activation>
void launch_gr_gate(sycl::queue& q, const Activation* lq, const uint16_t* w_up, const float* xn,
                    int hc_dim, int hc_lr, float* gated) {
    const size_t grid = (size_t)((hc_dim + WARPS - 1) / WARPS);
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> red(sycl::range<1>(THREADS), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(grid * THREADS),
                                         sycl::range<1>(THREADS)),
                       [=](sycl::nd_item<1> it) {
            const int i = (int)(it.get_group(0) * WARPS + it.get_local_id(0) / 32);
            const int lane = (int)(it.get_local_id(0) % 32);
            const int base = ((int)it.get_local_id(0) / 32) * 32;
            // CUDA returns early here (no barriers in its warp-only kernel); this
            // port has work-group barriers, so the tail block's chunks must ALL
            // reach them: inactive chunks reduce row 0 harmlessly and skip the store.
            const bool active = i < hc_dim;
            const uint16_t* row = w_up + (size_t)(active ? i : 0) * hc_lr;
            float acc = 0.0f;
            if (active)
                for (int k = lane; k < hc_lr; k += 32)
                    acc += activation_f32(lq[k]) * f32_from_bf16(row[k]);
            red[base + lane] = acc;
            for (int off = 16; off > 0; off >>= 1) {
                it.barrier();
                if (lane < off) red[base + lane] += red[base + lane + off];
            }
            if (active && lane == 0) gated[i] = xn[i] * sigmoid_f(red[base]);
        });
    });
}

void launch_gr_mean(sycl::queue& q, const float* gated, int n_embd, int hc, float* mixed) {
    const size_t grid = (size_t)((n_embd + THREADS - 1) / THREADS);
    q.parallel_for(sycl::range<1>(grid * THREADS), [=](sycl::id<1> iid) {
        const int d = (int)iid[0];
        if (d >= n_embd) return;
        float m = 0.0f;
        for (int c = 0; c < hc; ++c) m += gated[(size_t)c * n_embd + d];
        mixed[d] = m / (float)hc;
    });
}

template <typename Activation>
void launch_gr_inject(sycl::queue& q, const Activation* xq, const uint16_t* w_inject, int hc_dim,
                      int hc, float* inject) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> red(sycl::range<1>(32 * (size_t)hc), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t)hc * 32), sycl::range<1>(32)),
                       [=](sycl::nd_item<1> it) {
            const int c = (int)it.get_group(0);
            const int lane = (int)it.get_local_id(0);
            const uint16_t* row = w_inject + (size_t)c * hc_dim;
            float acc = 0.0f;
            for (int i = lane; i < hc_dim; i += 32)
                acc += activation_f32(xq[i]) * f32_from_bf16(row[i]);
            red[c * 32 + lane] = acc;
            for (int off = 16; off > 0; off >>= 1) {
                it.barrier();
                if (lane < off) red[c * 32 + lane] += red[c * 32 + lane + off];
            }
            if (lane == 0) inject[c] = red[c * 32];
        });
    });
}

void launch_gr_write(sycl::queue& q, const float* R, const float* block_out, const float* inject,
                     int n_embd, int hc, float* out) {
    const long long n = (long long)hc * n_embd;
    const size_t grid = (size_t)((n + THREADS - 1) / THREADS);
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> w(sycl::range<1>((size_t)hc), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(grid * THREADS),
                                         sycl::range<1>(THREADS)),
                       [=](sycl::nd_item<1> it) {
            const int tid = (int)it.get_local_id(0);
            if (tid < hc) w[tid] = 2.0f * sigmoid_f(inject[tid] / (float)hc);
            it.barrier();
            for (long long i = (long long)it.get_group(0) * THREADS + tid; i < n;
                 i += (long long)grid * THREADS) {
                const int c = (int)(i / n_embd), d = (int)(i % n_embd);
                out[i] = R[i] + block_out[d] * w[c];
            }
        });
    });
}

}  // namespace

void gr_set_fp32_activations(bool enabled) { fp32_activations = enabled; }
void gr_set_native_mmvf(bool enabled) { native_mmvf = enabled; }

size_t gr_workspace_init(const GrShapes& s, void* base, GrWorkspace& out) {
    const size_t hc_dim = (size_t)s.hc * (size_t)s.n_embd;
    const size_t sz[5] = {
        hc_dim * sizeof(float),               // 0: xn
        hc_dim * sizeof(uint16_t),            // 1: xq
        (size_t)s.hc_lr * sizeof(uint16_t),   // 2: lq
        hc_dim * sizeof(float),               // 3: gated
        (size_t)s.hc_lr * sizeof(float),      // 4: lo
    };
    size_t al[5], bytes = 0;
    for (int k = 0; k < 5; ++k) {
        al[k] = (sz[k] + 15) & ~(size_t)15;
        bytes += al[k];
    }
    out.bytes = bytes;
    if (base != nullptr) {
        unsigned char* p = (unsigned char*)base;
        void* ptr[5];
        for (int k = 0; k < 5; ++k) {
            ptr[k] = p;
            p += al[k];
        }
        out.xn = (float*)ptr[0];
        out.xq = (uint16_t*)ptr[1];
        out.lq = (uint16_t*)ptr[2];
        out.gated = (float*)ptr[3];
        out.lo = (float*)ptr[4];
    }
    return bytes;
}

void gr_read(const float* R, const float* w_norm, const uint16_t* w_down, const uint16_t* w_up,
             const uint16_t* w_inject, float eps, const GrShapes& s, const GrWorkspace& ws,
             float* mixed, float* inject, void* stream) {
    if (s.n_embd <= 0 || s.hc <= 0 || s.hc_lr <= 0) return;
    if (ws.xn == nullptr || ws.xq == nullptr || ws.lq == nullptr || ws.gated == nullptr ||
        ws.lo == nullptr) {
        std::fprintf(stderr, "gr_read: GrWorkspace is not initialised (see gr_workspace_init)\n");
        std::exit(1);
    }
    if (ws.bytes < gr_workspace_bytes(s)) {
        std::fprintf(stderr, "gr_read: GrWorkspace is %zu bytes but this geometry needs %zu\n",
                     ws.bytes, gr_workspace_bytes(s));
        std::exit(1);
    }
    const int n_embd = (int)s.n_embd, hc = (int)s.hc, hc_lr = (int)s.hc_lr;
    const int hc_dim = (int)(s.hc * s.n_embd);
    auto& q = strata::sycl_compat::q_for(stream);

    const bool use_native = native_mmvf;
    const bool use_fp32 = fp32_activations || use_native;
    try {
        if (use_native) {
            if ((hc_dim & 1) != 0 || (hc_lr & 1) != 0)
                throw std::invalid_argument(
                    "gr_read native MMVF requires even hc*n_embd and hc_lr");
            native_gr_rms_norm_weighted(R, w_norm, ws.xn, n_embd, hc, eps, stream);
            bf16_gemv_fp32_mmvf(ws.xn, w_down, ws.lo, hc_dim, hc_lr, stream);
            native_gr_down_silu(ws.lo, hc_lr, hc, stream);
            bf16_gemv_fp32_mmvf(ws.lo, w_up, ws.gated, hc_lr, hc_dim, stream);
            native_gr_pre_gated(ws.xn, ws.gated, mixed, n_embd, hc, w_inject != nullptr, stream);
        } else if (use_fp32) {
            launch_gr_norm<true>(q, R, w_norm, eps, n_embd, ws.xn, ws.xq, hc);
            launch_gr_down<float>(q, ws.xn, w_down, hc_dim, hc_lr, hc, ws.lo);
            launch_gr_gate<float>(q, ws.lo, w_up, ws.xn, hc_dim, hc_lr, ws.gated);
        } else {
            launch_gr_norm<false>(q, R, w_norm, eps, n_embd, ws.xn, ws.xq, hc);
            launch_gr_down<uint16_t>(q, ws.xq, w_down, hc_dim, hc_lr, hc, ws.lq);
            launch_gr_gate<uint16_t>(q, ws.lq, w_up, ws.xn, hc_dim, hc_lr, ws.gated);
        }
        if (!use_native) launch_gr_mean(q, ws.gated, n_embd, hc, mixed);
        if (w_inject != nullptr) {
            if (use_native)
                bf16_gemv_fp32_mmvf(ws.xn, w_inject, inject, hc_dim, hc, stream);
            else if (use_fp32)
                launch_gr_inject<float>(q, ws.xn, w_inject, hc_dim, hc, inject);
            else
                launch_gr_inject<uint16_t>(q, ws.xq, w_inject, hc_dim, hc, inject);
        }
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "gr_read launch: %s\n", e.what());
        std::exit(1);
    }
    if (stream == nullptr) {
        const cudaError_t se = cudaDeviceSynchronize();
        if (se != cudaSuccess) {
            std::fprintf(stderr, "gr_read: %s\n", cudaGetErrorString(se));
            std::exit(1);
        }
    }
}

void gr_write(const float* R, const float* block_out, const float* inject, const GrShapes& s,
              float* R_out, void* stream) {
    if (s.n_embd <= 0 || s.hc <= 0) return;
    try {
        if (native_mmvf)
            native_gr_post(R, block_out, inject, R_out, (int)s.n_embd, (int)s.hc, stream);
        else {
            auto& q = strata::sycl_compat::q_for(stream);
            launch_gr_write(q, R, block_out, inject, (int)s.n_embd, (int)s.hc, R_out);
        }
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "gr_write: %s\n", e.what());
        std::exit(1);
    }
    if (stream == nullptr) {
        const cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            std::fprintf(stderr, "gr_write: %s\n", cudaGetErrorString(e));
            std::exit(1);
        }
    }
}

}  // namespace strata::kernels
