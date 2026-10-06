// src/kernels/sycl/ple.cpp — SYCL port of src/kernels/cuda/ple.cu: the PLE
// block composition (key/value projections, grouped norms, gate, broadcast,
// dilated conv + SiLU, add3) plus the history advance. The block-wide double
// sums keep their order (strided partials, down-tree, warp-partial tree);
// bf16 round-trips use bf16_bits.hpp's pinned rounding.
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_gemv_q8.hpp"

#include <cuda_runtime.h>  // sycl_compat shim

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {
constexpr int THREADS = 256;
bool native_bf16 = false;
bool native_postops = false;

using sycl::local_accessor;
using sycl::nd_item;

inline sycl::queue& Q(void* stream) { return strata::sycl_compat::q_for(stream); }

void ck(void* stream, cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "ple_block: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
    (void) stream;
}

// block-wide double sum, broadcast (the CUDA block_sum's order: strided
// partials, per-warp down-tree, warp-0 down-tree over the 8 partials)
inline double block_sum(double v, int tx, local_accessor<double, 1> scratch, nd_item<1> it) {
    it.barrier(sycl::access::fence_space::local_space);
    const int lane = tx & 31, warp = tx >> 5;
    scratch[tx] = v;
    it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const double other = lane + off < 32 ? scratch[(tx & ~31) + lane + off] : 0.0;
        it.barrier(sycl::access::fence_space::local_space);
        if (lane + off < 32) v += other;
        scratch[tx] = v;
        it.barrier(sycl::access::fence_space::local_space);
    }
    if (lane == 0) scratch[warp] = v;
    it.barrier(sycl::access::fence_space::local_space);
    v = tx < 8 ? scratch[tx] : 0.0;
    scratch[tx] = v;
    it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const double other = lane + off < 32 ? scratch[(tx & ~31) + lane + off] : 0.0;
        it.barrier(sycl::access::fence_space::local_space);
        if (lane + off < 32) v += other;
        scratch[tx] = v;
        it.barrier(sycl::access::fence_space::local_space);
    }
    if (tx == 0) scratch[0] = v;
    it.barrier(sycl::access::fence_space::local_space);
    return scratch[0];
}

void gnorm_launch(const float* x, const float* w, float* y, int n_embd, int hc, float eps, void* stream) {
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<double, 1> scratch(sycl::range<1>(THREADS), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) hc * THREADS), sycl::range<1>(THREADS)),
                         [=](nd_item<1> it) {
                             const int c = (int) it.get_group(0);
                             const int tx = (int) it.get_local_id(0);
                             const float* xc = x + (size_t) c * n_embd;
                             const float* wc = w + (size_t) c * n_embd;
                             float* yc = y + (size_t) c * n_embd;
                             double acc = 0.0;
                             for (int d = tx; d < n_embd; d += THREADS) {
                                 const float sq = xc[d] * xc[d];
                                 acc += (double) sq;
                             }
                             const float mean = (float) (block_sum(acc, tx, scratch, it) / (double) n_embd);
                             const float scale = 1.0f / sycl::sqrt(mean + eps);
                             for (int d = tx; d < n_embd; d += THREADS) yc[d] = xc[d] * scale * wc[d];
                         });
    });
}

void gate_launch(const float* key, const float* query, float* gate, int n_embd, int hc, float inv_sqrt_n,
                 void* stream) {
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<double, 1> scratch(sycl::range<1>(THREADS), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) hc * THREADS), sycl::range<1>(THREADS)),
                         [=](nd_item<1> it) {
                             const int c = (int) it.get_group(0);
                             const int tx = (int) it.get_local_id(0);
                             const float* kc = key + (size_t) c * n_embd;
                             const float* qc = query + (size_t) c * n_embd;
                             double acc = 0.0;
                             for (int d = tx; d < n_embd; d += THREADS) acc += (double) (kc[d] * qc[d]);
                             const float s = (float) block_sum(acc, tx, scratch, it) * inv_sqrt_n;
                             const float mag = sycl::sqrt(sycl::fmax(sycl::fabs(s), 1e-6f));
                             const float sgn = (s > 0.0f) ? 1.0f : ((s < 0.0f) ? -1.0f : 0.0f);
                             if (tx == 0) gate[c] = 1.0f / (1.0f + sycl::exp(-(sgn * mag)));
                         });
    });
}

}  // namespace

void ple_set_native_bf16(bool enabled) { native_bf16 = enabled; }
void ple_set_native_postops(bool enabled) { native_postops = enabled; }
bool ple_native_postops_enabled() { return native_postops; }

void ple_history_advance(float* hist, const float* normalized, void* stream) {
    if (hist == nullptr || normalized == nullptr)
        throw std::invalid_argument("ple_history_advance: null history or normalized input");
    Q(stream).parallel_for((size_t) NG_HC_DIM, [=](size_t channel) {
        float* column = hist + (size_t) channel * NG_HIST;
        for (int row = 0; row + 1 < NG_HIST; ++row) column[row] = column[row + 1];
        column[NG_HIST - 1] = normalized[channel];
    });
}

bool ple_block_available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

uint64_t ple_block_scratch_bytes() {
    const size_t f = (size_t) (5 * NG_HC_DIM + NG_N_EMBD + NG_HC) * sizeof(float);
    const size_t q = (size_t) (NG_N_EMBD / 32) * 34;
    const size_t e = (size_t) NG_N_EMBD * sizeof(uint16_t);
    return ((f + 15) & ~(size_t) 15) + ((q + 15) & ~(size_t) 15) + e + 256;
}

void ple_block(const float* emb, const float* hidden, const float* hist_rows, const PleWeights& w, PleOut& out,
               void* scratch, void* stream) {
    const bool native_key = w.key_native_data != nullptr && w.key_bf16 == nullptr;
    if (native_key && (!emb || !hidden || !hist_rows || !out.result || !scratch || !stream || !w.key_native_q8_1 ||
                       (w.key_native_type != 42 && w.key_native_type != 18 && w.key_native_type != 23 &&
                        w.key_native_type != 8)))
        throw std::invalid_argument(
            "ple_block: native key requires Q2_0, IQ3_XXS, IQ4_XS or Q8_0 weights, input/output, private scratch "
            "and explicit stream");
    if (emb == nullptr || hidden == nullptr || hist_rows == nullptr || out.result == nullptr) return;
    const int n_embd = NG_N_EMBD, hc = NG_HC, hc_dim = NG_HC_DIM;
    if (scratch == nullptr) {
        std::fprintf(stderr, "ple_block: scratch is null; the caller owns it (see ple_block_scratch_bytes)\n");
        std::exit(1);
    }
    const size_t float_bytes = (size_t) (5 * hc_dim + n_embd + hc) * sizeof(float);
    auto dbg_sync = [&](const char* n) {
        if (std::getenv("STRATA_PLE_SYNC_EACH") == nullptr) return;
        const cudaError_t e = cudaStreamSynchronize((cudaStream_t) stream);
        std::fprintf(stderr, "[dbg] ple sync %-14s %s\n", n, cudaGetErrorString(e));
    };
    const size_t q8_bytes = (size_t) (n_embd / 32) * 34;
    uint8_t* base = (uint8_t*) scratch;
    float* d_scratch = (float*) base;
    uint8_t* d_act = base + ((float_bytes + 15) & ~(size_t) 15);
    uint16_t* d_emb16 = (uint16_t*) (d_act + ((q8_bytes + 15) & ~(size_t) 15));
    float* d_key = d_scratch;
    float* d_query = d_key + hc_dim;
    float* d_norm = d_query + hc_dim;
    float* d_gated = d_norm + hc_dim;
    float* d_conv = d_gated + hc_dim;
    float* d_value = d_conv + hc_dim;
    float* d_gate = d_value + n_embd;

    if (w.key_bf16 != nullptr) {
        bf16_gemv_fp32_mmvf(emb, w.key_bf16, d_key, n_embd, hc_dim, stream);
    } else if (native_key) {
        native_quantize_q8_1(emb, w.key_native_q8_1, n_embd, 1, stream);
        native_mmvq(w.key_native_type, w.key_native_data, w.key_native_q8_1, d_key, n_embd, hc_dim, 1, stream);
    } else {
        quantize_q8_0(emb, d_act, n_embd, stream);
        s2_gemv_q8(d_act, w.key_codes, w.key_scales, d_key, n_embd, hc_dim, 8, stream);
    }
    dbg_sync("keyproj");
    if (!native_postops) {
        gnorm_launch(d_key, w.norm_key, d_key, n_embd, hc, NG_RMS_EPS, stream);
        gnorm_launch(hidden, w.norm_query, d_query, n_embd, hc, NG_RMS_EPS, stream);
    }
    dbg_sync("gnorms");
    if (native_bf16) {
        bf16_gemv_fp32_mmvf(emb, w.value_bf16, d_value, n_embd, n_embd, stream);
    } else {
        const int n = n_embd;
        Q(stream).parallel_for((size_t) n, [=](size_t i) { d_emb16[i] = bf16_from_f32(emb[i]); });
        Q(stream).parallel_for((size_t) n, [=](size_t o) {
            const uint16_t* row = w.value_bf16 + (size_t) o * n;
            double acc = 0.0;
            for (int i = 0; i < n; ++i) acc += (double) f32_from_bf16(d_emb16[i]) * (double) f32_from_bf16(row[i]);
            d_value[o] = (float) acc;
        });
    }

    dbg_sync("value");
    const float* normalized_key = d_key;
    if (native_postops) {
        NativePlePostopsBuffers buffers{d_query, d_norm, d_gate, d_gated, d_norm, d_conv, out.result};
        native_ple_postops(d_key, hidden, d_value, hist_rows, w, buffers, stream);
        normalized_key = d_query;
    } else {
        gate_launch(d_key, d_query, d_gate, n_embd, hc, 1.0f / sycl::sqrt((float) n_embd), stream);
        const size_t total = (size_t) hc_dim;
        gate_launch(d_key, d_query, d_gate, n_embd, hc, 1.0f / sycl::sqrt((float) n_embd), stream);
        dbg_sync("gate");
        Q(stream).parallel_for(total, [=](size_t i) { d_gated[i] = d_value[i % n_embd] * d_gate[i / n_embd]; });
        dbg_sync("bcast");
        gnorm_launch(d_gated, w.norm_conv, d_norm, n_embd, hc, NG_RMS_EPS, stream);
        dbg_sync("gnorm3");
        Q(stream).parallel_for(total, [=](size_t c) {
            float acc = 0.0f;
            for (int k = 0; k < PLE_CONV_KERNEL; ++k) {
                const int row = NG_HIST - (PLE_CONV_KERNEL - 1 - k) * NGRAM_SIZE;
                const float v = (row == NG_HIST) ? d_norm[c] : hist_rows[(size_t) row + (size_t) NG_HIST * c];
                acc += f32_from_f16(w.conv1d_f16[k + PLE_CONV_KERNEL * c]) * v;
            }
            d_conv[c] = acc / (1.0f + sycl::exp(-acc));
        });
        dbg_sync("conv");
        Q(stream).parallel_for(total,
                               [=](size_t i) { out.result[i] = hidden[i] + d_gated[i] + d_conv[i]; });
    }

    dbg_sync("add3");
    if (out.key) ck(stream, cudaMemcpyAsync(out.key, normalized_key, hc_dim * sizeof(float), cudaMemcpyDeviceToDevice,
                                            (cudaStream_t) stream), "key");
    if (out.value)
        ck(stream, cudaMemcpyAsync(out.value, d_value, n_embd * sizeof(float), cudaMemcpyDeviceToDevice,
                                   (cudaStream_t) stream), "value");
    if (out.gate)
        ck(stream, cudaMemcpyAsync(out.gate, d_gate, hc * sizeof(float), cudaMemcpyDeviceToDevice,
                                   (cudaStream_t) stream), "gate");
    if (out.gated)
        ck(stream, cudaMemcpyAsync(out.gated, d_gated, hc_dim * sizeof(float), cudaMemcpyDeviceToDevice,
                                   (cudaStream_t) stream), "gated");
    if (out.normalized)
        ck(stream, cudaMemcpyAsync(out.normalized, d_norm, hc_dim * sizeof(float), cudaMemcpyDeviceToDevice,
                                   (cudaStream_t) stream), "norm");
    if (out.conv)
        ck(stream, cudaMemcpyAsync(out.conv, d_conv, hc_dim * sizeof(float), cudaMemcpyDeviceToDevice,
                                   (cudaStream_t) stream), "conv");
    dbg_sync("export");
    ck(stream, cudaGetLastError(), "launch");
}

}  // namespace strata::kernels
