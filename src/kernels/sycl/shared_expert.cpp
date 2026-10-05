// src/kernels/sycl/shared_expert.cpp — SYCL port of the verify-window shared
// expert (src/kernels/cuda/shared_expert.cu): native swiglu (fast-math silu,
// as pinned), the double-precision bf16 scalar gate, the scalar sigmoids and
// the per-token output scaling, composed exactly as shared_expert_multi does
// on CUDA (quantize -> gate/up MMVQ -> swiglu -> quantize -> down MMVQ -> gate
// -> scale rows).
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/shared_expert.hpp"

#include <cuda_runtime.h>  // sycl_compat shim

#include <atomic>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {
constexpr int THREADS = 256;
std::atomic<bool> native_bf16{false};

using sycl::local_accessor;
using sycl::nd_item;

inline sycl::queue& Q(void* stream) { return strata::sycl_compat::q_for(stream); }

void check_throw(void* stream, const char* what) {
    if (stream != nullptr) return;
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

}  // namespace

void shared_expert_set_native_bf16(bool enabled) { native_bf16.store(enabled, std::memory_order_relaxed); }

void shared_expert_multi(int n_tok, const float* x, const uint16_t* x_bf16, const NativeSharedWeights& nw,
                         const uint16_t* gate_inp_bf16, float* gate, float* up, float* g, float* out, int64_t n_embd,
                         int64_t n_ff, void* stream) {
    if (n_tok < 1 || n_tok > 8 || !nw.q8_1 || !nw.gate_data || !nw.up_data || !nw.down_data || !stream)
        throw std::invalid_argument("shared_expert_multi: needs 1..8 tokens, native weights, scratch and a stream");
    native_quantize_q8_1(x, nw.q8_1, (int) n_embd, n_tok, stream);
    native_mmvq(nw.gate_type, nw.gate_data, nw.q8_1, gate, (int) n_embd, (int) n_ff, n_tok, stream);
    native_mmvq(nw.up_type, nw.up_data, nw.q8_1, up, (int) n_embd, (int) n_ff, n_tok, stream);
    const int64_t n = n_ff * n_tok;
    // native swiglu: pinned fast-math silu(x) * up
    Q(stream).parallel_for((size_t) n, [=](size_t i) {
        const float xv = gate[i];
        gate[i] = (xv / (1.0f + sycl::exp(-xv))) * up[i];
    });
    native_quantize_q8_1(gate, nw.q8_1, (int) n_ff, n_tok, stream);
    native_mmvq(nw.down_type, nw.down_data, nw.q8_1, out, (int) n_ff, (int) n_embd, n_tok, stream);
    if (native_bf16.load(std::memory_order_relaxed)) {
        for (int t = 0; t < n_tok; ++t) {
            bf16_gemv_fp32_mmvf(x + (size_t) t * n_embd, gate_inp_bf16, g + t, n_embd, 1, stream);
        }
        Q(stream).parallel_for((size_t) n_tok, [=](size_t t) {
            g[t] = (1.0f / (1.0f + sycl::exp(-g[t])));
        });
    } else {
        // scalar_gate: double-precision bf16 dot, then sigmoid
        Q(stream).submit([&](sycl::handler& hnd) {
            local_accessor<float, 1> scratch(sycl::range<1>(THREADS), hnd);
            local_accessor<float, 1> warpsum(sycl::range<1>(THREADS / 32), hnd);
            hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) n_tok * THREADS), sycl::range<1>(THREADS)),
                             [=](nd_item<1> it) {
                                 const int t = (int) it.get_group(0);
                                 const int tx = (int) it.get_local_id(0);
                                 const int lane = tx & 31, warp = tx >> 5;
                                 const uint16_t* xb = x_bf16 + (size_t) t * n_embd;
                                 double acc = 0.0;
                                 for (int64_t i = tx; i < n_embd; i += THREADS)
                                     acc += (double) f32_from_bf16(xb[i]) * (double) f32_from_bf16(gate_inp_bf16[i]);
#pragma unroll
                                 for (int off = 16; off > 0; off >>= 1) {
                                     scratch[tx] = (float) acc;
                                     it.barrier(sycl::access::fence_space::local_space);
                                     if (lane + off < 32) acc += (double) scratch[tx + off];
                                     it.barrier(sycl::access::fence_space::local_space);
                                 }
                                 if (lane == 0) warpsum[warp] = (float) acc;
                                 it.barrier(sycl::access::fence_space::local_space);
                                 // every warp runs the 8-value merge redundantly
                                 double v = lane < THREADS / 32 ? (double) warpsum[lane] : 0.0;
#pragma unroll
                                 for (int off = 16; off > 0; off >>= 1) {
                                     scratch[tx] = (float) v;
                                     it.barrier(sycl::access::fence_space::local_space);
                                     if (lane + off < 32) v += (double) scratch[tx + off];
                                     it.barrier(sycl::access::fence_space::local_space);
                                 }
                                 if (tx == 0) g[t] = (float) (1.0 / (1.0 + sycl::exp(-v)));
                             });
        });
    }
    Q(stream).parallel_for(sycl::range<2>((size_t) n_tok, (size_t) n_embd), [=](sycl::id<2> id) {
        out[id[0] * (size_t) n_embd + id[1]] *= g[id[0]];
    });
    check_throw(stream, "shared_expert_multi");
}

}  // namespace strata::kernels
