// src/kernels/sycl/native_gr_postops.cpp — SYCL port of src/kernels/cuda/native_gr_postops.cu.
// __fmaf_rn -> sycl::fma (explicit; contract is off backend-wide),
// __fmul_rn/__fadd_rn -> plain ops (exact under -ffp-contract=off),
// expf -> sycl::exp. Structure 1:1.
#include "strata/kernels/native_gr_postops.hpp"

#include <cuda_runtime.h>  // sycl_compat shim

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

constexpr int THREADS = 256;

inline float sigmoid(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }
inline float scale_zero_bias(float x, float scale) { return sycl::fma(scale, x, 0.0f); }

void check_pointer(const void* p) {
    if (!p || reinterpret_cast<std::uintptr_t>(p) % alignof(float))
        throw std::invalid_argument(
            "native GR postops require non-null four-byte aligned pointers");
}

void check_shape(int n, int hc) {
    if (n <= 0 || hc <= 0 || std::uint64_t(n) * hc > std::uint64_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("native GR postops require positive bounded dimensions");
}

size_t blocks(size_t n) { return (size_t)((n + THREADS - 1) / THREADS); }

void check_launch() {
    const auto error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("native GR postops launch: ") +
                                 cudaGetErrorString(error));
}

}  // namespace

void native_gr_down_silu(float* lo, int hc_lr, int hc, void* stream) {
    check_shape(hc_lr, hc);
    check_pointer(lo);
    const float scale = 1.0f / (float)hc;
    auto& q = strata::sycl_compat::q_for(stream);
    try {
        q.parallel_for(sycl::range<1>(blocks((size_t)hc_lr) * THREADS), [=](sycl::id<1> iid) {
            const size_t i = iid[0];
            if (i >= (size_t)hc_lr) return;
            const float x = scale_zero_bias(lo[i], scale);
            lo[i] = x / (1.0f + sycl::exp(-x));
        });
    } catch (const sycl::exception&) {
        strata::sycl_compat::last_error() = cudaErrorUnknown;
    }
    check_launch();
}

void native_gr_pre_gated(const float* xn, float* gate, float* mixed, int n_embd, int hc,
                         bool fused_layer, void* stream) {
    check_shape(n_embd, hc);
    check_pointer(xn);
    check_pointer(gate);
    check_pointer(mixed);
    const float scale = 1.0f / (float)hc;
    auto& q = strata::sycl_compat::q_for(stream);
    try {
        q.parallel_for(sycl::range<1>(blocks((size_t)n_embd) * THREADS), [=](sycl::id<1> iid) {
            const size_t d = iid[0];
            if (d >= (size_t)n_embd) return;
            float sum = 0.0f;
            for (int c = 0; c < hc; ++c) {
                const size_t i = (size_t)c * n_embd + d;
                const float x = xn[i], w = sigmoid(gate[i]);
                const float product = x * w;  // __fmul_rn: plain under contract-off
                gate[i] = product;
                if (fused_layer)
                    sum = sycl::fma(x, w, sum);
                else
                    sum = c == 0 ? product : sum + product;  // __fadd_rn
            }
            if (fused_layer)
                mixed[d] = scale * sum;
            else
                mixed[d] = scale_zero_bias(sum, scale);
        });
    } catch (const sycl::exception&) {
        strata::sycl_compat::last_error() = cudaErrorUnknown;
    }
    check_launch();
}

void native_gr_post(const float* residual, const float* block_out, const float* inject,
                    float* output, int n_embd, int hc, void* stream) {
    check_shape(n_embd, hc);
    check_pointer(residual);
    check_pointer(block_out);
    check_pointer(inject);
    check_pointer(output);
    const float scale = 1.0f / (float)hc;
    auto& q = strata::sycl_compat::q_for(stream);
    try {
        q.parallel_for(sycl::range<1>(blocks((size_t)n_embd * hc) * THREADS), [=](sycl::id<1> iid) {
            const size_t i = iid[0];
            if (i >= (size_t)n_embd * hc) return;
            const int c = (int)(i / (size_t)n_embd), d = (int)(i % (size_t)n_embd);
            const float weight = scale_zero_bias(sigmoid(scale_zero_bias(inject[c], scale)), 2.0f);
            output[i] = sycl::fma(block_out[d], weight, residual[i]);
        });
    } catch (const sycl::exception&) {
        strata::sycl_compat::last_error() = cudaErrorUnknown;
    }
    check_launch();
}

}  // namespace strata::kernels
