// src/kernels/sycl/native_bf16.cpp — SYCL port of src/kernels/cuda/native_bf16.cu.
// The fp32-activation bf16 MMVF GEMVs (the native projections' workhorse).
//
// Numerical contract preserved:
//   * __fmaf_rn -> sycl::fma (EXPLICIT single-rounding FMA; the library's
//     -ffp-contract=off makes plain a*b+c a double rounding, which is NOT
//     what ggml_cuda_mad does — the two ordered fmas per pair stay ordered)
//   * the xor-butterfly warp sums become per-32-lane-chunk local-memory xor
//     trees with (lane & off) == 0 updaters — the same pairing, so lane 0's
//     value is the CUDA kernel's bit for bit
//   * mmvf_block_size dispatch (32..256 by 32) and all guards verbatim
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <cuda_runtime.h>  // sycl_compat shim

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

int mmvf_block_size(int64_t n_in) {
    int best = 32;
    int64_t best_iterations = (n_in + 63) / 64;
    for (int candidate = 64; candidate <= 256; candidate += 32) {
        const int64_t iterations = (n_in + 2 * candidate - 1) / (2 * candidate);
        if (iterations < best_iterations) {
            best_iterations = iterations;
            best = candidate;
        }
    }
    return best;
}

template <int BLOCK_SIZE>
void launch_mmvf(sycl::queue& q, const float* x, const uint16_t* w, float* y, int n_in,
                 int64_t n_out) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> red(sycl::range<1>(BLOCK_SIZE), h);
        sycl::local_accessor<float, 1> partials(sycl::range<1>(32), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t)n_out * BLOCK_SIZE),
                                         sycl::range<1>(BLOCK_SIZE)),
                       [=](sycl::nd_item<1> it) {
            constexpr int NW = BLOCK_SIZE / 32;
            const int t = (int)it.get_local_id(0);
            const int chunk = t / 32, lane = t % 32;
            const long long row = (long long)it.get_group(0);
            const uint32_t* weights2 =
                reinterpret_cast<const uint32_t*>(w + (size_t)row * n_in);
            const sycl::float2* inputs2 = reinterpret_cast<const sycl::float2*>(x);
            if constexpr (NW > 1) {
                if (t < 32) partials[t] = 0.0f;
                it.barrier();
            }
            float acc = 0.0f;
            for (int pair = t; pair < n_in / 2; pair += BLOCK_SIZE) {
                const uint32_t weight = weights2[pair];
                const sycl::float2 input = inputs2[pair];
                // the two ordered multiply-adds of ggml_cuda_mad, not a pair sum
                acc = sycl::fma(f32_from_bf16((uint16_t)weight), input.x(), acc);
                acc = sycl::fma(f32_from_bf16((uint16_t)(weight >> 16)), input.y(), acc);
            }
            const int base = chunk * 32;
            red[t] = acc;
            for (int off = 16; off > 0; off >>= 1) {
                it.barrier();
                if ((lane & off) == 0) red[base + lane] += red[base + lane + off];
            }
            if constexpr (NW > 1) {
                if (lane == 0) partials[chunk] = red[base];
                it.barrier();
                // barriers must be reached by ALL chunks; only chunk 0's slots are live
                if (chunk == 0) red[lane] = (lane < NW) ? partials[lane] : 0.0f;
                for (int off = 16; off > 0; off >>= 1) {
                    it.barrier();
                    if (chunk == 0 && (lane & off) == 0) red[lane] += red[lane + off];
                }
                if (chunk == 0 && lane == 0) y[row] = red[0];
            } else {
                if (lane == 0) y[row] = red[0];
            }
        });
    });
}

template <int BLOCK_SIZE, int NT>
void launch_mmvf_multi(sycl::queue& q, const float* x, int64_t ldx, const uint16_t* w, float* y,
                       int64_t ldy, int n_in, int n_tok, int64_t n_out) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> red(sycl::range<1>(BLOCK_SIZE * NT), h);
        sycl::local_accessor<float, 1> partials(sycl::range<1>(NT * 32), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t)n_out * BLOCK_SIZE),
                                         sycl::range<1>(BLOCK_SIZE)),
                       [=](sycl::nd_item<1> it) {
            constexpr int NW = BLOCK_SIZE / 32;
            const int t = (int)it.get_local_id(0);
            const int chunk = t / 32, lane = t % 32;
            const long long row = (long long)it.get_group(0);
            const uint32_t* weights2 =
                reinterpret_cast<const uint32_t*>(w + (size_t)row * n_in);
            if constexpr (NW > 1) {
                if (t < 32)
                    for (int k = 0; k < NT; ++k) partials[k * 32 + t] = 0.0f;
                it.barrier();
            }
            float acc[NT];
#pragma unroll
            for (int k = 0; k < NT; ++k) acc[k] = 0.0f;
            for (int pair = t; pair < n_in / 2; pair += BLOCK_SIZE) {
                const uint32_t weight = weights2[pair];
                const float w0 = f32_from_bf16((uint16_t)weight);
                const float w1 = f32_from_bf16((uint16_t)(weight >> 16));
#pragma unroll
                for (int k = 0; k < NT; ++k) {
                    if (k < n_tok) {
                        const sycl::float2 input =
                            reinterpret_cast<const sycl::float2*>(x + (size_t)k * ldx)[pair];
                        acc[k] = sycl::fma(w0, input.x(), acc[k]);
                        acc[k] = sycl::fma(w1, input.y(), acc[k]);
                    }
                }
            }
            const int base = chunk * 32;
            for (int k = 0; k < NT; ++k) red[k * BLOCK_SIZE + t] = acc[k];
            for (int k = 0; k < NT; ++k) {
                for (int off = 16; off > 0; off >>= 1) {
                    it.barrier();
                    if ((lane & off) == 0)
                        red[k * BLOCK_SIZE + base + lane] +=
                            red[k * BLOCK_SIZE + base + lane + off];
                }
            }
            if constexpr (NW > 1) {
                if (lane == 0)
                    for (int k = 0; k < NT; ++k)
                        partials[k * 32 + chunk] = red[k * BLOCK_SIZE + base];
                it.barrier();
                // barriers must be reached by ALL chunks; only chunk 0's slots are live
                if (chunk == 0)
                    for (int k = 0; k < NT; ++k)
                        red[k * 32 + lane] = (lane < NW) ? partials[k * 32 + lane] : 0.0f;
                for (int k = 0; k < NT; ++k) {
                    for (int off = 16; off > 0; off >>= 1) {
                        it.barrier();
                        if (chunk == 0 && (lane & off) == 0)
                            red[k * 32 + lane] += red[k * 32 + lane + off];
                    }
                }
                if (chunk == 0 && lane == 0)
                    for (int k = 0; k < NT; ++k)
                        if (k < n_tok) y[(size_t)k * ldy + row] = red[k * 32];
            } else {
                if (lane == 0)
                    for (int k = 0; k < NT; ++k)
                        if (k < n_tok) y[(size_t)k * ldy + row] = red[k * BLOCK_SIZE];
            }
        });
    });
}

}  // namespace

void bf16_gemv_fp32_mmvf(const float* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out,
                         void* stream) {
    if (n_in <= 0 || (n_in & 1) != 0 || n_in > std::numeric_limits<int>::max() || n_out <= 0 ||
        n_out > std::numeric_limits<int>::max())
        throw std::invalid_argument(
            "bf16_gemv_fp32_mmvf: require positive even n_in and positive n_out <= INT_MAX");
    if (x == nullptr || w == nullptr || y == nullptr ||
        (reinterpret_cast<uintptr_t>(x) & 7u) != 0 ||
        (reinterpret_cast<uintptr_t>(w) & 3u) != 0 ||
        (reinterpret_cast<uintptr_t>(y) & 3u) != 0)
        throw std::invalid_argument("bf16_gemv_fp32_mmvf: null or misaligned pointer");
    auto& q = strata::sycl_compat::q_for(stream);
    try {
#define STRATA_MMVF_CASE(N) \
    case N: launch_mmvf<N>(q, x, w, y, (int)n_in, n_out); break
        switch (mmvf_block_size(n_in)) {
            STRATA_MMVF_CASE(32);
            STRATA_MMVF_CASE(64);
            STRATA_MMVF_CASE(96);
            STRATA_MMVF_CASE(128);
            STRATA_MMVF_CASE(160);
            STRATA_MMVF_CASE(192);
            STRATA_MMVF_CASE(224);
            STRATA_MMVF_CASE(256);
        }
#undef STRATA_MMVF_CASE
    } catch (const sycl::exception& e) {
        throw std::runtime_error(std::string("bf16_gemv_fp32_mmvf launch: ") + e.what());
    }
}

void bf16_gemv_fp32_mmvf_multi(const float* x, int64_t ldx, const uint16_t* w, float* y,
                               int64_t ldy, int64_t n_in, int64_t n_out, int n_tok, void* stream) {
    if (n_tok == 1 && ldy >= n_out) {
        bf16_gemv_fp32_mmvf(x, w, y, n_in, n_out, stream);
        return;
    }
    if (n_tok < 1 || n_tok > 8 || n_in <= 0 || (n_in & 1) != 0 || n_out <= 0 || (ldx & 1) != 0 ||
        x == nullptr || w == nullptr || y == nullptr ||
        (reinterpret_cast<uintptr_t>(x) & 7u) != 0)
        throw std::invalid_argument(
            "bf16_gemv_fp32_mmvf_multi: 1..8 rows, even n_in/ldx, aligned pointers");
    auto& q = strata::sycl_compat::q_for(stream);
    try {
#define STRATA_MMVF_M(N)                                             \
    case N:                                                          \
        if (n_tok <= 4)                                              \
            launch_mmvf_multi<N, 4>(q, x, ldx, w, y, ldy, (int)n_in, \
                                    n_tok, n_out);                   \
        else                                                         \
            launch_mmvf_multi<N, 8>(q, x, ldx, w, y, ldy, (int)n_in, \
                                    n_tok, n_out);                   \
        break
        switch (mmvf_block_size(n_in)) {
            STRATA_MMVF_M(32);
            STRATA_MMVF_M(64);
            STRATA_MMVF_M(96);
            STRATA_MMVF_M(128);
            STRATA_MMVF_M(160);
            STRATA_MMVF_M(192);
            STRATA_MMVF_M(224);
            STRATA_MMVF_M(256);
        }
#undef STRATA_MMVF_M
    } catch (const sycl::exception& e) {
        throw std::runtime_error(std::string("bf16_gemv_fp32_mmvf_multi launch: ") + e.what());
    }
}

}  // namespace strata::kernels
