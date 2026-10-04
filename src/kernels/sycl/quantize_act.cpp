// src/kernels/sycl/quantize_act.cpp — SYCL port of src/kernels/cuda/quantize_act.cu.
//
// Port rules (milestone 3, Q4/Q8-first directive):
//   * arithmetic IDENTICAL to the CUDA source, including the double-precision
//     division + rint in quantize_q8_0 (subtlety 2 of that file) and the
//     round-half-away-from-zero rule of the _scaled variant — parity gates on it
//   * __fmul_rn(a,b) -> plain a*b; the SYCL library is compiled -ffp-contract=off
//     so no FMA contraction can change the rounding step (same reason the CUDA
//     source pins __fmul_rn)
//   * <<<grid, block>>> -> parallel_for over blocks (one work-item per CUDA thread)
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cuda_runtime.h>  // sycl_compat shim

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int QK8_0 = 32;
constexpr int QK_K = 256;
constexpr int Q8K_BYTES = 292;

// ggml's nearest_int (magic-add), bit_cast instead of memcpy for device code
inline int nearest_int_dev(float fval) {
    const float val = fval + 12582912.0f;
    const int i = sycl::bit_cast<int>(val);
    return (i & 0x007fffff) - 0x00400000;
}

}  // namespace

void quantize_q8_0(const float* x, uint8_t* blocks, int64_t n, void* stream) {
    if (n <= 0) return;
    if (n % QK8_0 != 0) {
        std::fprintf(stderr, "quantize_q8_0: n %lld is not a multiple of %d\n", (long long)n, QK8_0);
        std::exit(1);
    }
    const long long nb = n / QK8_0;
    auto& q = strata::sycl_compat::q_for(stream);
    try {
        q.parallel_for(sycl::range<1>((size_t)nb), [=](sycl::id<1> bid) {
            const long long b = bid[0];
            const float* xb = x + b * QK8_0;
            uint8_t* out = blocks + b * 34;  // { fp16 d ; int8 qs[32] }

            float amax = 0.0f;
            for (int i = 0; i < QK8_0; ++i) amax = sycl::fmax(amax, sycl::fabs(xb[i]));
            if (amax == 0.0f) {
                const uint16_t zb = f16_from_f32(0.0f);
                out[0] = (uint8_t)(zb & 0xFF);
                out[1] = (uint8_t)(zb >> 8);
                for (int i = 0; i < QK8_0; ++i) out[2 + i] = 0;
                return;
            }
            const float d32 = amax / 127.0f;
            const uint16_t d16bits = f16_from_f32(d32);
            out[0] = (uint8_t)(d16bits & 0xFF);
            out[1] = (uint8_t)(d16bits >> 8);

            // double division, matching the reference exactly (subtlety 2)
            for (int i = 0; i < QK8_0; ++i) {
                double qv = sycl::rint((double)xb[i] / (double)d32);
                if (qv > 127.0) qv = 127.0;
                if (qv < -128.0) qv = -128.0;
                out[2 + i] = (uint8_t)(int8_t)qv;
            }
        });
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "quantize_q8_0 launch: %s\n", e.what());
        std::exit(1);
    }
    if (stream == nullptr) cudaDeviceSynchronize();
}

void quantize_q8_0_scaled(const float* x, uint8_t* blocks, float* scales, int64_t n, void* stream) {
    if (n <= 0) return;
    if (n % QK8_0 != 0) {
        std::fprintf(stderr, "quantize_q8_0_scaled: n %lld is not a multiple of %d\n", (long long)n,
                     QK8_0);
        std::exit(1);
    }
    if (scales == nullptr) {
        std::fprintf(stderr, "quantize_q8_0_scaled: scales is null\n");
        std::exit(1);
    }
    const long long nb = n / QK8_0;
    auto& q = strata::sycl_compat::q_for(stream);
    try {
        q.parallel_for(sycl::range<1>((size_t)nb), [=](sycl::id<1> bid) {
            const long long b = bid[0];
            const float* xb = x + b * QK8_0;
            uint8_t* out = blocks + b * 34;

            float amax = 0.0f;
            for (int i = 0; i < QK8_0; ++i) amax = sycl::fmax(amax, sycl::fabs(xb[i]));
            // VERBATIM from cpu/expert.cpp:144-145 (IEEE division guaranteed by
            // -foffload-fp32-prec-div)
            const float s = amax > 0.f ? amax / 127.f : 0.f;
            const float inv = s > 0.f ? 1.f / s : 0.f;
            scales[b] = s;

            const uint16_t d16bits = f16_from_f32(s);
            out[0] = (uint8_t)(d16bits & 0xFF);
            out[1] = (uint8_t)(d16bits >> 8);
            for (int i = 0; i < QK8_0; ++i) {
                // VERBATIM from cpu/expert.cpp:159-162: reciprocal multiply, round half AWAY from zero
                const float t = xb[i] * inv;
                const float r = t + (t >= 0.f ? 0.5f : -0.5f);
                int v = (int)r;
                v = v < -127 ? -127 : (v > 127 ? 127 : v);
                out[2 + i] = (uint8_t)(int8_t)v;
            }
        });
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "quantize_q8_0_scaled launch: %s\n", e.what());
        std::exit(1);
    }
    if (stream == nullptr) cudaDeviceSynchronize();
}

void dequant_q8_0(const uint8_t* blocks, float* x, int64_t n, void* stream) {
    if (n <= 0) return;
    const long long nb = n / QK8_0;
    auto& q = strata::sycl_compat::q_for(stream);
    try {
        q.parallel_for(sycl::range<1>((size_t)nb), [=](sycl::id<1> bid) {
            const long long b = bid[0];
            const uint8_t* blk = blocks + b * 34;
            const uint16_t dbits = (uint16_t)(blk[0] | (blk[1] << 8));
            const float d = f32_from_f16(dbits);
            float* out = x + b * QK8_0;
            for (int i = 0; i < QK8_0; ++i) out[i] = (float)(int8_t)blk[2 + i] * d;
        });
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "dequant_q8_0 launch: %s\n", e.what());
        std::exit(1);
    }
    if (stream == nullptr) cudaDeviceSynchronize();
}

void quantize_q8_K(const float* x, uint8_t* blocks, int64_t n, void* stream) {
    if (n <= 0) return;
    if (n % QK_K != 0) {
        std::fprintf(stderr, "quantize_q8_K: n %lld is not a multiple of %d\n", (long long)n, QK_K);
        std::exit(1);
    }
    const long long nb = n / QK_K;
    auto& q = strata::sycl_compat::q_for(stream);
    try {
        q.parallel_for(sycl::range<1>((size_t)nb), [=](sycl::id<1> bid) {
            const long long b = bid[0];
            const float* xb = x + b * QK_K;
            uint8_t* out = blocks + b * Q8K_BYTES;
            float* d = (float*)out;
            int8_t* qs = (int8_t*)(out + 4);
            int16_t* bsums = (int16_t*)(out + 4 + QK_K);

            float max = 0.0f, amax = 0.0f;
            for (int j = 0; j < QK_K; ++j) {
                const float ax = sycl::fabs(xb[j]);
                if (ax > amax) {
                    amax = ax;
                    max = xb[j];
                }
            }
            if (amax == 0.0f) {
                *d = 0.0f;
                for (int j = 0; j < QK_K; ++j) qs[j] = 0;
                for (int j = 0; j < QK_K / 16; ++j) bsums[j] = 0;
                return;
            }
            // fp32 division, verbatim from the CUDA source — IEEE-exact because
            // the backend compiles with -foffload-fp32-prec-div (icpx's default
            // device division is a ~28%-1ulp-off reciprocal; see div_sweep note
            // in cmake/sycl_backend.cmake)
            const float iscale = -127.0f / max;
            for (int j = 0; j < QK_K; ++j) {
                // __fmul_rn == plain product under -ffp-contract=off
                const int v = nearest_int_dev(iscale * xb[j]);
                qs[j] = (int8_t)sycl::min(127, v);  // MIN only — no lower clamp
            }
            for (int j = 0; j < QK_K / 16; ++j) {
                int sum = 0;
                for (int ii = 0; ii < 16; ++ii) sum += qs[j * 16 + ii];
                bsums[j] = (int16_t)sum;
            }
            *d = 1.0f / iscale;
        });
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "quantize_q8_K launch: %s\n", e.what());
        std::exit(1);
    }
    if (stream == nullptr) cudaDeviceSynchronize();
}

void dequant_q8_K(const uint8_t* blocks, float* x, int64_t n, void* stream) {
    if (n <= 0) return;
    const long long nb = n / QK_K;
    auto& q = strata::sycl_compat::q_for(stream);
    try {
        q.parallel_for(sycl::range<1>((size_t)nb), [=](sycl::id<1> bid) {
            const long long b = bid[0];
            const uint8_t* blk = blocks + b * Q8K_BYTES;
            float d = sycl::bit_cast<float>(*(const uint32_t*)blk);
            const int8_t* qs = (const int8_t*)(blk + 4);
            float* out = x + b * QK_K;
            for (int i = 0; i < QK_K; ++i) out[i] = (float)qs[i] * d;
        });
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "dequant_q8_K launch: %s\n", e.what());
        std::exit(1);
    }
    if (stream == nullptr) cudaDeviceSynchronize();
}

}  // namespace strata::kernels
