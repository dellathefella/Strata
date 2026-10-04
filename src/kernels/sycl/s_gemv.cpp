// src/kernels/sycl/s_gemv.cpp — SYCL port of the s_gemv family
// (src/kernels/cuda/s_gemv.cu): the canonical-form S-family GEMV
//     value = cb[code] * scale + offset
// over fp16 activations. Ported entry points (milestone 3 first increment):
//   s_gemv            — naive, one work-item per output row (the reference)
//   s_gemv_split      — one work-group per row, four accumulators, tree reduce
//   s_gemv_split_async — same on a caller stream, no sync (doorbell overlap)
// The Q8_K-activation variants (s_gemv_q8k*) land with the K-quant milestone.
//
// Port notes:
//   * __shared__ codebook staging (s_iq4nl) is dropped: kIq4Nl is a
//     device-visible constexpr array — one implementation, no shared-memory
//     copy, same values (the CUDA staging existed for __constant__ semantics).
//   * __half22float2 -> sycl::half2::convert<float> (exact fp16->fp32).
//   * reduction order preserved exactly (parity tolerance assumes it).
#include "strata/kernels/s_gemv.hpp"

#include <cuda_fp16.h>     // sycl_compat shim
#include <cuda_runtime.h>  // sycl_compat shim

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr signed char kIq4Nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                      1,   13,   25,  38,  53,  69,  89,  113};

template <int CODE_BITS>
inline float decode(int code, int bias, int codebook) {
    if (codebook == (int)Codebook::Iq4Nl) return (float)kIq4Nl[code & 0x0F];
    return (float)(code + bias);  // bias applies to the CODE, integer domain
}

// ---- naive kernel: one work-item per output row -----------------------------
template <int CODE_BITS>
void launch_naive(sycl::queue& q, const uint16_t* x, const uint8_t* codes, const float* scales,
                  const float* offset, float* y, long long n_in, long long n_out, int bias,
                  int codebook, int group_elems, int has_offset) {
    constexpr int PER_BYTE = 8 / CODE_BITS;
    q.parallel_for(sycl::range<1>((size_t)n_out), [=](sycl::id<1> oid) {
        const long long o = oid[0];
        const long long n_groups = n_in / group_elems;
        const long long codes_per_row = n_in / PER_BYTE;
        const uint8_t* c = codes + o * codes_per_row;
        const float* s = scales + o * n_groups;
        const float* off = has_offset ? offset + o * n_groups : nullptr;

        float acc = 0.0f;
        for (long long g = 0; g < n_groups; ++g) {
            const float d = s[g];
            const float b = off ? off[g] : 0.0f;
            const long long base = g * (long long)group_elems;
            for (int j = 0; j < group_elems; ++j) {
                const long long i = base + j;
                const int code =
                    (c[i / PER_BYTE] >> ((int)(i % PER_BYTE) * CODE_BITS)) & ((1 << CODE_BITS) - 1);
                // offset belongs to the WEIGHT: applied BEFORE the activation multiply
                const float w = decode<CODE_BITS>(code, bias, codebook) * d + b;
                acc += w * __half2float(__ushort_as_half(x[i]));
            }
        }
        y[o] = acc;
    });
}

// ---- split kernel: one work-group per row, 4 accumulators -------------------
template <int CODE_BITS>
void launch_split(sycl::queue& q, const uint16_t* x, const uint8_t* codes, const float* scales,
                  const float* offset, float* y, long long n_in, long long n_out, int bias,
                  int codebook, int group_elems, int group_shift, int has_offset, int tpr) {
    constexpr int PER_BYTE = 8 / CODE_BITS;
    constexpr int QE = 4;
    constexpr int QB = QE * CODE_BITS / 8;  // bytes per quad: 1 for S2, 2 for S4, 4 for S8
    constexpr unsigned MASK = (1u << CODE_BITS) - 1u;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> partial(sycl::range<1>((size_t)tpr), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t)n_out * tpr),
                                         sycl::range<1>((size_t)tpr)),
                       [=](sycl::nd_item<1> it) {
                           const long long o = (long long)it.get_group(0);
                           if (o >= n_out) return;
                           const int tid = (int)it.get_local_id(0);

                           const long long n_groups = n_in / group_elems;
                           const uint8_t* c = codes + o * (n_in / PER_BYTE);
                           const float* s = scales + o * n_groups;
                           const float* off = has_offset ? offset + o * n_groups : nullptr;

                           float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
                           long long i = (long long)tid * QE;
                           for (; i + QE <= n_in; i += (long long)tpr * QE) {
                               const long long g = i >> group_shift;
                               const float d = s[g];
                               const float b = off ? off[g] : 0.0f;
                               const uint8_t* cp = c + i / PER_BYTE;
                               unsigned v;
                               if (QB == 1)
                                   v = cp[0];
                               else if (QB == 2)
                                   v = (unsigned)cp[0] | ((unsigned)cp[1] << 8);
                               else
                                   v = (unsigned)cp[0] | ((unsigned)cp[1] << 8) |
                                       ((unsigned)cp[2] << 16) | ((unsigned)cp[3] << 24);
                               const sycl::float2 f01 =
                                   (*reinterpret_cast<const sycl::half2*>(x + i)).convert<float>();
                               const sycl::float2 f23 =
                                   (*reinterpret_cast<const sycl::half2*>(x + i + 2)).convert<float>();
                               acc0 += (decode<CODE_BITS>((int)(v & MASK), bias, codebook) * d + b) *
                                       f01.x();
                               acc1 +=
                                   (decode<CODE_BITS>((int)((v >> CODE_BITS) & MASK), bias, codebook) *
                                    d + b) *
                                   f01.y();
                               acc2 += (decode<CODE_BITS>((int)((v >> (2 * CODE_BITS)) & MASK), bias,
                                                          codebook) *
                                          d + b) *
                                       f23.x();
                               acc3 += (decode<CODE_BITS>((int)((v >> (3 * CODE_BITS)) & MASK), bias,
                                                          codebook) *
                                          d + b) *
                                       f23.y();
                           }
                           for (; i < n_in; i += (long long)tpr * QE) {
                               for (int k = 0; k < QE && i + k < n_in; ++k) {
                                   const long long e = i + k;
                                   const long long g = e >> group_shift;
                                   const int code =
                                       (c[e / PER_BYTE] >> ((int)(e % PER_BYTE) * CODE_BITS)) & MASK;
                                   acc0 += (decode<CODE_BITS>(code, bias, codebook) * s[g] +
                                            (off ? off[g] : 0.0f)) *
                                           __half2float(__ushort_as_half(x[e]));
                               }
                           }
                           partial[tid] = (acc0 + acc1) + (acc2 + acc3);
                           it.barrier();
                           for (int step = tpr / 2; step > 0; step >>= 1) {
                               if (tid < step) partial[tid] += partial[tid + step];
                               it.barrier();
                           }
                           if (tid == 0) y[o] = partial[0];
                       });
    });
}

}  // namespace

void s_gemv(const uint16_t* x, const uint8_t* codes, const float* scales, const float* offset,
            float* y, int64_t n_in, int64_t n_out, const SForm& form) {
    if (n_in <= 0 || n_out <= 0) return;
    if (form.group_elems <= 0 || n_in % form.group_elems != 0) {
        std::fprintf(stderr, "s_gemv: n_in %lld is not a multiple of group_elems %d\n",
                     (long long)n_in, form.group_elems);
        std::exit(1);
    }
    if (form.has_offset && offset == nullptr) {
        std::fprintf(stderr, "s_gemv: form says has_offset but offset is null\n");
        std::exit(1);
    }
    auto& q = strata::sycl_compat::default_queue();
    const int cb = (int)form.codebook;
    try {
        switch (form.code_bits) {
            case 2:
                launch_naive<2>(q, x, codes, scales, offset, y, n_in, n_out, form.code_bias, cb,
                                form.group_elems, form.has_offset ? 1 : 0);
                break;
            case 4:
                launch_naive<4>(q, x, codes, scales, offset, y, n_in, n_out, form.code_bias, cb,
                                form.group_elems, form.has_offset ? 1 : 0);
                break;
            case 8:
                launch_naive<8>(q, x, codes, scales, offset, y, n_in, n_out, form.code_bias, cb,
                                form.group_elems, form.has_offset ? 1 : 0);
                break;
            default:
                std::fprintf(stderr, "s_gemv: unsupported code_bits %d\n", form.code_bits);
                std::exit(1);
        }
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "s_gemv: %s\n", e.what());
        std::exit(1);
    }
    cudaDeviceSynchronize();
}

static void s_gemv_split_impl(const uint16_t* x, const uint8_t* codes, const float* scales,
                              const float* offset, float* y, int64_t n_in, int64_t n_out,
                              const SForm& form, int threads_per_row, void* stream, bool sync) {
    if (n_in <= 0 || n_out <= 0) return;
    if (threads_per_row < 1 || (threads_per_row & (threads_per_row - 1)) != 0 ||
        threads_per_row > 1024) {
        std::fprintf(stderr,
                     "s_gemv_split: threads_per_row must be a power of two in 1..1024, got %d\n",
                     threads_per_row);
        std::exit(1);
    }
    const int cb = (int)form.codebook;
    if (form.group_elems <= 0 || (form.group_elems & (form.group_elems - 1)) != 0) {
        std::fprintf(stderr, "s_gemv_split: group_elems must be a power of two, got %d\n",
                     form.group_elems);
        std::exit(1);
    }
    if (form.group_elems % 4 != 0) {
        std::fprintf(stderr, "s_gemv_split: group_elems %d is not a multiple of 4\n",
                     form.group_elems);
        std::exit(1);
    }
    int group_shift = 0;
    while ((1 << group_shift) < form.group_elems) ++group_shift;
    auto& q = strata::sycl_compat::q_for(stream);
    try {
        switch (form.code_bits) {
            case 2:
                launch_split<2>(q, x, codes, scales, offset, y, n_in, n_out, form.code_bias, cb,
                                form.group_elems, group_shift, form.has_offset ? 1 : 0,
                                threads_per_row);
                break;
            case 4:
                launch_split<4>(q, x, codes, scales, offset, y, n_in, n_out, form.code_bias, cb,
                                form.group_elems, group_shift, form.has_offset ? 1 : 0,
                                threads_per_row);
                break;
            case 8:
                launch_split<8>(q, x, codes, scales, offset, y, n_in, n_out, form.code_bias, cb,
                                form.group_elems, group_shift, form.has_offset ? 1 : 0,
                                threads_per_row);
                break;
            default:
                std::fprintf(stderr, "s_gemv_split: unsupported code_bits %d\n", form.code_bits);
                std::exit(1);
        }
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "s_gemv_split: %s\n", e.what());
        std::exit(1);
    }
    if (sync) cudaDeviceSynchronize();
}

void s_gemv_split(const uint16_t* x, const uint8_t* codes, const float* scales, const float* offset,
                  float* y, int64_t n_in, int64_t n_out, const SForm& form, int threads_per_row) {
    s_gemv_split_impl(x, codes, scales, offset, y, n_in, n_out, form, threads_per_row, nullptr,
                      true);
}

void s_gemv_split_async(const uint16_t* x, const uint8_t* codes, const float* scales,
                        const float* offset, float* y, int64_t n_in, int64_t n_out,
                        const SForm& form, int threads_per_row, void* stream) {
    s_gemv_split_impl(x, codes, scales, offset, y, n_in, n_out, form, threads_per_row, stream,
                      false);
}

}  // namespace strata::kernels
