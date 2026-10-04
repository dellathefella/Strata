// src/kernels/sycl/elementwise.cpp — SYCL port of src/kernels/cuda/elementwise.cu
// (P2.S5's glue kernels + the doorbell primitives).
//
// Port notes:
//   * __fmul_rn/__fadd_rn -> plain ops (library compiled -ffp-contract=off)
//   * warp-shuffle rms_norm reduction -> per-32-lane-chunk LOCAL-memory tree
//     with the same 16,8,4,2,1 pairing: bit-identical order, no sub_group
//     size assumption, and the row guard keeps participating in barriers
//     (a returned work-item would hang the group's barrier — the QSA-bug
//     comment in the CUDA source is about the guard, this is about its SYCL
//     hazard)
//   * __threadfence_system -> sycl::atomic_fence(system scope);
//     volatile RMW on mapped memory -> atomic_ref fetch_add/acquire loads
//     (milestone-1 measured these paths on B60: 335 us ring, 1.0 ms visible)
//   * volatile float4 copies -> scalar volatile loads into sycl::float4
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/dp4a.hpp"

#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cuda_runtime.h>  // sycl_compat shim

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int THREADS = 256;

using sys_ref_u32 =
    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::system,
                     sycl::access::address_space::global_space>;

inline float softplus_dev(float x) {
    return x > 20.0f ? x : sycl::log1p(sycl::exp(x));
}

inline size_t grid_items(int64_t n) { return (size_t)((n + THREADS - 1) / THREADS) * THREADS; }

void sync_if_needed(void* stream, const char* what) {
    if (stream != nullptr) return;
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

bool check_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s launch: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}

}  // namespace

void embedding_gather(const uint8_t* codes, const float* scales, const float* offsets, int64_t n,
                      int code_bits, int code_bias, int group_elems, float* out, void* stream) {
    if (n <= 0) return;
    auto& q = strata::sycl_compat::q_for(stream);
    try {
        q.parallel_for(sycl::range<1>((size_t)n), [=](sycl::id<1> iid) {
            const int64_t i = iid[0];
            const int per_byte = 8 / code_bits;
            const unsigned mask = (1u << code_bits) - 1u;
            const int code = (codes[i / per_byte] >> ((i % per_byte) * code_bits)) & mask;
            const int64_t group = i / group_elems;
            const float product = (float)(code + code_bias) * scales[group];
            out[i] = product + (offsets ? offsets[group] : 0.0f);
        });
    } catch (const sycl::exception&) {
        strata::sycl_compat::last_error() = cudaErrorUnknown;
    }
    check_launch("embedding_gather");
}

void gdn_gate(const float* alpha, const float* dt, const float* ssm_a, float* gate, int64_t n_tokens,
              int64_t h_v, void* stream) {
    if (n_tokens <= 0 || h_v <= 0) return;
    const int64_t n = n_tokens * h_v;
    auto& q = strata::sycl_compat::q_for(stream);
    try {
        q.parallel_for(sycl::range<1>((size_t)n), [=](sycl::id<1> iid) {
            const int64_t i = iid[0];
            const int64_t h = i % h_v;
            gate[i] = softplus_dev(alpha[i] + dt[h]) * ssm_a[h];
        });
    } catch (const sycl::exception&) {
        strata::sycl_compat::last_error() = cudaErrorUnknown;
    }
    check_launch("gdn_gate");
    sync_if_needed(stream, "gdn_gate");
}

void scale_inplace(float* x, int64_t n, float s, void* stream) {
    if (n <= 0) return;
    auto& q = strata::sycl_compat::q_for(stream);
    try {
        q.parallel_for(sycl::range<1>((size_t)n), [=](sycl::id<1> iid) { x[(int64_t)iid[0]] *= s; });
    } catch (const sycl::exception&) {
        strata::sycl_compat::last_error() = cudaErrorUnknown;
    }
    check_launch("scale_inplace");
    sync_if_needed(stream, "scale_inplace");
}

void add_inplace(float* dst, const float* src, int64_t n, void* stream) {
    if (n <= 0) return;
    auto& q = strata::sycl_compat::q_for(stream);
    try {
        q.parallel_for(sycl::range<1>((size_t)n),
                       [=](sycl::id<1> iid) { dst[(int64_t)iid[0]] += src[(int64_t)iid[0]]; });
    } catch (const sycl::exception&) {
        strata::sycl_compat::last_error() = cudaErrorUnknown;
    }
    check_launch("add_inplace");
    sync_if_needed(stream, "add_inplace");
}

void f32_to_f16_bulk(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    auto& q = strata::sycl_compat::q_for(stream);
    try {
        q.parallel_for(sycl::range<1>((size_t)n),
                       [=](sycl::id<1> iid) { y[(int64_t)iid[0]] = f16_from_f32(x[(int64_t)iid[0]]); });
    } catch (const sycl::exception&) {
        strata::sycl_compat::last_error() = cudaErrorUnknown;
    }
    check_launch("f32_to_f16_bulk");
    sync_if_needed(stream, "f32_to_f16_bulk");
}

void f32_to_bf16_bulk(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    auto& q = strata::sycl_compat::q_for(stream);
    try {
        q.parallel_for(sycl::range<1>((size_t)n),
                       [=](sycl::id<1> iid) { y[(int64_t)iid[0]] = bf16_from_f32(x[(int64_t)iid[0]]); });
    } catch (const sycl::exception&) {
        strata::sycl_compat::last_error() = cudaErrorUnknown;
    }
    check_launch("f32_to_bf16_bulk");
    sync_if_needed(stream, "f32_to_bf16_bulk");
}

void silu_inplace(float* x, int64_t n, void* stream) {
    if (n <= 0) return;
    auto& q = strata::sycl_compat::q_for(stream);
    try {
        q.parallel_for(sycl::range<1>((size_t)n), [=](sycl::id<1> iid) {
            const int64_t i = iid[0];
            // DOUBLE then cast, matching ref/gdn.py's numpy (the oracle)
            const double v = (double)x[i];
            x[i] = (float)(v / (1.0 + sycl::exp(-v)));
        });
    } catch (const sycl::exception&) {
        strata::sycl_compat::last_error() = cudaErrorUnknown;
    }
    check_launch("silu_inplace");
    sync_if_needed(stream, "silu_inplace");
}

void doorbell_ring(uint32_t* d_seq, void* stream) {
    if (d_seq == nullptr) return;
    auto& q = strata::sycl_compat::q_for(stream);
    try {
        q.single_task([=]() {
            sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
            sys_ref_u32 seq(*reinterpret_cast<uint32_t*>(d_seq));
            seq.fetch_add(1u);
        });
    } catch (const sycl::exception&) {
        strata::sycl_compat::last_error() = cudaErrorUnknown;
    }
    check_launch("doorbell_ring");
    sync_if_needed(stream, "doorbell_ring");
}

void doorbell_wait(const uint32_t* d_flag, const uint32_t* d_seq, void* stream) {
    if (d_flag == nullptr || d_seq == nullptr) return;
    auto& q = strata::sycl_compat::q_for(stream);
    try {
        q.single_task([=]() {
            sys_ref_u32 seq(*const_cast<uint32_t*>(d_seq));
            sys_ref_u32 flag(*const_cast<uint32_t*>(d_flag));
            const uint32_t want = seq.load();
            while (flag.load() != want) strata_spin_pause();
            sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
        });
    } catch (const sycl::exception&) {
        strata::sycl_compat::last_error() = cudaErrorUnknown;
    }
    check_launch("doorbell_wait");
}

void copy_from_mapped(float* dst, const float* src, int64_t n, void* stream) {
    if (n <= 0) return;
    if ((n & 3) != 0 || ((uintptr_t)dst & 15) != 0 || ((uintptr_t)src & 15) != 0) {
        std::fprintf(stderr,
                     "copy_from_mapped: n must be a multiple of 4 and both pointers 16-byte "
                     "aligned\n");
        std::exit(1);
    }
    const int64_t n4 = n / 4;
    const int64_t blocks = (n4 + 255) / 256 < 64 ? (n4 + 255) / 256 : 64;
    auto& q = strata::sycl_compat::q_for(stream);
    try {
        q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t)blocks * 256),
                                         sycl::range<1>(256)),
                       [=](sycl::nd_item<1> it) {
                           const volatile float* s = src;
                           sycl::float4* d = reinterpret_cast<sycl::float4*>(dst);
                           for (int64_t i = (int64_t)it.get_global_id(0); i < n4;
                                i += blocks * 256) {
                               d[i] = sycl::float4(s[i * 4 + 0], s[i * 4 + 1], s[i * 4 + 2],
                                                   s[i * 4 + 3]);
                           }
                       });
    } catch (const sycl::exception&) {
        strata::sycl_compat::last_error() = cudaErrorUnknown;
    }
    check_launch("copy_from_mapped");
}

void copy_rows_from_mapped(float* dst, const float* src, int64_t rows, int64_t width,
                           const int32_t* hit_rows, const int32_t* count, void* stream) {
    if (rows <= 0) return;
    if ((width & 3) != 0 || ((uintptr_t)dst & 15) != 0 || ((uintptr_t)src & 15) != 0) {
        std::fprintf(stderr,
                     "copy_rows_from_mapped: width must be a multiple of 4 and both pointers "
                     "16-byte aligned\n");
        std::exit(1);
    }
    const int64_t row4 = width / 4;
    auto& q = strata::sycl_compat::q_for(stream);
    try {
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<int, 1> hit(sycl::range<1>(1), h);
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t)rows * 128),
                                             sycl::range<1>(128)),
                           [=](sycl::nd_item<1> it) {
                               const int row = (int)it.get_group(0);
                               const int tid = (int)it.get_local_id(0);
                               if (tid == 0) {
                                   int h = 0;
                                   const int c = *count;
                                   for (int i = 0; i < c; ++i) h |= hit_rows[i] == row;
                                   hit[0] = h;
                               }
                               it.barrier();
                               sycl::float4* d = reinterpret_cast<sycl::float4*>(dst) + (int64_t)row * row4;
                               if (hit[0]) {
                                   for (int64_t i = tid; i < row4; i += 128)
                                       d[i] = sycl::float4(0.0f, 0.0f, 0.0f, 0.0f);
                               } else {
                                   const volatile float* s = src + (int64_t)row * row4 * 4;
                                   for (int64_t i = tid; i < row4; i += 128)
                                       d[i] = sycl::float4(s[i * 4 + 0], s[i * 4 + 1], s[i * 4 + 2],
                                                           s[i * 4 + 3]);
                               }
                           });
        });
    } catch (const sycl::exception&) {
        strata::sycl_compat::last_error() = cudaErrorUnknown;
    }
}

void doorbell_publish(const float* x, const int32_t* ids, const float* weights, int64_t n, int64_t k,
                      float* x_out, int32_t* ids_out, float* weights_out, uint32_t* d_seq,
                      void* stream) {
    if (k > 1024) {
        std::fprintf(stderr, "doorbell_publish: k too large\n");
        std::exit(1);
    }
    auto& q = strata::sycl_compat::q_for(stream);
    const int nn = (int)n, kk = (int)k;
    try {
        q.parallel_for(sycl::nd_range<1>(sycl::range<1>(1024), sycl::range<1>(1024)),
                       [=](sycl::nd_item<1> it) {
                           const int tid = (int)it.get_local_id(0);
                           for (int i = tid; i < nn; i += 1024) x_out[i] = x[i];
                           if (tid < kk) {
                               ids_out[tid] = ids[tid];
                               weights_out[tid] = weights[tid];
                           }
                           it.barrier();
                           sycl::atomic_fence(sycl::memory_order::release,
                                                     sycl::memory_scope::system);
                           if (tid == 0) {
                               sys_ref_u32 seq(*reinterpret_cast<uint32_t*>(d_seq));
                               seq.fetch_add(1u);
                           }
                       });
    } catch (const sycl::exception&) {
        strata::sycl_compat::last_error() = cudaErrorUnknown;
    }
    check_launch("doorbell_publish");
}

void copy_i32_from_mapped(int32_t* dst, const int32_t* src, int64_t n, void* stream) {
    if (n <= 0) return;
    auto& q = strata::sycl_compat::q_for(stream);
    const int nn = (int)n;
    try {
        q.parallel_for(sycl::nd_range<1>(sycl::range<1>(128), sycl::range<1>(128)),
                       [=](sycl::nd_item<1> it) {
                           const volatile int32_t* s = src;
                           for (int i = (int)it.get_local_id(0); i < nn; i += 128) dst[i] = s[i];
                       });
    } catch (const sycl::exception&) {
        strata::sycl_compat::last_error() = cudaErrorUnknown;
    }
    check_launch("copy_i32_from_mapped");
}

void rms_norm_weighted(float* x, const float* w, int64_t rows, int64_t cols, float eps,
                       void* stream) {
    if (rows <= 0 || cols <= 0) return;
    // 4 rows per 128-thread work-group (CUDA: 4 warps per block)
    const size_t grid = (size_t)((rows + 3) / 4);
    auto& q = strata::sycl_compat::q_for(stream);
    try {
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> red(sycl::range<1>(128), h);
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>(grid * 128), sycl::range<1>(128)),
                           [=](sycl::nd_item<1> it) {
                               const int tid = (int)it.get_local_id(0);
                               const int chunk = tid / 32;  // the CUDA "warp" slot
                               const int lane = tid % 32;
                               const int64_t row = (int64_t)it.get_group(0) * 4 + chunk;
                               // the row guard (the QSA bug) — but guarded rows still reach every barrier
                               const bool active = row < rows;
                               float* r = x + row * cols;
                               float acc = 0.0f;
                               if (active)
                                   for (int64_t c = lane; c < cols; c += 32) acc += r[c] * r[c];
                               // per-chunk tree, the CUDA butterfly's exact pairing: 16,8,4,2,1
                               red[tid] = acc;
                               for (int step = 16; step > 0; step >>= 1) {
                                   it.barrier();
                                   if (lane < step)
                                       red[chunk * 32 + lane] += red[chunk * 32 + lane + step];
                               }
                               // the MEAN's reciprocal from lane 0, broadcast (one value, no divergence)
                               float inv = 0.0f;
                               if (lane == 0) {
                                   inv = sycl::rsqrt(red[chunk * 32] / (float)cols + eps);
                                   red[chunk * 32] = inv;
                               }
                               it.barrier();
                               inv = red[chunk * 32];
                               if (active)
                                   for (int64_t c = lane; c < cols; c += 32)
                                       r[c] = (w ? r[c] * w[c] : r[c]) * inv;
                           });
        });
    } catch (const sycl::exception&) {
        strata::sycl_compat::last_error() = cudaErrorUnknown;
    }
    check_launch("rms_norm_weighted");
    sync_if_needed(stream, "rms_norm_weighted");
}

}  // namespace strata::kernels
