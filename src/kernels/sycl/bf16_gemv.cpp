// src/kernels/sycl/bf16_gemv.cpp — SYCL port of src/kernels/cuda/bf16_gemv.cu.
// The router-logits GEMV over bf16. naive/warp/split dispatch thresholds and
// accumulation order preserved; the warp butterfly (order-DEPENDENT adds) is
// reproduced with per-32-lane-chunk local-memory trees — the same pairing, so
// the parity tolerances see the same numbers. No sub_group size assumption.
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <cuda_runtime.h>  // sycl_compat shim

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int THREADS = 256;

inline void finish(void* stream, const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s launch: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
    if (stream != nullptr) return;
    const cudaError_t s = cudaDeviceSynchronize();
    if (s != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(s));
        std::exit(1);
    }
}

// one work-item per row, contiguous walk (the CUDA "naive" kernel)
void launch_naive(sycl::queue& q, const uint16_t* x, const uint16_t* w, float* y, long long n_in,
                  long long n_out) {
    q.parallel_for(sycl::range<1>((size_t)n_out), [=](sycl::id<1> oid) {
        const long long o = oid[0];
        const uint16_t* row = w + o * n_in;
        float acc = 0.0f;
        for (long long i = 0; i < n_in; ++i)
            acc += f32_from_bf16(x[i]) * f32_from_bf16(row[i]);
        y[o] = acc;
    });
}

// 32-lane chunks per row, butterfly-order local tree (the CUDA "warp" kernel).
// grid: ceil(n_out/8) groups of 256 = 8 rows each, matching the CUDA launch.
void launch_warp(sycl::queue& q, const uint16_t* x, const uint16_t* w, float* y, long long n_in,
                 long long n_out) {
    const size_t grid = (size_t)((n_out + THREADS / 32 - 1) / (THREADS / 32));
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> red(sycl::range<1>(THREADS), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(grid * THREADS),
                                         sycl::range<1>(THREADS)),
                       [=](sycl::nd_item<1> it) {
                           const int tid = (int)it.get_local_id(0);
                           const int chunk = tid / 32, lane = tid % 32;
                           const long long o = (long long)it.get_group(0) * (THREADS / 32) + chunk;
                           float acc = 0.0f;
                           if (o < n_out) {
                               const uint16_t* row = w + o * n_in;
                               for (long long i = lane; i < n_in; i += 32)
                                   acc += f32_from_bf16(x[i]) * f32_from_bf16(row[i]);
                           }
                           red[tid] = acc;
                           for (int off = 16; off > 0; off >>= 1) {
                               it.barrier();
                               if (lane < off)
                                   red[chunk * 32 + lane] += red[chunk * 32 + lane + off];
                           }
                           if (lane == 0 && o < n_out) y[o] = red[chunk * 32];
                       });
    });
}

// one work-group per row, tpr lanes, shared tree (the CUDA "split" kernel)
void launch_split(sycl::queue& q, const uint16_t* x, const uint16_t* w, float* y, long long n_in,
                  long long n_out, int tpr) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> scratch(sycl::range<1>((size_t)tpr), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t)n_out * tpr),
                                         sycl::range<1>((size_t)tpr)),
                       [=](sycl::nd_item<1> it) {
                           const long long o = (long long)it.get_group(0);
                           const int t = (int)it.get_local_id(0);
                           float acc = 0.0f;
                           if (o < n_out) {
                               const uint16_t* row = w + o * n_in;
                               for (long long i = t; i < n_in; i += tpr)
                                   acc += f32_from_bf16(x[i]) * f32_from_bf16(row[i]);
                           }
                           scratch[t] = acc;
                           for (int off = tpr >> 1; off > 0; off >>= 1) {
                               it.barrier();
                               if (t < off) scratch[t] += scratch[t + off];
                           }
                           if (t == 0 && o < n_out) y[o] = scratch[0];
                       });
    });
}

}  // namespace

void bf16_gemv(const uint16_t* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out,
               void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    auto& q = strata::sycl_compat::q_for(stream);
    try {
        // the CUDA dispatch: warp-per-row (coalesced) once there are enough rows
        // to fill the machine; naive below the threshold
        if (n_out >= 64)
            launch_warp(q, x, w, y, n_in, n_out);
        else
            launch_naive(q, x, w, y, n_in, n_out);
    } catch (const sycl::exception&) {
        strata::sycl_compat::last_error() = cudaErrorUnknown;
    }
    finish(stream, n_out >= 64 ? "bf16_gemv(warp)" : "bf16_gemv");
}

void bf16_gemv_split(const uint16_t* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out,
                     int threads_per_row, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    auto& q = strata::sycl_compat::q_for(stream);
    try {
        if (threads_per_row == 32) {
            launch_warp(q, x, w, y, n_in, n_out);
        } else {
            if (threads_per_row <= 0 || (threads_per_row & (threads_per_row - 1)) != 0) {
                std::fprintf(stderr,
                             "bf16_gemv_split: threads_per_row %d must be a power of two (32 "
                             "selects the warp-per-row path)\n",
                             threads_per_row);
                std::exit(1);
            }
            launch_split(q, x, w, y, n_in, n_out, threads_per_row);
        }
    } catch (const sycl::exception&) {
        strata::sycl_compat::last_error() = cudaErrorUnknown;
    }
    finish(stream, threads_per_row == 32 ? "bf16_gemv_split(warp)" : "bf16_gemv_split");
}

}  // namespace strata::kernels
