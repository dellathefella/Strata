// src/kernels/sycl/router_top10.cpp — SYCL port of src/kernels/cuda/router_top10.cu.
//
// The CUDA kernel's NUMBERS are order-independent by construction (its own
// comments are the proof): the max is an exact fmaxf tree, the selection
// reduces under a total order (value desc, index asc as tiebreak), the double
// sum is serial-ascending on one thread, and renorm is serial on one thread.
// The 32-lane warp machinery is only a performance structure, so this port
// replaces __shfl_down butterflies with work-group local-memory trees and
// keeps every value-producing expression verbatim — bit-identical outputs
// without depending on a specific sub_group size (B60: 16 or 32).
//
// The HIP fast-kernel variants stay HIP-only: router_top10_variant returns
// false here, exactly as in a CUDA build.
#include "strata/kernels/router_top10.hpp"

#include <cuda_runtime.h>  // sycl_compat shim

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int RT_MAX_THREADS = 512;

// (value, index) under the CUDA kernel's total order: higher value wins;
// on equal values the LOWER index wins.
struct Cand {
    float v;
    int i;
};

inline Cand better(Cand a, Cand b) {
    if (a.v > b.v) return a;
    if (b.v > a.v) return b;
    return a.i <= b.i ? a : b;
}

}  // namespace

bool router_top10_variant(const float*, int, int, int, int*, float*, void*, int) {
    return false;  // HIP-only fast kernels; CUDA and SYCL builds take the portable path
}

void router_top10(const float* logits, int n_tokens, int n_expert, int k, int* ids, float* weights,
                  void* stream) {
    if (n_tokens <= 0 || n_expert <= 0 || k <= 0) return;
    if (k > 64) {
        std::fprintf(stderr, "router_top10: k %d exceeds the kernel's 64\n", k);
        std::exit(1);
    }
    if (n_expert > RT_MAX_THREADS * 64) {
        std::fprintf(stderr, "router_top10: n_expert %d is past the kernel's %d\n", n_expert,
                     RT_MAX_THREADS * 64);
        std::exit(1);
    }
    int threads = n_expert < RT_MAX_THREADS ? n_expert : RT_MAX_THREADS;
    threads = (threads + 31) & ~31;
    const int nt = threads;

    auto& q = strata::sycl_compat::q_for(stream);
    try {
        q.submit([&](sycl::handler& h) {
            // the same one-allocation layout the CUDA kernel carves out of extern __shared__:
            // taken mask (16-aligned) | s_ex doubles | s_p floats, plus reduction scratch
            sycl::local_accessor<unsigned char, 1> s_taken(sycl::range<1>((size_t)n_expert), h);
            sycl::local_accessor<double, 1> s_ex(sycl::range<1>((size_t)n_expert), h);
            sycl::local_accessor<float, 1> s_p(sycl::range<1>((size_t)n_expert), h);
            sycl::local_accessor<float, 1> s_red(sycl::range<1>((size_t)nt), h);
            sycl::local_accessor<int, 1> s_rid(sycl::range<1>((size_t)nt), h);
            sycl::local_accessor<double, 1> s_sum(sycl::range<1>(1), h);

            h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t)n_tokens * nt),
                                             sycl::range<1>((size_t)nt)),
                           [=](sycl::nd_item<1> it) {
                const int t = (int)it.get_group(0);
                if (t >= n_tokens) return;
                const int tid = (int)it.get_local_id(0);
                const float* l = logits + (size_t)t * n_expert;

                // ---- max over ALL experts: exact fmaxf tree (order-independent)
                float mx = -INFINITY;
                for (int e = tid; e < n_expert; e += nt) mx = sycl::fmax(mx, l[e]);
                s_red[tid] = mx;
                for (int step = nt / 2; step > 0; step >>= 1) {
                    it.barrier();
                    if (tid < step) s_red[tid] = sycl::fmax(s_red[tid], s_red[tid + step]);
                }
                it.barrier();
                mx = s_red[0];
                // ---- the 512 exponentials, once each, in parallel (same expression)
                for (int e = tid; e < n_expert; e += nt)
                    s_ex[e] = sycl::exp((double)l[e] - (double)mx);
                it.barrier();

                // ---- the sum, ascending, on one thread (order preserved verbatim)
                if (tid == 0) {
                    double sum = 0.0;
                    for (int e = 0; e < n_expert; ++e) sum += s_ex[e];
                    s_sum[0] = sum;
                }
                it.barrier();
                const float inv = (float)(1.0 / s_sum[0]);

                // ---- p[] once, the same expression as the CUDA kernel
                for (int e = tid; e < n_expert; e += nt) s_p[e] = (float)(s_ex[e] * inv);
                for (int e = tid; e < n_expert; e += nt) s_taken[e] = 0;
                it.barrier();

                // ---- selection in k passes: strided scan + total-order tree reduce
                for (int i = 0; i < k; ++i) {
                    Cand best{-INFINITY, n_expert};  // the sentinel that loses to every real index
                    for (int e = tid; e < n_expert; e += nt) {
                        if (s_taken[e]) continue;
                        const float pe = s_p[e];
                        if (pe > best.v) { best.v = pe; best.i = e; }
                    }
                    s_red[tid] = best.v;
                    s_rid[tid] = best.i;
                    for (int step = nt / 2; step > 0; step >>= 1) {
                        it.barrier();
                        if (tid < step) {
                            const Cand a{s_red[tid], s_rid[tid]};
                            const Cand b{s_red[tid + step], s_rid[tid + step]};
                            const Cand w = better(a, b);
                            s_red[tid] = w.v;
                            s_rid[tid] = w.i;
                        }
                    }
                    it.barrier();
                    if (tid == 0 && s_rid[0] < n_expert) {
                        ids[(size_t)t * k + i] = s_rid[0];
                        weights[(size_t)t * k + i] = s_red[0];
                        s_taken[s_rid[0]] = 1;
                    }
                    it.barrier();
                }

                // ---- renormalise with ggml's lower clamp, serial on thread 0, order preserved
                if (tid == 0) {
                    double s = 0.0;
                    for (int i = 0; i < k; ++i) s += (double)weights[(size_t)t * k + i];
                    const double sc = sycl::fmax(s, 6.103515625e-05);  // 2**-14
                    for (int i = 0; i < k; ++i)
                        weights[(size_t)t * k + i] = (float)((double)weights[(size_t)t * k + i] / sc);
                }
            });
        });
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "router_top10 launch: %s\n", e.what());
        std::exit(1);
    }
    if (stream == nullptr) {
        const cudaError_t s = cudaDeviceSynchronize();
        if (s != cudaSuccess) {
            std::fprintf(stderr, "router_top10: %s\n", cudaGetErrorString(s));
            std::exit(1);
        }
    }
}

}  // namespace strata::kernels
