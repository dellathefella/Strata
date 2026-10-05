// src/kernels/sycl/sampler.cpp — SYCL port of the greedy path of
// src/kernels/cuda/sampler.cu (sampler_greedy_kernel + sample_tokens dispatch).
// The penalty bitmap becomes local memory with atomic_ref fetch_or; the two
// shfl_down pair-trees (value,index) become local-memory down-trees with the
// identical merge rule (larger value wins, ties to the smaller index); the
// cross-warp merge runs redundantly in every warp (SYCL barriers are
// work-group-wide) and warp 0 lane 0 writes the token. The temperature-bearing
// sampled paths are not ported yet and throw.
#include "strata/kernels/sampler.hpp"

#include <cuda_runtime.h>  // sycl_compat shim

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {
constexpr int GTHREADS = 1024;

using sycl::local_accessor;
using sycl::nd_item;
using sycl::atomic_ref;
using sycl::memory_order;
using sycl::memory_scope;

inline sycl::queue& Q(void* stream) { return strata::sycl_compat::q_for(stream); }

inline float neg_inf() { return sycl::bit_cast<float>(0xff800000u); }

inline int history_count(const int* h, int n, int v) {
    int c = 0;
    for (int i = 0; i < n; ++i)
        if (h[i] == v) ++c;
    return c;
}

inline float apply_penalties(float logit, int count, const SamplerParams& p) {
    if (count <= 0) return logit;
    if (logit <= 0.0f)
        logit *= p.penalty_repeat;
    else
        logit /= p.penalty_repeat;
    logit -= (float) count * p.penalty_freq + (count > 0 ? 1.0f : 0.0f) * p.penalty_present;
    return logit;
}

}  // namespace

bool sample_greedy_cluster(const float*, int, int, int*, void*) { return false; }  // no clusters on SYCL

void sample_tokens(const float* logits, int n_tokens, int n_vocab, const int* history, int history_len,
                   const SamplerParams& p, int* out, void* stream) {
    if (n_tokens <= 0 || n_vocab <= 0) return;
    if (p.penalty_last_n > 0 && (history == nullptr || history_len <= 0)) {
        std::fprintf(stderr, "sample_tokens: penalty_last_n %d needs a history (got %p, len %d)\n", p.penalty_last_n,
                     (const void*) history, history_len);
        std::exit(1);
    }
    if (!(p.greedy || p.temperature <= 0.0f)) {
        throw std::runtime_error("SYCL backend: the sampled (temperature > 0) sampler path is not ported yet");
    }
    const bool use_bits = history != nullptr && history_len > 0 && p.penalty_last_n > 0;
    const int bits_words = (n_vocab + 31) / 32;
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<unsigned, 1> bits(sycl::range<1>((size_t) bits_words), hnd);
        local_accessor<float, 1> redV(sycl::range<1>(GTHREADS), hnd);
        local_accessor<int, 1> redI(sycl::range<1>(GTHREADS), hnd);
        local_accessor<float, 1> sv(sycl::range<1>(32), hnd);
        local_accessor<int, 1> si(sycl::range<1>(32), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) n_tokens * GTHREADS), sycl::range<1>(GTHREADS)),
                         [=](nd_item<1> it) {
                             const int t = (int) it.get_group(0);
                             const int tx = (int) it.get_local_id(0);
                             const float* l = logits + (size_t) t * n_vocab;
                             const int* hrow = history ? history + (size_t) t * history_len : nullptr;
                             int hlen = 0;
                             if (hrow) {
                                 hlen = p.penalty_last_n < history_len ? p.penalty_last_n : history_len;
                                 if (hlen < 0) hlen = 0;
                                 hrow += history_len - hlen;
                             }
                             if (use_bits) {
                                 for (int w = tx; w < bits_words; w += GTHREADS) bits[w] = 0u;
                                 it.barrier(sycl::access::fence_space::local_space);
                                 for (int i = tx; i < hlen; i += GTHREADS)
                                     if (hrow[i] >= 0 && hrow[i] < n_vocab) {
                                         atomic_ref<unsigned, memory_order::relaxed, memory_scope::work_group> b(
                                             bits[hrow[i] >> 5]);
                                         b.fetch_or(1u << (hrow[i] & 31));
                                     }
                                 it.barrier(sycl::access::fence_space::local_space);
                             }
                             auto hit_count = [&](int v) -> int {
                                 if (!use_bits || !(bits[v >> 5] & (1u << (v & 31)))) return 0;
                                 return history_count(hrow, hlen, v);
                             };
                             float bv = neg_inf();
                             int best = n_vocab;
                             for (int v = tx; v < n_vocab; v += GTHREADS) {
                                 const float s = apply_penalties(l[v], hit_count(v), p);
                                 if (s > bv) {
                                     bv = s;
                                     best = v;
                                 }
                             }
                             const int lane = tx & 31;
#pragma unroll
                             for (int off = 16; off > 0; off >>= 1) {
                                 redV[tx] = bv;
                                 redI[tx] = best;
                                 it.barrier(sycl::access::fence_space::local_space);
                                 const float ov = lane + off < 32 ? redV[tx + off] : neg_inf();
                                 const int oi = lane + off < 32 ? redI[tx + off] : n_vocab;
                                 it.barrier(sycl::access::fence_space::local_space);
                                 if (ov > bv || (ov == bv && oi < best)) {
                                     bv = ov;
                                     best = oi;
                                 }
                             }
                             const int warp = tx >> 5;
                             if (lane == 0) {
                                 sv[warp] = bv;
                                 si[warp] = best;
                             }
                             it.barrier(sycl::access::fence_space::local_space);
                             // the warp-0 merge, run redundantly by every warp
                             const int nw = GTHREADS / 32;
                             float wv = lane < nw ? sv[lane] : neg_inf();
                             int wi = lane < nw ? si[lane] : n_vocab;
#pragma unroll
                             for (int off = 16; off > 0; off >>= 1) {
                                 redV[tx] = wv;
                                 redI[tx] = wi;
                                 it.barrier(sycl::access::fence_space::local_space);
                                 const float ov = lane + off < 32 ? redV[tx + off] : neg_inf();
                                 const int oi = lane + off < 32 ? redI[tx + off] : n_vocab;
                                 it.barrier(sycl::access::fence_space::local_space);
                                 if (ov > wv || (ov == wv && oi < wi)) {
                                     wv = ov;
                                     wi = oi;
                                 }
                             }
                             if (warp == 0 && lane == 0) out[t] = (wi < n_vocab) ? wi : 0;
                         });
    });
    if (stream == nullptr) {
        const cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            std::fprintf(stderr, "sample_tokens: %s\n", cudaGetErrorString(e));
            std::exit(1);
        }
    }
}

}  // namespace strata::kernels
