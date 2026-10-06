// src/kernels/sycl/kv_q8.cpp — SYCL port of src/kernels/cuda/kv_q8.cu
// (Q8 KV cache append with host-pool mirror, and FP16 scratch gather).
// The 64-value amax reduction is two 32-lane chunks combined through local
// memory in the fixed order warp_max[0], warp_max[1]; quantization rounds
// against the STORED f16 scale exactly as the CUDA kernel does.
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q8.hpp"

#include <cuda_runtime.h>  // sycl_compat shim

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

using sycl::local_accessor;
using sycl::nd_item;

inline sycl::queue& Q(void* stream) { return strata::sycl_compat::q_for(stream); }

void check(void* stream, const char* what) {
    if (stream != nullptr) return;
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "kv_q8: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

void validate(const QsaShapes& s, const char* what) {
    if (s.head_dim % KV_Q8_GROUP != 0 || s.n_head_kv <= 0 || s.page_size <= 0) {
        std::fprintf(stderr, "kv_q8: %s: head_dim %lld must be a multiple of %d\n", what, (long long) s.head_dim,
                     KV_Q8_GROUP);
        std::exit(1);
    }
}

}  // namespace

void kv_append_q8_step(int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale, const int32_t* page_table,
                       const int32_t* step, const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                       const KvHostPools* host) {
    validate(s, "kv_append_q8");
    const KvHostPools pools = host ? *host : KvHostPools{};
    const int kv_heads = (int) s.n_head_kv;
    const int head_dim = (int) s.head_dim;
    const int page_size = (int) s.page_size;
    const int groups = head_dim / KV_Q8_GROUP;
    const size_t gy = (size_t) groups;
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<float, 1> warp_max(sycl::range<1>(2), hnd);
        local_accessor<float, 1> red(sycl::range<1>(KV_Q8_GROUP), hnd);
        // CUDA grid (n_head_kv, groups, 2) x 64 threads -> flat groups
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) kv_heads * gy * 2 * KV_Q8_GROUP),
                                           sycl::range<1>(KV_Q8_GROUP)),
                         [=](nd_item<1> it) {
                             const size_t g = it.get_group(0);
                             const int h = (int) (g / (gy * 2));
                             const int gg = (int) ((g / 2) % gy);
                             const bool is_v = (g % 2) == 1;
                             const int t = (int) it.get_local_id(0);
                             const long long pos = (long long) step[kStepPos];
                             const float x = (is_v ? vcur : kcur)[(size_t) h * head_dim + gg * KV_Q8_GROUP + t];
                             float a = sycl::fabs(x);
                             red[t] = a;
                             it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
                             for (int o = 16; o > 0; o >>= 1) {
                                 const float other = red[t ^ o];
                                 it.barrier(sycl::access::fence_space::local_space);
                                 a = sycl::fmax(a, other);
                                 red[t] = a;
                                 it.barrier(sycl::access::fence_space::local_space);
                             }
                             if ((t & 31) == 0) warp_max[t >> 5] = a;
                             it.barrier(sycl::access::fence_space::local_space);
                             const float amax = sycl::fmax(warp_max[0], warp_max[1]);
                             const uint16_t sbits = f16_from_f32(amax / 127.0f);
                             const float sf = f32_from_f16(sbits);
                             int q = 0;
                             if (sf > 0.0f) {
                                 q = (int) sycl::rint(x / sf);
                                 q = q < -127 ? -127 : (q > 127 ? 127 : q);
                             }
                             const long long page = (long long) page_table[pos / page_size];
                             if (page >= 0) {
                                 const long long row = (page * kv_heads + h) * page_size + (pos % page_size);
                                 (is_v ? v_q : k_q)[row * head_dim + gg * KV_Q8_GROUP + t] = (int8_t) q;
                                 if (t == 0) (is_v ? v_scale : k_scale)[row * groups + gg] = sbits;
                             }
                             if (pools.k_q != nullptr) {
                                 const long long row =
                                     ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size);
                                 (is_v ? pools.v_q : pools.k_q)[row * head_dim + gg * KV_Q8_GROUP + t] = (int8_t) q;
                                 if (t == 0) (is_v ? pools.v_scale : pools.k_scale)[row * groups + gg] = sbits;
                             }
                         });
    });
    check(stream, "kv_append_q8 launch");
}

void kv_gather_q8_step(const int8_t* k_q, const int8_t* v_q, const uint16_t* k_scale, const uint16_t* v_scale,
                       const int32_t* page_table, const int32_t* ids, const int32_t* step, int64_t max_ids,
                       const QsaShapes& s, uint16_t* k_scratch, uint16_t* v_scratch, void* stream) {
    validate(s, "kv_gather_q8");
    if (max_ids <= 0) return;
    const int kv_heads = (int) s.n_head_kv;
    const int head_dim = (int) s.head_dim;
    const int page_size = (int) s.page_size;
    const int per = head_dim / 4;
    // the width lives in DEVICE memory (step[kStepWidth]): size the launch with
    // max_ids and cull inside, exactly as the CUDA kernel does
    Q(stream).parallel_for((size_t) max_ids * kv_heads * (long long) per, [=](size_t i) {
        const long long n_ids = (long long) step[kStepWidth];
        if ((long long) i >= n_ids * kv_heads * (long long) per) return;
        const long long id = (long long) i / (kv_heads * (long long) per);
        const int rem = (int) (i % (kv_heads * (long long) per));
        const int h = rem / per, q4 = rem - h * per;
        const int cell = ids[id];
        const long long page = (long long) page_table[cell / page_size];
        const long long row = (page * kv_heads + h) * page_size + (cell % page_size);
        const int d = q4 * 4;
        const int groups = head_dim / KV_Q8_GROUP;
        const float ks = f32_from_f16(k_scale[row * groups + d / KV_Q8_GROUP]);
        const float vs = f32_from_f16(v_scale[row * groups + d / KV_Q8_GROUP]);
        const int8_t* krow = k_q + row * head_dim + d;
        const int8_t* vrow = v_q + row * head_dim + d;
        const long long dst = (id * kv_heads + h) * (long long) per + q4;
        uint16_t ko[4], vo[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            ko[e] = f16_from_f32((float) krow[e] * ks);
            vo[e] = f16_from_f32((float) vrow[e] * vs);
        }
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            k_scratch[dst * 4 + e] = ko[e];
            v_scratch[dst * 4 + e] = vo[e];
        }
    });
    check(stream, "kv_gather_q8 launch");
}

}  // namespace strata::kernels
