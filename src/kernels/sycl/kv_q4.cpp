// src/kernels/sycl/kv_q4.cpp — SYCL port of src/kernels/cuda/kv_q4.cu
// (Hadamard-mixed Q4_0 KV cache: fwht256, append step/batch, gather).
// The fwht butterflies keep the CUDA pairing exactly: the low-5-bit stages
// exchange across the 32-lane chunk (local tree, sign rule (lane&h)==0 ?
// val+val2 : val2-val), the high-3-bit stages stay inside each lane's 8
// registers. q4_group's (amax,mval) pair butterfly and the shfl_down(qc,16)
// nibble pack use the same local-memory exchange pattern.
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_stream.hpp"

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
        std::fprintf(stderr, "kv_q4: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

void need_256(const QsaShapes& s, const char* what) {
    if (s.head_dim != 256) {
        std::fprintf(stderr, "%s: head_dim must be 256 (the Hadamard transform's size)\n", what);
        std::exit(1);
    }
}

// one work-group = 32 lanes x 4 rows; per-lane 8-register fwht over 256 floats
void fwht256_launch(const float* src, float* dst, int64_t n_rows, void* stream) {
    const int64_t rows_per_block = 4;
    const int64_t num_blocks = (n_rows + rows_per_block - 1) / rows_per_block;
    const float scale = 1.0f / 16.0f;
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<float, 1> redA(sycl::range<1>(128), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) num_blocks * 128), sycl::range<1>(128)),
                         [=](nd_item<1> it) {
                             const int lid = (int) it.get_local_id(0);
                             const int64_t r = (int64_t) it.get_group(0) * rows_per_block + (int64_t) (lid / 32);
                             // predicate, do not return: the butterflies below
                             // carry work-group-wide barriers
                             const bool live = r < n_rows;
                             const int lane = lid & 31;
                             const float* row_src = src + (live ? r : 0) * 256;
                             float* row_dst = dst + (live ? r : 0) * 256;
                             float reg[8];
#pragma unroll
                             for (int i = 0; i < 8; ++i) reg[i] = (live ? row_src[i * 32 + lane] : 0.0f) * scale;
#pragma unroll
                             for (int hstep = 1; hstep < 32; hstep *= 2) {
#pragma unroll
                                 for (int j = 0; j < 8; ++j) {
                                     // one slot per THREAD (the work-group holds 4 rows of 32 lanes; a
                                     // 32-slot array made the rows overwrite each other's butterflies)
                                     redA[lid] = reg[j];
                                     it.barrier(sycl::access::fence_space::local_space);
                                     const float val2 = redA[lid ^ hstep];
                                     it.barrier(sycl::access::fence_space::local_space);
                                     const float val = reg[j];
                                     reg[j] = (lane & hstep) == 0 ? val + val2 : val2 - val;
                                 }
                             }
#pragma unroll
                             for (int hstep = 32; hstep < 256; hstep *= 2) {
                                 const int step = hstep / 32;
#pragma unroll
                                 for (int j = 0; j < 8; j += 2 * step) {
#pragma unroll
                                     for (int k = 0; k < step; ++k) {
                                         const float x = reg[j + k];
                                         const float y = reg[j + k + step];
                                         reg[j + k] = x + y;
                                         reg[j + k + step] = x - y;
                                     }
                                 }
                             }
#pragma unroll
                             for (int i = 0; i < 8; ++i)
                                 if (live) row_dst[i * 32 + lane] = reg[i];
                         });
    });
    check(stream, "fwht256 launch");
}

// the 32-value group quant: (amax,mval) pair butterfly, then the nibble pack
// pairs lane with lane+16 (shfl_down). Returns the fp16 scale; `byte` is the
// packed nibbles for lanes < 16.
struct Q4Group {
    uint16_t d;
    uint8_t byte;
};
inline Q4Group q4_group(float x, int lane, local_accessor<float, 1> redA, local_accessor<float, 1> redB,
                        local_accessor<uint8_t, 1> qc_sh, nd_item<1> it) {
    float amax = sycl::fabs(x), mval = x;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        redA[lane] = amax;
        redB[lane] = mval;
        it.barrier(sycl::access::fence_space::local_space);
        const float a = redA[lane ^ o];
        const float v = redB[lane ^ o];
        it.barrier(sycl::access::fence_space::local_space);
        if (a > amax || (a == amax && v > mval)) {
            amax = a;
            mval = v;
        }
    }
    const float d = mval / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    int q = (int) sycl::trunc(x * id + 8.5f);
    const uint8_t qc = (uint8_t) (q < 0 ? 0 : (q > 15 ? 15 : q));
    qc_sh[lane] = qc;
    it.barrier(sycl::access::fence_space::local_space);
    const uint8_t qhi = lane < 16 ? qc_sh[lane + 16] : (uint8_t) 0;
    it.barrier(sycl::access::fence_space::local_space);
    Q4Group g;
    g.d = f16_from_f32(d);
    g.byte = (uint8_t) (qc | (qhi << 4));
    return g;
}

inline void q4_store(uint8_t* pool, long long row, int b, int lane, uint16_t d, uint8_t byte) {
    block_q4_0* blk = reinterpret_cast<block_q4_0*>(pool + row * (long long) sizeof(block_q4_0) * 8) + b;
    if (lane == 0) blk->d = d;
    if (lane < 16) blk->qs[lane] = byte;
}

}  // namespace

void fwht256_cuda(const float* src, float* dst, int64_t n_rows, void* stream) {
    if (n_rows <= 0) return;
    fwht256_launch(src, dst, n_rows, stream);
}

void kv_append_q4_step(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, const int32_t* step,
                       const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                       const KvHostPools* host) {
    need_256(s, "kv_append_q4");
    const KvHostPools pools = host ? *host : KvHostPools{};
    const int kv_heads = (int) s.n_head_kv;
    const int head_dim = (int) s.head_dim;
    const int page_size = (int) s.page_size;
    const size_t gy = (size_t) (head_dim / QK4_0);
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<float, 1> redA(sycl::range<1>(32), hnd);
        local_accessor<float, 1> redB(sycl::range<1>(32), hnd);
        local_accessor<uint8_t, 1> qc_sh(sycl::range<1>(32), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) kv_heads * gy * 2 * 32), sycl::range<1>(32)),
                         [=](nd_item<1> it) {
                             const size_t g = it.get_group(0);
                             const int h = (int) (g / (gy * 2));
                             const int b = (int) ((g / 2) % gy);
                             const bool is_v = (g % 2) == 1;
                             const int t = (int) it.get_local_id(0);
                             const long long pos = (long long) step[kStepPos];
                             const float x = (is_v ? vcur : kcur)[(size_t) h * head_dim + b * QK4_0 + t];
                             const Q4Group gr = q4_group(x, t, redA, redB, qc_sh, it);
                             const long long page = (long long) page_table[pos / page_size];
                             if (page >= 0)
                                 q4_store(is_v ? v_q4 : k_q4, (page * kv_heads + h) * page_size + (pos % page_size),
                                          b, t, gr.d, gr.byte);
                             if (pools.k_q4 != nullptr)
                                 q4_store(is_v ? pools.v_q4 : pools.k_q4,
                                          ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size), b, t,
                                          gr.d, gr.byte);
                         });
    });
    check(stream, "kv_append_q4 launch");
}

void kv_append_q4(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, int64_t pos0, int64_t T, const float* K,
                  const float* V, const QsaShapes& s, void* stream, const KvHostPools* host, const KvHostPools* stage) {
    if (T <= 0) return;
    need_256(s, "kv_append_q4");
    const KvHostPools hp = host ? *host : KvHostPools{};
    const KvHostPools stp = stage ? *stage : KvHostPools{};
    const int kv_heads = (int) s.n_head_kv;
    const int head_dim = (int) s.head_dim;
    const int page_size = (int) s.page_size;
    const size_t gy = (size_t) (head_dim / QK4_0);
    for (int is_v = 0; is_v < 2; ++is_v) {
        Q(stream).submit([&](sycl::handler& hnd) {
            local_accessor<float, 1> redA(sycl::range<1>(32), hnd);
            local_accessor<float, 1> redB(sycl::range<1>(32), hnd);
            local_accessor<uint8_t, 1> qc_sh(sycl::range<1>(32), hnd);
            hnd.parallel_for(
                sycl::nd_range<1>(sycl::range<1>((size_t) T * kv_heads * gy * 32), sycl::range<1>(32)),
                [=](nd_item<1> it) {
                    const size_t g = it.get_group(0);
                    const long long t = (long long) (g / (kv_heads * gy));
                    const int h = (int) ((g / gy) % (size_t) kv_heads);
                    const int b = (int) (g % gy);
                    const int th = (int) it.get_local_id(0);
                    const long long pos = pos0 + t;
                    const float x =
                        (is_v ? V : K)[(size_t) t * (kv_heads * head_dim) + (size_t) h * head_dim + b * QK4_0 + th];
                    const Q4Group gr = q4_group(x, th, redA, redB, qc_sh, it);
                    const long long page = (long long) page_table[pos / page_size];
                    const long long row_id =
                        ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size);
                    if (page >= 0)
                        q4_store(is_v ? v_q4 : k_q4, (page * kv_heads + h) * page_size + (pos % page_size), b, th,
                                 gr.d, gr.byte);
                    if (hp.k_q4 != nullptr) q4_store(is_v ? hp.v_q4 : hp.k_q4, row_id, b, th, gr.d, gr.byte);
                    if (stp.k_q4 != nullptr) q4_store(is_v ? stp.v_q4 : stp.k_q4, row_id, b, th, gr.d, gr.byte);
                });
        });
    }
    check(stream, "kv_append_q4 batch launch");
}

void kv_gather_q4_step(const uint8_t* k_q4, const uint8_t* v_q4, const int32_t* page_table, const int32_t* ids,
                       const int32_t* step, int64_t max_ids, const QsaShapes& s, uint16_t* k_scratch,
                       uint16_t* v_scratch, void* stream) {
    if (max_ids <= 0) return;
    const int kv_heads = (int) s.n_head_kv;
    const int head_dim = (int) s.head_dim;
    const int page_size = (int) s.page_size;
    const int blocks_per_head = head_dim / QK4_0;
    const int bytes_per_head = blocks_per_head * (int) sizeof(block_q4_0);
    // the width lives in DEVICE memory (step[kStepWidth]): size the launch with
    // max_ids and cull inside, exactly as the CUDA kernel does.  Reading it
    // here on the host dereferences a device pointer.
    Q(stream).parallel_for(sycl::range<2>((size_t) max_ids * kv_heads * blocks_per_head, 32), [=](sycl::id<2> id) {
        const long long n_ids = (long long) step[kStepWidth];
        const long long blk_idx = (long long) id[0];
        if (blk_idx >= n_ids * kv_heads * blocks_per_head) return;
        const int t = (int) id[1];
        const long long bid = blk_idx / (kv_heads * blocks_per_head);
        const int rem = (int) (blk_idx % (kv_heads * blocks_per_head));
        const int h = rem / blocks_per_head;
        const int b = rem % blocks_per_head;
        const int cell = ids[bid];
        const long long page = (long long) page_table[cell / page_size];
        const long long row = (page * kv_heads + h) * page_size + (cell % page_size);
        const block_q4_0* k_blk = reinterpret_cast<const block_q4_0*>(k_q4 + row * bytes_per_head) + b;
        const block_q4_0* v_blk = reinterpret_cast<const block_q4_0*>(v_q4 + row * bytes_per_head) + b;
        const float kd = f32_from_f16(k_blk->d);
        const float vd = f32_from_f16(v_blk->d);
        const int j = t < 16 ? t : (t - 16);
        const uint8_t k_byte = k_blk->qs[j];
        const uint8_t v_byte = v_blk->qs[j];
        const int kq = (t < 16) ? ((k_byte & 0x0F) - 8) : ((k_byte >> 4) - 8);
        const int vq = (t < 16) ? ((v_byte & 0x0F) - 8) : ((v_byte >> 4) - 8);
        const long long dst_offset = ((bid * kv_heads + h) * head_dim) + (b * QK4_0 + t);
        k_scratch[dst_offset] = f16_from_f32((float) kq * kd);
        v_scratch[dst_offset] = f16_from_f32((float) vq * vd);
    });
    check(stream, "kv_gather_q4 launch");
}

}  // namespace strata::kernels
