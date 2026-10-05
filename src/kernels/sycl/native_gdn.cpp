// src/kernels/sycl/native_gdn.cpp — SYCL port of src/kernels/cuda/native_gdn.cu
// (single-token GDN state step, lane-sharded over the 128 state rows).
// The CUDA block was dim3(32,4) with blockIdx.z selecting the column quad;
// here one work-group of 128 lanes covers all four quads of one head
// (col = lid/4 * ... mapping kept: col = (lid & 31) + 32 * 0 ... see below),
// preserving the per-lane 4-row shard and the two xor-sums' pairing.
#include "strata/kernels/native_gdn.hpp"

#include <cuda_runtime.h>  // sycl_compat shim

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace strata::kernels {
namespace {
std::atomic<bool> enabled{false};
constexpr int S = 128;

using sycl::local_accessor;
using sycl::nd_item;

inline sycl::queue& Q(void* stream) { return strata::sycl_compat::q_for(stream); }

// CUDA geometry: block (32 lanes, 4 cols) x grid (h_v, 1, S/4): one warp per
// (head, column quad). SYCL: one work-group of 128 = 4 warps = the 4 column
// quads of one head; lane = lid & 31, quad = lid >> 5, col = quad * 32 + lane.
void step_launch(float* state, const float* q, const float* k, const float* v, const float* gate, const float* beta,
                 float* output, int h_k, int h_v, float scale, void* stream) {
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<float, 1> red(sycl::range<1>(128), hnd);
        // one work-group = one CUDA block (32 lanes x 4 column quads); the
        // 32-lane chunk shares ONE column and shards the 128 state rows
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) h_v * 32 * 128), sycl::range<1>(128)),
                         [=](nd_item<1> it) {
                             const int head = (int) (it.get_group(0) / 32);
                             const int bz = (int) (it.get_group(0) % 32);
                             const int lid = (int) it.get_local_id(0);
                             const int lane = lid & 31;
                             const int col = bz * 4 + (lid >> 5);
                             const int q_head = head % h_k;
                             float s_shard[4], k_reg[4], q_reg[4];
#pragma unroll
                             for (int r = 0; r < 4; ++r) {
                                 const int i = r * 32 + lane;
                                 s_shard[r] = state[(size_t(i) * h_v + head) * S + col];
                                 k_reg[r] = k[q_head * S + i];
                                 q_reg[r] = q[q_head * S + i];
                             }
                             const float g_val = sycl::exp(gate[head]);
                             float kv_shard = 0.0f;
#pragma unroll
                             for (int r = 0; r < 4; ++r) kv_shard += s_shard[r] * k_reg[r];
                             // warp xor-sum over the 32-lane chunk
                             red[lid] = kv_shard;
                             it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
                             for (int o = 16; o > 0; o >>= 1) {
                                 const float add = red[lid ^ o];
                                 it.barrier(sycl::access::fence_space::local_space);
                                 kv_shard += add;
                                 red[lid] = kv_shard;
                                 it.barrier(sycl::access::fence_space::local_space);
                             }
                             const float kv_col = kv_shard;
                             const float delta_col = (v[head * S + col] - g_val * kv_col) * beta[head];
                             float attn_partial = 0.0f;
#pragma unroll
                             for (int r = 0; r < 4; ++r) {
                                 s_shard[r] = g_val * s_shard[r] + k_reg[r] * delta_col;
                                 attn_partial += s_shard[r] * q_reg[r];
                             }
                             red[lid] = attn_partial;
                             it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
                             for (int o = 16; o > 0; o >>= 1) {
                                 const float add = red[lid ^ o];
                                 it.barrier(sycl::access::fence_space::local_space);
                                 attn_partial += add;
                                 red[lid] = attn_partial;
                                 it.barrier(sycl::access::fence_space::local_space);
                             }
                             const float attn_col = attn_partial;
                             if (lane == 0) output[head * S + col] = attn_col * scale;
#pragma unroll
                             for (int r = 0; r < 4; ++r) {
                                 const int i = r * 32 + lane;
                                 state[(size_t(i) * h_v + head) * S + col] = s_shard[r];
                             }
                         });
    });
}

bool valid_span(const void* pointer, size_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(pointer);
    return pointer && address % sizeof(float) == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const auto ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}
}  // namespace

void native_gdn_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_gdn_enabled() { return enabled.load(std::memory_order_relaxed); }

void native_gdn_step(float* state, const float* q, const float* k, const float* v, const float* gate,
                     const float* beta, float* output, const GdnShapes& shape, void* stream) {
    if (!stream || shape.S != S || shape.h_k <= 0 || shape.h_v <= 0 || shape.h_v > 65535 ||
        shape.h_v % shape.h_k != 0)
        throw std::invalid_argument("native GDN requires a stream, S=128 and positive divisible head counts <=65535");
    const size_t state_bytes = size_t(S) * S * size_t(shape.h_v) * sizeof(float);
    const size_t qk_bytes = size_t(S) * size_t(shape.h_k) * sizeof(float);
    const size_t output_bytes = size_t(S) * size_t(shape.h_v) * sizeof(float);
    const size_t head_bytes = size_t(shape.h_v) * sizeof(float);
    if (!valid_span(state, state_bytes) || !valid_span(output, output_bytes) ||
        overlap(state, state_bytes, output, output_bytes))
        throw std::invalid_argument("native GDN requires aligned, disjoint state and output spans");
    const void* inputs[] = {q, k, v, gate, beta};
    const size_t bytes[] = {qk_bytes, qk_bytes, output_bytes, head_bytes, head_bytes};
    for (int i = 0; i < 5; ++i) {
        if (!valid_span(inputs[i], bytes[i]) || overlap(state, state_bytes, inputs[i], bytes[i]) ||
            overlap(output, output_bytes, inputs[i], bytes[i]))
            throw std::invalid_argument("native GDN requires aligned input spans disjoint from state and output");
    }
    const float scale = 1.0f / sqrtf(float(S));
    step_launch(state, q, k, v, gate, beta, output, (int) shape.h_k, (int) shape.h_v, scale, stream);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

}  // namespace strata::kernels
