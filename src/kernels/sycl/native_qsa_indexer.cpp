// src/kernels/sycl/native_qsa_indexer.cpp — SYCL port of the single-token
// append of src/kernels/cuda/native_qsa_indexer.cu (indexer key cache +
// pooled-key rms/rotation). The batch (prompt) kernels remain stubs: they are
// prefill-path only.
//
// 256 threads: d < 128 owns a channel, the upper half only participates in
// the two reduction trees (the CUDA kernel's warp sums run over all 8 warps
// with zero contributions above 128). f16 round-trip on the incoming key and
// the __fadd_rn/__fmaf_rn mean chain are kept exactly (plain + and * under
// -ffp-contract=off; the 0.25 multiply is a single rn product).
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/rope_scaling.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>  // sycl_compat shim

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strata::kernels {
namespace {
std::atomic<bool> enabled{false};
constexpr int D = 128, R = 4, ROT = 64, THREADS = 256;

using sycl::local_accessor;
using sycl::nd_item;

inline sycl::queue& Q(void* stream) { return strata::sycl_compat::q_for(stream); }

template <bool TAB>
void append_launch(const float* raw, const int32_t* pos_dev, int pos_base, const float* gamma, float epsilon,
                   float* tail, float* dead, float* pooled, int32_t* block_pos, int max_cells, float theta_scale,
                   float freq_scale, float corr_low, float corr_high, float ext_factor, float mscale,
                   const int32_t* mtab, RopeTab rt, void* stream) {
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<float, 1> values(sycl::range<1>(D), hnd);
        local_accessor<float, 1> partials(sycl::range<1>(THREADS / 32), hnd);
        local_accessor<float, 1> red(sycl::range<1>(THREADS), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>(THREADS), sycl::range<1>(THREADS)), [=](nd_item<1> it) {
            const int d = (int) it.get_local_id(0);
            const int pos = pos_dev[0];
            if (pos < 0 || pos >= max_cells) return;  // uniform: whole group exits
            const int slot = pos % R;
            float incoming = 0.0f;
            if (d < D) {
                incoming = (float) sycl::half(raw[d]);  // f16 round-trip, as pinned
                if (slot < R - 1) tail[slot * D + d] = incoming;
            }
            if (pos != 0 && slot != R - 1) return;  // uniform
            it.barrier(sycl::access::fence_space::local_space);
            float mean = 0.0f;
            if (d < D) {
                float sum = pos == 0 ? incoming : tail[d];
#pragma unroll
                for (int j = 1; j < R; ++j)
                    sum = sum + (pos == 0 || j == R - 1 ? incoming : tail[j * D + d]);
                mean = 0.25f * sum;
            }
            float square_sum = 0.0f;
            if (d < D) square_sum += mean * mean;
            const int lane = d & 31;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                red[d] = square_sum;
                it.barrier(sycl::access::fence_space::local_space);
                square_sum += red[d ^ off];
                it.barrier(sycl::access::fence_space::local_space);
            }
            if (lane == 0) partials[d / 32] = square_sum;
            it.barrier(sycl::access::fence_space::local_space);
            square_sum = lane < THREADS / 32 ? partials[lane] : 0.0f;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                red[d] = square_sum;
                it.barrier(sycl::access::fence_space::local_space);
                square_sum += red[d ^ off];
                it.barrier(sycl::access::fence_space::local_space);
            }
            const float scale = sycl::rsqrt(square_sum / D + epsilon);
            if (d < D) values[d] = scale * mean * gamma[d];
            it.barrier(sycl::access::fence_space::local_space);
            if (d >= D) return;
            const int b = pos / R;
            const int rope_pos = pos == 0 ? 0 : pos_base + R * b;
            float y = values[d];
            if (d < ROT) {
                const int pair = d % (ROT / 2);
                float c, s;
                if (!(TAB && rope_tab_cs(rt, pos == 0 ? 0 : mrope_pos(mtab, rope_pos, pair), pair, c, s))) {
                    const float theta_extrap =
                        (pos == 0 ? 0 : mrope_pos(mtab, rope_pos, pair)) * sycl::pow(theta_scale, (float) pair);
                    rope_scaled_angle(theta_extrap, freq_scale, corr_low, corr_high, ext_factor, mscale, pair, c, s);
                }
                const float a = values[pair], z = values[pair + ROT / 2];
                y = d < ROT / 2 ? a * c - z * s : a * s + z * c;
            }
            pooled[(size_t) b * D + d] = y;
            if (pos == 0)
                dead[d] = y;
            else
                pooled[(size_t) (b + 1) * D + d] = dead[d];
            if (d == 0 && pos != 0) *block_pos = rope_pos;
        });
    });
}

}  // namespace

void native_qsa_indexer_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_qsa_indexer_enabled() { return enabled.load(std::memory_order_relaxed); }

void native_qsa_indexer_append(const float* raw, const int32_t* relative_pos_device, int32_t pos_base,
                               const float* gamma, float epsilon, const QsaIndexerBuffers& b, const QsaShapes& s,
                               int64_t max_cells, const RopeScaling& scaling, void* stream) {
    if (!stream || s.idx_dim != D || s.idx_block != R || s.n_rot != ROT || max_cells < 1 ||
        max_cells > INT32_MAX || pos_base < 0 || pos_base % R || int64_t(pos_base) + max_cells > INT32_MAX ||
        !std::isfinite(epsilon) || epsilon <= 0.0f || rope_scaling_invalid(scaling) != nullptr)
        throw std::invalid_argument(
            "native QSA indexer requires fixed geometry, aligned position base, positive capacity/epsilon, valid "
            "frequency/scaling and explicit stream");
    if (!raw || !relative_pos_device || !gamma || !b.tail || !b.dead || !b.pooled || !b.block_pos)
        throw std::invalid_argument("native QSA indexer requires nonnull buffers");
    const float theta_scale = sycl::pow((float) scaling.freq_base, -2.0f / ROT);
    const RopeKernelArgs k = scaling.kernel_args(ROT);
    const RopeTab rt = rope_table_for(scaling);
    if (rt.cos != nullptr)
        append_launch<true>(raw, relative_pos_device, pos_base, gamma, epsilon, b.tail, b.dead, b.pooled, b.block_pos,
                            (int) max_cells, theta_scale, k.freq_scale, k.corr_low, k.corr_high, k.ext_factor,
                            k.attn_factor, mrope_table(), rt, stream);
    else
        append_launch<false>(raw, relative_pos_device, pos_base, gamma, epsilon, b.tail, b.dead, b.pooled,
                             b.block_pos, (int) max_cells, theta_scale, k.freq_scale, k.corr_low, k.corr_high,
                             k.ext_factor, k.attn_factor, mrope_table(), rt, stream);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

void native_qsa_indexer_append_batch(const float*, int64_t, int64_t, int32_t, const float*, float,
                                     const QsaIndexerBuffers&, const QsaShapes&, int64_t, const RopeScaling&,
                                     void*) {
    throw std::runtime_error("SYCL backend: native_qsa_indexer_append_batch (prompt path) is not ported yet");
}

}  // namespace strata::kernels
