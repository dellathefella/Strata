// src/kernels/sycl/iq_embed.cpp — SYCL port of the embedding/dequant slice of
// src/kernels/cuda/iq_kernels.cu that the native pack's token path needs:
// iq_dequant_f32/f16 for Q8_0 and BF16 rows, and iq_embed_rows (device token
// ids -> dequantized rows). The grouped-expert and K/IQ-format dequantizers
// remain in the stub layer.
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>  // sycl_compat shim

#include <stdexcept>

namespace strata::kernels {
namespace {
inline sycl::queue& Q(void* stream) { return strata::sycl_compat::q_for(stream); }

// Q8_0 block: fp16 d then 32 int8 codes; value = code * d
inline void dequant_q8_0_f32(const uint8_t* src, int64_t n, float* dst, size_t i) {
    const int64_t blk = i / 32;
    const int j = (int) (i % 32);
    const uint16_t dbits = (uint16_t) src[blk * 34] | ((uint16_t) src[blk * 34 + 1] << 8);
    const float d = f32_from_f16(dbits);
    dst[i] = (float) ((int8_t) src[blk * 34 + 2 + j]) * d;
}
}  // namespace

bool embed_type_supported(int ggml_type) noexcept { return ggml_type == 8 || ggml_type == 30; }

void iq_dequant_f32(int ggml_type, const void* src, int64_t n, float* dst, void* stream) {
    if (n <= 0) return;
    const uint8_t* s = static_cast<const uint8_t*>(src);
    if (ggml_type == 8) {
        Q(stream).parallel_for((size_t) n, [=](size_t i) { dequant_q8_0_f32(s, n, dst, i); });
    } else if (ggml_type == 30) {
        const uint16_t* h = static_cast<const uint16_t*>(src);
        Q(stream).parallel_for((size_t) n, [=](size_t i) { dst[i] = f32_from_bf16(h[i]); });
    } else {
        throw std::runtime_error("SYCL backend: iq_dequant_f32 for this GGML type is not ported yet");
    }
}

void iq_dequant_f16(int ggml_type, const void* src, int64_t n, uint16_t* dst, void* stream) {
    if (n <= 0) return;
    const uint8_t* s = static_cast<const uint8_t*>(src);
    if (ggml_type == 8) {
        Q(stream).parallel_for((size_t) n, [=](size_t i) {
            float v;
            dequant_q8_0_f32(s, n, &v, i);
            dst[i] = f16_from_f32(v);
        });
    } else if (ggml_type == 30) {
        const uint16_t* h = static_cast<const uint16_t*>(src);
        Q(stream).parallel_for((size_t) n, [=](size_t i) { dst[i] = f16_from_f32(f32_from_bf16(h[i])); });
    } else {
        throw std::runtime_error("SYCL backend: iq_dequant_f16 for this GGML type is not ported yet");
    }
}

void iq_embed_rows(int ggml_type, const void* table, size_t row_bytes, const int32_t* tokens, int64_t n_tok,
                   int64_t n_embd, float* out, void* stream) {
    if (n_tok <= 0 || n_embd <= 0) return;
    const uint8_t* t = static_cast<const uint8_t*>(table);
    if (ggml_type == 8) {
        Q(stream).parallel_for(sycl::range<2>((size_t) n_tok, (size_t) n_embd), [=](sycl::id<2> id) {
            const int64_t tok = tokens[id[0]];
            dequant_q8_0_f32(t + (size_t) tok * row_bytes, n_embd, out + (size_t) id[0] * n_embd, id[1]);
        });
    } else if (ggml_type == 30) {
        Q(stream).parallel_for(sycl::range<2>((size_t) n_tok, (size_t) n_embd), [=](sycl::id<2> id) {
            const int64_t tok = tokens[id[0]];
            const uint16_t* row = reinterpret_cast<const uint16_t*>(t + (size_t) tok * row_bytes);
            out[(size_t) id[0] * n_embd + id[1]] = f32_from_bf16(row[id[1]]);
        });
    } else {
        throw std::runtime_error("SYCL backend: iq_embed_rows for this GGML type is not ported yet");
    }
}

}  // namespace strata::kernels
