// src/prefill/sycl_stub.cpp — link-complete SYCL stand-ins for the batched
// prompt path (gemm.cu's cuBLAS GEMMs and kernels.cu's chunk kernels). The
// engine's short-prompt path (verify-window replays, --short-read) and the
// per-token path do not call these; a prompt that needs the batched path
// fails loudly here until the oneMKL GEMM + batched kernel port lands
// (docs/SYCL_PORT.md milestone 5).
#include "strata/prefill/kernels.hpp"

#include <cuda_runtime.h>  // sycl_compat shim
#include "strata/prefill/kernels.hpp"

#include <stdexcept>
#include <string>

namespace strata::prefill {
namespace {
[[noreturn]] void unported(const char* what) {
    throw std::runtime_error(std::string("SYCL backend prefill: ") + what + " is not ported yet");
}
}  // namespace

// Init must succeed even when every prompt goes through the verify windows:
// Prefill::init builds the Gemm at startup. The GEMM calls themselves still
// throw until the oneMKL port lands (milestone 5).


void gr_norm(const float*, const float*, float, float*, uint16_t*, int64_t, void*, uint16_t*) { unported("gr_norm"); }
void gr_norm_rs(const float*, const float*, float, float*, uint16_t*, int64_t, void*, uint16_t*) {
    unported("gr_norm_rs");
}
void gr_mix_r(const float*, const float*, const float*, const float*, float*, uint16_t*, int64_t, void*, uint16_t*,
              uint16_t*) {
    unported("gr_mix_r");
}
void gr_write_norm_rs(float*, const float*, const float*, int64_t, const float*, float, float*, uint16_t*, int64_t,
                      void*, uint16_t*) {
    unported("gr_write_norm_rs");
}
void gr_silu(const float*, uint16_t*, int64_t, void*, uint16_t*) { unported("gr_silu"); }
void gr_mix(const float*, const float*, float*, uint16_t*, int64_t, void*, uint16_t*, uint16_t*) {
    unported("gr_mix");
}
void gr_write(float*, const float*, const float*, int64_t, int64_t, void*) { unported("gr_write"); }
void gr_broadcast(const float*, float*, int64_t, void*) { unported("gr_broadcast"); }
void gdn_gates(const float*, const float*, const float*, float*, float*, int64_t, void*) { unported("gdn_gates"); }
void gdn_conv(float*, const float*, const float*, float*, int64_t, float, void*) { unported("gdn_conv"); }
void gdn_recurrence(float*, const float*, const float*, const float*, const float*, const float*, float, float*,
                    uint16_t*, int64_t, void*) {
    unported("gdn_recurrence");
}
void route(const float*, int32_t*, float*, int64_t, int64_t, void*) { unported("route"); }
void blob_dequant(const uint8_t*, uint16_t*, uint16_t*, void*) { unported("blob_dequant"); }
void swiglu_interleaved(const float*, uint16_t*, int64_t, void*) { unported("swiglu_interleaved"); }
void swiglu_pair(const float*, const float*, uint16_t*, int64_t, void*) { unported("swiglu_pair"); }
void copy_i32(int32_t*, const int32_t*, int64_t, void*) { unported("copy_i32"); }
void gather_rows16(const uint16_t*, const int32_t*, uint16_t*, int64_t, int64_t, void*) {
    unported("gather_rows16");
}
void moe_combine(const float*, const int32_t*, const float*, const float*, const float*, float*, int64_t, void*) {
    unported("moe_combine");
}
void rms_rows(float*, const float*, int64_t, int64_t, int64_t, float, void*) { unported("rms_rows"); }
void rope(float*, int64_t, int64_t, int64_t, int64_t, int64_t, const strata::kernels::RopeScaling&, void*) {
    unported("rope");
}
void split_q(const float*, float*, int64_t, void*) { unported("split_q"); }
void gate_attn(const float*, const float*, uint16_t*, int64_t, void*) { unported("gate_attn"); }
void kv_append(const float*, const float*, int64_t, int64_t, const int32_t*, int64_t, uint16_t*, uint16_t*, int8_t*,
               int8_t*, uint16_t*, uint16_t*, void*, const strata::kernels::KvHostPools*,
               const strata::kernels::KvHostPools*) {
    unported("kv_append");
}
void to_f16(const float*, uint16_t*, int64_t, void*) { unported("to_f16"); }
void to_bf16(const float*, uint16_t*, int64_t, void*, uint16_t*) { unported("to_bf16"); }
void round_f16(const float*, float*, int64_t, void*) { unported("round_f16"); }
void blob_dequant_f16(const uint8_t*, uint16_t*, uint16_t*, void*) { unported("blob_dequant_f16"); }

}  // namespace strata::prefill
