// src/kernels/sycl/stubs_not_ported.cpp — link-complete definitions for the
// kernel families the SYCL backend has not ported yet. Every entry point
// throws at CALL time with the family name, so a code path that genuinely
// needs one fails loudly instead of silently computing nothing; the linker
// stays happy and the engine binary builds while the ports land family by
// family (docs/SYCL_PORT.md milestone 3+). Replace a stub with its real port
// file and delete it from here.
#include "strata/kernels/cvec.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_flash_attn.hpp"
#include "strata/kernels/native_gdn_preprocess.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/s_gemv.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/shared_expert.hpp"

#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {
[[noreturn]] void unported(const char* what) {
    throw std::runtime_error(std::string("SYCL backend: ") + what + " is not ported yet");
}
}  // namespace

// ---- PLE (n-gram table + block) ----
void coupled_draft_sample(float*, int, const int32_t*, const int32_t*, int, const SamplerParams*, int32_t*, int, int,
                          const int32_t*, void*, int32_t*, float*, void*) {
    unported("coupled_draft_sample");
}

// ---- shared expert / moe combine / hit path ----
void moe_combine(const float*, const float*, const float*, float*, int64_t, int64_t, void*) {
    unported("moe_combine");
}

// ---- native (GGUF-block) grouped experts + iq dequant/embed ----
void iq_set_old_kernels(bool) {}
bool iq_old_kernels() { return false; }

// ---- control vectors ----
void cvec_set_enabled(bool) {}
bool cvec_enabled() { return false; }
void cvec_apply(float*, int64_t, int64_t, int64_t, const float*, int64_t, const float*, int64_t, bool, void*) {
    unported("cvec_apply");
}

// ---- canonical K-quant GEMVs (canonical packs only) ----
void s_gemv_q8k(const uint8_t*, const uint8_t*, const float*, const float*, float*, int64_t, int64_t, const SForm&,
                void*) {
    unported("s_gemv_q8k");
}

}  // namespace strata::kernels

// ---- second link round: host-side definitions that live in the CUDA TUs ----
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_select.hpp"

namespace strata::kernels {

void qsa_decode_attn_step(const float*, const QsaAttnPools&, const int32_t*, const int32_t*, int64_t, const QsaShapes&,
                          float*, float*, void*) {
    unported("qsa_decode_attn_step");
}
void s_gemv_q8k_split(const uint8_t*, const uint8_t*, const float*, const float*, float*, int64_t, int64_t,
                      const SForm&, void*) {
    unported("s_gemv_q8k_split");
}
void s_gemv_q8_0_split(const uint8_t*, const uint8_t*, const float*, const float*, float*, int64_t, int64_t,
                       const SForm&, void*) {
    unported("s_gemv_q8_0_split");
}
uint64_t shared_expert_scratch_bytes(int64_t) { return uint64_t(1) << 20; }
void moe_hit_grouped_s2_cpu_order(const uint8_t*, const int32_t*, const int32_t*, int64_t, int64_t, const uint8_t*,
                                  void*, float*, void*, const float*, float*) {
    unported("moe_hit_grouped_s2_cpu_order");
}


const Cvec& cvec() {
    static Cvec c;
    return c;
}
bool cvec_upload(const std::vector<float>&, const std::vector<float>&, int, int, int, int64_t, int64_t,
                 std::string& err) {
    err = "SYCL backend: control vectors are not ported yet";
    return false;
}
bool cvec_replicate(std::string& err) {
    // CUDA: !loaded() || upload_here() - nothing can be loaded on this
    // backend (cvec_upload refuses), so the replicate is a no-op success;
    // a stage init calls it unconditionally
    (void) err;
    return true;
}


}  // namespace strata::kernels

namespace strata::kernels {
bool qsa_prompt_attn_batch(const float*, const QsaAttnPools&, const int32_t*, const int32_t*, int64_t,
                           const QsaShapes&, float*, int64_t, void*) {
    // false = "this device cannot": prefill falls back to chunked
    // qsa_decode_attn_batch (ported, parity-clean).  A native flash port can
    // replace this later for prompt speed.
    return false;
}
}  // namespace strata::kernels

namespace strata::kernels {
size_t coupled_draft_scratch_bytes(int) { unported("coupled_draft_scratch_bytes"); }
void coupled_draft_stage(const SamplerParams*, const int32_t*, SamplerParams*, int32_t*, int, void*) {
    unported("coupled_draft_stage");
}
}  // namespace strata::kernels

namespace strata::kernels {
void native_flash_attn_short_step(const float*, const uint16_t*, const uint16_t*, const int32_t*, int64_t, int,
                                  const QsaShapes&, float*, int32_t*, const uint16_t*, void*) {
    unported("native_flash_attn_short_step");
}
}  // namespace strata::kernels
