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
uint64_t ple_block_scratch_bytes() { unported("ple_block_scratch_bytes"); }
void ple_block(const float*, const float*, const float*, const PleWeights&, PleOut&, void*, void*) {
    unported("ple_block");
}
void ple_history_advance(float*, const float*, void*) { unported("ple_history_advance"); }
bool ple_block_available() { return false; }
void native_ple_postops(const float*, const float*, const float*, const float*, const PleWeights&,
                        const NativePlePostopsBuffers&, void*) {
    unported("native_ple_postops");
}
void native_ple_postops_batch(float*, float*, const float*, float*, const PleWeights&, float*, float*, float*, int,
                              void*) {
    unported("native_ple_postops_batch");
}
bool PleTable::issue(const uint32_t*) { unported("PleTable::issue"); }
bool PleTable::collect(float*, std::string&) { unported("PleTable::collect"); }
bool PleTable::gather_batch(const uint32_t*, size_t, float*, std::string&) { unported("PleTable::gather_batch"); }
bool PleTable::is_open() const { return false; }

// ---- sampler ----
void sample_tokens(const float*, int, int, const int*, int, const SamplerParams&, int*, void*) {
    unported("sample_tokens");
}
bool sample_greedy_cluster(const float*, int, int, int*, void*) { unported("sample_greedy_cluster"); }
size_t coupled_draft_scratch_bytes(int) { unported("coupled_draft_scratch_bytes"); }
void coupled_draft_stage(const SamplerParams*, const int32_t*, SamplerParams*, int32_t*, int, void*) {
    unported("coupled_draft_stage");
}
void coupled_draft_sample(float*, int, const int32_t*, const int32_t*, int, const SamplerParams*, int32_t*, int, int,
                          const int32_t*, void*, int32_t*, float*, void*) {
    unported("coupled_draft_sample");
}

// ---- shared expert / moe combine / hit path ----
void shared_expert(const uint8_t*, const uint8_t*, const uint16_t*, const SForm&, const uint8_t*, const float*,
                   const float*, const SForm&, const uint8_t*, const float*, const float*, const SForm&,
                   const uint8_t*, const float*, const float*, const uint16_t*, float*, float*, int64_t, int64_t, int,
                   void*, const float*, const NativeSharedWeights*) {
    unported("shared_expert");
}
void shared_expert_multi(int, const float*, const uint16_t*, const NativeSharedWeights&, const uint16_t*, float*,
                         float*, float*, float*, int64_t, int64_t, void*) {
    unported("shared_expert_multi");
}
void moe_combine(const float*, const float*, const float*, float*, int64_t, int64_t, void*) {
    unported("moe_combine");
}
uint64_t moe_hit_grouped_scratch_bytes(int64_t, int64_t, int64_t) { unported("moe_hit_grouped_scratch_bytes"); }
void moe_hit_grouped_s2(const uint8_t*, const int32_t*, const int32_t*, int64_t, int64_t, const uint8_t*, void*,
                        float*, void*, const float*) {
    unported("moe_hit_grouped_s2");
}
void moe_hit_select(const int32_t*, const int32_t*, int, int, int32_t*, int32_t*, int32_t*, void*) {
    unported("moe_hit_select");
}
void moe_hit_grouped_s2_dev(const uint8_t*, const int32_t*, const int32_t*, const int32_t*, int64_t, int64_t,
                            const uint8_t*, void*, float*, void*, const float*) {
    unported("moe_hit_grouped_s2_dev");
}
void moe_hit_add(float*, const float*, const int32_t*, const int32_t*, int64_t, int64_t, void*) {
    unported("moe_hit_add");
}
void moe_grouped_s2(const unsigned long long*, const int32_t*, const int32_t*, const int32_t*, const int32_t*, int64_t,
                    int64_t, const uint8_t*, const float*, void*, float*, void*) {
    unported("moe_grouped_s2");
}
void moe_group_resident(const int32_t*, int, int, const uint8_t*, int64_t, unsigned long long*, int32_t*, int32_t*,
                        int32_t*, int32_t*, void*) {
    unported("moe_group_resident");
}
void moe_hit_select_multi(const int32_t*, const int32_t*, int, int, int32_t*, int32_t*, int32_t*, void*) {
    unported("moe_hit_select_multi");
}

// ---- native (GGUF-block) grouped experts + iq dequant/embed ----
NativeExpertLayout native_expert_layout(int, int, int64_t, int64_t) { unported("native_expert_layout"); }
bool native_expert_supported(int, int, int64_t, int64_t) noexcept { return false; }
size_t native_expert_scratch_bytes(int64_t, int64_t) { unported("native_expert_scratch_bytes"); }
void native_expert_grouped(const NativeExpertLayout&, const unsigned long long*, const int32_t*, const int32_t*,
                           const int32_t*, const int32_t*, int64_t, int64_t, const void*, void*, float*, void*) {
    unported("native_expert_grouped");
}
void iq_set_old_kernels(bool) {}
bool iq_old_kernels() { return false; }
void iq_mmvq(int, const void*, const void*, float*, int, int, int, void*) { unported("iq_mmvq"); }
void iq_dequant_f32(int, const void*, int64_t, float*, void*) { unported("iq_dequant_f32"); }
void iq_dequant_f16(int, const void*, int64_t, uint16_t*, void*) { unported("iq_dequant_f16"); }
void iq_embed_rows(int, const void*, size_t, const int32_t*, int64_t, int64_t, float*, void*) {
    unported("iq_embed_rows");
}
void iq_dequant_gu_f16(int, const void*, const void*, int64_t, int64_t, uint16_t*, void*) {
    unported("iq_dequant_gu_f16");
}
bool embed_type_supported(int) noexcept { return false; }
void quantize_q8_1_rows(const float*, int64_t, int64_t, void*, void*) { unported("quantize_q8_1_rows"); }

// ---- native GDN preprocess ----
void native_gdn_conv_silu(float*, const float*, const float*, float*, float*, int64_t, int64_t, void*) {
    unported("native_gdn_conv_silu");
}
void native_gdn_l2_norm(float*, int64_t, int64_t, float, void*) { unported("native_gdn_l2_norm"); }
void native_gdn_beta_gate(float*, int64_t, void*) { unported("native_gdn_beta_gate"); }
void native_gdn_gate(const float*, const float*, const float*, float*, int64_t, void*) {
    unported("native_gdn_gate");
}
void native_gdn_out_norm(const float*, const float*, const float*, float*, int64_t, int64_t, float, void*) {
    unported("native_gdn_out_norm");
}

// ---- native QSA indexer ----
void native_qsa_indexer_append(const float*, const int32_t*, int32_t, const float*, float,
                               const QsaIndexerBuffers&, const QsaShapes&, int64_t, const RopeScaling&, void*) {
    unported("native_qsa_indexer_append");
}
void native_qsa_indexer_append_batch(const float*, int64_t, int64_t, int32_t, const float*, float,
                                     const QsaIndexerBuffers&, const QsaShapes&, int64_t, const RopeScaling&,
                                     void*) {
    unported("native_qsa_indexer_append_batch");
}

// ---- flash attention short step ----
void native_flash_attn_short_step(const float*, const uint16_t*, const uint16_t*, const int32_t*, int64_t, int,
                                  const QsaShapes&, float*, int32_t*, const uint16_t*, void*) {
    unported("native_flash_attn_short_step");
}

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

void qsa_block_scores(const float*, const float*, const float*, const int32_t*, int64_t, int64_t, const QsaShapes&,
                      float*, void*, int64_t) {
    unported("qsa_block_scores");
}
void qsa_block_topk(const float*, const int32_t*, int64_t, int64_t, int64_t, const QsaShapes&, int32_t*, void*,
                    int64_t) {
    unported("qsa_block_topk");
}
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
void shared_expert_set_native_bf16(bool) {}
void moe_hit_grouped_s2_cpu_order(const uint8_t*, const int32_t*, const int32_t*, int64_t, int64_t, const uint8_t*,
                                  void*, float*, void*, const float*, float*) {
    unported("moe_hit_grouped_s2_cpu_order");
}

void ple_set_native_bf16(bool) {}
void ple_set_native_postops(bool) {}
bool ple_native_postops_enabled() { return false; }
PleConsts ple_artifact_consts() { return PleConsts{}; }
void ple_prefetch_enable(bool) {}
bool ple_prefetch_enabled() { return false; }
void ngram_rows(const int32_t*, const int32_t*, int, const PleConsts&, uint32_t*) { unported("ngram_rows"); }

PleTable::PleTable() = default;
PleTable::~PleTable() = default;
bool PleTable::open(const std::string&, std::string& err, const PleIoOptions&) {
    err = "SYCL backend: the PLE table reader is not ported yet";
    return false;
}
bool PleTable::open(const std::string& p, std::string& err) { return open(p, err, PleIoOptions{}); }
void PleTable::set_injected_delay_us(double) {}
bool PleTable::locked() const { return false; }
uint64_t PleTable::rows() const { return 0; }
std::string PleTable::io_report() const { return "sycl stub"; }

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
    err = "SYCL backend: control vectors are not ported yet";
    return false;
}

void native_qsa_indexer_set_enabled(bool) {}
bool native_qsa_indexer_enabled() { return false; }

}  // namespace strata::kernels

namespace strata::kernels {
bool qsa_block_scores_tc(const float*, const float*, const float*, const int32_t*, int64_t, int64_t, const QsaShapes&,
                         float*, void*, int64_t) {
    return false;  // tensor-core variant: never selected on SYCL
}
void qsa_prompt_attn_batch(const float*, const QsaAttnPools&, const int32_t*, const int32_t*, int64_t,
                           const QsaShapes&, float*, int64_t, void*) {
    unported("qsa_prompt_attn_batch");
}
}  // namespace strata::kernels
