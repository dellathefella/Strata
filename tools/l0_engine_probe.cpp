// tools/l0_engine_probe.cpp — eager (uncaptured) sequential call of every
// kernel family the verify window runs before its first doorbell publish,
// with a stream sync and a print after each. The last line printed before a
// hang names the deadlocking kernel. Shapes are the real Flash-Next geometry
// (N=2560, HC=4, LR=320, h_k=16, h_v=48, S=128, C=(h_k+2*h_v)*S), T=1.
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/rope_scaling.hpp"
#include "strata/kernels/native_gdn_preprocess.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/gdn.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>  // sycl_compat shim

#include <cstdio>
#include <cstdlib>
#include <cstdlib>
#include <vector>

namespace sk = strata::kernels;

static void* zalloc(size_t bytes) {
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) { std::fprintf(stderr, "alloc %zu failed\n", bytes); std::exit(1); }
    cudaMemset(p, 0, bytes);
    return p;
}

#define STEP(name)                                                  \
    do {                                                            \
        if (cudaStreamSynchronize(cs) != cudaSuccess) {        \
            std::fprintf(stderr, "SYNC FAIL before " name "\n");    \
            return 1;                                               \
        }                                                           \
        std::fprintf(stderr, "OK " name "\n");                      \
    } while (0)

int main() {
    cudaStream_t cs = nullptr;
    cudaStreamCreate(&cs);
    void* S0 = cs;  // explicit stream: the quantize wrappers require one
    (void) S0;
    constexpr int64_t N = 2560, HC = 4, LR = 320, HK = 16, HV = 48, S = 128;
    constexpr int64_t C = (HK + 2 * HV) * S;
    constexpr int T = 1;
    std::vector<float> zeros;
    float* R = (float*) zalloc(N * HC * 4);
    float* bo = (float*) zalloc(N * 4);
    float* inj = (float*) zalloc(HC * 4);
    float* w_norm = (float*) zalloc(N * HC * 4);
    uint16_t* w_down = (uint16_t*) zalloc(LR * N * HC * 2);
    uint16_t* w_up = (uint16_t*) zalloc(N * HC * LR * 2);
    float* lo = (float*) zalloc(LR * 4);
    float* rs = (float*) zalloc(HC * 4);
    float* mixed = (float*) zalloc(N * 4);
    sk::FusedGrArgs a;
    a.R = R; a.R_out = R; a.apply = false; a.bo_prev = bo; a.inj_prev = inj; a.w_norm = w_norm;
    a.w_down = w_down; a.w_up = w_up; a.w_inject = nullptr; a.eps = 1e-6f;
    a.lo = lo; a.rs = rs; a.inject_out = inj; a.mixed = mixed;
    float* xn = (float*) zalloc(T * N * HC * 4);
    sk::fused_gr_read_multi(&a, T, xn, cs);
    STEP("fused_gr_read_multi");

    float* qkv = (float*) zalloc(T * C * 4);
    float* hist = (float*) zalloc(C * 3 * 4);
    float* convw = (float*) zalloc(C * 4 * 4);
    float* hbuf = (float*) zalloc(T * C * 4);
    sk::gdn_conv_l2_multi(hist, qkv, convw, hbuf, (int) C, (int) (HK + HV), 1e-6f, T, cs, 0);
    STEP("gdn_conv_l2_multi");
    int32_t* nkeep = (int32_t*) zalloc(4);
    sk::gdn_conv_commit(hist, qkv, (int) C, nkeep, cs);
    STEP("gdn_conv_commit");

    float* x = (float*) zalloc(N * 4 * T);
    uint16_t* wa = (uint16_t*) zalloc(HV * N * 2);
    uint16_t* wb = (uint16_t*) zalloc(HV * N * 2);
    float* dt = (float*) zalloc(HV * 4);
    float* ssm_a = (float*) zalloc(HV * 4);
    float* gate = (float*) zalloc(T * HV * 4);
    float* beta = (float*) zalloc(T * HV * 4);
    sk::gdn_ab_multi(x, wa, wb, dt, ssm_a, gate, beta, (int) N, (int) HV, T, cs);
    STEP("gdn_ab_multi");

    float* state = (float*) zalloc(S * S * HV * 4);
    float* z = (float*) zalloc(T * S * HV * 4);
    float* gamma = (float*) zalloc(S * 4);
    float* y = (float*) zalloc(S * HV * S * 4);
    sk::gdn_step_norm_multi(state, hbuf, (int) C, gate, beta, z, gamma, 1e-6f, y, (int) HK, (int) HV, T, nullptr, cs);
    STEP("gdn_step_norm_multi");

    uint8_t* q81 = (uint8_t*) zalloc((N / 32) * 36 * T);
    sk::native_quantize_q8_1(x, q81, (int) N, T, cs);
    STEP("native_quantize_q8_1");
    uint8_t* wq8 = (uint8_t*) zalloc(10240 * N / 32 * 34);
    float* ymm = (float*) zalloc(N * 4 * T);
    sk::native_mmvq(8, wq8, q81, ymm, (int) N, (int) N, T, cs);
    STEP("native_mmvq q8_0");

    float* attn = (float*) zalloc(S * HV * S * 4);
    float* qfull = (float*) zalloc(2 * S * HV * 4);
    sk::native_qsa_rms_norm_weighted(y, gamma, attn, (int) S, (int) (S * HV), 1e-6f, cs);
    STEP("native_qsa_rms_norm_weighted");
    float* gatedf = (float*) zalloc(S * HV * 4);
    sk::native_qsa_gate_apply(attn, qfull, gatedf, (int) HV, (int) S, cs);
    STEP("native_qsa_gate_apply");

    float* logits = (float*) zalloc(T * 512 * 4);
    int32_t* ids = (int32_t*) zalloc(T * 10 * 4);
    float* rw = (float*) zalloc(T * 10 * 4);
    sk::native_router_top10_multi(logits, ids, rw, T, cs);
    STEP("native_router_top10_multi");

    // ---- the QSA select + resident-plan kernels that run before publish ----
    int32_t* h_step = (int32_t*) malloc(64);
    sk::QsaShapes qs{};
    qs.n_head = 24; qs.n_head_kv = 2; qs.head_dim = 256; qs.idx_dim = S; qs.idx_n_head = 4;
    qs.idx_block = 4; qs.page_size = 64; qs.n_rot = 64;
    int32_t* d_step = (int32_t*) zalloc(64);
    h_step[0] = 7; h_step[1] = 8; h_step[2] = 2; h_step[3] = 8;  // pos, n_kv, n_bid, width
    cudaMemcpy(d_step, h_step, 64, cudaMemcpyHostToDevice);
    float* tail = (float*) zalloc(3 * S * 4);
    float* dead = (float*) zalloc(S * 4);
    float* pooled = (float*) zalloc(8 * S * 4);
    int32_t* bpos = (int32_t*) zalloc(4);
    sk::QsaIndexerBuffers ib{tail, dead, pooled, bpos};
    float* raw = (float*) zalloc(S * 4);
    int32_t* posdev = d_step;  // kStepPos word
    float* cosb = (float*) zalloc(64 * 64 * 4);
    float* sinb = (float*) zalloc(64 * 64 * 4);
    sk::RopeScaling scaling{};
    sk::native_qsa_indexer_append(raw, posdev, 0, gamma, 1e-6f, ib, qs, 8, scaling, cs);
    STEP("native_qsa_indexer_append");
    float* qidx = (float*) zalloc(4 * S * 4);
    float* scores = (float*) zalloc(8 * 4);
    sk::qsa_block_scores(pooled, dead, qidx, d_step, 1, 8, qs, scores, cs, 0);
    STEP("qsa_block_scores");
    int32_t* sel = (int32_t*) zalloc(8 * 4);
    sk::qsa_block_topk(scores, d_step, 1, 8, 8, qs, sel, cs, 0);
    STEP("qsa_block_topk");
    sk::QsaAttnPools pools{};
    pools.page_table = (int32_t*) zalloc(8 * 4);
    pools.k_pool = (uint16_t*) zalloc(8 * 64 * HV * S * 2);
    pools.v_pool = (uint16_t*) zalloc(8 * 64 * HV * S * 2);
    float* kscr = (float*) zalloc(8 * HV * S * 4);
    float* vscr = (float*) zalloc(8 * HV * S * 4);
    float* q = (float*) zalloc(24 * 256 * 4);
    float* attno = (float*) zalloc(24 * 256 * 4);
    sk::qsa_decode_attn_batch(q, pools, sel, d_step, 8, qs, (float*) kscr, attno, 1, cs);
    STEP("qsa_decode_attn_batch");
    int32_t* plan = (int32_t*) zalloc(4096);
    uint32_t* skip = (uint32_t*) zalloc(64);
    uint8_t* cbase = (uint8_t*) zalloc(4096);
    sk::resident_plan(ids, 10, 10, (const int32_t*) rw, 512, cbase, nullptr, 4096, plan, 64, skip, 1, cs);
    STEP("resident_plan");
    uint32_t* flag = (uint32_t*) zalloc(64);
    sk::doorbell_publish(mixed, ids, rw, N, 10, mixed, ids, rw, flag, cs);
    STEP("doorbell_publish");

    // ---- families the window runs that the first probe omitted ----
    float* histg = (float*) zalloc(C * 3 * 4);
    float* rawo = (float*) zalloc(C * 4);
    float* silo = (float*) zalloc(C * 4);
    sk::native_gdn_conv_silu(histg, x, convw, rawo, silo, C, 4, cs);
    STEP("native_gdn_conv_silu");
    sk::native_gdn_l2_norm(rawo, T * HV, S, 1e-6f, cs);
    STEP("native_gdn_l2_norm");
    sk::native_gdn_beta_gate(beta, T * HV, cs);
    STEP("native_gdn_beta_gate");
    sk::native_gdn_gate(gate, dt, ssm_a, gate, HV, cs);
    STEP("native_gdn_gate");
    sk::native_gdn_out_norm(y, z, gamma, y, HV, S, 1e-6f, cs);
    STEP("native_gdn_out_norm");
    float* ybig = (float*) zalloc(10240 * 4 * T);
    sk::native_mmvq(8, wq8, q81, ybig, (int) N, 10240, T, cs);
    STEP("native_mmvq q8_0 n_out=10240");
    sk::native_mmvq(20, wq8, q81, ybig, (int) N, 10240, T, cs);
    STEP("native_mmvq iq4_nl n_out=10240");
    float* pnorm = (float*) zalloc(S * HV * 4);
    sk::ple_history_advance(histg, pnorm, cs);
    STEP("ple_history_advance");
    sk::PleWeights pw{};
    pw.key_bf16 = (uint16_t*) zalloc(N * HC * N * 2);
    pw.value_bf16 = (uint16_t*) zalloc(N * N * 2);
    pw.norm_key = (float*) zalloc(N * HC * 4);
    pw.norm_query = (float*) zalloc(N * HC * 4);
    pw.norm_conv = (float*) zalloc(N * HC * 4);
    pw.conv1d_f16 = (uint16_t*) zalloc(4 * N * HC * 2);
    sk::PleOut po{};
    po.result = (float*) zalloc(N * HC * 4);
    po.key = (float*) zalloc(N * HC * 4);
    po.value = (float*) zalloc(N * 4);
    po.gate = (float*) zalloc(HC * 4);
    po.gated = (float*) zalloc(N * HC * 4);
    po.normalized = (float*) zalloc(N * HC * 4);
    po.conv = (float*) zalloc(N * HC * 4);
    void* pscr = zalloc(sk::ple_block_scratch_bytes());
    float* embrow = (float*) zalloc(N * 4 * 16);
    pw.key_bf16 = nullptr;
    pw.key_native_data = wq8;
    pw.key_native_q8_1 = q81;
    pw.key_native_type = 8;
    sk::ple_block(embrow, (float*) zalloc(N * HC * 4), histg, pw, po, pscr, cs);
    STEP("ple_block native-key (eager)");
    STEP("ple_block");
    sk::NativePlePostopsBuffers pb{po.key, po.normalized, po.gate, po.gated, po.normalized, po.conv, po.result};
    sk::native_ple_postops(po.key, po.gated, po.value, histg, pw, pb, cs);
    STEP("native_ple_postops");
    int32_t* toks = (int32_t*) zalloc(64);
    float* embout = (float*) zalloc(4 * N * 4);
    sk::embedding_gather_dev((const uint8_t*) wq8, (const float*) gate, nullptr, toks, 4, N, 4, 0, 32, 64, 8, embout, cs);
    STEP("embedding_gather_dev");
    int32_t* dmap = (int32_t*) zalloc(64);
    sk::copy_i32_from_mapped(dmap, d_step, 4, cs);
    STEP("copy_i32_from_mapped");
    cudaMemcpy2DAsync(kscr, 256 * 4, vscr, 256 * 4, 256 * 4, 4, cudaMemcpyDeviceToDevice, cs);
    STEP("cudaMemcpy2DAsync");
    // ---- layer-0 replica at engine shapes (C=14336, inject mixer, ZV out) ----
    {
        constexpr int64_t CE = (16 + 2 * 48) * 128;   // 14336
        constexpr int64_t ZV = 48 * 128;              // 6144
        float* R = (float*) zalloc(N * HC * 4);
        float* qkvE = (float*) zalloc(CE * 4);
        float* hbE = (float*) zalloc(CE * 4);
        float* histE = (float*) zalloc(CE * 3 * 4);
        float* convwE = (float*) zalloc(CE * 4 * 4);
        float* gateE = (float*) zalloc(T * 48 * 4);
        float* betaE = (float*) zalloc(T * 48 * 4);
        float* zE = (float*) zalloc(T * ZV * 4);
        float* yE = (float*) zalloc(T * ZV * 4);
        float* stateE = (float*) zalloc(128 * 128 * 48 * 4);
        uint8_t* wCE = (uint8_t*) zalloc(CE * N / 32 * 34);
        uint8_t* wZE = (uint8_t*) zalloc(ZV * N / 32 * 34);
        uint8_t* wOE = (uint8_t*) zalloc(N * ZV / 32 * 34);
        uint8_t* q81E = (uint8_t*) zalloc((N / 32) * 36 * T);
        float* mixedE = (float*) zalloc(N * 4);
        sk::FusedGrArgs fa{};
        fa.R = R; fa.R_out = R; fa.apply = false; fa.bo_prev = bo; fa.inj_prev = inj;
        fa.w_norm = w_norm; fa.w_down = w_down; fa.w_up = w_up; fa.w_inject = w_down;  // inject present
        fa.eps = 1e-6f; fa.lo = lo; fa.rs = rs; fa.inject_out = inj; fa.mixed = mixedE;
        float* xnE = (float*) zalloc(T * N * HC * 4);
        sk::fused_gr_read_multi(&fa, T, xnE, cs);
        STEP("L0 fused_gr (inject)");
        sk::native_quantize_q8_1(mixedE, q81E, (int) N, T, cs);
        sk::native_mmvq(8, wCE, q81E, qkvE, (int) N, (int) CE, T, cs);
        STEP("L0 mmvq qkv C=14336");
        sk::gdn_conv_l2_multi(histE, qkvE, convwE, hbE, (int) CE, 32, 1e-6f, T, cs, 0);
        sk::gdn_conv_commit(histE, qkvE, (int) CE, nkeep, cs);
        sk::gdn_ab_multi(mixedE, wa, wb, dt, ssm_a, gateE, betaE, (int) N, 48, T, cs);
        sk::native_mmvq(8, wZE, q81E, zE, (int) N, (int) ZV, T, cs);
        sk::gdn_step_norm_multi(stateE, hbE, (int) CE, gateE, betaE, zE, gamma, 1e-6f, yE, 16, 48, T, nullptr, cs);
        STEP("L0 gdn chain C=14336");
        sk::native_quantize_q8_1(yE, q81E, (int) ZV, T, cs);
        sk::native_mmvq(8, wOE, q81E, mixedE, (int) ZV, (int) N, T, cs);
        STEP("L0 out mmvq");
    }
    std::fprintf(stderr, "PROBE COMPLETE (eager)\n");
    // ---- now the same chain CAPTURED, then replayed: the verify window's shape ----
    uint32_t* flag2_raw = nullptr;
    cudaHostAlloc(&flag2_raw, 64, cudaHostAllocMapped);
    flag2_raw[0] = 0;
    volatile uint32_t* flag2 = flag2_raw;
    if (cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        std::fprintf(stderr, "capture begin failed\n"); return 1;
    }
    int32_t* h_step_host = nullptr;
    cudaHostAlloc(&h_step_host, 64, cudaHostAllocMapped);
    h_step_host[0] = 7; h_step_host[1] = 8; h_step_host[2] = 2; h_step_host[3] = 8;
    cudaMemcpyAsync(d_step, h_step_host, 64, cudaMemcpyHostToDevice, cs);
    sk::fused_gr_read_multi(&a, T, xn, cs);
    sk::gdn_conv_l2_multi(hist, qkv, convw, hbuf, (int) C, (int) (HK + HV), 1e-6f, T, cs, 0);
    sk::gdn_conv_commit(hist, qkv, (int) C, nkeep, cs);
    sk::gdn_ab_multi(x, wa, wb, dt, ssm_a, gate, beta, (int) N, (int) HV, T, cs);
    sk::gdn_step_norm_multi(state, hbuf, (int) C, gate, beta, z, gamma, 1e-6f, y, (int) HK, (int) HV, T, nullptr, cs);
    sk::native_quantize_q8_1(x, q81, (int) N, T, cs);
    sk::native_mmvq(8, wq8, q81, ymm, (int) N, (int) N, T, cs);
    sk::native_qsa_rms_norm_weighted(y, gamma, attn, (int) S, (int) (S * HV), 1e-6f, cs);
    sk::native_qsa_gate_apply(attn, qfull, gatedf, (int) HV, (int) S, cs);
    sk::native_router_top10_multi(logits, ids, rw, T, cs);
    sk::native_qsa_indexer_append(raw, posdev, 0, gamma, 1e-6f, ib, qs, 8, scaling, cs);
    sk::qsa_block_scores(pooled, dead, qidx, d_step, 1, 8, qs, scores, cs, 0);
    sk::qsa_block_topk(scores, d_step, 1, 8, 8, qs, sel, cs, 0);
    sk::qsa_decode_attn_batch(q, pools, sel, d_step, 8, qs, (float*) kscr, attno, 1, cs);
    sk::resident_plan(ids, 10, 10, (const int32_t*) rw, 512, cbase, nullptr, 4096, plan, 64, skip, 1, cs);
    sk::native_gdn_conv_silu(histg, x, convw, rawo, silo, C, 4, cs);
    sk::native_gdn_l2_norm(rawo, T * HV, S, 1e-6f, cs);
    sk::native_gdn_beta_gate(beta, T * HV, cs);
    sk::native_gdn_gate(gate, dt, ssm_a, gate, HV, cs);
    sk::native_gdn_out_norm(y, z, gamma, y, HV, S, 1e-6f, cs);
    sk::ple_history_advance(histg, pnorm, cs);
    sk::ple_block(embrow, (float*) zalloc(N * HC * 4), histg, pw, po, pscr, cs);
    sk::NativePlePostopsBuffers pb2{po.key, po.normalized, po.gate, po.gated, po.normalized, po.conv, po.result};
    sk::native_ple_postops(po.key, po.gated, po.value, histg, pw, pb2, cs);
    sk::embedding_gather_dev((const uint8_t*) wq8, (const float*) gate, nullptr, toks, 4, N, 4, 0, 32, 64, 8, embout, cs);
    sk::copy_i32_from_mapped(dmap, d_step, 4, cs);
    sk::kv_append_step((uint16_t*) kscr, (uint16_t*) vscr, (const int32_t*) plan, d_step, x, x, qs, cs);
    sk::iq_embed_rows(8, wq8, 34 * 80, toks, 4, N, embout, cs);
    sk::copy_indexed(mixed, mixed, N, d_step, N, cs);
    sk::broadcast_streams(x, mixed, N / HC, HC, T, cs);
    sk::add_streams_broadcast(mixed, x, mixed, N / HC, HC, T, cs);
    sk::gpu_stamp((unsigned long long*) cbase, 0, cs);
    cudaMemcpy2DAsync(kscr, 256 * 4, vscr, 256 * 4, 256 * 4, 4, cudaMemcpyDeviceToDevice, cs);
    sk::doorbell_publish(mixed, ids, rw, N, 10, mixed, ids, rw, flag2_raw, cs);
    cudaGraph_t g = nullptr;
    if (cudaStreamEndCapture(cs, &g) != cudaSuccess) { std::fprintf(stderr, "end capture failed\n"); return 1; }
    cudaGraphExec_t ex = nullptr;
    if (cudaGraphInstantiate(&ex, g, 0) != cudaSuccess) { std::fprintf(stderr, "instantiate failed\n"); return 1; }
    std::fprintf(stderr, "OK captured+instantiated\n");
    std::fprintf(stderr, "launching graph...\n");
    const auto tl = std::chrono::steady_clock::now();
    if (cudaGraphLaunch(ex, cs) != cudaSuccess) { std::fprintf(stderr, "graph launch failed\n"); return 1; }
    std::fprintf(stderr, "launch returned in %.1f ms\n",
                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tl).count());
    const auto t0 = std::chrono::steady_clock::now();
    int last = 0;
    while (flag2[0] == 0) {
        const int el = (int) std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (el != last) { last = el; std::fprintf(stderr, "poll %ds flag=%u\n", el, flag2[0]); }
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(30)) {
            std::fprintf(stderr, "REPLAY HANG: publish never rang in 30s\n");
            return 2;
        }
    }
    cudaStreamSynchronize(cs);
    std::fprintf(stderr, "PROBE COMPLETE (captured replay)\n");
    return 0;
}
