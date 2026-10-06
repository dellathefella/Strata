// src/prefill/gemm_sycl.cpp — the batched projection GEMMs on oneMKL
// (SYCL milestone 5). Same maths as the cuBLAS calls in gemm.cu:
//   Y[T, N] (f32, row stride ldy) = X[T, K] . W[N, K]^T, beta accumulates,
// expressed column-major as C[N, T] = W[K, N]^T . X[K, T] with bf16/f16
// inputs and fp32 compute. `native` dequantizes the GGUF block weights into
// the fp16 scratch with the ported dequant_f16 and reuses the f16 GEMM, row
// slices and all, exactly as gemm.cu does.
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/prefill/gemm.hpp"

#include <cuda_runtime.h>  // sycl_compat shim

#include <oneapi/mkl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>

using BF16 = sycl::ext::oneapi::bfloat16;

#include <cstdio>
#include <cstdlib>

namespace strata::prefill {
namespace {

inline sycl::queue& Q(void* stream) { return strata::sycl_compat::q_for(stream); }

template <typename HT>
void gemm_tn(void* stream, const HT* X, const HT* W, float* Y, int64_t Tn, int64_t N, int64_t K, int64_t ldy,
             float beta) {
    const float alpha = 1.0f;
    oneapi::mkl::blas::column_major::gemm(Q(stream), oneapi::mkl::transpose::trans,
                                          oneapi::mkl::transpose::nontrans, N, Tn, K, alpha, W, K, X, K, beta, Y,
                                          ldy);
}

}  // namespace

Gemm::~Gemm() {
    if (scratch_ && !external_) cudaFree(scratch_);
    if (workspace_ && !external_) cudaFree(workspace_);
}

bool Gemm::init(void* stream, int64_t scratch_elems, std::string& err) {
    stream_ = stream;
    scratch_elems_ = scratch_elems;
    if (scratch_elems > 0 && cudaMalloc(&scratch_, (size_t) scratch_elems * sizeof(uint16_t)) != cudaSuccess) {
        err = "SYCL backend: prefill scratch allocation failed";
        return false;
    }
    if (cudaMalloc(&workspace_, 4u << 20) != cudaSuccess) {
        err = "SYCL backend: prefill workspace allocation failed";
        return false;
    }
    return true;
}

bool Gemm::init_external(void* stream, uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes,
                         std::string& err) {
    stream_ = stream;
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
    workspace_ = workspace;
    external_ = true;
    (void) ws_bytes;
    (void) err;
    return true;
}

void Gemm::rebind(uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes) {
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
    workspace_ = workspace;
    (void) ws_bytes;
}

void Gemm::bf16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                float beta) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    gemm_tn<BF16>(stream_, reinterpret_cast<const BF16*>(X), reinterpret_cast<const BF16*>(W), Y, T, N, K,
                        ldy, beta);
}

void Gemm::f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
               float beta) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    gemm_tn<sycl::half>(stream_, reinterpret_cast<const sycl::half*>(X), reinterpret_cast<const sycl::half*>(W),
                        Y, T, N, K, ldy, beta);
}

void Gemm::native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                  int64_t ldy, float beta) {
    if (N * K > scratch_elems_) {
        const int64_t rows = scratch_elems_ / K;
        if (rows <= 0) {
            std::fprintf(stderr, "prefill gemm: scratch too small for K=%lld\n", (long long) K);
            std::exit(1);
        }
        if (ldy <= 0) ldy = N;
        for (int64_t r0 = 0; r0 < N; r0 += rows) {
            const int64_t n = (N - r0 < rows) ? N - r0 : rows;
            strata::kernels::dequant_f16(ggml_type, W_blocks, r0, n, K, scratch_, stream_);
            f16(X, scratch_, Y + r0, T, n, K, ldy, beta);
        }
        return;
    }
    strata::kernels::dequant_f16(ggml_type, W_blocks, 0, N, K, scratch_, stream_);
    f16(X, scratch_, Y, T, N, K, ldy, beta);
}

}  // namespace strata::prefill
