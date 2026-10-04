// src/kernels/sycl/gdn.cpp — SYCL port of src/kernels/cuda/gdn.cu (gated delta
// net: state step, causal conv step, L2 norm, beta gate, output norm).
// Warp butterflies (double sums — ORDER-DEPENDENT) become per-32-lane-chunk
// local-memory trees with the identical 16,8,4,2,1 pairing. expf/log1pf ->
// sycl::; the decay-before-update property and modulo head pairing are verbatim.
#include "strata/kernels/gdn.hpp"

#include <cuda_runtime.h>  // sycl_compat shim

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int JTHREADS = 32;  // threads along j, the state's fast axis
constexpr int MAX_H = 1;      // heads staged in local memory per group

inline float softplus_f(float x) { return x > 20.0f ? x : sycl::log1p(sycl::exp(x)); }
inline float sigmoid_f(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }

void sync_err(void* stream, const char* what) {
    if (stream != nullptr) return;
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace

void gdn_step(float* state, const float* q, const float* k, const float* v, const float* gate,
              const float* beta, float* o, const GdnShapes& s, void* stream) {
    if (s.S <= 0 || s.h_k <= 0 || s.h_v <= 0) return;
    if (s.S > 128) {
        std::fprintf(stderr, "gdn_step: S = %lld exceeds the staged 128 (`ks`/`qs` are [1][128])\n",
                     (long long)s.S);
        std::exit(1);
    }
    const int S = (int)s.S, h_k = (int)s.h_k, h_v = (int)s.h_v;
    const size_t gx = (size_t)((S + JTHREADS - 1) / JTHREADS);
    const size_t gy = (size_t)((h_v + MAX_H - 1) / MAX_H);
    auto& q_ = strata::sycl_compat::q_for(stream);
    try {
        q_.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 2> ks(sycl::range<2>(MAX_H, 128), h);
            sycl::local_accessor<float, 2> qs(sycl::range<2>(MAX_H, 128), h);
            sycl::local_accessor<float, 1> dec_s(sycl::range<1>(MAX_H), h);
            sycl::local_accessor<float, 1> beta_s(sycl::range<1>(MAX_H), h);
            h.parallel_for(sycl::nd_range<2>(sycl::range<2>(gx, gy * JTHREADS),
                                             sycl::range<2>(1, JTHREADS)),
                           [=](sycl::nd_item<2> it) {
                const int h0 = (int)it.get_group(1) * MAX_H;
                const int nh = MAX_H < (h_v - h0) ? MAX_H : (h_v - h0);
                const int j = (int)(it.get_group(0) * JTHREADS + it.get_local_id(1));
                const int tid = (int)it.get_local_id(1);

                for (int hi = 0; hi < nh; ++hi) {
                    const int hh = h0 + hi;
                    const int src = hh % h_k;  // MODULO head pairing
                    for (int i = tid; i < S; i += JTHREADS) {
                        ks[hi][i] = k[(size_t)src * S + i];
                        qs[hi][i] = q[(size_t)src * S + i];
                    }
                    if (tid == 0) {
                        dec_s[hi] = sycl::exp(gate[hh]);
                        beta_s[hi] = beta[hh];
                    }
                }
                it.barrier();
                if (j >= S) return;

                for (int hi = 0; hi < nh; ++hi) {
                    const int hh = h0 + hi;
                    const float dec = dec_s[hi], b = beta_s[hi];
                    float* col = state + (size_t)hh * S + j;  // (S, h_v, S) with j fastest
                    const size_t stride = (size_t)h_v * S;

                    // pass 1: decay BEFORE the update — PROPERTY 5
                    float sk = 0.0f;
                    for (int i = 0; i < S; ++i) {
                        const float sv = col[(size_t)i * stride] * dec;
                        col[(size_t)i * stride] = sv;
                        sk += sv * ks[hi][i];
                    }
                    const float d = (v[(size_t)hh * S + j] - sk) * b;
                    // pass 2: rank-1 update, then read out against q with the UPDATED state
                    float dot = 0.0f;
                    for (int i = 0; i < S; ++i) {
                        const float sv = col[(size_t)i * stride] + ks[hi][i] * d;
                        col[(size_t)i * stride] = sv;
                        dot += sv * qs[hi][i];
                    }
                    o[(size_t)hh * S + j] = dot;
                }
            });
        });
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "gdn_step: %s\n", e.what());
        std::exit(1);
    }
    sync_err(stream, "gdn_step");
}

void gdn_conv_step(float* conv_state, const float* x, const float* kW, float* out, int64_t channels,
                   int64_t d_conv, void* stream) {
    if (channels <= 0 || d_conv < 1) return;
    const int C = (int)channels, dc = (int)d_conv;
    auto& q = strata::sycl_compat::q_for(stream);
    try {
        q.parallel_for(sycl::range<1>((size_t)C), [=](sycl::id<1> cid) {
            const int c = (int)cid[0];
            float* st = conv_state + (size_t)c * (dc - 1);
            const float* w = kW + (size_t)c * dc;
            float acc = 0.0f;
            for (int i = 0; i < dc - 1; ++i) acc += st[i] * w[i];  // kernel[0] reads the OLDEST row
            acc += x[c] * w[dc - 1];                               // new input lands in the LAST tap
            out[c] = acc;
            for (int i = 0; i < dc - 2; ++i) st[i] = st[i + 1];    // slide
            st[dc - 2] = x[c];
        });
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "gdn_conv_step: %s\n", e.what());
        std::exit(1);
    }
    sync_err(stream, "gdn_conv_step");
}

void gdn_l2_norm(float* x, int64_t rows, int64_t cols, float eps, void* stream) {
    if (rows <= 0 || cols <= 0) return;
    if (cols > 1024) {
        std::fprintf(stderr, "gdn_l2_norm: cols = %lld exceeds the 1024-wide reduction\n",
                     (long long)cols);
        std::exit(1);
    }
    const int ncols = (int)cols;
    auto& q = strata::sycl_compat::q_for(stream);
    try {
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<double, 1> red(sycl::range<1>(32), h);
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t)rows * 32),
                                             sycl::range<1>(32)),
                           [=](sycl::nd_item<1> it) {
                const int row = (int)it.get_group(0);
                const int lane = (int)it.get_local_id(0);
                float* p = x + (size_t)row * ncols;
                double acc = 0.0;
                for (int i = lane; i < ncols; i += 32) acc += (double)p[i] * (double)p[i];
                red[lane] = acc;
                for (int off = 16; off > 0; off >>= 1) {
                    it.barrier();
                    if (lane < off) red[lane] += red[lane + off];
                }
                // lane 0 holds the sum after its own last combine; broadcast via red[0]
                it.barrier();
                const float inv = (float)(1.0 / sycl::sqrt(red[0] + (double)eps));
                for (int i = lane; i < ncols; i += 32) p[i] *= inv;
            });
        });
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "gdn_l2_norm: %s\n", e.what());
        std::exit(1);
    }
    sync_err(stream, "gdn_l2_norm");
}

void gdn_beta_gate(float* beta, int64_t h_v, void* stream) {
    if (beta == nullptr || h_v <= 0) return;
    const int n = (int)h_v;
    auto& q = strata::sycl_compat::q_for(stream);
    try {
        q.parallel_for(sycl::range<1>((size_t)n),
                       [=](sycl::id<1> i) { beta[i[0]] = sigmoid_f(beta[i[0]]); });
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "gdn_beta_gate: %s\n", e.what());
        std::exit(1);
    }
    sync_err(stream, "gdn_beta_gate");
}

void gdn_out_norm(const float* o, const float* z, const float* ssm_norm, float* y, int64_t h_v,
                  int64_t S, float eps, void* stream) {
    if (h_v <= 0 || S <= 0) return;
    const int nS = (int)S;
    auto& q = strata::sycl_compat::q_for(stream);
    try {
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<double, 1> red(sycl::range<1>(32), h);
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t)h_v * 32),
                                             sycl::range<1>(32)),
                           [=](sycl::nd_item<1> it) {
                const int hh = (int)it.get_group(0);
                const int lane = (int)it.get_local_id(0);
                const float* po = o + (size_t)hh * nS;
                const float* pz = z + (size_t)hh * nS;
                float* py = y + (size_t)hh * nS;
                double acc = 0.0;
                for (int i = lane; i < nS; i += 32) acc += (double)po[i] * (double)po[i];
                red[lane] = acc;
                for (int off = 16; off > 0; off >>= 1) {
                    it.barrier();
                    if (lane < off) red[lane] += red[lane + off];
                }
                it.barrier();
                const float inv = (float)(1.0 / sycl::sqrt(red[0] / (double)nS + (double)eps));
                for (int i = lane; i < nS; i += 32)
                    py[i] = po[i] * inv * ssm_norm[i] * sigmoid_f(pz[i]);
            });
        });
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "gdn_out_norm: %s\n", e.what());
        std::exit(1);
    }
    sync_err(stream, "gdn_out_norm");
}

}  // namespace strata::kernels
