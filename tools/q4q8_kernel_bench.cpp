// q4q8_kernel_bench — Strata SYCL port, Q4/Q8 kernel spike (milestone 3).
//
// Establishes the hardware-accelerated quant matmul path on Intel Arc (Xe2/BMG):
//   T1  Q8_0 GEMV, float-dequant baseline          vs CPU reference
//   T2  Q8_0 GEMV, packed int8 dot (dpct-style)    vs CPU reference
//   T3  Q4_0 GEMV, float-dequant baseline          vs CPU reference
//   T4  joint_matrix INT8 DPAS microtest (ones)    — hardware matrix path proof
//   T5  joint_matrix INT8 DPAS tiled GEMM          vs CPU int32 reference
//
// Findings (B60, 2026-10-03):
// - device int8 DPAS combos: M=8 N=16 K=32, s8/u8 x s8/u8 -> s32
// - ggml-sycl uses an EMULATED dp4a (unpack+scalar muls, = T2 style) and no
//   DPAS at all — T4/T5 are the untapped hardware path.
// - first kernel launch includes SPIR-V->native JIT; warmup reps excluded.
//
// Build: icpx -fsycl -O2 q4q8_kernel_bench.cpp -o q4q8_kernel_bench
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/sycl.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

namespace mx = sycl::ext::oneapi::experimental::matrix;

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

// device-callable: raw USM ptr -> global-space undecorated multi_ptr
template <typename T>
static sycl::multi_ptr<T, sycl::access::address_space::global_space,
                       sycl::access::decorated::no>
gmp(T* p) {
    return sycl::multi_ptr<T, sycl::access::address_space::global_space,
                           sycl::access::decorated::no>(
        sycl::detail::dynamic_address_cast<
            sycl::access::address_space::global_space, true>(p));
}

// dpct/ggml-sycl-style packed int8x4 dot (the EMULATED dp4a baseline)
static inline int32_t dot4x8(int w, int x, int32_t acc) {
    auto vw = sycl::vec<int, 1>(w).as<sycl::vec<int8_t, 4>>().convert<int>();
    auto vx = sycl::vec<int, 1>(x).as<sycl::vec<int8_t, 4>>().convert<int>();
    return acc + vw[0] * vx[0] + vw[1] * vx[1] + vw[2] * vx[2] + vw[3] * vx[3];
}

// ---------------- host quantization (ggml-compatible semantics) ----------
// Q8_0: 32-elem blocks, fp16 scale d, qs = round(x/d) clamp [-127,127]
// Q4_0: 32-elem blocks, fp16 scale d, q nibble = round(x/d)+8 clamp [0,15]
static void quantize_q8_0(const float* w, int nrows, int K,
                          std::vector<sycl::half>& d,
                          std::vector<int8_t>& qs) {
    const int KB = K / 32;
    d.resize((size_t)nrows * KB);
    qs.resize((size_t)nrows * KB * 32);
    for (int r = 0; r < nrows; r++)
        for (int b = 0; b < KB; b++) {
            const float* x = w + (size_t)r * K + b * 32;
            float amax = 0.f;
            for (int j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(x[j]));
            float scale = amax / -127.f;
            if (scale == 0.f) scale = 1.f;
            float inv = 1.f / scale;
            d[(size_t)r * KB + b] = sycl::half(scale);
            for (int j = 0; j < 32; j++) {
                int v = (int)nearbyintf(x[j] * inv);
                qs[((size_t)r * KB + b) * 32 + j] =
                    (int8_t)(v < -127 ? -127 : (v > 127 ? 127 : v));
            }
        }
}

static void quantize_q4_0(const float* w, int nrows, int K,
                          std::vector<sycl::half>& d,
                          std::vector<uint8_t>& qs) {
    const int KB = K / 32;
    d.resize((size_t)nrows * KB);
    qs.resize((size_t)nrows * KB * 16);
    for (int r = 0; r < nrows; r++)
        for (int b = 0; b < KB; b++) {
            const float* x = w + (size_t)r * K + b * 32;
            float amax = 0.f;
            for (int j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(x[j]));
            float scale = amax / -8.f;
            if (scale == 0.f) scale = 1.f;
            float inv = 1.f / scale;
            d[(size_t)r * KB + b] = sycl::half(scale);
            for (int j = 0; j < 16; j++) {
                auto cl = [&](float v) {
                    int q = (int)nearbyintf(v * inv) + 8;
                    return (uint8_t)(q < 0 ? 0 : (q > 15 ? 15 : q));
                };
                qs[((size_t)r * KB + b) * 16 + j] =
                    (uint8_t)(cl(x[j]) | (cl(x[j + 16]) << 4));
            }
        }
}

// ---------------- DPAS GEMM config -----------------------------------------
// device capability query (mx_caps): int8 = M(max)8 x N16 x K32, s8/u8 -> s32
constexpr size_t TM = 8, TN = 16, TK = 32;
using JA = mx::joint_matrix<sycl::sub_group, int8_t, mx::use::a, TM, TK,
                            mx::layout::row_major>;
using JB = mx::joint_matrix<sycl::sub_group, int8_t, mx::use::b, TK, TN,
                            mx::layout::row_major>;
using JC = mx::joint_matrix<sycl::sub_group, int32_t, mx::use::accumulator,
                            TM, TN, mx::layout::dynamic>;

int main() {
    sycl::queue q(sycl::gpu_selector_v,
                  sycl::property_list{sycl::property::queue::in_order{}});
    auto dev = q.get_device();
    printf("device: %s\n", dev.get_info<sycl::info::device::name>().c_str());

    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.f, 0.05f);

    const int N = 4096, K = 4096, KB = K / 32;
    std::vector<float> W((size_t)N * K), x(K);
    for (auto& v : W) v = nd(rng);
    for (auto& v : x) v = nd(rng);

    std::vector<sycl::half> d8, d4;
    std::vector<int8_t> q8;
    std::vector<uint8_t> q4;
    quantize_q8_0(W.data(), N, K, d8, q8);
    quantize_q4_0(W.data(), N, K, d4, q4);

    // CPU references (dequantized semantics = what the kernels must match)
    std::vector<float> y8(N, 0.f), y4(N, 0.f);
    for (int r = 0; r < N; r++)
        for (int b = 0; b < KB; b++) {
            float s8 = (float)d8[(size_t)r * KB + b];
            float s4 = (float)d4[(size_t)r * KB + b];
            const int8_t* w8 = &q8[((size_t)r * KB + b) * 32];
            const uint8_t* w4 = &q4[((size_t)r * KB + b) * 16];
            const float* xv = &x[b * 32];
            for (int j = 0; j < 32; j++) y8[r] += s8 * (float)w8[j] * xv[j];
            for (int j = 0; j < 16; j++) {
                y4[r] += s4 * (float)((int)(w4[j] & 0xF) - 8) * xv[j];
                y4[r] += s4 * (float)((int)(w4[j] >> 4) - 8) * xv[j + 16];
            }
        }

    // quantize x for the int8 path (Q8_1-style, float scale per 32-block)
    std::vector<int8_t> xq((size_t)KB * 32);
    std::vector<float> xd(KB);
    for (int b = 0; b < KB; b++) {
        float amax = 0.f;
        for (int j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(x[b * 32 + j]));
        float s = amax / -127.f;
        if (s == 0.f) s = 1.f;
        xd[b] = s;
        for (int j = 0; j < 32; j++) {
            int v = (int)nearbyintf(x[b * 32 + j] / s);
            xq[(size_t)b * 32 + j] =
                (int8_t)(v < -127 ? -127 : (v > 127 ? 127 : v));
        }
    }
    std::vector<float> y8i(N, 0.f);
    for (int r = 0; r < N; r++)
        for (int b = 0; b < KB; b++) {
            int32_t sum = 0;
            const int8_t* w8 = &q8[((size_t)r * KB + b) * 32];
            for (int j = 0; j < 32; j++)
                sum += (int32_t)w8[j] * xq[(size_t)b * 32 + j];
            y8i[r] += (float)d8[(size_t)r * KB + b] * xd[b] * (float)sum;
        }

    auto err = [](const std::vector<float>& a, const std::vector<float>& b) {
        double m = 0;
        for (size_t i = 0; i < a.size(); i++)
            m = fmax(m, fabs((double)a[i] - b[i]));
        return m;
    };

    int8_t* dq8 = sycl::malloc_device<int8_t>(q8.size(), q);
    int8_t* dxq = sycl::malloc_device<int8_t>(xq.size(), q);
    uint8_t* dq4 = sycl::malloc_device<uint8_t>(q4.size(), q);
    sycl::half* dd8 = sycl::malloc_device<sycl::half>(d8.size(), q);
    sycl::half* dd4 = sycl::malloc_device<sycl::half>(d4.size(), q);
    float* dxd = sycl::malloc_device<float>(xd.size(), q);
    float* dx = sycl::malloc_device<float>(K, q);
    float* dy = sycl::malloc_device<float>(N, q);
    q.memcpy(dq8, q8.data(), q8.size()).wait();
    q.memcpy(dxq, xq.data(), xq.size()).wait();
    q.memcpy(dq4, q4.data(), q4.size()).wait();
    q.memcpy(dd8, d8.data(), d8.size() * 2).wait();
    q.memcpy(dd4, d4.data(), d4.size() * 2).wait();
    q.memcpy(dxd, xd.data(), xd.size() * 4).wait();
    q.memcpy(dx, x.data(), K * 4).wait();

    const int REPS = 50;
    const double bytes8 = (double)q8.size() + d8.size() * 2;
    const double bytes4 = (double)q4.size() + d4.size() * 2;

    auto gemv_q8_float = [&](void) {
        q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> r) {
            float acc = 0.f;
            size_t rb = (size_t)r[0] * KB;
            for (int b = 0; b < KB; b++) {
                float s = (float)dd8[rb + b];
                const int8_t* w = dq8 + (rb + b) * 32;
                const float* xv = dx + b * 32;
#pragma unroll
                for (int j = 0; j < 32; j++) acc += s * (float)w[j] * xv[j];
            }
            dy[r[0]] = acc;
        }).wait();
    };
    auto gemv_q8_int = [&](void) {
        q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> r) {
            float acc = 0.f;
            size_t rb = (size_t)r[0] * KB;
            for (int b = 0; b < KB; b++) {
                const int* w = (const int*)(dq8 + (rb + b) * 32);
                const int* xv = (const int*)(dxq + (size_t)b * 32);
                int32_t sum = 0;
#pragma unroll
                for (int i = 0; i < 8; i++) sum = dot4x8(w[i], xv[i], sum);
                acc += (float)dd8[rb + b] * dxd[b] * (float)sum;
            }
            dy[r[0]] = acc;
        }).wait();
    };
    auto gemv_q4_float = [&](void) {
        q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> r) {
            float acc = 0.f;
            size_t rb = (size_t)r[0] * KB;
            for (int b = 0; b < KB; b++) {
                float s = (float)dd4[rb + b];
                const uint8_t* w = dq4 + (rb + b) * 16;
                const float* xv = dx + b * 32;
#pragma unroll
                for (int j = 0; j < 16; j++) {
                    acc += s * (float)((int)(w[j] & 0xF) - 8) * xv[j];
                    acc += s * (float)((int)(w[j] >> 4) - 8) * xv[j + 16];
                }
            }
            dy[r[0]] = acc;
        }).wait();
    };

    auto run = [&](const char* name, double bytes,
                   const std::function<void()>& kernel,
                   const std::vector<float>& ref) {
        kernel();  // warmup: first launch pays SPIR-V JIT, exclude it
        auto t0 = clk::now();
        for (int it = 0; it < REPS; it++) kernel();
        double t = ms_since(t0) / REPS;
        std::vector<float> y(N);
        q.memcpy(y.data(), dy, N * 4).wait();
        printf("%s: %.3f ms, %.1f GB/s, maxerr %.2e\n", name, t,
               bytes / t / 1e6, err(y, ref));
    };
    run("T1 Q8_0 gemv float-dequant ", bytes8, gemv_q8_float, y8);
    run("T2 Q8_0 gemv packed-int8   ", bytes8, gemv_q8_int, y8i);
    run("T3 Q4_0 gemv float-dequant ", bytes4, gemv_q4_float, y4);

    sycl::free(dq8, q); sycl::free(dxq, q); sycl::free(dq4, q);
    sycl::free(dd8, q); sycl::free(dd4, q); sycl::free(dxd, q);
    sycl::free(dx, q); sycl::free(dy, q);

    // ---- T4: DPAS microtest: A=1(8x32 s8), B=1(32x16 s8) => C=K ----
    {
        int32_t* dC = sycl::malloc_device<int32_t>(TM * TN, q);
        q.memset(dC, 0xFF, TM * TN * 4).wait();
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(16, 16), [=](sycl::nd_item<1> it) {
                auto sg = it.get_sub_group();
                JA a; JB b; JC c, d;
                mx::joint_matrix_fill(sg, a, (int8_t)1);
                mx::joint_matrix_fill(sg, b, (int8_t)1);
                mx::joint_matrix_fill(sg, c, 0);
                mx::joint_matrix_mad(sg, d, a, b, c);
                mx::joint_matrix_store(sg, d, gmp(dC), TN, mx::layout::row_major);
            });
        }).wait_and_throw();
        std::vector<int32_t> C(TM * TN);
        q.memcpy(C.data(), dC, TM * TN * 4).wait();
        bool ok = true;
        for (auto v : C) ok = ok && (v == (int32_t)TK);
        printf("T4 DPAS int8 microtest (ones, K=%zu): %s (C[0]=%d)\n", TK,
               ok ? "PASS" : "FAIL", C[0]);
        sycl::free(dC, q);
    }

    // ---- T5: DPAS tiled int8 GEMM vs CPU ----
    {
        const int M = 1024, Nn = 1024, Kk = 1024;
        std::vector<int8_t> A((size_t)M * Kk), B((size_t)Kk * Nn);
        std::uniform_int_distribution<int> ud(-8, 8);
        for (auto& v : A) v = (int8_t)ud(rng);
        for (auto& v : B) v = (int8_t)ud(rng);
        int8_t* dA = sycl::malloc_device<int8_t>(A.size(), q);
        int8_t* dB = sycl::malloc_device<int8_t>(B.size(), q);
        int32_t* dC = sycl::malloc_device<int32_t>((size_t)M * Nn, q);
        q.memcpy(dA, A.data(), A.size()).wait();
        q.memcpy(dB, B.data(), B.size()).wait();

        const size_t tiles = (M / TM) * (Nn / TN);
        const int Ntiles = Nn / TN;
        auto gemm = [&]() {
            q.submit([&](sycl::handler& h) {
                h.parallel_for(sycl::nd_range<1>(tiles * 16, 16),
                               [=](sycl::nd_item<1> it2) {
                    size_t tile = it2.get_group(0);
                    size_t tm = tile / Ntiles, tn = tile % Ntiles;
                    auto sg = it2.get_sub_group();
                    JC c1, c2;
                    mx::joint_matrix_fill(sg, c1, 0);
                    for (int k0 = 0; k0 < Kk; k0 += 2 * TK) {
                        JA a; JB b;
                        mx::joint_matrix_load(
                            sg, a, gmp(dA + (tm * TM) * Kk + k0), Kk);
                        mx::joint_matrix_load(
                            sg, b, gmp(dB + (size_t)k0 * Nn + tn * TN), Nn);
                        mx::joint_matrix_mad(sg, c2, a, b, c1);
                        mx::joint_matrix_load(
                            sg, a, gmp(dA + (tm * TM) * Kk + k0 + TK), Kk);
                        mx::joint_matrix_load(
                            sg, b, gmp(dB + (size_t)(k0 + TK) * Nn + tn * TN),
                            Nn);
                        mx::joint_matrix_mad(sg, c1, a, b, c2);
                    }
                    mx::joint_matrix_store(
                        sg, c1, gmp(dC + (tm * TM) * Nn + tn * TN), Nn,
                        mx::layout::row_major);
                });
            }).wait();
        };
        gemm();  // warmup (JIT)
        auto t0 = clk::now();
        const int GREPS = 20;
        for (int it = 0; it < GREPS; it++) gemm();
        double t = ms_since(t0) / GREPS;
        double tops = 2.0 * M * Nn * Kk / (t * 1e-3) / 1e12;

        std::vector<int32_t> C((size_t)M * Nn);
        q.memcpy(C.data(), dC, C.size() * 4).wait();
        bool ok = true;
        for (int s = 0; s < 128; s++) {
            int i = rng() % M, j = rng() % Nn;
            int32_t ref = 0;
            for (int k = 0; k < Kk; k++)
                ref += (int32_t)A[(size_t)i * Kk + k] * B[(size_t)k * Nn + j];
            if (C[(size_t)i * Nn + j] != ref) ok = false;
        }
        printf("T5 DPAS int8 GEMM %dx%dx%d: %.3f ms, %.2f TOPS, parity %s\n",
               M, Nn, Kk, t, tops, ok ? "PASS" : "FAIL");
        sycl::free(dA, q); sycl::free(dB, q); sycl::free(dC, q);
    }

    return 0;
}