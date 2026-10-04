// src/kernels/sycl/rope.cpp — SYCL port of src/kernels/cuda/rope.cu.
// NEOX partial RoPE; the host-side float64 table builders are verbatim (pure
// C++), the kernel is one work-item per row (same as CUDA: one thread per row,
// which is what makes the tail copy race-free).
#include "strata/kernels/rope.hpp"
#include "strata/kernels/mrope.hpp"

#include <cuda_runtime.h>  // sycl_compat shim

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

namespace {
// The process's rope config (rope_scaling.hpp). One writer — the engine's
// startup thread — and readers after it.
RopeScaling g_rope_scaling;
}  // namespace

void rope_scaling_set(const RopeScaling& scaling) { g_rope_scaling = scaling; }
const RopeScaling& rope_scaling() { return g_rope_scaling; }

void build_rope_table(int n_rot, double theta, int max_pos, float* cos_tab, float* sin_tab) {
    const int half = n_rot / 2;
    for (int p = 0; p < max_pos; ++p) {
        for (int i = 0; i < half; ++i) {
            // float64 throughout, in the reference's order: inv, then ang, then cos/sin
            const double inv = std::pow(theta, -2.0 * (double)i / (double)n_rot);
            const double ang = (double)p * inv;
            cos_tab[(size_t)p * half + i] = (float)std::cos(ang);
            sin_tab[(size_t)p * half + i] = (float)std::sin(ang);
        }
    }
}

void build_rope_table(int n_rot, const RopeScaling& sc, int max_pos, float* cos_tab, float* sin_tab) {
    if (sc.type == RopeScalingType::None) {
        build_rope_table(n_rot, sc.freq_base, max_pos, cos_tab, sin_tab);  // verbatim
        return;
    }
    const int half = n_rot / 2;
    const double fs = sc.freq_scale();
    const double ms = sc.mscale();
    double cd[2];
    sc.corr_dims(n_rot, cd);
    const bool correct = sc.ext_factor != 0;
    for (int p = 0; p < max_pos; ++p) {
        for (int i = 0; i < half; ++i) {
            const double inv = std::pow(sc.freq_base, -2.0 * (double)i / (double)n_rot);
            const double extrap = (double)p * inv;
            const double interp = fs * extrap;
            double ang = interp;
            if (correct) {
                const double ramp =
                    (double)rope_yarn_ramp((float)cd[0], (float)cd[1], i) * sc.ext_factor;
                ang = interp * (1.0 - ramp) + extrap * ramp;
            }
            cos_tab[(size_t)p * half + i] = (float)(std::cos(ang) * ms);
            sin_tab[(size_t)p * half + i] = (float)(std::sin(ang) * ms);
        }
    }
}

void rope_neox_apply(const float* x, float* out, int64_t rows, int head_dim, int n_rot,
                     const float* cos_tab, const float* sin_tab, const int* pos, void* stream) {
    if (rows <= 0 || n_rot <= 0) return;
    if (n_rot % 2 != 0 || n_rot > head_dim) {
        std::fprintf(stderr, "rope_neox_apply: n_rot %d must be even and <= head_dim %d\n", n_rot,
                     head_dim);
        std::exit(1);
    }
    const int32_t* mtab = mrope_table();
    auto& q = strata::sycl_compat::q_for(stream);
    try {
        q.parallel_for(sycl::range<1>((size_t)rows), [=](sycl::id<1> rid) {
            const long long r = rid[0];
            const int half = n_rot / 2;
            const float* xr = x + r * head_dim;
            float* orow = out + r * head_dim;

            for (int d = n_rot; d < head_dim; ++d) orow[d] = xr[d];  // untouched tail
            for (int i = 0; i < half; ++i) {
                const size_t toff = (size_t)mrope_pos(mtab, pos[r], i) * half;
                rope_neox_pair(xr[i], xr[half + i], cos_tab[toff + i], sin_tab[toff + i], orow[i],
                               orow[half + i]);
            }
        });
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "rope_neox_apply launch: %s\n", e.what());
        std::exit(1);
    }
    if (stream == nullptr) cudaDeviceSynchronize();
}

}  // namespace strata::kernels
