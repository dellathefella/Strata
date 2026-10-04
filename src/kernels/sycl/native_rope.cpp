// src/kernels/sycl/native_rope.cpp — SYCL port of src/kernels/cuda/native_rope.cu.
// Numerical contract unchanged (pinned ggml rope_multi/rope_yarn, MIT ggml
// attribution applies to the transcribed arithmetic). Includes the per-device
// mrope/rope-table registries (single device 0 under sycl_compat for now).
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/mrope.hpp"

#include <cuda_runtime.h>  // sycl_compat shim

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace strata::kernels {
namespace {

std::atomic<bool> enabled{false};

bool overlaps(const void* a, size_t an, const void* b, size_t bn) {
    auto x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
    return x <= y ? y - x < an : x - y < bn;
}

// TAB (#280): angles from the session's float64 table; the host launches the
// <false> instantiation whenever no table applies, exactly as the CUDA port.
template <bool TAB>
void launch_apply(sycl::queue& q, const float* x, float* out, int rows, int width, int n_rot,
                  float theta_scale, float freq_scale, float corr_low, float corr_high,
                  float ext_factor, float mscale, const int* positions, const int32_t* mtab,
                  RopeTab rt) {
    // CUDA grid was dim3(ceil(width/2/128), rows), block 128, pair = bx*128+tx, row = by
    const size_t pair_blocks = ((size_t)width / 2 + 127) / 128;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t)rows, pair_blocks * 128),
                                         sycl::range<2>(1, 128)),
                       [=](sycl::nd_item<2> it) {
                           const int row = (int)it.get_group(0);
                           const int pair = (int)(it.get_group(1) * 128 + it.get_local_id(1));
                           if (row >= rows || pair >= width / 2) return;
                           const size_t start = (size_t)row * width;
                           if (pair >= n_rot / 2) {
                               if (x != out) {
                                   out[start + 2 * pair] = x[start + 2 * pair];
                                   out[start + 2 * pair + 1] = x[start + 2 * pair + 1];
                               }
                               return;
                           }
                           float c, s;
                           if (!(TAB &&
                                 rope_tab_cs(rt, mrope_pos(mtab, positions[row], pair), pair, c, s))) {
                               const float theta_extrap =
                                   mrope_pos(mtab, positions[row], pair) *
                                   sycl::powr(theta_scale, (float)pair);
                               rope_scaled_angle(theta_extrap, freq_scale, corr_low, corr_high,
                                                 ext_factor, mscale, pair, c, s);
                           }
                           const float a = x[start + pair], b = x[start + pair + n_rot / 2];
                           out[start + pair] = a * c - b * s;
                           out[start + pair + n_rot / 2] = a * s + b * c;
                       });
    });
}

// one table per device (a layer split runs the rope kernels on several)
constexpr int kMropeDevices = 64;
std::atomic<const int32_t*> mrope_tab[kMropeDevices] = {};
int mrope_dev() {
    int d = 0;
    if (cudaGetDevice(&d) != cudaSuccess || d < 0 || d >= kMropeDevices) d = 0;
    return d;
}

// #280: the float64 angle table of each device, set at session init
struct RopeReg {
    RopeTab tab;
    RopeScaling scaling;
};
RopeReg rope_tab[kMropeDevices] = {};

bool same_scaling(const RopeScaling& a, const RopeScaling& b) {
    return a.type == b.type && a.freq_base == b.freq_base && a.factor == b.factor &&
           a.freq_scale_in == b.freq_scale_in && a.orig_ctx == b.orig_ctx &&
           a.ext_factor == b.ext_factor && a.attn_factor == b.attn_factor &&
           a.beta_fast == b.beta_fast && a.beta_slow == b.beta_slow;
}

bool rope_table_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_ROPE_TABLE");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}

}  // namespace

void mrope_table_set(const int32_t* device_table) {
    mrope_tab[mrope_dev()].store(device_table, std::memory_order_relaxed);
}
const int32_t* mrope_table() { return mrope_tab[mrope_dev()].load(std::memory_order_relaxed); }

void rope_table_set(const float* cos_tab, const float* sin_tab, int max_pos,
                    const RopeScaling& scaling) {
    rope_tab[mrope_dev()] = RopeReg{RopeTab{cos_tab, sin_tab, max_pos}, scaling};
}

void rope_table_release(const float* cos_tab) {
    for (RopeReg& r : rope_tab)
        if (cos_tab != nullptr && r.tab.cos == cos_tab) r = RopeReg{};
}

RopeTab rope_table_for(const RopeScaling& scaling) {
    if (!rope_table_enabled()) return {};
    const RopeReg& r = rope_tab[mrope_dev()];
    return r.tab.cos != nullptr && same_scaling(r.scaling, scaling) ? r.tab : RopeTab{};
}

void native_rope_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_rope_enabled() { return enabled.load(std::memory_order_relaxed); }

void native_rope_apply(const float* x, float* out, int rows, int head_dim, int n_rot,
                       const RopeScaling& scaling, const int* positions, void* stream) {
    if (!x || !out || !positions || !stream || rows < 1 || rows > 65535 ||
        (head_dim != 128 && head_dim != 256) || n_rot != 64 ||
        rope_scaling_invalid(scaling) != nullptr ||
        reinterpret_cast<uintptr_t>(x) % 4 || reinterpret_cast<uintptr_t>(out) % 4 ||
        reinterpret_cast<uintptr_t>(positions) % 4) {
        throw std::invalid_argument(
            "native RoPE requires aligned F32 rows, width 128/256, rotation 64, valid "
            "base/scaling and explicit stream");
    }
    const size_t bytes = (size_t)rows * head_dim * sizeof(float);
    if ((x != out && overlaps(x, bytes, out, bytes)) ||
        overlaps(positions, (size_t)rows * sizeof(int), out, bytes) ||
        overlaps(positions, (size_t)rows * sizeof(int), x, bytes)) {
        throw std::invalid_argument("native RoPE buffers partially overlap");
    }
    // Match pinned host-side float powf before device powf/trigonometry.
    const float theta_scale = std::pow((float)scaling.freq_base, -2.0f / n_rot);
    const RopeKernelArgs k = scaling.kernel_args(n_rot);
    const RopeTab rt = rope_table_for(scaling);
    auto& q = strata::sycl_compat::q_for(stream);
    try {
        if (rt.cos != nullptr)
            launch_apply<true>(q, x, out, rows, head_dim, n_rot, theta_scale, k.freq_scale,
                               k.corr_low, k.corr_high, k.ext_factor, k.attn_factor, positions,
                               mrope_table(), rt);
        else
            launch_apply<false>(q, x, out, rows, head_dim, n_rot, theta_scale, k.freq_scale,
                                k.corr_low, k.corr_high, k.ext_factor, k.attn_factor, positions,
                                mrope_table(), rt);
    } catch (const sycl::exception& e) {
        throw std::runtime_error(e.what());
    }
}

}  // namespace strata::kernels
