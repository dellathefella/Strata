// src/kernels/sycl/native_mmvq.cpp — SYCL port of the 32-element ("small")
// formats of src/kernels/cuda/native_mmvq.cu: Q8_1 activation quantization and
// the Q4_0/Q5_0/Q8_0/IQ4_NL mat-vecs, single- and multi-column, in the PINNED
// exact layout (NW=4 warps, ROWS=1 or 4). The K/IQ4-XS formats throw until
// their ports land; the Q8_0 native pack does not touch them.
//
// The CUDA kernel's cross-warp finish (warp 0 adds partial[] then warp-sums)
// is computed REDUNDANTLY by every warp here, because SYCL barriers are
// work-group-wide: partial0 holds warp 0's own partial, every warp forms
// v[lane] = partial0 + sum(partial) and runs the identical xor tree, and only
// warp 0's lane i writes row i. Same values, same order, same result.
//
// Block layouts and dot expressions are transcribed verbatim (ggml-derived,
// MIT, see the CUDA file): load_int_b2 two-byte packs, STRATA_DP4A signed-byte
// dots, the affine Q4_0/Q5_0 corrections against the Q8_1 input sum.
#include "strata/kernels/dp4a.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>  // sycl_compat shim

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

constexpr int QK = 256;
constexpr int Q8K = 32;
constexpr int WARPS = 4;
constexpr int WARP = 32;
constexpr int QUANT_THREADS = 256;
constexpr int MAX_NCOLS = 8;

using sycl::local_accessor;
using sycl::nd_item;

inline sycl::queue& Q(void* stream) { return strata::sycl_compat::q_for(stream); }

struct Q81Block {
    __half2 ds;
    int8_t qs[32];
};
struct Q40Block {
    __half d;
    uint8_t qs[16];
};
struct Q50Block {
    __half d;
    uint8_t qh[4];
    uint8_t qs[16];
};
struct Q80Block {
    __half d;
    int8_t qs[32];
};
struct IQ4NLBlock {
    __half d;
    uint8_t qs[16];
};
static_assert(sizeof(Q81Block) == 36 && sizeof(Q40Block) == 18 && sizeof(Q50Block) == 22 &&
              sizeof(Q80Block) == 34 && sizeof(IQ4NLBlock) == 18, "block layouts are pinned");

bool g_multi_exact = true;

inline int load_int_b2(const void* ptr, int i32) {
    const auto* x = static_cast<const uint16_t*>(ptr);
    int value = x[2 * i32] << 0;
    value |= x[2 * i32 + 1] << 16;
    return value;
}

// CUDA __byte_perm: byte i of the result is byte (sel>>4i)&7 of {b,a}
inline uint32_t byte_perm(uint32_t a, uint32_t b, uint32_t sel) {
    uint32_t r = 0;
    for (int i = 0; i < 4; ++i) {
        const uint32_t which = (sel >> (4 * i)) & 0x7;
        const uint32_t src = which < 4 ? a : b;
        const uint32_t byte = (src >> (8 * (which & 3))) & 0xffu;
        r |= byte << (8 * i);
    }
    return r;
}

constexpr int8_t kIq4nlValues[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                     1,    13,   25,  38,  53,  69,  89,  113};

struct Int2 {
    int x, y;
};
inline Int2 iq4_table_lookup(int q4) {
    uint32_t table32[4];
    for (int i = 0; i < 4; ++i) {
        const int8_t* b = kIq4nlValues + i * 4;
        table32[i] = (uint32_t) (uint8_t) b[0] | ((uint32_t) (uint8_t) b[1] << 8) |
                     ((uint32_t) (uint8_t) b[2] << 16) | ((uint32_t) (uint8_t) b[3] << 24);
    }
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = 0x32103210 | ((q4 & 0x88888888) >> 1);
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low = byte_perm(table32[0], table32[1], (uint32_t) q4 >> shift);
        const uint32_t high = byte_perm(table32[2], table32[3], (uint32_t) q4 >> shift);
        tmp[i] = byte_perm(low, high, low_high_selection_indices >> shift);
    }
    return Int2{(int) byte_perm(tmp[0], tmp[1], 0x6420), (int) byte_perm(tmp[0], tmp[1], 0x7531)};
}

inline float small_q8_dot(const Q40Block* w, const Q81Block* x, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v = load_int_b2(w->qs, iqs + i);
        const int vi0 = (v >> 0) & 0x0f0f0f0f;
        const int vi1 = (v >> 4) & 0x0f0f0f0f;
        sumi = STRATA_DP4A(vi0, reinterpret_cast<const int*>(x->qs)[iqs + i], sumi);
        sumi = STRATA_DP4A(vi1, reinterpret_cast<const int*>(x->qs)[iqs + i + 4], sumi);
    }
    const sycl::float2 ds = __half22float2(x->ds);
    const float d = (float) w->d;
    return d * (sumi * ds.x() - 4 * ds.y());
}

inline float small_q8_dot(const Q50Block* w, const Q81Block* x, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vl = load_int_b2(w->qs, iqs + i);
        const int vh = load_int_b2(w->qh, 0) >> (4 * (iqs + i));
        int vi0 = (vl >> 0) & 0x0f0f0f0f;
        vi0 |= (vh << 4) & 0x00000010;
        vi0 |= (vh << 11) & 0x00001000;
        vi0 |= (vh << 18) & 0x00100000;
        vi0 |= (vh << 25) & 0x10000000;
        sumi = STRATA_DP4A(vi0, reinterpret_cast<const int*>(x->qs)[iqs + i], sumi);
        int vi1 = (vl >> 4) & 0x0f0f0f0f;
        vi1 |= (vh >> 12) & 0x00000010;
        vi1 |= (vh >> 5) & 0x00001000;
        vi1 |= (vh << 2) & 0x00100000;
        vi1 |= (vh << 9) & 0x10000000;
        sumi = STRATA_DP4A(vi1, reinterpret_cast<const int*>(x->qs)[iqs + i + 4], sumi);
    }
    const sycl::float2 ds = __half22float2(x->ds);
    const float d = (float) w->d;
    return d * (sumi * ds.x() - 8 * ds.y());
}

inline float small_q8_dot(const Q80Block* w, const Q81Block* x, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v = load_int_b2(w->qs, iqs + i);
        const int u = reinterpret_cast<const int*>(x->qs)[iqs + i];
        sumi = STRATA_DP4A(v, u, sumi);
    }
    const float d0 = (float) w->d;
    const float d1 = __low2float(x->ds);
    return d0 * d1 * float(sumi);
}

inline float small_q8_dot(const IQ4NLBlock* w, const Q81Block* x, int iqs) {
    const int* q8 = reinterpret_cast<const int*>(x->qs) + iqs;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const Int2 v = iq4_table_lookup(load_int_b2(w->qs, iqs + i));
        sumi = STRATA_DP4A(v.x, q8[i], sumi);
        sumi = STRATA_DP4A(v.y, q8[i + 4], sumi);
    }
    const float d = __half2float(w->d) * __low2float(x->ds);
    return d * sumi;
}

// the pinned exact layout, NCOLS columns (1 = the single-column kernel)
template <typename Weight, int Qi, int NCOLS, int ROWS>
void small_mmvq_launch(const Weight* w, const Q81Block* x, float* y, int n_in, int n_out, void* stream) {
    constexpr int BPI = 2 * WARPS * WARP / Qi;
    const int blocks_per_row = n_in / 32;
    const int x_stride = n_in / Q8K;
    const unsigned blocks = ROWS == 1 ? (unsigned) n_out : (unsigned) ((size_t) n_out + WARPS - 1) / WARPS;
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<float, 3> partial(sycl::range<3>(WARPS, NCOLS * ROWS, WARP), hnd);
        local_accessor<float, 2> partial0(sycl::range<2>(NCOLS * ROWS, WARP), hnd);
        local_accessor<float, 1> red(sycl::range<1>(WARPS * WARP), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) blocks * WARPS * WARP),
                                           sycl::range<1>(WARPS * WARP)),
                         [=](nd_item<1> it) {
                             const int tid = (int) it.get_local_id(0);
                             const int warp = tid >> 5, lane = tid & 31;
                             const int row0 = ROWS * (int) it.get_group(0);
                             float tmp[NCOLS][ROWS];
#pragma unroll
                             for (int j = 0; j < NCOLS; ++j)
#pragma unroll
                                 for (int i = 0; i < ROWS; ++i) tmp[j][i] = 0.0f;
                             for (int kbx = tid / (Qi / 2); kbx < blocks_per_row; kbx += BPI) {
                                 const int kqs = 2 * (tid % (Qi / 2));
#pragma unroll
                                 for (int i = 0; i < ROWS; ++i) {
                                     if (row0 + i >= n_out) continue;
                                     const size_t block = (size_t) (row0 + i) * blocks_per_row + kbx;
#pragma unroll
                                     for (int j = 0; j < NCOLS; ++j)
                                         tmp[j][i] += small_q8_dot(w + block, x + (size_t) j * x_stride + kbx, kqs);
                                 }
                             }
                             if (warp == 0) {
#pragma unroll
                                 for (int j = 0; j < NCOLS; ++j)
#pragma unroll
                                     for (int i = 0; i < ROWS; ++i) partial0[j * ROWS + i][lane] = tmp[j][i];
                             } else {
#pragma unroll
                                 for (int j = 0; j < NCOLS; ++j)
#pragma unroll
                                     for (int i = 0; i < ROWS; ++i)
                                         partial[warp - 1][j * ROWS + i][lane] = tmp[j][i];
                             }
                             it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
                             for (int j = 0; j < NCOLS; ++j) {
#pragma unroll
                                 for (int i = 0; i < ROWS; ++i) {
                                     float v = partial0[j * ROWS + i][lane];
#pragma unroll
                                     for (int l = 0; l < WARPS - 1; ++l) v += partial[l][j * ROWS + i][lane];
                                     // the warp-0 xor tree, run redundantly by every warp
                                     red[tid] = v;
                                     it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
                                     for (int o = 16; o > 0; o >>= 1) {
                                         const float other = red[tid ^ o];
                                         it.barrier(sycl::access::fence_space::local_space);
                                         v += other;
                                         red[tid] = v;
                                         it.barrier(sycl::access::fence_space::local_space);
                                     }
                                     if (warp == 0 && lane == i && row0 + i < n_out)
                                         y[(size_t) j * n_out + row0 + i] = v;
                                 }
                             }
                         });
    });
}

template <typename Weight, int Qi>
void small_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    if (n_in % 32 != 0 || n_in < 32) throw std::invalid_argument("native MMVQ requires n_in a multiple of 32");
    if (ncols < 1 || ncols > MAX_NCOLS) throw std::invalid_argument("native MMVQ requires 1 <= ncols <= 8");
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    if (!weights || !x_q8_1 || !y || !stream) throw std::invalid_argument("native MMVQ requires valid pointers");
    const auto* w = static_cast<const Weight*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const bool small_k = n_in / 32 < 2 * WARPS * WARP / Qi;
    switch (ncols) {
        case 1:
            if (small_k) small_mmvq_launch<Weight, Qi, 1, WARPS>(w, x, y, n_in, n_out, stream);
            else small_mmvq_launch<Weight, Qi, 1, 1>(w, x, y, n_in, n_out, stream);
            break;
        case 2:
            if (small_k) small_mmvq_launch<Weight, Qi, 2, WARPS>(w, x, y, n_in, n_out, stream);
            else small_mmvq_launch<Weight, Qi, 2, 1>(w, x, y, n_in, n_out, stream);
            break;
        case 3:
            if (small_k) small_mmvq_launch<Weight, Qi, 3, WARPS>(w, x, y, n_in, n_out, stream);
            else small_mmvq_launch<Weight, Qi, 3, 1>(w, x, y, n_in, n_out, stream);
            break;
        case 4:
            if (small_k) small_mmvq_launch<Weight, Qi, 4, WARPS>(w, x, y, n_in, n_out, stream);
            else small_mmvq_launch<Weight, Qi, 4, 1>(w, x, y, n_in, n_out, stream);
            break;
        case 5:
            if (small_k) small_mmvq_launch<Weight, Qi, 5, WARPS>(w, x, y, n_in, n_out, stream);
            else small_mmvq_launch<Weight, Qi, 5, 1>(w, x, y, n_in, n_out, stream);
            break;
        case 6:
            if (small_k) small_mmvq_launch<Weight, Qi, 6, WARPS>(w, x, y, n_in, n_out, stream);
            else small_mmvq_launch<Weight, Qi, 6, 1>(w, x, y, n_in, n_out, stream);
            break;
        case 7:
            if (small_k) small_mmvq_launch<Weight, Qi, 7, WARPS>(w, x, y, n_in, n_out, stream);
            else small_mmvq_launch<Weight, Qi, 7, 1>(w, x, y, n_in, n_out, stream);
            break;
        default:
            if (small_k) small_mmvq_launch<Weight, Qi, 8, WARPS>(w, x, y, n_in, n_out, stream);
            else small_mmvq_launch<Weight, Qi, 8, 1>(w, x, y, n_in, n_out, stream);
            break;
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("native MMVQ launch: ") + cudaGetErrorString(e));
}

template <typename Weight, int Qi>
void small_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out, int ncols,
               void* stream) {
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    small_mmvq<Weight, Qi>(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

[[noreturn]] void not_ported(const char* fmt) {
    throw std::runtime_error(std::string("native MMVQ: the ") + fmt +
                             " format is not ported to the SYCL backend yet");
}

}  // namespace

void native_mmvq_set_multi_exact(bool exact) { g_multi_exact = exact; }
bool native_mmvq_multi_exact() { return g_multi_exact; }

std::size_t native_q8_1_bytes(int n_in, int ncols) {
    if (n_in % Q8K != 0 || n_in <= 0 || ncols < 1) throw std::invalid_argument("native Q8_1 shape");
    return std::size_t(ncols) * std::size_t(n_in / Q8K) * sizeof(Q81Block);
}

void native_quantize_q8_1(const float* x, void* x_q8_1, int n_in, int ncols, void* stream) {
    if (!x || !x_q8_1 || !stream || n_in % Q8K != 0 || ncols < 1)
        throw std::invalid_argument("native Q8_1 quantize requires valid buffers and shapes");
    Q81Block* y = static_cast<Q81Block*>(x_q8_1);
    const size_t total = (size_t) ncols * n_in;
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<float, 1> redA(sycl::range<1>(32), hnd);
        local_accessor<float, 1> redB(sycl::range<1>(32), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>(total), sycl::range<1>(32)), [=](nd_item<1> it) {
            const size_t i = it.get_global_id(0);
            const int lane = (int) it.get_local_id(0);
            const float xi = x[i];
            float amax = sycl::fabs(xi);
            redA[lane] = amax;
            it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const float other = redA[lane ^ o];
                it.barrier(sycl::access::fence_space::local_space);
                amax = sycl::fmax(amax, other);
                redA[lane] = amax;
                it.barrier(sycl::access::fence_space::local_space);
            }
            float sum = xi;
            redB[lane] = sum;
            it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const float other = redB[lane ^ o];
                it.barrier(sycl::access::fence_space::local_space);
                sum += other;
                redB[lane] = sum;
                it.barrier(sycl::access::fence_space::local_space);
            }
            const float d = amax / 127.0f;
            const int8_t q = amax == 0.0f ? 0 : (int8_t) sycl::round(xi / d);
            y[i / Q8K].qs[i % Q8K] = q;
            if (i % Q8K == 0) y[i / Q8K].ds = __floats2half2_rn(d, sum);
        });
    });
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("native Q8_1 quantize: ") + cudaGetErrorString(e));
}

void quantize_q8_1_rows(const float* x, int64_t n_rows, int64_t n_cols, void* y, void* stream) {
    // rows are contiguous and n_cols is a multiple of 32, so the flat
    // element-wise kernel is exactly the row-wise one
    native_quantize_q8_1(x, y, (int) n_cols, (int) n_rows, stream);
}

void native_q4_0_mmvq(const void* w, const void* x, float* y, int n_in, int n_out, int nc, void* s) {
    small_mmvq<Q40Block, 4>(w, x, y, n_in, n_out, nc, s);
}
void native_q4_0_f32(const void* w, const float* x, void* sc, float* y, int n_in, int n_out, int nc, void* s) {
    small_f32<Q40Block, 4>(w, x, sc, y, n_in, n_out, nc, s);
}
void native_q5_0_mmvq(const void* w, const void* x, float* y, int n_in, int n_out, int nc, void* s) {
    small_mmvq<Q50Block, 4>(w, x, y, n_in, n_out, nc, s);
}
void native_q5_0_f32(const void* w, const float* x, void* sc, float* y, int n_in, int n_out, int nc, void* s) {
    small_f32<Q50Block, 4>(w, x, sc, y, n_in, n_out, nc, s);
}
void native_q8_0_mmvq(const void* w, const void* x, float* y, int n_in, int n_out, int nc, void* s) {
    small_mmvq<Q80Block, 8>(w, x, y, n_in, n_out, nc, s);
}
void native_q8_0_f32(const void* w, const float* x, void* sc, float* y, int n_in, int n_out, int nc, void* s) {
    small_f32<Q80Block, 8>(w, x, sc, y, n_in, n_out, nc, s);
}
void native_iq4_nl_mmvq(const void* w, const void* x, float* y, int n_in, int n_out, int nc, void* s) {
    small_mmvq<IQ4NLBlock, 4>(w, x, y, n_in, n_out, nc, s);
}
void native_iq4_nl_f32(const void* w, const float* x, void* sc, float* y, int n_in, int n_out, int nc, void* s) {
    small_f32<IQ4NLBlock, 4>(w, x, sc, y, n_in, n_out, nc, s);
}

void native_q2_0_mmvq(const void*, const void*, float*, int, int, int, void*) { not_ported("Q2_0"); }
void native_q2_0_f32(const void*, const float*, void*, float*, int, int, int, void*) { not_ported("Q2_0"); }
void native_q3_k_mmvq(const void*, const void*, float*, int, int, int, void*) { not_ported("Q3_K"); }
void native_q3_k_f32(const void*, const float*, void*, float*, int, int, int, void*) { not_ported("Q3_K"); }
void native_q4_k_mmvq(const void*, const void*, float*, int, int, int, void*) { not_ported("Q4_K"); }
void native_q4_k_f32(const void*, const float*, void*, float*, int, int, int, void*) { not_ported("Q4_K"); }
void native_q5_k_mmvq(const void*, const void*, float*, int, int, int, void*) { not_ported("Q5_K"); }
void native_q5_k_f32(const void*, const float*, void*, float*, int, int, int, void*) { not_ported("Q5_K"); }
void native_q6_k_mmvq(const void*, const void*, float*, int, int, int, void*) { not_ported("Q6_K"); }
void native_q6_k_f32(const void*, const float*, void*, float*, int, int, int, void*) { not_ported("Q6_K"); }
void native_iq4_xs_mmvq(const void*, const void*, float*, int, int, int, void*) { not_ported("IQ4_XS"); }
void native_iq4_xs_f32(const void*, const float*, void*, float*, int, int, int, void*) { not_ported("IQ4_XS"); }

size_t iq_row_bytes(int t, int64_t n) noexcept {
    // ggml block sizes (bytes per 256 elements unless noted); pinned public
    // layout constants, cross-checked against the static_asserts in this file
    // (iq4_nl 18/32, q8_0 34/32, q5_0 22/32, iq4_xs 136/256, q2_0 18/64).
    switch (t) {
        case 16: return (size_t) (n / 256) * 66;    // iq2_xxs
        case 17: return (size_t) (n / 256) * 74;    // iq2_xs
        case 18: return (size_t) (n / 256) * 98;    // iq3_xxs
        case 20: return (size_t) (n / 32) * 18;     // iq4_nl
        case 21: return (size_t) (n / 256) * 110;   // iq3_s
        case 22: return (size_t) (n / 256) * 80;   // iq2_s
        case 29: return (size_t) (n / 256) * 56;    // iq1_m
        case 23: return (size_t) (n / 256) * 136;   // iq4_xs
        case 11: return (size_t) (n / 256) * 110;   // q3_K
        case 42: return (size_t) (n / 64) * 18;     // q2_0
        case 12: return (size_t) (n / 256) * 144;   // q4_K
        case 13: return (size_t) (n / 256) * 176;   // q5_K
        case 7: return (size_t) (n / 32) * 22;      // q5_1
        case 8: return (size_t) (n / 32) * 34;      // q8_0
        case 30: return (size_t) n * 2;             // bf16 (embedding only)
        default: return 0;
    }
}

bool native_mmvq_supported(int ggml_type) noexcept {
    return ggml_type == 2 || ggml_type == 6 || ggml_type == 7 || ggml_type == 8 || ggml_type == 11 ||
           ggml_type == 12 || ggml_type == 13 || ggml_type == 14 || ggml_type == 20 ||
           ggml_type == 23 || ggml_type == 42 || ggml_type == 16 || ggml_type == 17 || ggml_type == 18 ||
           ggml_type == 21 || ggml_type == 22 || ggml_type == 29;
}

std::size_t native_mmvq_weight_bytes(int ggml_type, int n_in, int n_out) {
    int block_elems, block_bytes;
    switch (ggml_type) {
        case 2: block_elems = 32; block_bytes = 18; break;
        case 6: block_elems = 32; block_bytes = 22; break;
        case 7: block_elems = 32; block_bytes = 24; break;
        case 8: block_elems = 32; block_bytes = 34; break;
        case 20: block_elems = 32; block_bytes = 18; break;
        case 11: block_elems = 256; block_bytes = 110; break;
        case 12: block_elems = 256; block_bytes = 144; break;
        case 13: block_elems = 256; block_bytes = 176; break;
        case 14: block_elems = 256; block_bytes = 210; break;
        case 23: block_elems = 256; block_bytes = 136; break;
        case 42: block_elems = 64; block_bytes = 18; break;
        case 16: case 17: case 18: case 21: case 22: case 29:
            block_elems = 256; block_bytes = (int) iq_row_bytes(ggml_type, 256); break;
        default: throw std::invalid_argument("unsupported native MMVQ GGML type");
    }
    if (n_in % block_elems != 0 || n_in <= 0) throw std::invalid_argument("native MMVQ shape");
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    const std::size_t row_bytes = std::size_t(n_in / block_elems) * block_bytes;
    if (row_bytes > std::numeric_limits<std::size_t>::max() / std::size_t(n_out)) {
        throw std::length_error("native MMVQ weight byte count overflows size_t");
    }
    return row_bytes * std::size_t(n_out);
}

void native_mmvq(int ggml_type, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                 void* stream) {
    switch (ggml_type) {
        case 2: native_q4_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
        case 6: native_q5_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
        case 8: native_q8_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
        case 20: native_iq4_nl_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
        case 7: case 11: case 12: case 13: case 14: case 23: case 42:
        case 16: case 17: case 18: case 21: case 22: case 29:
            not_ported("K/IQ native MMVQ");
        default: throw std::invalid_argument("unsupported native MMVQ GGML type");
    }
}

}  // namespace strata::kernels
