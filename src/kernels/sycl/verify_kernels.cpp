// src/kernels/sycl/verify_kernels.cpp — SYCL port of src/kernels/cuda/verify_kernels.cu
// (the verify window's per-token-multi kernels + doorbell/plumbing copies).
//
// Same transcription rules as the other ports in this directory:
//   * warp xor-butterflies become per-32-lane local-memory trees with the
//     IDENTICAL 16,8,4,2,1 pairing and order (the sums are order-dependent)
//   * __expf-based sigmoid/softplus -> sycl::exp / sycl::log1p (house spelling
//     from gdn.cpp; the parity gates accept it)
//   * __fmul_rn/__fadd_rn -> plain * and + (the backend compiles with
//     -ffp-contract=off, so nothing contracts and changes the rounding)
//   * volatile mapped-memory reads -> atomic_ref loads (system scope); the
//     doorbell spins are the milestone-1 pattern from elementwise.cpp
//   * __threadfence/_system -> sycl::atomic_fence(system)
//   * CUDA warps that early-return before a shuffle become PREDICATED chunks
//     here: SYCL barriers are work-group-wide, so out-of-range chunks stay in
//     the reduction with dummy values and skip only the write
// gpu_stamp writes zeros: L0 exposes no device global timer to kernels and the
// stamps only feed the optional --profile stage report.
#include "strata/kernels/verify_kernels.hpp"

#include "strata/kernels/dp4a.hpp"

#include <cuda_runtime.h>  // sycl_compat shim

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int S = 128;  // GDN state size
constexpr int RG = 4;
constexpr int RPG = S / RG;

using sycl::local_accessor;
using sycl::nd_item;
using sycl::atomic_ref;
using sycl::memory_order;
using sycl::memory_scope;

inline sycl::queue& Q(void* stream) { return strata::sycl_compat::q_for(stream); }

void check_launch(void* stream, const char* what) {
    if (stream != nullptr) return;
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

inline float sigmoid_f(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }
inline float softplus_f(float x) { return x > 20.0f ? x : sycl::log1p(sycl::exp(x)); }

// v += v(lane^o) for o = 16,8,4,2,1 within a 32-lane chunk. `red` must cover
// the work-group's local range; pairing and order match the CUDA shuffle.
inline float xor_reduce32(float v, int lane, local_accessor<float, 1> red, nd_item<1> it) {
    red[lane] = v;
    it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float add = red[lane ^ o];
        it.barrier(sycl::access::fence_space::local_space);
        v += add;
        red[lane] = v;
        it.barrier(sycl::access::fence_space::local_space);
    }
    return v;
}

inline uint32_t ld_u32_mapped(const uint32_t* p) {
    atomic_ref<uint32_t, memory_order::seq_cst, memory_scope::system> r(*const_cast<uint32_t*>(p));
    return r.load(memory_order::relaxed);
}

inline int32_t ld_i32_mapped(const int32_t* p) {
    atomic_ref<int32_t, memory_order::seq_cst, memory_scope::system> r(*const_cast<int32_t*>(p));
    return r.load(memory_order::relaxed);
}

inline void st_u32_mapped(uint32_t* p, uint32_t v, bool release = false) {
    atomic_ref<uint32_t, memory_order::seq_cst, memory_scope::system> r(*p);
    r.store(v, release ? memory_order::release : memory_order::relaxed);
}

}  // namespace

void gdn_conv_l2_multi(const float* history, const float* qkv, const float* conv_w, float* h, int channels,
                       int qk_heads, float eps, int n_tok, void* stream, int t_begin) {
    if (!history || !qkv || !conv_w || !h || channels % S != 0 || n_tok < 1 || n_tok > kVerifyMaxT) {
        std::fprintf(stderr, "gdn_conv_l2_multi: invalid arguments\n");
        std::exit(1);
    }
    const size_t gx = (size_t) (channels / S);
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<float, 1> red(sycl::range<1>(S), hnd);
        local_accessor<float, 1> part(sycl::range<1>(S / 32), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>(gx * (size_t) n_tok * S), sycl::range<1>(S)),
                         [=](nd_item<1> it) {
                             const size_t g = it.get_group(0);
                             const int bx = (int) (g % gx);
                             const int t = t_begin + (int) (g / gx);
                             const int tx = (int) it.get_local_id(0);
                             const int c = bx * S + tx;
                             float win[3];
#pragma unroll
                             for (int j = 0; j < 3; ++j) {
                                 const int src = t + j;
                                 win[j] = src < 3 ? history[(size_t) c * 3 + src]
                                                  : qkv[(size_t) (src - 3) * channels + c];
                             }
                             const float v0 = win[0], v1 = win[1], v2 = win[2];
                             const float x = qkv[(size_t) t * channels + c];
                             const float sum = v0 * conv_w[c * 4] + v1 * conv_w[c * 4 + 1] +
                                               v2 * conv_w[c * 4 + 2] + x * conv_w[c * 4 + 3];
                             float y = sum / (1.0f + sycl::exp(-sum));
                             if (bx < qk_heads) {
                                 float sq = xor_reduce32(y * y, tx, red, it);
                                 if ((tx & 31) == 0) part[tx >> 5] = sq;
                                 it.barrier(sycl::access::fence_space::local_space);
                                 const float ss = part[0] + part[1] + part[2] + part[3];
                                 y *= sycl::rsqrt(ss + eps);
                             }
                             h[(size_t) t * channels + c] = y;
                         });
    });
    check_launch(stream, "gdn_conv_l2_multi");
}

void gdn_conv_commit(float* history, const float* qkv, int channels, const int32_t* n_keep, void* stream) {
    const size_t C = (size_t) channels;
    Q(stream).parallel_for(C, [=](size_t c) {
        const int n = *n_keep;
        if (n <= 0) return;
        float seq[3];
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            const int src = n + j;
            seq[j] = src < 3 ? history[c * 3 + src] : qkv[(size_t) (src - 3) * C + c];
        }
        history[c * 3] = seq[0];
        history[c * 3 + 1] = seq[1];
        history[c * 3 + 2] = seq[2];
    });
    check_launch(stream, "gdn_conv_commit");
}

void gdn_ab_multi(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt, const float* ssm_a,
                  float* gate, float* beta, int n_embd, int h_v, int n_tok, void* stream) {
    if (n_embd % 8 != 0 || n_tok < 1 || n_tok > kVerifyMaxT) {
        std::fprintf(stderr, "gdn_ab_multi: invalid arguments\n");
        std::exit(1);
    }
    const int n = n_embd;
    const int T = n_tok;
    const size_t blocks = (size_t) ((2 * h_v + 7) / 8);
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<float, 1> red(sycl::range<1>(256), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>(blocks * 256), sycl::range<1>(256)),
                         [=](nd_item<1> it) {
                             const int lid = (int) it.get_local_id(0);
                             const int row = (int) it.get_group(0) * 8 + (lid >> 5);
                             const int lane = lid & 31;
                             const bool live = row < 2 * h_v;
                             const bool is_beta = row >= h_v;
                             const int r = is_beta ? row - h_v : row;
                             const uint16_t* w = (is_beta ? w_beta : w_alpha) + (size_t) r * n;
                             float acc[kVerifyMaxT];
#pragma unroll
                             for (int t = 0; t < kVerifyMaxT; ++t) acc[t] = 0.0f;
                             for (int j = lane; j < n / 8; j += 32) {
                                 uint32_t wv[4];
#pragma unroll
                                 for (int e = 0; e < 4; ++e) {
                                     const uint16_t lo = w[j * 8 + e * 2];
                                     const uint16_t hi = w[j * 8 + e * 2 + 1];
                                     wv[e] = (uint32_t) lo | ((uint32_t) hi << 16);
                                 }
#pragma unroll
                                 for (int t = 0; t < kVerifyMaxT; ++t) {
                                     if (t >= T) break;
                                     const float* xt = x + (size_t) t * n + (size_t) j * 8;
#pragma unroll
                                     for (int e = 0; e < 4; ++e) {
                                         const float a0 = xt[e * 2], a1 = xt[e * 2 + 1];
                                         float a = acc[t];
                                         a = sycl::fma(sycl::bit_cast<float>(wv[e] << 16), a0, a);
                                         a = sycl::fma(sycl::bit_cast<float>(wv[e] & 0xffff0000u), a1, a);
                                         acc[t] = a;
                                     }
                                 }
                             }
#pragma unroll
                             for (int t = 0; t < kVerifyMaxT; ++t) {
                                 if (t >= T) break;
                                 // index the scratch by full local id: 8 rows
                                 // (32-lane chunks) share this work-group
                                 const float a = xor_reduce32(acc[t], lid, red, it);
                                 if (lane != 0 || !live) continue;
                                 if (is_beta) {
                                     beta[(size_t) t * h_v + r] = sigmoid_f(a);
                                 } else {
                                     const float v = a + dt[r];
                                     gate[(size_t) t * h_v + r] = softplus_f(v) * ssm_a[r];
                                 }
                             }
                         });
    });
    check_launch(stream, "gdn_ab_multi");
}

void gdn_step_norm_multi(float* state, const float* h, int conv_channels, const float* gate, const float* beta,
                         const float* z, const float* gamma, float eps, float* y, int h_k, int h_v, int n_tok,
                         const int32_t* n_keep, void* stream, int t_out_begin) {
    if (!state || !h || !gate || !beta || !z || !gamma || !y || h_k <= 0 || h_v % h_k || n_tok < 1 ||
        n_tok > kVerifyMaxT) {
        std::fprintf(stderr, "gdn_step_norm_multi: invalid arguments\n");
        std::exit(1);
    }
    const int C = conv_channels;
    const int T = n_tok;
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<float, 1> sk(sycl::range<1>(S), hnd);
        local_accessor<float, 1> sq(sycl::range<1>(S), hnd);
        local_accessor<float, 1> red(sycl::range<1>(RG * S), hnd);
        local_accessor<float, 1> wsum(sycl::range<1>(S * RG / 32), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) h_v * S * RG), sycl::range<1>(S * RG)),
                         [=](nd_item<1> it) {
                             const int head = (int) it.get_group(0);
                             const int tid = (int) it.get_local_id(0);
                             const int col = tid % S;
                             const int rg = tid / S;
                             const int qh = head % h_k;
                             const int qk = S * h_k;
                             const int value_dim = S * h_v;
                             const int n = n_keep ? *n_keep : T;
                             float s[RPG];
                             float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
                             const size_t row_stride = (size_t) h_v * S;
#pragma unroll
                             for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
                             for (int t = 0; t < n; ++t) {
                                 const float* ht = h + (size_t) t * C;
                                 it.barrier(sycl::access::fence_space::local_space);
                                 if (tid < S) {
                                     sk[tid] = ht[qk + qh * S + tid];
                                     sq[tid] = ht[qh * S + tid];
                                 }
                                 it.barrier(sycl::access::fence_space::local_space);
                                 const float g = sycl::exp(gate[(size_t) t * h_v + head]);
                                 float kv = 0.0f;
#pragma unroll
                                 for (int r = 0; r < RPG; ++r) kv = sycl::fma(s[r], sk[rg * RPG + r], kv);
                                 red[rg * S + col] = kv;
                                 it.barrier(sycl::access::fence_space::local_space);
                                 const float kv_col = red[col] + red[S + col] + red[2 * S + col] + red[3 * S + col];
                                 const float delta =
                                     (ht[2 * qk + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
                                 float o = 0.0f;
#pragma unroll
                                 for (int r = 0; r < RPG; ++r) {
                                     s[r] = sycl::fma(g, s[r], sk[rg * RPG + r] * delta);
                                     o = sycl::fma(s[r], sq[rg * RPG + r], o);
                                 }
                                 it.barrier(sycl::access::fence_space::local_space);
                                 red[rg * S + col] = o;
                                 it.barrier(sycl::access::fence_space::local_space);
                                 float oc = 0.0f, sq_part = 0.0f;
                                 if (rg == 0) {
                                     oc = (red[col] + red[S + col] + red[2 * S + col] + red[3 * S + col]) *
                                          sycl::rsqrt((float) S);
                                     sq_part = oc * oc;
                                 }
                                 if (t < t_out_begin) continue;  // uniform: replayed token, no output
                                 sq_part = xor_reduce32(sq_part, tid, red, it);
                                 if ((tid & 31) == 0) wsum[tid >> 5] = sq_part;
                                 it.barrier(sycl::access::fence_space::local_space);
                                 if (rg == 0) {
                                     const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
                                     const float scale = sycl::rsqrt(ss / (float) S + eps);
                                     const float zz = z[(size_t) t * value_dim + head * S + col];
                                     y[(size_t) t * value_dim + head * S + col] =
                                         oc * scale * gamma[col] * sigmoid_f(zz);
                                 }
                             }
                             if (n_keep != nullptr && n > 0) {
#pragma unroll
                                 for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
                             }
                         });
    });
    check_launch(stream, "gdn_step_norm_multi");
}

namespace {
// Under segmented capture a wait is a CUT, not a kernel: end the current
// graph, hand it to the sink, and start recording the next segment. Outside
// capture (eager parity runs) keep the spin kernel so semantics match CUDA.
bool eager_wait(uint32_t value, void* stream) {
    auto feed = strata::sycl_compat::eager_feed;
    if (!feed) return false;
    // The feed callback reads mapped buffers that the kernels queued so far
    // write (R rows, flags).  Those submissions are asynchronous: drain the
    // queue first, or the host consumes stale/garbage data (crash in the
    // expert pool with garbage job counts).
    if (stream) cudaStreamSynchronize((cudaStream_t) stream);
    else cudaDeviceSynchronize();
    return feed(value);
}

bool segment_cut(void* stream) {
    auto* sink = strata::sycl_compat::seg_sink;
    if (!sink) return false;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(s, &st);
    if (st != cudaStreamCaptureStatusActive) return false;
    cudaGraph_t g = nullptr;
    if (cudaStreamEndCapture(s, &g) != cudaSuccess) return false;
    sink->push_back(g);
    return cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
}
}  // namespace

void wait_flag_ge(const uint32_t* flag, uint32_t value, void* stream) {
    if (eager_wait(value, stream)) return;
    if (segment_cut(stream)) return;
    Q(stream).single_task([=] {
        while (ld_u32_mapped(flag) < value) strata_spin_pause();
        sycl::atomic_fence(memory_order::acquire, memory_scope::system);
    });
    check_launch(stream, "wait_flag_ge");
}

void wait_flag_ge_or(const uint32_t* flag, uint32_t value, const uint32_t* skip, void* stream) {
    if (eager_wait(value, stream)) return;
    if (segment_cut(stream)) return;
    Q(stream).single_task([=] {
        if (ld_u32_mapped(skip) == value) return;
        while (ld_u32_mapped(flag) < value) strata_spin_pause();
        sycl::atomic_fence(memory_order::acquire, memory_scope::system);
    });
    check_launch(stream, "wait_flag_ge_or");
}

void gpu_stamp(unsigned long long* buf, int i, void* stream) {
    Q(stream).single_task([=] { buf[i] = 0ull; });  // no device global timer on L0
}

void resident_plan(const int32_t* ids, int n_entries, int k, const int32_t* res_layer, int n_expert,
                   const uint8_t* cache_base, const unsigned long long* slot_off, long long blob, int32_t* plan,
                   long long capx, uint32_t* skip, uint32_t ring, void* stream) {
    const int n = n_entries;
    const bool force_skip = std::getenv("STRATA_HIT_SKIP") != nullptr;
    Q(stream).single_task([=] {
        if (force_skip) {
            sycl::atomic_fence(memory_order::release, memory_scope::system);
            st_u32_mapped(skip, 0);
            return;
        }
        for (int i = 0; i < n; ++i) {
            const int32_t e = ids[i];
            if (e < 0 || e >= n_expert || res_layer[e] < 0) {
                st_u32_mapped(skip, 0);
                return;
            }
        }
        int32_t* counts = plan;
        int32_t* start = plan + 4;
        int32_t* dst = start + capx + 1;
        int32_t* tok = dst + capx;
        const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
        unsigned long long* ptr = (unsigned long long*) (plan + ptr_off);
        int32_t* start2 = plan + ptr_off + 4 * capx;
        int groups = 0, entries = 0;
        for (int i0 = 0; i0 < n; ++i0) {
            bool first = true;
            for (int j = 0; j < i0; ++j)
                if (ids[j] == ids[i0]) {
                    first = false;
                    break;
                }
            if (!first) continue;
            const int32_t slot = res_layer[ids[i0]];
            ptr[groups] = (unsigned long long) (cache_base +
                                                (slot_off ? (size_t) slot_off[slot] : (size_t) slot * (size_t) blob));
            start[groups] = entries;
            for (int i = i0; i < n; ++i)
                if (ids[i] == ids[i0]) {
                    dst[entries] = i;
                    tok[entries] = i / k;
                    ++entries;
                }
            ++groups;
        }
        start[groups] = entries;
        start2[0] = entries;
        counts[0] = groups;
        counts[1] = entries;
        counts[2] = 0;
        sycl::atomic_fence(memory_order::release, memory_scope::system);
        st_u32_mapped(skip, ring, true);
    });
    if (std::getenv("STRATA_HIT_SYNC")) {
        const cudaError_t e = cudaDeviceSynchronize();
        std::fprintf(stderr, "[dbg] hit %-10s %s\n", "resident_plan", cudaGetErrorString(e));
    }
}

void copy_i32_from_mapped_unless(int32_t* dst, const int32_t* src, long long n, const uint32_t* skip, uint32_t value,
                                 void* stream) {
    if (n <= 0) return;
    const long long nn = n;
    Q(stream).submit([&](sycl::handler& hnd) {
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>(128), sycl::range<1>(128)), [=](nd_item<1> it) {
            if (ld_u32_mapped(skip) == value) return;
            for (long long i = (long long) it.get_local_id(0); i < nn; i += 128) dst[i] = ld_i32_mapped(src + i);
        });
    });
    check_launch(stream, "copy_i32_from_mapped_unless");
}

void copy_or_zero_from_mapped(float* dst, const float* src, long long n, const uint32_t* skip, uint32_t value,
                              void* stream) {
    if (n <= 0) return;
    const long long n4 = n / 4;
    const size_t blocks = (size_t) ((n4 + 255) / 256 < 64 ? (n4 + 255) / 256 : 64);
    Q(stream).submit([&](sycl::handler& hnd) {
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>(blocks * 256), sycl::range<1>(256)), [=](nd_item<1> it) {
            const bool zero = ld_u32_mapped(skip) == value;
            for (long long i = (long long) it.get_global_id(0); i < n4;
                 i += (long long) it.get_global_range(0)) {
                if (zero) {
                    dst[i * 4] = 0.f;
                    dst[i * 4 + 1] = 0.f;
                    dst[i * 4 + 2] = 0.f;
                    dst[i * 4 + 3] = 0.f;
                } else {
                    const uint32_t* w = reinterpret_cast<const uint32_t*>(src) + i * 4;
                    dst[i * 4] = sycl::bit_cast<float>(ld_u32_mapped(w));
                    dst[i * 4 + 1] = sycl::bit_cast<float>(ld_u32_mapped(w + 1));
                    dst[i * 4 + 2] = sycl::bit_cast<float>(ld_u32_mapped(w + 2));
                    dst[i * 4 + 3] = sycl::bit_cast<float>(ld_u32_mapped(w + 3));
                }
            }
        });
    });
    check_launch(stream, "copy_or_zero_from_mapped");
}

void embedding_gather_dev(const uint8_t* codes, const float* scales, const float* offsets, const int32_t* tokens,
                          int n_tok, int64_t n, int code_bits, int code_bias, int group_elems, uint64_t row_codes,
                          uint64_t row_groups, float* out, void* stream) {
    const int64_t nn = n;
    Q(stream).parallel_for(sycl::range<2>((size_t) n_tok, (size_t) n), [=](sycl::id<2> id) {
        const int t = (int) id[0];
        const int64_t i = (int64_t) id[1];
        const uint64_t token = (uint64_t) tokens[t];
        const uint8_t* c = codes + token * row_codes;
        const float* sc = scales + token * row_groups;
        const float* of = offsets ? offsets + token * row_groups : nullptr;
        const int per_byte = 8 / code_bits;
        const unsigned mask = (1u << code_bits) - 1u;
        const int code = (c[i / per_byte] >> ((i % per_byte) * code_bits)) & mask;
        const int64_t group = i / group_elems;
        const float product = (float) (code + code_bias) * sc[group];
        out[(size_t) t * nn + i] = product + (of ? of[group] : 0.0f);
    });
    check_launch(stream, "embedding_gather_dev");
}

void broadcast_streams(const float* x, float* R, int64_t n_embd, int hc, int n_tok, void* stream) {
    const int64_t n = n_embd;
    Q(stream).parallel_for(sycl::range<2>((size_t) n_tok, (size_t) (n * hc)), [=](sycl::id<2> id) {
        const int t = (int) id[0];
        const int64_t i = (int64_t) id[1];
        R[(size_t) t * n * hc + i] = x[(size_t) t * n + i % n];
    });
    check_launch(stream, "broadcast_streams");
}

void copy_indexed(float* dst, const float* src, int64_t stride, const int32_t* index, int64_t n, void* stream) {
    const size_t blocks = (size_t) ((n + 255) / 256 < 64 ? (n + 255) / 256 : 64);
    Q(stream).submit([&](sycl::handler& hnd) {
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>(blocks * 256), sycl::range<1>(256)), [=](nd_item<1> it) {
            const int idx = *index;
            if (idx < 0) return;
            for (int64_t i = (int64_t) it.get_global_id(0); i < n; i += (int64_t) it.get_global_range(0))
                dst[i] = src[(size_t) idx * stride + i];
        });
    });
    check_launch(stream, "copy_indexed");
}

void fetch_blobs(const unsigned long long* src, const int32_t* n, uint8_t* dst, int64_t blob_bytes, int cap,
                 void* stream) {
    if (cap <= 0) return;
    if (blob_bytes % 16 != 0) {
        std::fprintf(stderr, "fetch_blobs: blob size must be a multiple of 16\n");
        std::exit(1);
    }
    const long long per = (long long) (blob_bytes / 16);
    using V16 = sycl::vec<uint8_t, 16>;
    V16* dstv = reinterpret_cast<V16*>(dst);
    Q(stream).submit([&](sycl::handler& hnd) {
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>(48 * 8 * 256), sycl::range<1>(256)), [=](nd_item<1> it) {
            const long long total = (long long) (*n) * per;
            for (long long i = (long long) it.get_global_id(0); i < total; i += (long long) it.get_global_range(0)) {
                const long long k = i / per, off = i - k * per;
                const V16* row = reinterpret_cast<const V16*>((const uint8_t*) (uintptr_t) src[k]);
                dstv[i] = row[off];
            }
        });
    });
    check_launch(stream, "fetch_blobs");
}

void rebase_ptrs(unsigned long long* ptr, const int32_t* n, uint8_t* base, int64_t blob_bytes, void* stream) {
    const unsigned long long b = (unsigned long long) (uintptr_t) base;
    const long long bytes = blob_bytes;
    Q(stream).single_task([=] {
        const int k = 0;
        (void) k;
        for (int i = 0; i < *n; ++i) ptr[i] = b + (unsigned long long) i * (unsigned long long) bytes;
    });
    check_launch(stream, "rebase_ptrs");
}

void add_streams_broadcast(const float* h, const float* e, float* R, int64_t n_embd, int hc, int n_tok, void* stream) {
    const int64_t n = n_embd;
    Q(stream).parallel_for(sycl::range<2>((size_t) n_tok, (size_t) (n * hc)), [=](sycl::id<2> id) {
        const int t = (int) id[0];
        const int64_t i = (int64_t) id[1];
        R[(size_t) t * n * hc + i] = h[(size_t) t * n * hc + i] + e[(size_t) t * n + i % n];
    });
    check_launch(stream, "add_streams_broadcast");
}

void ident_hits(const int32_t* ids, int n, int32_t* slot, int32_t* dst, int32_t* count, void* stream) {
    if (n < 1 || n > 1024) {
        std::fprintf(stderr, "ident_hits: n out of range\n");
        std::exit(1);
    }
    const int nn = n;
    Q(stream).parallel_for(sycl::range<1>(1024), [=](size_t i) {
        if ((int) i < nn) {
            slot[i] = ids[i];
            dst[i] = (int32_t) i;
        }
        if (i == 0) *count = nn;
    });
    check_launch(stream, "ident_hits");
}

void mtp_select(const float* R_src, int64_t R_stride, const int32_t* ids, const int32_t* row_dev, float* R_dst,
                int32_t* tok_dst, int32_t* out, int j, void* stream, const float* probs, float* out_p) {
    Q(stream).submit([&](sycl::handler& hnd) {
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>(16 * 256), sycl::range<1>(256)), [=](nd_item<1> it) {
            const int row = *row_dev;
            for (int64_t i = (int64_t) it.get_global_id(0); i < R_stride; i += (int64_t) it.get_global_range(0))
                R_dst[i] = R_src[(size_t) row * R_stride + i];
            if (it.get_group(0) == 0 && it.get_local_id(0) == 0) {
                const int32_t tok = ids[row];
                *tok_dst = tok;
                if (out != nullptr) st_u32_mapped(reinterpret_cast<uint32_t*>(out) + j, (uint32_t) tok);
                if (probs != nullptr && out_p != nullptr)
                    st_u32_mapped(reinterpret_cast<uint32_t*>(out_p) + j,
                                  sycl::bit_cast<uint32_t>(probs[row]));
            }
        });
    });
    check_launch(stream, "mtp_select");
}

void gather_rows(const uint8_t* src, int64_t row_bytes, const int32_t* ids, int64_t n, uint8_t* dst, void* stream) {
    const long long total = n * row_bytes;
    Q(stream).submit([&](sycl::handler& hnd) {
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>(48 * 8 * 256), sycl::range<1>(256)), [=](nd_item<1> it) {
            for (long long i = (long long) it.get_global_id(0); i < total; i += (long long) it.get_global_range(0)) {
                const long long r = i / row_bytes, o = i - r * row_bytes;
                dst[i] = src[(long long) ids[r] * row_bytes + o];
            }
        });
    });
    check_launch(stream, "gather_rows");
}

void map_ids(int32_t* ids, const int32_t* table, int n, void* stream) {
    const int nn = n;
    Q(stream).parallel_for(sycl::range<1>(64), [=](size_t i) {
        if ((int) i < nn) ids[i] = table[ids[i]];
    });
    check_launch(stream, "map_ids");
}

void row_top_prob(const float* logits, int n_rows, int n_vocab, const int32_t* ids, float* probs, void* stream) {
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<float, 1> red(sycl::range<1>(1024), hnd);
        local_accessor<float, 1> part(sycl::range<1>(32), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) n_rows * 1024), sycl::range<1>(1024)),
                         [=](nd_item<1> it) {
                             const int t = (int) it.get_group(0);
                             const int tx = (int) it.get_local_id(0);
                             const float* l = logits + (size_t) t * n_vocab;
                             const float m = l[ids[t]];
                             float s = 0.0f;
                             for (int i = tx; i < n_vocab; i += 1024) s += sycl::exp(l[i] - m);
                             s = xor_reduce32(s, tx, red, it);
                             if ((tx & 31) == 0) part[tx >> 5] = s;
                             it.barrier(sycl::access::fence_space::local_space);
                             if (tx == 0) {
                                 float tot = 0.0f;
                                 for (int w = 0; w < 32; ++w) tot += part[w];
                                 probs[t] = 1.0f / tot;
                             }
                         });
    });
    check_launch(stream, "row_top_prob");
}

void dense_steps(const int32_t* cells, int n, int32_t* steps, void* stream) {
    const int nn = n;
    Q(stream).parallel_for(sycl::range<1>(64), [=](size_t i) {
        if ((int) i >= nn) return;
        const int c = cells[i];
        steps[i * 4 + 0] = c;
        steps[i * 4 + 1] = c + 1;
        steps[i * 4 + 2] = (c + 1) / 4;
        steps[i * 4 + 3] = c + 1;
    });
    check_launch(stream, "dense_steps");
}

void window_ids(int32_t* steps, int n, int window, int32_t* ids, int64_t ids_stride, void* stream) {
    Q(stream).submit([&](sycl::handler& hnd) {
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) n * 8 * 256), sycl::range<1>(256)),
                         [=](nd_item<1> it) {
                             const size_t g = it.get_group(0);
                             const int q = (int) (g / 8);
                             const int bx = (int) (g % 8);
                             const int tx = (int) it.get_local_id(0);
                             int32_t* st = steps + (size_t) q * 4;
                             const int n_kv = st[1];
                             const int start = n_kv > window ? n_kv - window : 0;
                             const int width = n_kv - start;
                             for (int j = bx * 256 + tx; j < width; j += 8 * 256)
                                 ids[(size_t) q * ids_stride + j] = start + j;
                             it.barrier(sycl::access::fence_space::local_space);
                             if (bx == 0 && tx == 0) st[3] = width;
                         });
    });
    check_launch(stream, "window_ids");
}

}  // namespace strata::kernels
