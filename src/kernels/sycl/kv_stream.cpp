// src/kernels/sycl/kv_stream.cpp — SYCL port of src/kernels/cuda/kv_stream.cu
// (resident-KV streaming: clock-sweep resolve, slot copies, ring table).
// block_scan's shfl_up inclusive scan becomes a local-memory up-sweep with the
// identical (lane >= o) guard; atomicCAS/atomicAdd become atomic_ref RMWs on
// the same USM words.
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_stream.hpp"

#include <cuda_runtime.h>  // sycl_compat shim

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {
constexpr int RT = 1024;  // the resolve block

using sycl::local_accessor;
using sycl::nd_item;
using sycl::atomic_ref;
using sycl::memory_order;
using sycl::memory_scope;

inline sycl::queue& Q(void* stream) { return strata::sycl_compat::q_for(stream); }

void check(void* stream, const char* what) {
    if (stream != nullptr) return;
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "kv_stream: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

struct Runs {
    const uint8_t* src[4];
    uint8_t* dst[4];
    int len[4];
    int n;
};
Runs runs_of(const QsaAttnPools& slots, const KvHostPools& host, int fmt, const QsaShapes& s) {
    const int rows = (int) (s.n_head_kv * s.page_size);
    Runs r{};
    if (fmt == kKvQ4) {
        const int bytes = rows * (int) kv_q4_bytes_per_head((int) s.head_dim);
        r.src[0] = (const uint8_t*) host.k_q4;
        r.dst[0] = (uint8_t*) slots.k_q4;
        r.len[0] = bytes;
        r.src[1] = (const uint8_t*) host.v_q4;
        r.dst[1] = (uint8_t*) slots.v_q4;
        r.len[1] = bytes;
        r.n = 2;
    } else if (fmt == kKvInt8) {
        const int codes = rows * (int) s.head_dim, scales = rows * (int) (s.head_dim / KV_Q8_GROUP) * 2;
        r.src[0] = (const uint8_t*) host.k_q;
        r.dst[0] = (uint8_t*) slots.k_q;
        r.len[0] = codes;
        r.src[1] = (const uint8_t*) host.v_q;
        r.dst[1] = (uint8_t*) slots.v_q;
        r.len[1] = codes;
        r.src[2] = (const uint8_t*) host.k_scale;
        r.dst[2] = (uint8_t*) slots.k_scale;
        r.len[2] = scales;
        r.src[3] = (const uint8_t*) host.v_scale;
        r.dst[3] = (uint8_t*) slots.v_scale;
        r.len[3] = scales;
        r.n = 4;
    } else {
        const int bytes = rows * (int) s.head_dim * 2;
        r.src[0] = (const uint8_t*) host.k_pool;
        r.dst[0] = (uint8_t*) slots.k_pool;
        r.len[0] = bytes;
        r.src[1] = (const uint8_t*) host.v_pool;
        r.dst[1] = (uint8_t*) slots.v_pool;
        r.len[1] = bytes;
        r.n = 2;
    }
    return r;
}

}  // namespace

uint64_t kv_block_bytes(const QsaShapes& s, int fmt) {
    const uint64_t rows = (uint64_t) (s.n_head_kv * s.page_size);
    if (fmt == kKvQ4) return rows * kv_q4_bytes_per_head((int) s.head_dim) * 2;
    return fmt == kKvInt8 ? rows * (uint64_t) s.head_dim * 2 + rows * (uint64_t) (s.head_dim / KV_Q8_GROUP) * 2 * 2
                          : rows * (uint64_t) s.head_dim * 2 * 2;
}

void kv_stream_reset(const KvStreamMap& m, void* stream) {
    const long long nb = m.n_blocks, ns = m.n_slots;
    Q(stream).parallel_for(sycl::range<1>(128 * 256), [=](size_t i0) {
        for (long long i = (long long) i0; i < nb; i += 128 * 256) m.page_table[i] = -1;
        for (long long i = (long long) i0; i < ns; i += 128 * 256) {
            m.slot_block[i] = -1;
            m.slot_stamp[i] = -1;
            m.slot_ref[i] = 0;
        }
        if ((int) i0 < kKvCtlInts) m.ctl[i0] = 0;
    });
    check(stream, "reset");
}

void kv_stream_resolve(const KvStreamMap& m, const QsaAttnPools& slots, const KvHostPools& host, int fmt,
                       const int32_t* ids, const int32_t* steps, int64_t n_q, int64_t cap, const QsaShapes& s,
                       void* stream) {
    if (n_q <= 0) return;
    if (s.n_head_kv * s.page_size * (s.head_dim / KV_Q8_GROUP) * 2 % 16 != 0) {
        std::fprintf(stderr, "kv_stream: a block's scale run must be a multiple of 16 bytes\n");
        std::exit(1);
    }
    if (m.n_slots < RT) {
        std::fprintf(stderr, "kv_stream: %lld slots is fewer than the resolve block (%d): the clock sweep would take a "
                             "slot twice\n",
                     (long long) m.n_slots, RT);
        std::exit(1);
    }
    const int nq = (int) n_q;
    const int capp = (int) cap;
    const int page_size = (int) s.page_size;
    Q(stream).submit([&](sycl::handler& hnd) {
        local_accessor<int, 1> s_nmiss(sycl::range<1>(1), hnd);
        local_accessor<int, 1> s_lookups(sycl::range<1>(1), hnd);
        local_accessor<int, 1> s_cut(sycl::range<1>(1), hnd);
        local_accessor<int, 1> warp_sums(sycl::range<1>(32), hnd);
        local_accessor<int, 1> scan(sycl::range<1>(RT), hnd);
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>(RT), sycl::range<1>(RT)), [=](nd_item<1> it) {
            const int tx = (int) it.get_local_id(0);
            const int epoch = m.ctl[0] + 1;
            if (tx == 0) {
                s_nmiss[0] = 0;
                s_lookups[0] = 0;
            }
            it.barrier(sycl::access::fence_space::local_space);
            int lookups = 0;
            for (int q = 0; q < nq; ++q) {
                const int width = steps[q * kStepCount + kStepWidth];
                const int32_t* qi = ids + (long long) q * capp;
                for (int i = tx; i < width; i += RT) {
                    const int b = qi[i] / page_size;
                    if (i > 0 && qi[i - 1] / page_size == b) continue;
                    ++lookups;
                    atomic_ref<int32_t, memory_order::seq_cst, memory_scope::device> pt(m.page_table[b]);
                    const int sl = pt.load();
                    if (sl >= 0) {
                        m.slot_stamp[sl] = epoch;
                        m.slot_ref[sl] = 1;
                    } else if (sl == -1) {
                        int expect = -1;
                        if (pt.compare_exchange_strong(expect, -2)) {
                            atomic_ref<int, memory_order::relaxed, memory_scope::device> nm(s_nmiss[0]);
                            const int idx = nm.fetch_add(1);
                            m.miss_block[idx] = b;
                        }
                    }
                }
            }
            atomic_ref<int, memory_order::relaxed, memory_scope::device> lk(s_lookups[0]);
            lk.fetch_add(lookups);
            it.barrier(sycl::access::fence_space::local_space);
            const int need = s_nmiss[0];
            const int n = (int) m.n_slots;
            int hand = m.ctl[1];
            int got = 0;
            for (int scanned = 0; got < need && scanned < 3 * n; scanned += RT) {
                const int j = (int) (((long long) hand + tx) % n);
                const bool mine = m.slot_stamp[j] == epoch;
                const bool cand = !mine && (m.slot_block[j] < 0 || m.slot_ref[j] == 0);
                // inclusive scan of cand over the block (shfl_up semantics):
                // per-warp up-sweep into warp_sums, then a 32-lane sweep of
                // the warp sums; reads happen before writes at every step
                int x = cand ? 1 : 0;
                const int v = x;
                const int lane = tx & 31, w = tx >> 5;
                scan[tx] = x;
                it.barrier(sycl::access::fence_space::local_space);
                for (int o = 1; o < 32; o <<= 1) {
                    const int y = lane >= o ? scan[(tx & ~31) + (lane - o)] : 0;
                    it.barrier(sycl::access::fence_space::local_space);
                    if (lane >= o) x += y;
                    scan[tx] = x;
                    it.barrier(sycl::access::fence_space::local_space);
                }
                if (lane == 31) warp_sums[w] = x;
                it.barrier(sycl::access::fence_space::local_space);
                if (w == 0) {
                    int t2 = warp_sums[tx];
                    scan[tx] = t2;
                    it.barrier(sycl::access::fence_space::local_space);
                    for (int o = 1; o < 32; o <<= 1) {
                        const int y = lane >= o ? scan[tx - o] : 0;
                        it.barrier(sycl::access::fence_space::local_space);
                        if (lane >= o) t2 += y;
                        scan[tx] = t2;
                        it.barrier(sycl::access::fence_space::local_space);
                    }
                    warp_sums[tx] = t2;
                }
                it.barrier(sycl::access::fence_space::local_space);
                const int total = warp_sums[31];
                const int excl = x - v + (w > 0 ? warp_sums[w - 1] : 0);
                const int rank = excl;
                const int want = need - got;
                if (tx == 0) s_cut[0] = RT;
                it.barrier(sycl::access::fence_space::local_space);
                if (cand && rank == want - 1) s_cut[0] = tx + 1;
                it.barrier(sycl::access::fence_space::local_space);
                const int cut = s_cut[0];
                if (cand && rank < want) {
                    m.miss_slot[got + rank] = j;
                    m.slot_stamp[j] = epoch;
                } else if (tx < cut && !mine) {
                    m.slot_ref[j] = 0;
                }
                got += total < want ? total : want;
                hand = (int) (((long long) hand + cut) % n);
                it.barrier(sycl::access::fence_space::local_space);
            }
            const int placed = got < need ? got : need;
            for (int k = tx; k < need; k += RT) {
                const int b = m.miss_block[k];
                if (k >= placed) {
                    m.page_table[b] = -1;
                    continue;
                }
                const int sl = m.miss_slot[k];
                const int old = m.slot_block[sl];
                if (old >= 0) m.page_table[old] = -1;
                m.slot_block[sl] = b;
                m.slot_stamp[sl] = epoch;
                m.slot_ref[sl] = 1;
                m.page_table[b] = sl;
            }
            it.barrier(sycl::access::fence_space::local_space);
            if (tx == 0) {
                m.ctl[0] = epoch;
                m.ctl[1] = hand;
                m.ctl[2] = placed;
                if (placed < need) m.ctl[3] = 1;
                unsigned long long* c = reinterpret_cast<unsigned long long*>(m.ctl + 4);
                atomic_ref<unsigned long long, memory_order::relaxed, memory_scope::device> c0(c[0]);
                atomic_ref<unsigned long long, memory_order::relaxed, memory_scope::device> c1(c[1]);
                atomic_ref<unsigned long long, memory_order::relaxed, memory_scope::device> c2(c[2]);
                c0.fetch_add((unsigned long long) placed);
                c1.fetch_add((unsigned long long) s_lookups[0]);
                c2.fetch_add(1ull);
            }
        });
    });
    check(stream, "resolve");
    const Runs r = runs_of(slots, host, fmt, s);
    Q(stream).submit([&](sycl::handler& hnd) {
        hnd.parallel_for(sycl::nd_range<1>(sycl::range<1>(96 * 128), sycl::range<1>(128)), [=](nd_item<1> it) {
            const int need = m.ctl[2];
            for (int k = (int) it.get_group(0); k < need; k += 96) {
                const long long b = m.miss_block[k], sl = m.miss_slot[k];
                for (int a = 0; a < r.n; ++a) {
                    const uint8_t* src = r.src[a] + b * r.len[a];
                    uint8_t* dst = r.dst[a] + sl * r.len[a];
                    for (int i = (int) it.get_local_id(0); i < r.len[a] / 16; i += 128) {
                        for (int e = 0; e < 16; ++e) dst[i * 16 + e] = src[i * 16 + e];
                    }
                }
            }
        });
    });
    check(stream, "copy");
}

void kv_ring_table(int32_t* page_table, int64_t n_blocks, int64_t n_slots, void* stream) {
    const long long nb = n_blocks, ns = n_slots;
    Q(stream).parallel_for(sycl::range<1>(64 * 256), [=](size_t i0) {
        for (long long i = (long long) i0; i < nb; i += 64 * 256) page_table[i] = (int32_t) (i % ns);
    });
    check(stream, "ring table");
}

void kv_ring_restore(const QsaAttnPools& slots, const KvHostPools& host, int fmt, int64_t b0, int64_t b1,
                     int64_t n_slots, const QsaShapes& s, void* stream) {
    const Runs r = runs_of(slots, host, fmt, s);
    for (int64_t b = b0; b < b1;) {
        const int64_t sl = b % n_slots, run = std::min<int64_t>(b1 - b, n_slots - sl);
        for (int a = 0; a < r.n; ++a)
            if (cudaMemcpyAsync(r.dst[a] + sl * r.len[a], r.src[a] + b * r.len[a], (size_t) (run * r.len[a]),
                                cudaMemcpyDefault, (cudaStream_t) stream) != cudaSuccess)
                check(stream, "ring restore");
        b += run;
    }
}

void kv_stage_from_host(const QsaAttnPools& stage, const KvHostPools& host, int fmt, int64_t n_blocks,
                        const QsaShapes& s, void* stream) {
    if (n_blocks <= 0) return;
    const Runs r = runs_of(stage, host, fmt, s);
    for (int a = 0; a < r.n; ++a)
        if (cudaMemcpyAsync(r.dst[a], r.src[a], (size_t) (n_blocks * r.len[a]), cudaMemcpyDefault,
                            (cudaStream_t) stream) != cudaSuccess)
            check(stream, "stage");
}

KvStreamCounters kv_stream_counters(const KvStreamMap& m) {
    int32_t c[kKvCtlInts] = {};
    KvStreamCounters r;
    if (m.ctl == nullptr || cudaMemcpy(c, m.ctl, sizeof(c), cudaMemcpyDeviceToHost) != cudaSuccess) return r;
    const unsigned long long* u = reinterpret_cast<const unsigned long long*>(c + 4);
    r.misses = u[0];
    r.lookups = u[1];
    r.calls = u[2];
    r.overflow = c[3] != 0;
    return r;
}

}  // namespace strata::kernels
