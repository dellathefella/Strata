// include/strata/sycl_compat/cuda_runtime.h — CUDA runtime API subset on SYCL/Level-Zero.
//
// The SYCL backend compiles Strata's host translation units (parity tests today,
// src/core/*.cpp tomorrow) unmodified: this header shadows <cuda_runtime.h> via
// the include path and maps the runtime calls onto a single global in-order SYCL
// queue (USM). Semantics deliberately match CUDA's for the callers in this tree:
//   * all pointers are USM — cudaHostGetDevicePointer is the identity
//   * the default "stream" (nullptr/0) is the global in-order queue
//   * cudaMemcpy is synchronous; cudaMemcpyAsync enqueues on its stream's queue
//   * cudaGetLastError returns and clears a thread-local code
// Graphs (cudaGraph*) are NOT here yet — milestone 2 (GraphRegistry on L0).
#pragma once

#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/sycl.hpp>

#include <atomic>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <mutex>
#include <unordered_map>

using cudaError_t = int;
static constexpr cudaError_t cudaSuccess = 0;
static constexpr cudaError_t cudaErrorMemoryAllocation = 2;
static constexpr cudaError_t cudaErrorInvalidValue = 11;
static constexpr cudaError_t cudaErrorNotReady = 600;
static constexpr cudaError_t cudaErrorUnknown = 999;

static constexpr unsigned cudaEventDefault = 0;
static constexpr unsigned cudaEventBlockingSync = 1;
static constexpr unsigned cudaEventDisableTiming = 2;
static constexpr unsigned cudaEventInterprocess = 4;

static constexpr unsigned cudaHostRegisterDefault = 0;
static constexpr unsigned cudaHostRegisterPortable = 1;
static constexpr unsigned cudaHostRegisterMapped = 2;
static constexpr unsigned cudaHostRegisterIoMemory = 4;

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4,
};

enum cudaHostAllocFlags {
    cudaHostAllocDefault = 0,
    cudaHostAllocPortable = 1,
    cudaHostAllocMapped = 2,
    cudaHostAllocWriteCombined = 4,
};

enum cudaStreamFlags {
    cudaStreamDefault = 0,
    cudaStreamNonBlocking = 1,
};

namespace strata::sycl_compat {

namespace ex = sycl::ext::oneapi::experimental;

struct stream_wrap {
    sycl::queue q;
    // non-null between cudaStreamBeginCapture and cudaStreamEndCapture
    std::unique_ptr<ex::command_graph<ex::graph_state::modifiable>> rec;
    bool capturing = false;
    explicit stream_wrap(const sycl::device& d)
        : q(d, sycl::property_list{sycl::property::queue::in_order{},
                                   sycl::property::queue::enable_profiling{}}) {}
};

struct graph_wrap {
    std::unique_ptr<ex::command_graph<ex::graph_state::modifiable>> g;
};

// owns the finalized (executable) graph; cudaGraphExecDestroy frees it
struct graphexec_wrap {
    std::unique_ptr<ex::command_graph<ex::graph_state::executable>> g;
};

struct event_wrap {
    sycl::event e;
};

inline sycl::queue& default_queue() {
    static sycl::queue q(sycl::gpu_selector_v,
                         sycl::property_list{sycl::property::queue::in_order{},
                                             sycl::property::queue::enable_profiling{}});
    return q;
}

inline sycl::queue& q_for(void* stream) {
    return stream ? static_cast<stream_wrap*>(stream)->q : default_queue();
}

inline cudaError_t& last_error() {
    thread_local cudaError_t e = cudaSuccess;
    return e;
}

}  // namespace strata::sycl_compat

using cudaStream_t = strata::sycl_compat::stream_wrap*;
using cudaEvent_t = strata::sycl_compat::event_wrap*;

inline const char* cudaGetErrorString(cudaError_t e) {
    switch (e) {
        case cudaSuccess: return "no error";
        case cudaErrorMemoryAllocation: return "out of memory";
        case cudaErrorInvalidValue: return "invalid argument";
        default: return "sycl_compat: unknown error";
    }
}

inline cudaError_t cudaGetLastError() {
    cudaError_t e = strata::sycl_compat::last_error();
    strata::sycl_compat::last_error() = cudaSuccess;
    return e;
}

inline cudaError_t cudaPeekAtLastError() { return strata::sycl_compat::last_error(); }

namespace strata::sycl_compat {
inline std::atomic<size_t>& allocated() {
    static std::atomic<size_t> bytes{0};
    return bytes;
}
inline size_t allocated_bytes() { return allocated().load(); }

inline std::mutex& sizes_mutex() {
    static std::mutex m;
    return m;
}
inline std::unordered_map<void*, size_t>& sizes_map() {
    static std::unordered_map<void*, size_t> sizes;
    return sizes;
}
inline void remember_alloc(void* p, size_t n) {
    {
        std::lock_guard<std::mutex> g(sizes_mutex());
        sizes_map()[p] = n;
    }
    allocated().fetch_add(n);
}
inline void forget_alloc(void* p) {
    size_t n = 0;
    {
        std::lock_guard<std::mutex> g(sizes_mutex());
        auto it = sizes_map().find(p);
        if (it != sizes_map().end()) {
            n = it->second;
            sizes_map().erase(it);
        }
    }
    allocated().fetch_sub(n);
}

inline cudaError_t malloc_impl(void** p, size_t n) {
    if (!p) return last_error() = cudaErrorInvalidValue;
    try {
        *p = sycl::malloc_device(n, default_queue());
        if (!*p) return last_error() = cudaErrorMemoryAllocation;
        remember_alloc(*p, n);
    } catch (const sycl::exception& ex) {
        std::fprintf(stderr, "sycl_compat cudaMalloc: %s\n", ex.what());
        return last_error() = cudaErrorMemoryAllocation;
    }
    return last_error() = cudaSuccess;
}
}  // namespace strata::sycl_compat

// CUDA's cudaMalloc is a template on the pointee type
template <typename T>
inline cudaError_t cudaMalloc(T** p, size_t n) {
    return strata::sycl_compat::malloc_impl(reinterpret_cast<void**>(p), n);
}

inline cudaError_t cudaFree(void* p) {
    if (!p) return cudaSuccess;
    try {
        strata::sycl_compat::forget_alloc(p);
        sycl::free(p, strata::sycl_compat::default_queue());
    } catch (const sycl::exception& ex) {
        std::fprintf(stderr, "sycl_compat cudaFree: %s\n", ex.what());
        return strata::sycl_compat::last_error() = cudaErrorUnknown;
    }
    return strata::sycl_compat::last_error() = cudaSuccess;
}

inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t n, cudaMemcpyKind) {
    try {
        strata::sycl_compat::default_queue()
            .memcpy(dst, const_cast<void*>(src), n)
            .wait_and_throw();
    } catch (const sycl::exception& ex) {
        std::fprintf(stderr, "sycl_compat cudaMemcpy: %s\n", ex.what());
        return strata::sycl_compat::last_error() = cudaErrorUnknown;
    }
    return strata::sycl_compat::last_error() = cudaSuccess;
}

inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t n,
                                   cudaMemcpyKind, cudaStream_t s) {
    try {
        strata::sycl_compat::q_for(s).memcpy(dst, const_cast<void*>(src), n);
    } catch (const sycl::exception& ex) {
        std::fprintf(stderr, "sycl_compat cudaMemcpyAsync: %s\n", ex.what());
        return strata::sycl_compat::last_error() = cudaErrorUnknown;
    }
    return strata::sycl_compat::last_error() = cudaSuccess;
}

inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t n, cudaMemcpyKind k) {
    return cudaMemcpyAsync(dst, src, n, k, nullptr);
}

inline cudaError_t cudaMemset(void* dst, int v, size_t n) {
    try {
        strata::sycl_compat::default_queue().memset(dst, v, n).wait_and_throw();
    } catch (const sycl::exception& ex) {
        std::fprintf(stderr, "sycl_compat cudaMemset: %s\n", ex.what());
        return strata::sycl_compat::last_error() = cudaErrorUnknown;
    }
    return strata::sycl_compat::last_error() = cudaSuccess;
}

inline cudaError_t cudaMemsetAsync(void* dst, int v, size_t n, cudaStream_t s) {
    try {
        strata::sycl_compat::q_for(s).memset(dst, v, n);
    } catch (const sycl::exception& ex) {
        std::fprintf(stderr, "sycl_compat cudaMemsetAsync: %s\n", ex.what());
        return strata::sycl_compat::last_error() = cudaErrorUnknown;
    }
    return strata::sycl_compat::last_error() = cudaSuccess;
}

namespace strata::sycl_compat {
inline cudaError_t host_alloc_impl(void** p, size_t n) {
    if (!p) return last_error() = cudaErrorInvalidValue;
    try {
        // USM HOST: pinned host pages, device-visible over PCIe (the doorbell
        // pattern). NOT malloc_shared: on discrete Arc a shared allocation can
        // sit in device memory, and host syscalls (fread into a stage buffer)
        // fault on it with EFAULT (measured: WeightTable::load short read).
        *p = sycl::malloc_host(n, default_queue());
        if (!*p) return last_error() = cudaErrorMemoryAllocation;
    } catch (const sycl::exception& ex) {
        std::fprintf(stderr, "sycl_compat cudaHostAlloc: %s\n", ex.what());
        return last_error() = cudaErrorMemoryAllocation;
    }
    return last_error() = cudaSuccess;
}
}  // namespace strata::sycl_compat

template <typename T>
inline cudaError_t cudaHostAlloc(T** p, size_t n, unsigned /*flags*/) {
    return strata::sycl_compat::host_alloc_impl(reinterpret_cast<void**>(p), n);
}

inline cudaError_t cudaFreeHost(void* p) { return cudaFree(p); }

inline cudaError_t cudaHostGetDevicePointer(void** dev, void* host, unsigned /*flags*/) {
    if (!dev) return strata::sycl_compat::last_error() = cudaErrorInvalidValue;
    *dev = host;  // USM: one address space
    return cudaSuccess;
}

inline cudaError_t cudaDeviceSynchronize() {
    try {
        strata::sycl_compat::default_queue().wait_and_throw();
    } catch (const sycl::exception& ex) {
        std::fprintf(stderr, "sycl_compat cudaDeviceSynchronize: %s\n", ex.what());
        return strata::sycl_compat::last_error() = cudaErrorUnknown;
    }
    return strata::sycl_compat::last_error() = cudaSuccess;
}

inline cudaError_t cudaStreamCreate(cudaStream_t* s) {
    try {
        *s = new strata::sycl_compat::stream_wrap(
            strata::sycl_compat::default_queue().get_device());
    } catch (const sycl::exception& ex) {
        std::fprintf(stderr, "sycl_compat cudaStreamCreate: %s\n", ex.what());
        return strata::sycl_compat::last_error() = cudaErrorUnknown;
    }
    return cudaSuccess;
}

inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* s, unsigned /*flags*/) {
    return cudaStreamCreate(s);
}

inline cudaError_t cudaStreamDestroy(cudaStream_t s) {
    if (!s) return cudaSuccess;
    s->q.wait_and_throw();
    delete s;
    return cudaSuccess;
}

inline cudaError_t cudaStreamSynchronize(cudaStream_t s) {
    try {
        strata::sycl_compat::q_for(s).wait_and_throw();
    } catch (const sycl::exception& ex) {
        std::fprintf(stderr, "sycl_compat cudaStreamSynchronize: %s\n", ex.what());
        return strata::sycl_compat::last_error() = cudaErrorUnknown;
    }
    return cudaSuccess;
}

inline cudaError_t cudaEventCreate(cudaEvent_t* e) {
    *e = new strata::sycl_compat::event_wrap();
    return cudaSuccess;
}

inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e, unsigned /*flags*/) {
    return cudaEventCreate(e);
}

inline cudaError_t cudaEventDestroy(cudaEvent_t e) {
    delete e;
    return cudaSuccess;
}

inline cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t s) {
    if (!e) return strata::sycl_compat::last_error() = cudaErrorInvalidValue;
    // an event on an in-order queue: a marker task whose completion == all prior work
    e->e = strata::sycl_compat::q_for(s).submit(
        [&](sycl::handler& h) { h.single_task([]() {}); });
    return cudaSuccess;
}

inline cudaError_t cudaEventSynchronize(cudaEvent_t e) {
    e->e.wait_and_throw();
    return cudaSuccess;
}

inline cudaError_t cudaEventQuery(cudaEvent_t e) {
    // 0 = complete, 600 = not ready (cudaErrorNotReady)
    const auto st = e->e.get_info<sycl::info::event::command_execution_status>();
    return st == sycl::info::event_command_status::complete ? cudaSuccess : cudaErrorNotReady;
}

inline cudaError_t cudaEventRecord(cudaEvent_t e) { return cudaEventRecord(e, nullptr); }

inline cudaError_t cudaStreamQuery(cudaStream_t s) {
    // in-order queue: the engine uses this as a flush hint while host-spinning
    // on the doorbell; SYCL needs no flush (submissions reach the driver eagerly).
    (void) s;
    return cudaSuccess;
}

inline cudaError_t cudaStreamWaitEvent(cudaStream_t s, cudaEvent_t e, unsigned = 0) {
    if (!e) return strata::sycl_compat::last_error() = cudaErrorInvalidValue;
    try {
        // in-order queues serialize anyway; make the dependency explicit
        strata::sycl_compat::q_for(s).submit(
            [&](sycl::handler& h) { h.depends_on(e->e); h.single_task([]() {}); });
    } catch (const sycl::exception& exn) {
        std::fprintf(stderr, "sycl_compat cudaStreamWaitEvent: %s\n", exn.what());
        return strata::sycl_compat::last_error() = cudaErrorUnknown;
    }
    return cudaSuccess;
}

inline cudaError_t cudaLaunchHostFunc(cudaStream_t s, void (*fn)(void*), void* arg) {
    try {
        strata::sycl_compat::q_for(s).submit([&](sycl::handler& h) {
            h.host_task([fn, arg]() { fn(arg); });
        });
    } catch (const sycl::exception& exn) {
        std::fprintf(stderr, "sycl_compat cudaLaunchHostFunc: %s\n", exn.what());
        return strata::sycl_compat::last_error() = cudaErrorUnknown;
    }
    return cudaSuccess;
}

// row-major 2D copy with per-row pitch (USM: plain strided memcpy via a kernel)
inline cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src,
                                     size_t spitch, size_t width, size_t height,
                                     cudaMemcpyKind, cudaStream_t s) {
    try {
        auto& q = strata::sycl_compat::q_for(s);
        unsigned char* d = static_cast<unsigned char*>(dst);
        const unsigned char* sr = static_cast<const unsigned char*>(src);
        q.parallel_for(height, [=](size_t row) {
             const unsigned char* s_row = sr + row * spitch;
             unsigned char* d_row = d + row * dpitch;
             for (size_t i = 0; i < width; ++i) d_row[i] = s_row[i];
         })
            .wait_and_throw();
    } catch (const sycl::exception& exn) {
        std::fprintf(stderr, "sycl_compat cudaMemcpy2DAsync: %s\n", exn.what());
        return strata::sycl_compat::last_error() = cudaErrorUnknown;
    }
    return cudaSuccess;
}

inline cudaError_t cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch,
                                size_t width, size_t height, cudaMemcpyKind k) {
    return cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, k, nullptr);
}

template <typename T>
inline cudaError_t cudaMallocHost(T** p, size_t n) {
    return strata::sycl_compat::host_alloc_impl(reinterpret_cast<void**>(p), n);
}

// USM host/shared memory is already registered and device-visible: no-ops.
inline cudaError_t cudaHostRegister(void*, size_t, unsigned = cudaHostRegisterDefault) {
    return cudaSuccess;
}
inline cudaError_t cudaHostUnregister(void*) { return cudaSuccess; }

inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t a, cudaEvent_t b) {
    try {
        auto t0 = a->e.get_profiling_info<sycl::info::event_profiling::command_end>();
        auto t1 = b->e.get_profiling_info<sycl::info::event_profiling::command_start>();
        *ms = (double)(t1 - t0) / 1e6;
    } catch (const sycl::exception&) {
        // queues without profiling: fall back to wall-clock zero
        *ms = 0.f;
    }
    return cudaSuccess;
}

inline cudaError_t cudaGetDevice(int* dev) {
    // single-device backend for now; multi-device placement lands with the
    // 1+N milestone (streams will carry their device).
    if (!dev) return strata::sycl_compat::last_error() = cudaErrorInvalidValue;
    *dev = 0;
    return cudaSuccess;
}

inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }

static constexpr unsigned cudaDeviceScheduleSpin = 1;
static constexpr unsigned cudaDeviceMapHost = 8;
inline cudaError_t cudaInitDevice(int, unsigned, size_t) { return cudaSuccess; }

// ---------------- device query (src/core/device.cu) -------------------------
struct cudaDeviceProp {
    char name[256] = {0};
    int major = 9;             // the CC gate is bypassed under STRATA_USE_SYCL;
    int minor = 0;             // these exist so host formatting code compiles
    int multiProcessorCount = 0;
    size_t totalGlobalMem = 0;
    int warpSize = 32;         // Xe2 executes in SIMD32; the ported kernels use
                               // per-32-lane local trees, not sub_group shuffles
    char gcnArchName[64] = {0};
};

inline cudaError_t cudaGetDeviceCount(int* count) {
    if (!count) return strata::sycl_compat::last_error() = cudaErrorInvalidValue;
    try {
        *count = (int) sycl::device::get_devices(sycl::info::device_type::gpu).size();
        if (*count == 0) return strata::sycl_compat::last_error() = cudaErrorUnknown;
    } catch (const sycl::exception& exn) {
        std::fprintf(stderr, "sycl_compat cudaGetDeviceCount: %s\n", exn.what());
        return strata::sycl_compat::last_error() = cudaErrorUnknown;
    }
    return cudaSuccess;
}

inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p, int ordinal) {
    if (!p) return strata::sycl_compat::last_error() = cudaErrorInvalidValue;
    try {
        const auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
        if (ordinal < 0 || ordinal >= (int) devs.size())
            return strata::sycl_compat::last_error() = cudaErrorInvalidValue;
        const sycl::device& d = devs[ordinal];
        const std::string nm = d.get_info<sycl::info::device::name>();
        std::snprintf(p->name, sizeof(p->name), "%s", nm.c_str());
        p->multiProcessorCount =
            (int) d.get_info<sycl::info::device::max_compute_units>();
        p->totalGlobalMem = d.get_info<sycl::info::device::global_mem_size>();
    } catch (const sycl::exception& exn) {
        std::fprintf(stderr, "sycl_compat cudaGetDeviceProperties: %s\n", exn.what());
        return strata::sycl_compat::last_error() = cudaErrorUnknown;
    }
    return cudaSuccess;
}

// L0 exposes no free-memory query; track our own allocations (the engine takes
// one big DeviceArena plus a handful of side buffers, so this is close).
inline cudaError_t cudaMemGetInfo(size_t* free_b, size_t* total_b) {
    try {
        const sycl::device d = strata::sycl_compat::default_queue().get_device();
        const size_t total = d.get_info<sycl::info::device::global_mem_size>();
        const size_t used = strata::sycl_compat::allocated_bytes();
        if (total_b) *total_b = total;
        if (free_b) *free_b = total > used ? total - used : 0;
    } catch (const sycl::exception& exn) {
        std::fprintf(stderr, "sycl_compat cudaMemGetInfo: %s\n", exn.what());
        return strata::sycl_compat::last_error() = cudaErrorUnknown;
    }
    return cudaSuccess;
}

inline cudaError_t cudaDriverGetVersion(int* v) {
    if (v) *v = 12000;
    return cudaSuccess;
}
inline cudaError_t cudaRuntimeGetVersion(int* v) {
    if (v) *v = 12000;
    return cudaSuccess;
}

struct cudaFuncAttributes {
    int numRegs = 0;
    size_t sharedSizeBytes = 0;
    size_t localSizeBytes = 0;
    int maxThreadsPerBlock = 1024;
};
template <typename F>
inline cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* a, F) {
    if (!a) return strata::sycl_compat::last_error() = cudaErrorInvalidValue;
    *a = cudaFuncAttributes{};
    return cudaSuccess;
}

enum cudaDeviceAttr {
    cudaDevAttrMultiProcessorCount = 16,
    cudaDevAttrClockRate = 13,
};
inline cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int) {
    if (!value) return strata::sycl_compat::last_error() = cudaErrorInvalidValue;
    try {
        const sycl::device d = strata::sycl_compat::default_queue().get_device();
        switch (attr) {
            case cudaDevAttrMultiProcessorCount:
                *value = (int) d.get_info<sycl::info::device::max_compute_units>();
                break;
            case cudaDevAttrClockRate:
                *value = (int) d.get_info<sycl::info::device::max_clock_frequency>() * 1000;
                break;
            default:
                *value = 0;
        }
    } catch (const sycl::exception&) {
        *value = 0;
    }
    return cudaSuccess;
}

// ---------------- CUDA graphs on SYCL experimental command graphs -----------
using cudaGraph_t = strata::sycl_compat::graph_wrap*;
using cudaGraphExec_t = strata::sycl_compat::graphexec_wrap*;

enum cudaStreamCaptureMode {
    cudaStreamCaptureModeGlobal = 0,
    cudaStreamCaptureModeThreadLocal = 1,
    cudaStreamCaptureModeRelaxed = 2,
};

inline cudaError_t cudaStreamBeginCapture(cudaStream_t s, cudaStreamCaptureMode) {
    if (!s) return strata::sycl_compat::last_error() = cudaErrorInvalidValue;
    namespace ex = sycl::ext::oneapi::experimental;
    try {
        s->rec = std::make_unique<ex::command_graph<ex::graph_state::modifiable>>(s->q);
        s->rec->begin_recording(s->q);
        s->capturing = true;
    } catch (const sycl::exception& exn) {
        std::fprintf(stderr, "sycl_compat cudaStreamBeginCapture: %s\n", exn.what());
        return strata::sycl_compat::last_error() = cudaErrorUnknown;
    }
    return cudaSuccess;
}

inline cudaError_t cudaStreamEndCapture(cudaStream_t s, cudaGraph_t* graph) {
    if (!s || !graph || !s->rec)
        return strata::sycl_compat::last_error() = cudaErrorInvalidValue;
    try {
        s->rec->end_recording();
        s->capturing = false;
        *graph = new strata::sycl_compat::graph_wrap{std::move(s->rec)};
    } catch (const sycl::exception& exn) {
        std::fprintf(stderr, "sycl_compat cudaStreamEndCapture: %s\n", exn.what());
        return strata::sycl_compat::last_error() = cudaErrorUnknown;
    }
    return cudaSuccess;
}

inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t* exec, cudaGraph_t graph,
                                        unsigned long long) {
    if (!exec || !graph) return strata::sycl_compat::last_error() = cudaErrorInvalidValue;
    try {
        *exec = new strata::sycl_compat::graphexec_wrap{
            std::make_unique<sycl::ext::oneapi::experimental::command_graph<
                sycl::ext::oneapi::experimental::graph_state::executable>>(
                graph->g->finalize())};
    } catch (const sycl::exception& exn) {
        std::fprintf(stderr, "sycl_compat cudaGraphInstantiate: %s\n", exn.what());
        return strata::sycl_compat::last_error() = cudaErrorUnknown;
    }
    return cudaSuccess;
}

// the 5-arg CUDA <=11 spelling still used by graph.cpp / verify.cpp
inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t* exec, cudaGraph_t graph,
                                        void*, void*, size_t) {
    return cudaGraphInstantiate(exec, graph, 0ull);
}

// node enumeration: the engine only needs the COUNT (graph.cpp refuses
// zero-node captures); verify.cpp's per-node listing is CUDA/HIP-only and
// stays guarded out under STRATA_USE_SYCL.
using cudaGraphNode_t = void*;
enum cudaGraphNodeType {
    cudaGraphNodeTypeKernel = 0,
    cudaGraphNodeTypeMemcpy = 1,
    cudaGraphNodeTypeMemset = 2,
};

inline cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t* nodes,
                                     size_t* numNodes) {
    if (!graph || !numNodes) return strata::sycl_compat::last_error() = cudaErrorInvalidValue;
    size_t n = 0;
    try {
        n = graph->g->get_nodes().size();
    } catch (const sycl::exception& exn) {
        std::fprintf(stderr, "sycl_compat cudaGraphGetNodes: %s\n", exn.what());
        return strata::sycl_compat::last_error() = cudaErrorUnknown;
    }
    if (nodes) {
        for (size_t i = 0; i < n; ++i) nodes[i] = reinterpret_cast<cudaGraphNode_t>(i + 1);
    }
    *numNodes = n;
    return cudaSuccess;
}

inline cudaError_t cudaGraphLaunch(cudaGraphExec_t exec, cudaStream_t s) {
    if (!exec) return strata::sycl_compat::last_error() = cudaErrorInvalidValue;
    try {
        // in-order queue: the returned event is dropped, ordering is implicit
        strata::sycl_compat::q_for(s).ext_oneapi_graph(*exec->g);
    } catch (const sycl::exception& exn) {
        std::fprintf(stderr, "sycl_compat cudaGraphLaunch: %s\n", exn.what());
        return strata::sycl_compat::last_error() = cudaErrorUnknown;
    }
    return cudaSuccess;
}

inline cudaError_t cudaGraphUpload(cudaGraphExec_t, cudaStream_t) { return cudaSuccess; }

inline cudaError_t cudaGraphExecDestroy(cudaGraphExec_t exec) {
    delete exec;
    return cudaSuccess;
}

inline cudaError_t cudaGraphDestroy(cudaGraph_t graph) {
    delete graph;
    return cudaSuccess;
}

enum cudaStreamCaptureStatus {
    cudaStreamCaptureStatusNone = 0,
    cudaStreamCaptureStatusActive = 1,
    cudaStreamCaptureStatusInvalidated = 2,
};

inline cudaError_t cudaStreamIsCapturing(cudaStream_t s, cudaStreamCaptureStatus* status) {
    if (!status) return strata::sycl_compat::last_error() = cudaErrorInvalidValue;
    *status = (s && s->capturing) ? cudaStreamCaptureStatusActive : cudaStreamCaptureStatusNone;
    return cudaSuccess;
}

// Math spellings: Strata's portable HD headers call the C names (cosf, ...).
// Those are NOT device-callable on the SYCL pass, and neither global aliases
// nor macros work (they collide with glibc's declarations, which the device
// pass still parses). Instead, HD headers gain a guarded SYCL branch calling
// sycl:: builtins directly — see rope_scaling.hpp (STRATA_USE_SYCL).
