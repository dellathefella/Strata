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

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdio>

using cudaError_t = int;
static constexpr cudaError_t cudaSuccess = 0;
static constexpr cudaError_t cudaErrorMemoryAllocation = 2;
static constexpr cudaError_t cudaErrorInvalidValue = 11;
static constexpr cudaError_t cudaErrorUnknown = 999;

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

namespace strata::sycl_compat {

struct stream_wrap {
    sycl::queue q;
    explicit stream_wrap(const sycl::device& d)
        : q(d, sycl::property_list{sycl::property::queue::in_order{}}) {}
};

struct event_wrap {
    sycl::event e;
};

inline sycl::queue& default_queue() {
    static sycl::queue q(sycl::gpu_selector_v,
                         sycl::property_list{sycl::property::queue::in_order{}});
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
inline cudaError_t malloc_impl(void** p, size_t n) {
    if (!p) return last_error() = cudaErrorInvalidValue;
    try {
        *p = sycl::malloc_device(n, default_queue());
        if (!*p) return last_error() = cudaErrorMemoryAllocation;
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
        // shared USM: host-writable and device-visible (the doorbell pattern)
        *p = sycl::malloc_shared(n, default_queue());
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
    return st == sycl::info::event_command_status::complete ? cudaSuccess : 600;
}

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
