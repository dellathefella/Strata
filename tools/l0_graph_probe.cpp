// tools/l0_graph_probe.cpp — does a captured SYCL command graph actually
// replay, and are its stores to USM-host memory visible to the polling host?
// Mirrors the verify window's doorbell pattern at minimum size.
#include "strata/sycl_compat/cuda_runtime.h"
#include <chrono>
#include <cstdio>
int main() {
    uint32_t* flag_raw = nullptr;
    if (cudaHostAlloc(&flag_raw, 64, cudaHostAllocMapped) != cudaSuccess) { std::printf("hostalloc fail\n"); return 1; }
    volatile uint32_t* flag = flag_raw;
    flag[0] = 0;
    cudaStream_t s = nullptr;
    cudaStreamCreate(&s);
    if (cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { std::printf("begin fail\n"); return 1; }
    uint32_t* f2 = flag_raw;
    sycl::queue& q = strata::sycl_compat::q_for(s);
    q.single_task([=] {
        sycl::atomic_ref<uint32_t, sycl::memory_order::seq_cst, sycl::memory_scope::system> a(*f2);
        a.store(1u, sycl::memory_order::release);
    });
    cudaGraph_t g = nullptr;
    if (cudaStreamEndCapture(s, &g) != cudaSuccess) { std::printf("end fail\n"); return 1; }
    cudaGraphExec_t ex = nullptr;
    if (cudaGraphInstantiate(&ex, g, 0) != cudaSuccess) { std::printf("inst fail\n"); return 1; }
    if (cudaGraphLaunch(ex, s) != cudaSuccess) { std::printf("launch fail\n"); return 1; }
    const auto t0 = std::chrono::steady_clock::now();
    while (flag[0] == 0) {
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(10)) { std::printf("RESULT: no publish in 10s\n"); return 2; }
    }
    const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("RESULT: graph replay published in %.2f ms\n", ms);
    cudaStreamSynchronize(s);
    return 0;
}
