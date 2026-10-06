// tools/l0_mmap_probe.cpp — can a SYCL kernel read ordinary mmap'd file
// pages (the engine's cudaHostRegister no-op path on SYCL)? On CUDA the
// register makes them device-visible; on Intel discrete GPUs only USM is.
#include "strata/sycl_compat/cuda_runtime.h"
#include <cstdio>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
int main() {
    const char* path = "/data/models/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf";
    int fd = open(path, O_RDONLY);
    if (fd < 0) { std::printf("open fail\n"); return 1; }
    struct stat st; fstat(fd, &st);
    void* m = mmap(nullptr, 1 << 20, PROT_READ, MAP_SHARED, fd, 0);
    if (m == MAP_FAILED) { std::printf("mmap fail\n"); return 1; }
    cudaHostRegister(m, 1 << 20, cudaHostRegisterMapped | cudaHostRegisterPortable);
    float* out = nullptr;
    cudaMalloc(&out, 4096);
    const float* src = (const float*) m;
    sycl::queue& q = strata::sycl_compat::default_queue();
    q.parallel_for(1024, [=](size_t i) { out[i] = src[i] * 2.0f; });
    const cudaError_t e = cudaStreamSynchronize(nullptr);
    std::printf("RESULT sync after mmap read: %s\n", cudaGetErrorString(e));
    float h[4]; q.memcpy(h, out, 16).wait();
    std::printf("values %f %f\n", h[0], h[1]);
    return 0;
}
