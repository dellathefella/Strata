// src/core/poison_sycl.cpp — the DeviceArena poison fill for the SYCL backend.
// device.cu keeps its <<<>>> kernel for CUDA/HIP; icpx cannot parse launch
// syntax, so the SYCL build links this translation unit instead (declared in
// device.cu under STRATA_USE_SYCL). Same NaN bit pattern, same semantics.
#include <sycl/sycl.hpp>

#include <cstdint>

#include "strata/sycl_compat/cuda_runtime.h"

namespace strata::core {

void poison_fill(void* p, uint64_t n_floats) {
    float* f = static_cast<float*>(p);
    strata::sycl_compat::default_queue().parallel_for(
        (size_t) n_floats, [=](size_t i) {
            f[i] = sycl::bit_cast<float>((unsigned int) 0x7fc00000u);
        });
}

}  // namespace strata::core
