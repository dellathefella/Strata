// include/strata/sycl_compat/cuda_fp16.h — the fp16 spellings Strata's CUDA
// kernels use, on sycl::half. Host- and device-callable (icpx compiles both
// passes from one source). Exactness note: f16_bits.hpp remains the authority
// for byte-level conversions (round 193/198); these are for value math only.
#pragma once

#include <sycl/sycl.hpp>

using __half = sycl::half;
using __half2 = sycl::half2;

inline float __half2float(__half h) { return static_cast<float>(h); }
inline __half __float2half(float f) { return sycl::half(f); }
inline __half __float2half_rn(float f) { return sycl::half(f); }

inline unsigned short __half_as_ushort(__half h) {
    return sycl::bit_cast<unsigned short>(h);
}
inline __half __ushort_as_half(unsigned short u) {
    return sycl::bit_cast<sycl::half>(u);
}

inline sycl::float2 __half22float2(__half2 h) { return h.convert<float>(); }
inline __half2 __float22half2_rn(sycl::float2 f) {
    return sycl::half2(sycl::half(f.x()), sycl::half(f.y()));
}
