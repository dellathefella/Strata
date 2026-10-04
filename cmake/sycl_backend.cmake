# =============================================================================
# cmake/sycl_backend.cmake — Intel Arc (SYCL / Level-Zero) backend
#
# Configure with:
#   cmake -DCMAKE_CXX_COMPILER=icpx -DSTRATA_ENABLE_SYCL=ON ..
#
# Mirrors cmake/hip_backend.cmake's role: defines the runtime interface target
# the ported sources compile against. Unlike HIP (a rename shim), SYCL gets
# hand-ported kernels in src/kernels/sycl/ (dpct is not shipped with oneAPI
# 2026.1; hand ports also target DPAS directly — see docs/SYCL_PORT.md).
# Host translation units keep including <cuda_runtime.h>; the sycl_compat
# directory shadows it with a USM/in-order-queue implementation.
# =============================================================================

find_program(STRATA_ICPX icpx REQUIRED)
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM|IntelDPCPP")
  message(FATAL_ERROR
    "STRATA_ENABLE_SYCL needs icpx as the C++ compiler; got ${CMAKE_CXX_COMPILER_ID}. "
    "Reconfigure with -DCMAKE_CXX_COMPILER=icpx")
endif()
message(STATUS "Strata SYCL backend: ${CMAKE_CXX_COMPILER}")

set(STRATA_SYCL_COMPAT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/strata/sycl_compat")

add_library(strata_sycl_runtime INTERFACE)
# BEFORE: sycl_compat shadows <cuda_runtime.h>/<cuda_fp16.h> for every consumer
target_include_directories(strata_sycl_runtime BEFORE INTERFACE
  "${STRATA_SYCL_COMPAT_INCLUDE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_options(strata_sycl_runtime INTERFACE
  -fsycl -ffp-contract=off)   # no FMA contraction: __fmul_rn fidelity (quantize_act)
target_link_options(strata_sycl_runtime INTERFACE -fsycl)
target_compile_definitions(strata_sycl_runtime INTERFACE STRATA_USE_SYCL=1)

# ---- ported kernel families (milestone 3: Q4/Q8 activation path first) ----
add_library(strata_kernels_sycl STATIC
  src/kernels/sycl/quantize_act.cpp
  src/kernels/sycl/s2_gemv_q8.cpp
  src/kernels/sycl/s_gemv.cpp)
target_link_libraries(strata_kernels_sycl PUBLIC strata_sycl_runtime strata_warnings)

# ---- parity gate: the CUDA parity test, compiled unmodified against the shim ----
add_executable(s2_gemv_q8_parity_sycl src/kernels/s2_gemv_q8_parity.cpp)
target_link_libraries(s2_gemv_q8_parity_sycl PRIVATE strata_kernels_sycl strata_sycl_runtime)
enable_testing()
add_test(NAME s2_gemv_q8_parity_sycl COMMAND s2_gemv_q8_parity_sycl --selftest)

# ---- milestone benches ----
add_executable(l0_spin_bench tools/l0_spin_bench.cpp)
target_compile_options(l0_spin_bench PRIVATE -fsycl -O2)
target_link_options(l0_spin_bench PRIVATE -fsycl)

add_executable(q4q8_kernel_bench tools/q4q8_kernel_bench.cpp)
target_compile_options(q4q8_kernel_bench PRIVATE -fsycl -O2)
target_link_options(q4q8_kernel_bench PRIVATE -fsycl)

add_executable(mx_caps tools/mx_caps.cpp)
target_compile_options(mx_caps PRIVATE -fsycl -O2)
target_link_options(mx_caps PRIVATE -fsycl)

message(STATUS "Strata: SYCL enabled (kernels: quantize_act, s2_gemv_q8, s_gemv/s_gemv_split)")
