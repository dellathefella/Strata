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
  -fsycl -ffp-contract=off    # no FMA contraction: __fmul_rn fidelity (quantize_act)
  -foffload-fp32-prec-div)    # icpx DEFAULTS device fp32 `/` to a ~28%-1ulp-off
                              # reciprocal sequence (measured, 1M-sample sweep);
                              # Strata's byte-exact contracts need IEEE division
target_link_options(strata_sycl_runtime INTERFACE -fsycl -foffload-fp32-prec-div)
target_compile_definitions(strata_sycl_runtime INTERFACE STRATA_USE_SYCL=1)

# ---- ported kernel families (milestone 3: Q4/Q8 activation path first) ----
add_library(strata_kernels_sycl STATIC
  src/kernels/sycl/quantize_act.cpp
  src/kernels/sycl/s2_gemv_q8.cpp
  src/kernels/sycl/s_gemv.cpp
  src/kernels/sycl/router_top10.cpp
  src/kernels/sycl/rope.cpp
  src/kernels/sycl/native_rope.cpp
  src/kernels/sycl/elementwise.cpp
  src/kernels/sycl/dequant_bf16.cpp
  src/kernels/sycl/s2_gemv.cpp
  src/kernels/sycl/s2_gemv_quads.cpp
  src/kernels/sycl/s2_gemv_fast.cpp
  src/kernels/sycl/dequant_s2.cpp
  src/kernels/sycl/bf16_gemv.cpp
  src/kernels/sycl/native_bf16.cpp
  src/kernels/sycl/native_gr_norm.cpp
  src/kernels/sycl/native_gr_postops.cpp
  src/kernels/sycl/gdn.cpp
  src/kernels/sycl/gr.cpp
  src/kernels/sycl/verify_kernels.cpp
  src/kernels/sycl/fused_gr.cpp)
target_link_libraries(strata_kernels_sycl PUBLIC strata_sycl_runtime strata_warnings)

# ---- parity gates: the CUDA parity tests, compiled unmodified against the shim ----
add_executable(s2_gemv_q8_parity_sycl src/kernels/s2_gemv_q8_parity.cpp)
target_link_libraries(s2_gemv_q8_parity_sycl PRIVATE strata_kernels_sycl strata_sycl_runtime)

add_executable(router_top10_parity_sycl src/kernels/router_top10_parity.cpp)
target_link_libraries(router_top10_parity_sycl PRIVATE strata_kernels_sycl strata_sycl_runtime)

add_executable(rope_parity_sycl src/kernels/rope_parity.cpp)
target_link_libraries(rope_parity_sycl PRIVATE strata_kernels_sycl strata_sycl_runtime)

add_executable(elementwise_parity_sycl src/kernels/elementwise_parity.cpp)
target_link_libraries(elementwise_parity_sycl PRIVATE strata_kernels_sycl strata_sycl_runtime)

add_executable(s2_gemv_parity_sycl src/kernels/s2_gemv_parity.cpp)
target_link_libraries(s2_gemv_parity_sycl PRIVATE strata_kernels_sycl strata_sycl_runtime)

add_executable(s_gemv_parity_sycl src/kernels/s_gemv_parity.cpp)
target_link_libraries(s_gemv_parity_sycl PRIVATE strata_kernels_sycl strata_sycl_runtime)

add_executable(dequant_s2_parity_sycl src/kernels/dequant_s2_parity.cpp)
target_link_libraries(dequant_s2_parity_sycl PRIVATE strata_kernels_sycl strata_sycl_runtime)

add_executable(bf16_gemv_parity_sycl src/kernels/bf16_gemv_parity.cpp)
target_link_libraries(bf16_gemv_parity_sycl PRIVATE strata_kernels_sycl strata_sycl_runtime)

add_executable(gdn_parity_sycl src/kernels/gdn_parity.cpp)
target_link_libraries(gdn_parity_sycl PRIVATE strata_kernels_sycl strata_sycl_runtime)

add_executable(quantize_act_parity_sycl src/kernels/quantize_act_parity.cpp)
target_link_libraries(quantize_act_parity_sycl PRIVATE strata_kernels_sycl strata_sycl_runtime)
# Two icpx host-codegen quirks, both measured (dbg repros in the port log):
#   1. -O1..-O3 inline+fold `want[i] = x[i]*s` into a 1-ulp-off double-rounded
#      constant (466/1024 false diffs vs the IEEE-exact device kernel);
#      -fno-inline stops it.
#   2. on baseline x86-64 std::fma lowers to mul+add (the test's fma_diff
#      self-check then sees 0 and fails); -march=native emits vfmadd.
# The DEVICE kernel is IEEE-exact at every level; this is host-reference hygiene.
target_compile_options(elementwise_parity_sycl PRIVATE -O2 -fno-inline -march=native)

enable_testing()
add_test(NAME s2_gemv_q8_parity_sycl COMMAND s2_gemv_q8_parity_sycl --selftest)
add_test(NAME router_top10_parity_sycl COMMAND router_top10_parity_sycl --selftest)
add_test(NAME rope_parity_sycl COMMAND rope_parity_sycl --selftest)
add_test(NAME elementwise_parity_sycl COMMAND elementwise_parity_sycl --selftest)
add_test(NAME s2_gemv_parity_sycl COMMAND s2_gemv_parity_sycl --selftest)
add_test(NAME s_gemv_parity_sycl COMMAND s_gemv_parity_sycl --selftest)
add_test(NAME dequant_s2_parity_sycl COMMAND dequant_s2_parity_sycl --selftest)
add_test(NAME bf16_gemv_parity_sycl COMMAND bf16_gemv_parity_sycl --selftest)
add_test(NAME quantize_act_parity_sycl COMMAND quantize_act_parity_sycl --selftest)
add_test(NAME gdn_parity_sycl COMMAND gdn_parity_sycl --selftest)

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

# ---- engine bring-up on SYCL (milestone 2) ----------------------------------
# The main CMakeLists gates strata_core / strata_kernels / strata_engine behind
# CUDA|HIP. Under SYCL the same target names are defined HERE: the two runtime
# .cu files are host-shaped enough to compile as C++ against the sycl_compat
# shim (device.cu's single <<<>>> poison fill lives in poison_sycl.cpp), and
# strata_kernels is the ported SYCL kernel library under its CUDA name so the
# engine's link line is backend-neutral. Link failures against this alias are
# the kernel port worklist (docs/SYCL_PORT.md milestone 3+).
set_source_files_properties(
  ${CMAKE_CURRENT_SOURCE_DIR}/src/core/device.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/core/pinned.cu
  PROPERTIES LANGUAGE CXX)

add_library(strata_core STATIC
  src/core/device.cu src/core/pinned.cu src/platform/memory.cpp
  src/core/graph.cpp src/core/weights.cpp src/core/layout.cpp
  src/core/poison_sycl.cpp)
target_include_directories(strata_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_core PUBLIC strata_plan strata_warnings strata_sycl_runtime)

add_library(strata_kernels ALIAS strata_kernels_sycl)

add_executable(strata-device src/core/device_main.cpp)
target_link_libraries(strata-device PRIVATE strata_core)

add_executable(strata-load src/core/load_main.cpp)
target_link_libraries(strata-load PRIVATE strata_core strata_kernels)

add_library(strata_engine STATIC
  src/core/layer.cpp src/core/session.cpp src/core/expert_source.cpp
  src/core/remote_experts.cpp src/core/expert_cache.cpp src/core/native_head.cpp
  src/core/native_dense.cpp src/core/verify.cpp src/core/mtp.cpp
  src/core/conversation_snapshot.cpp src/core/conversation_state.cpp
  src/core/conversation_memory.cpp)
target_include_directories(strata_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_engine PUBLIC strata_core strata_kernels strata_kernels_cpu
                                            strata_sycl_runtime)
