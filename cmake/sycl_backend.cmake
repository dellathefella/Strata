# =============================================================================
# cmake/sycl_backend.cmake — Intel Arc (SYCL / Level-Zero) backend seam
#
# Mirrors cmake/hip_backend.cmake: selected with -DSTRATA_ENABLE_SYCL=ON
# (mutually exclusive with CUDA/HIP). Sets the _strata_gpu_* variables the
# rest of the build links against, force-includes the sycl_compat shim, and
# registers the parity/bench tools.
#
# Milestone 1: toolchain detection + standalone benches only. Kernel
# translation of src/kernels/cuda lands in milestones 3+ (SYCLomatic bulk
# pass + hand-fix list in docs/SYCL_PORT.md).
# =============================================================================

find_program(STRATA_ICPX icpx REQUIRED)
message(STATUS "Strata SYCL backend: ${STRATA_ICPX}")

set(CMAKE_CXX_COMPILER "${STRATA_ICPX}" PARENT_SCOPE)
set(_strata_gpu_lang SYCL PARENT_SCOPE)
set(_strata_gpu_compile_flags -fsycl -fno-sycl-id-queries-fit-in-int PARENT_SCOPE)
set(_strata_gpu_link_flags -fsycl PARENT_SCOPE)
set(_strata_gpu_libs "" PARENT_SCOPE)          # L0 comes with the SYCL runtime
set(_strata_gpu_blas_libs "" PARENT_SCOPE)     # oneMKL wired in milestone 5
set(_strata_gpu_includes "${CMAKE_CURRENT_SOURCE_DIR}/include/strata/sycl_compat" PARENT_SCOPE)

# Level-Zero doorbell / USM benches (milestone 1 acceptance)
add_executable(l0_spin_bench tools/l0_spin_bench.cpp)
target_compile_options(l0_spin_bench PRIVATE -fsycl -O2)
target_link_options(l0_spin_bench PRIVATE -fsycl)
