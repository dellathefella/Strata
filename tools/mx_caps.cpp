#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/sycl.hpp>
#include <cstdio>

namespace mx = sycl::ext::oneapi::experimental::matrix;

static const char* tn(mx::matrix_type t) {
    switch (t) {
        case mx::matrix_type::bf16: return "bf16";
        case mx::matrix_type::fp16: return "fp16";
        case mx::matrix_type::tf32: return "tf32";
        case mx::matrix_type::fp32: return "fp32";
        case mx::matrix_type::sint8: return "s8";
        case mx::matrix_type::sint32: return "s32";
        case mx::matrix_type::uint8: return "u8";
        case mx::matrix_type::uint32: return "u32";
        default: return "?";
    }
}

int main() {
    sycl::queue q(sycl::gpu_selector_v);
    printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    auto combos = q.get_device().get_info<
        sycl::ext::oneapi::experimental::info::device::matrix_combinations>();
    printf("total combos: %zu\n", combos.size());
    for (auto& c : combos) {
        if (c.atype == mx::matrix_type::sint8 || c.atype == mx::matrix_type::uint8 ||
            c.btype == mx::matrix_type::sint8 || c.btype == mx::matrix_type::uint8)
            printf("int8: M=%zu N=%zu K=%zu (max %zu/%zu/%zu) A=%s B=%s C=%s D=%s\n",
                   c.msize, c.nsize, c.ksize, c.max_msize, c.max_nsize, c.max_ksize,
                   tn(c.atype), tn(c.btype), tn(c.ctype), tn(c.dtype));
    }
    // also list first few fp16/bf16 for reference
    int n = 0;
    for (auto& c : combos) {
        if ((c.atype == mx::matrix_type::fp16 || c.atype == mx::matrix_type::bf16) && n++ < 6)
            printf("fp:   M=%zu N=%zu K=%zu A=%s B=%s C=%s D=%s\n",
                   c.msize, c.nsize, c.ksize,
                   tn(c.atype), tn(c.btype), tn(c.ctype), tn(c.dtype));
    }
    return 0;
}
