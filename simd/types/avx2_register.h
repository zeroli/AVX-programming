#pragma once

#include "simd/types/avx_register.h"

namespace simd {
/// AVX2 instructions
struct AVX2 : AVX
{
    static constexpr bool supported() noexcept { return SIMD_WITH_AVX2; }
    static constexpr bool available() noexcept { return true; }
    static constexpr const char* name() noexcept { return "AVX2"; }
};
}  // namespace simd

#if SIMD_WITH_AVX2
namespace simd {
namespace types {
DECLARE_SIMD_REGISTER_ALIAS(AVX2, AVX)
}  // namespace types
}  // namespace simd
#endif  // SIMD_WITH_AVX2
