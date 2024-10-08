#pragma once

#include "simd/types/avx2_register.h"

namespace simd {
/// AVX2 + FMA instructions
struct FMA3_AVX2 : AVX2, FMA3_AVX
{
    static constexpr bool supported() noexcept { return SIMD_WITH_FMA3_AVX2; }
    static constexpr bool available() noexcept { return true; }
    static constexpr char const* name() noexcept { return "FMA3+AVX2"; }
};

}  // namespace simd

#if SIMD_WITH_FMA3_AVX2
namespace simd {
namespace types {

DECLARE_SIMD_REGISTER_ALIAS(FMA3_AVX2, AVX2);
}  // namespace types
}  // namespace simd
#endif  // SIMD_WITH_FMA3_AVX2
