#pragma once

#include "simd/types/avx_register.h"

namespace simd {
/// AVX + FMA instructions
struct FMA3_AVX : virtual AVX
{
    static constexpr bool supported() noexcept { return SIMD_WITH_FMA3_AVX; }
    static constexpr bool available() noexcept { return true; }
    static constexpr char const* name() noexcept { return "FMA3+AVX"; }
};

}  // namespace simd

#if SIMD_WITH_FMA3_AVX
namespace simd {
namespace types {

DECLARE_SIMD_REGISTER_ALIAS(FMA3_AVX, AVX);

}  // namespace types
}  // namespace simd
#endif  // SIMD_WITH_FMA3_AVX
