#pragma once

#include "simd/types/sse_register.h"

namespace simd {
/// SSE + FMA instructions
struct FMA3_SSE : SSE
{
    static constexpr bool supported() noexcept { return SIMD_WITH_FMA3_SSE; }
    static constexpr bool available() noexcept { return true; }
    static constexpr char const* name() noexcept { return "FMA3+SSE"; }
};

}  // namespace simd

#if SIMD_WITH_FMA3_SSE
namespace simd {
namespace types {

DECLARE_SIMD_REGISTER_ALIAS(FMA3_SSE, SSE);

}  // namespace types
}  // namespace simd
#endif  // SIMD_WITH_FMA3_SSE
