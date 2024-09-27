#pragma once

#include "simd/config/config.h"
#include "simd/types/register.h"

#include <cstddef>

namespace simd {
/// generic architecture
struct Generic {
    SIMD_INLINE
    static constexpr bool supported() noexcept { return true; }
    SIMD_INLINE
    static constexpr bool available() noexcept { return true; }
    SIMD_INLINE
    static constexpr size_t alignment() noexcept { return 1; }
    SIMD_INLINE
    static constexpr bool require_alignment() noexcept { return false; }
    SIMD_INLINE
    static constexpr const char* name() noexcept { return "generic"; }
};

struct unsupported { };

}  // namespace simd
