#pragma once

#include "simd/config/config.h"
#include "simd/types/register.h"

#include <cstddef>

namespace simd {
/// generic architecture
struct Generic {
    static constexpr bool supported() noexcept { return true; }
    static constexpr bool available() noexcept { return true; }
    static constexpr size_t alignment() noexcept { return 1; }
    static constexpr bool require_alignment() noexcept { return false; }
    static constexpr const char* name() noexcept { return "generic"; }
};

struct unsupported { };

}  // namespace simd
