#pragma once

#include "simd/api/detail.h"

namespace simd {
/// static_cast each element of T to U, with same width
template <typename U, typename T, size_t W>
Vec<U, W> cast(const Vec<T, W>& x) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::cast<U>(x, A{});
}
}  // namespace simd
