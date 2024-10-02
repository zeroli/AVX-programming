#pragma once

#include "simd/api/detail.h"

namespace simd {
/// arithmetic binary operations
DEFINE_API_BINARY_OP(add);
DEFINE_API_BINARY_OP(sub);
DEFINE_API_BINARY_OP(mul);
DEFINE_API_BINARY_OP(div);
DEFINE_API_BINARY_OP(mod);

DEFINE_API_UNARY_OP(neg);

/// compute `(x * y) + z` in one instructon when possible
template <typename T, size_t W,
    REQUIRES(std::is_floating_point<T>::value)
>
Vec<T, W> fmadd(const Vec<T, W>& x, const Vec<T, W>& y, const Vec<T, W>& z) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::fmadd<T, W>(x, y, z, A{});
}

/// compute `(x * y) - z` in one instructon when possible
template <typename T, size_t W,
    REQUIRES(std::is_floating_point<T>::value)
>
Vec<T, W> fmsub(const Vec<T, W>& x, const Vec<T, W>& y, const Vec<T, W>& z) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::fmsub<T, W>(x, y, z, A{});
}

/// compute `-(x * y) + z` in one instructon when possible
template <typename T, size_t W,
    REQUIRES(std::is_floating_point<T>::value)
>
Vec<T, W> fnmadd(const Vec<T, W>& x, const Vec<T, W>& y, const Vec<T, W>& z) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::fnmadd<T, W>(x, y, z, A{});
}

/// compute `-(x * y) - z` in one instructon when possible
template <typename T, size_t W,
    REQUIRES(std::is_floating_point<T>::value)
>
Vec<T, W> fnmsub(const Vec<T, W>& x, const Vec<T, W>& y, const Vec<T, W>& z) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::fnmsub<T, W>(x, y, z, A{});
}

/// compute `(x * y) - z` for even index (0, 2, 4, ...)
/// compute `(x * y) + z` for odd index (1, 3, 5, ...)
/// in one instructon when possible
template <typename T, size_t W,
    REQUIRES(std::is_floating_point<T>::value)
>
Vec<T, W> fmaddsub(const Vec<T, W>& x, const Vec<T, W>& y, const Vec<T, W>& z) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::fmaddsub<T, W>(x, y, z, A{});
}

/// compute `(x * y) + z` for even index (0, 2, 4, ...)
/// compute `(x * y) - z` for odd index (1, 3, 5, ...)
/// in one instructon when possible
template <typename T, size_t W,
    REQUIRES(std::is_floating_point<T>::value)
>
Vec<T, W> fmsubadd(const Vec<T, W>& x, const Vec<T, W>& y, const Vec<T, W>& z) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::fmsubadd<T, W>(x, y, z, A{});
}
}  // namespace simd
