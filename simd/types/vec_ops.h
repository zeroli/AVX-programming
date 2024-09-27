#pragma once

#include "simd/arch/isa.h"
#include "simd/types/vec.h"

namespace simd {
namespace ops {
template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> add(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::add<T, W>(lhs, rhs, A{});
}

template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> sub(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::sub<T, W>(lhs, rhs, A{});
}

template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> mul(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::mul<T, W>(lhs, rhs, A{});
}

template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> div(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::div<T, W>(lhs, rhs, A{});
}

template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> bitwise_and(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::bitwise_and<T, W>(lhs, rhs, A{});
}
template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> bitwise_or(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::bitwise_or<T, W>(lhs, rhs, A{});
}
template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> bitwise_xor(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::bitwise_xor<T, W>(lhs, rhs, A{});
}

template <typename T, size_t W>
SIMD_INLINE
VecBool<T, W> eq(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::eq<T, W>(lhs, rhs, A{});
}
template <typename T, size_t W>
SIMD_INLINE
VecBool<T, W> ne(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::ne<T, W>(lhs, rhs, A{});
}

template <typename T, size_t W>
SIMD_INLINE
VecBool<T, W> gt(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::gt<T, W>(lhs, rhs, A{});
}
template <typename T, size_t W>
SIMD_INLINE
VecBool<T, W> ge(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::ge<T, W>(lhs, rhs, A{});
}

template <typename T, size_t W>
SIMD_INLINE
VecBool<T, W> lt(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::lt<T, W>(lhs, rhs, A{});
}
template <typename T, size_t W>
SIMD_INLINE
VecBool<T, W> le(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::le<T, W>(lhs, rhs, A{});
}

}  // namespace ops
}  // namespace simd
