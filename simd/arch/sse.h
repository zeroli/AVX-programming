#pragma once

#include "simd/types/sse_register.h"
#include "simd/arch/sse/algorithm.h"
#include "simd/arch/sse/arithmetic.h"
#include "simd/arch/sse/cast.h"
#include "simd/arch/sse/complex.h"
#include "simd/arch/sse/logical.h"
#include "simd/arch/sse/math.h"
#include "simd/arch/sse/memory.h"
#include "simd/arch/sse/trigo.h"

namespace simd {
namespace kernel {
template <typename T, size_t W>
Vec<T, W> add(const Vec<T, W>& lhs, const Vec<T, W>& rhs, requires_arch<SSE>) noexcept
{
    return impl::add<T, W>::apply(lhs, rhs);
}

template <typename T, size_t W>
Vec<T, W> sub(const Vec<T, W>& lhs, const Vec<T, W>& rhs, requires_arch<SSE>) noexcept
{
    return impl::sub<T, W>::apply(lhs, rhs);
}

template <typename T, size_t W>
Vec<T, W> mul(const Vec<T, W>& lhs, const Vec<T, W>& rhs, requires_arch<SSE>) noexcept
{
    return impl::mul<T, W>::apply(lhs, rhs);
}

template <typename T, size_t W>
Vec<T, W> div(const Vec<T, W>& lhs, const Vec<T, W>& rhs, requires_arch<SSE>) noexcept
{
    return impl::div<T, W>::apply(lhs, rhs);
}

template <typename T, size_t W>
Vec<T, W> broadcast(T val, requires_arch<SSE>) noexcept
{
    return impl::broadcast<T, W>::apply(val);
}

template <typename T, size_t W>
Vec<T, W> bitwise_and(const Vec<T, W>& lhs, const Vec<T, W>& rhs, requires_arch<SSE>) noexcept
{
    return impl::bitwise_and<T, W>::apply(lhs, rhs);
}
template <typename T, size_t W>
Vec<T, W> bitwise_or(const Vec<T, W>& lhs, const Vec<T, W>& rhs, requires_arch<SSE>) noexcept
{
    return impl::bitwise_or<T, W>::apply(lhs, rhs);
}
template <typename T, size_t W>
Vec<T, W> bitwise_xor(const Vec<T, W>& lhs, const Vec<T, W>& rhs, requires_arch<SSE>) noexcept
{
    return impl::bitwise_xor<T, W>::apply(lhs, rhs);
}

template <typename T, size_t W>
Vec<T, W> max(const Vec<T, W>& lhs, const Vec<T, W>& rhs, requires_arch<SSE>) noexcept
{
    return impl::max<T, W>::apply(lhs, rhs);
}

template <typename T, size_t W>
Vec<T, W> min(const Vec<T, W>& lhs, const Vec<T, W>& rhs, requires_arch<SSE>) noexcept
{
    return impl::min<T, W>::apply(lhs, rhs);
}

/// math operations
/// abs
template <typename T, size_t W>
Vec<T, W> abs(const Vec<T, W>& self, requires_arch<SSE>) noexcept
{
    return impl::abs<T, W>::apply(self);
}
/// sqrt
template <typename T, size_t W>
Vec<T, W> sqrt(const Vec<T, W>& self, requires_arch<SSE>) noexcept
{
    return impl::sqrt<T, W>::apply(self);
}
}  // namespace kernel
}  // namespace simd
