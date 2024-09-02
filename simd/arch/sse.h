#pragma once

#include "simd/types/sse_register.h"
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
Vec<T, W> broadcast(T val, requires_arch<SSE>) noexcept
{
    return impl::broadcast<T, W>::apply(val);
}

}  // namespace kernel
}  // namespace simd
