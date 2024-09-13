#pragma once

#include "simd/types/sse_register.h"
#include "simd/arch/sse/algorithm.h"
#include "simd/arch/sse/arithmetic.h"
#include "simd/arch/sse/bitwise.h"
#include "simd/arch/sse/cast.h"
#include "simd/arch/sse/complex.h"
#include "simd/arch/sse/logical.h"
#include "simd/arch/sse/math.h"
#include "simd/arch/sse/memory.h"
#include "simd/arch/sse/trigo.h"

namespace simd {
namespace kernel {
#define DEFINE_SSE_BINARY_OP(OP) \
template <typename T, size_t W> \
Vec<T, W> OP(const Vec<T, W>& lhs, const Vec<T, W>& rhs, requires_arch<SSE>) noexcept \
{ \
    return impl::OP<T, W>::apply(lhs, rhs); \
} \
///

DEFINE_SSE_BINARY_OP(add);
DEFINE_SSE_BINARY_OP(sub);
DEFINE_SSE_BINARY_OP(mul);
DEFINE_SSE_BINARY_OP(div);

DEFINE_SSE_BINARY_OP(bitwise_and);
DEFINE_SSE_BINARY_OP(bitwise_or);
DEFINE_SSE_BINARY_OP(bitwise_xor);
DEFINE_SSE_BINARY_OP(bitwise_andnot);

DEFINE_SSE_BINARY_OP(logical_and);
DEFINE_SSE_BINARY_OP(logical_or);

DEFINE_SSE_BINARY_OP(max);
DEFINE_SSE_BINARY_OP(min);

#undef DEFINE_SSE_BINARY_OP

#define DEFINE_SSE_UNARY_OP(OP) \
template <typename T, size_t W> \
Vec<T, W> OP(const Vec<T, W>& self, requires_arch<SSE>) noexcept \
{ \
    return impl::OP<T, W>::apply(self); \
} \
///

DEFINE_SSE_UNARY_OP(bitwise_not);
DEFINE_SSE_UNARY_OP(neg);

DEFINE_SSE_UNARY_OP(abs);
DEFINE_SSE_UNARY_OP(sqrt);

#undef DEFINE_SSE_UNARY_OP

template <typename T, size_t W>
Vec<T, W> broadcast(T val, requires_arch<SSE>) noexcept
{
    return impl::broadcast<T, W>::apply(val);
}

}  // namespace kernel
}  // namespace simd
