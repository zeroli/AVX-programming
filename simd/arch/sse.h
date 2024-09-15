#pragma once

namespace simd {
namespace kernel {
namespace sse {
#include "simd/arch/kernel_impl.h"
}  // namespace sse
}  // namespace kernel
}  // namespace simd

#include "simd/types/sse_register.h"
#include "simd/arch/sse/algorithm.h"
#include "simd/arch/sse/arithmetic.h"
#include "simd/arch/sse/cast.h"
#include "simd/arch/sse/compare.h"
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
    return sse::OP<T, W>::apply(lhs, rhs); \
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

#define DEFINE_SSE_BINARY_COMP_OP(OP) \
template <typename T, size_t W> \
VecBool<T, W> OP(const Vec<T, W>& lhs, const Vec<T, W>& rhs, requires_arch<SSE>) noexcept \
{ \
    return sse::OP<T, W>::apply(lhs, rhs); \
} \
///

DEFINE_SSE_BINARY_COMP_OP(eq);
DEFINE_SSE_BINARY_COMP_OP(ne);
DEFINE_SSE_BINARY_COMP_OP(gt);
DEFINE_SSE_BINARY_COMP_OP(ge);
DEFINE_SSE_BINARY_COMP_OP(lt);
DEFINE_SSE_BINARY_COMP_OP(le);

#undef DEFINE_SSE_BINARY_COMP_OP

DEFINE_SSE_BINARY_OP(max);
DEFINE_SSE_BINARY_OP(min);

#undef DEFINE_SSE_BINARY_OP

#define DEFINE_SSE_UNARY_OP(OP) \
template <typename T, size_t W> \
Vec<T, W> OP(const Vec<T, W>& self, requires_arch<SSE>) noexcept \
{ \
    return sse::OP<T, W>::apply(self); \
} \
///

DEFINE_SSE_UNARY_OP(bitwise_not);
DEFINE_SSE_UNARY_OP(neg);

DEFINE_SSE_UNARY_OP(abs);
DEFINE_SSE_UNARY_OP(sqrt);

#undef DEFINE_SSE_UNARY_OP

template <typename T, size_t W>
VecBool<T, W> bitwise_not(const VecBool<T, W>& self, requires_arch<SSE>) noexcept
{
    return sse::bitwise_not<T, W>::apply(self);
}

template <typename T, size_t W>
bool all(const VecBool<T, W>& self, requires_arch<SSE>) noexcept
{
    return sse::all<T, W>::apply(self);
}

template <typename T, size_t W>
bool any(const VecBool<T, W>& self, requires_arch<SSE>) noexcept
{
    return sse::any<T, W>::apply(self);
}

template <typename T, size_t W>
Vec<T, W> broadcast(T val, requires_arch<SSE>) noexcept
{
    return sse::broadcast<T, W>::apply(val);
}

}  // namespace kernel
}  // namespace simd
