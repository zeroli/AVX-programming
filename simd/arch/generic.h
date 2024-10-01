#pragma once

namespace simd { namespace kernel { namespace generic {
#include "simd/arch/kernel_impl.h"
} } } // namespace simd::kernel::generic

#include "simd/types/generic_arch.h"
#include "simd/types/traits.h"
#include "simd/arch/generic/detail.h"
#include "simd/arch/generic/algorithm.h"
#include "simd/arch/generic/arithmetic.h"
#include "simd/arch/generic/cast.h"
#include "simd/arch/generic/compare.h"
#include "simd/arch/generic/logical.h"
#include "simd/arch/generic/math.h"
#include "simd/arch/generic/memory.h"
#include "simd/arch/generic/trigo.h"

namespace simd { namespace kernel {
#define DEFINE_GENERIC_UNARY_OP(OP) \
template <typename T, size_t W> \
SIMD_INLINE \
Vec<T, W> OP(const Vec<T, W>& x, requires_arch<Generic>) noexcept \
{ \
    return generic::OP<T, W>::apply(x); \
} \
template <typename T, size_t W> \
SIMD_INLINE \
VecBool<T, W> OP(const VecBool<T, W>& x, requires_arch<Generic>) noexcept \
{ \
    return generic::OP<T, W>::apply(x); \
} \
///###

#define DEFINE_GENERIC_BINARY_OP(OP) \
template <typename T, size_t W> \
SIMD_INLINE \
Vec<T, W> OP(const Vec<T, W>& lhs, const Vec<T, W>& rhs, requires_arch<Generic>) noexcept \
{ \
    return generic::OP<T, W>::apply(lhs, rhs); \
} \
///###

#define DEFINE_GENERIC_BINARY_CMP_OP(OP) \
template <typename T, size_t W> \
SIMD_INLINE \
VecBool<T, W> OP(const Vec<T, W>& lhs, const Vec<T, W>& rhs, requires_arch<Generic>) noexcept \
{ \
    return generic::OP<T, W>::apply(lhs, rhs); \
} \
///

DEFINE_GENERIC_UNARY_OP(sign);
DEFINE_GENERIC_UNARY_OP(bitofsign);
DEFINE_GENERIC_UNARY_OP(bitwise_not);

DEFINE_GENERIC_BINARY_OP(copysign);

DEFINE_GENERIC_BINARY_OP(bitwise_and);
DEFINE_GENERIC_BINARY_OP(bitwise_or);
DEFINE_GENERIC_BINARY_OP(bitwise_xor);
DEFINE_GENERIC_BINARY_OP(bitwise_lshift);
DEFINE_GENERIC_BINARY_OP(bitwise_rshift);

DEFINE_GENERIC_BINARY_CMP_OP(eq);
DEFINE_GENERIC_BINARY_CMP_OP(ne);
DEFINE_GENERIC_BINARY_CMP_OP(gt);
DEFINE_GENERIC_BINARY_CMP_OP(ge);
DEFINE_GENERIC_BINARY_CMP_OP(lt);
DEFINE_GENERIC_BINARY_CMP_OP(le);

template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> bitwise_andnot(const VecBool<T, W>& lhs, const Vec<T, W>& rhs, requires_arch<Generic>) noexcept
{
    return generic::bitwise_andnot<T, W>::apply(lhs, rhs);
}

template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> bitwise_lshift(const Vec<T, W>& lhs, int32_t rhs, requires_arch<Generic>) noexcept
{
    return generic::bitwise_lshift<T, W>::apply(lhs, rhs);
}

template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> bitwise_rshift(const Vec<T, W>& lhs, int32_t rhs, requires_arch<Generic>) noexcept
{
    return generic::bitwise_rshift<T, W>::apply(lhs, rhs);
}

template <typename T, size_t W>
SIMD_INLINE
bool all_of(const VecBool<T, W>& x, requires_arch<Generic>) noexcept
{
    return generic::all_of<T, W>::apply(x);
}

template <typename T, size_t W>
SIMD_INLINE
bool any_of(const VecBool<T, W>& x, requires_arch<Generic>) noexcept
{
    return generic::any_of<T, W>::apply(x);
}

template <typename T, size_t W>
SIMD_INLINE
bool none_of(const VecBool<T, W>& x, requires_arch<Generic>) noexcept
{
    return generic::none_of<T, W>::apply(x);
}

template <typename T, size_t W>
SIMD_INLINE
bool some_of(const VecBool<T, W>& x, requires_arch<Generic>) noexcept
{
    return generic::some_of<T, W>::apply(x);
}

template <typename T, size_t W>
SIMD_INLINE
T hadd(const Vec<T, W>& x, requires_arch<Generic>) noexcept
{
    return generic::hadd<T, W>::apply(x);
}

#undef DEFINE_GENERIC_UNARY_OP
#undef DEFINE_GENERIC_BINARY_OP
#undef DEFINE_GENERIC_BINARY_CMP_OP
} } // namespace simd::kernel
