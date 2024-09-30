#pragma once

namespace simd { namespace kernel { namespace avx {
#include "simd/arch/kernel_impl.h"
} } } // namespace simd::kernel::avx

#include "simd/types/avx_register.h"
#include "simd/types/traits.h"
#include "simd/arch/avx/detail.h"
#include "simd/arch/avx/algorithm.h"
#include "simd/arch/avx/arithmetic.h"
#include "simd/arch/avx/cast.h"
#include "simd/arch/avx/compare.h"
#include "simd/arch/avx/complex.h"
#include "simd/arch/avx/logical.h"
#include "simd/arch/avx/math.h"
#include "simd/arch/avx/memory.h"
#include "simd/arch/avx/trigo.h"

namespace simd { namespace kernel {

#define DEFINE_AVX_BINARY_OP(OP) \
template <typename T, size_t W> \
SIMD_INLINE \
Vec<T, W> OP(const Vec<T, W>& lhs, const Vec<T, W>& rhs, requires_arch<AVX>) noexcept \
{ \
    return avx::OP<T, W>::apply(lhs, rhs); \
} \
///

DEFINE_AVX_BINARY_OP(add);
DEFINE_AVX_BINARY_OP(sub);
DEFINE_AVX_BINARY_OP(mul);
DEFINE_AVX_BINARY_OP(div);
DEFINE_AVX_BINARY_OP(mod);

DEFINE_AVX_BINARY_OP(bitwise_and);
DEFINE_AVX_BINARY_OP(bitwise_or);
DEFINE_AVX_BINARY_OP(bitwise_xor);
DEFINE_AVX_BINARY_OP(bitwise_lshift);
DEFINE_AVX_BINARY_OP(bitwise_rshift);

template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> bitwise_andnot(const VecBool<T, W>& lhs, const Vec<T, W>& rhs, requires_arch<AVX>) noexcept
{
    return avx::bitwise_andnot<T, W>::apply(lhs, rhs);
}

template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> bitwise_lshift(const Vec<T, W>& lhs, int32_t rhs, requires_arch<AVX>) noexcept
{
    return avx::bitwise_lshift<T, W>::apply(lhs, rhs);
}
template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> bitwise_rshift(const Vec<T, W>& lhs, int32_t rhs, requires_arch<AVX>) noexcept
{
    return avx::bitwise_rshift<T, W>::apply(lhs, rhs);
}

#define DEFINE_AVX_UNARY_OP(OP) \
template <typename T, size_t W> \
SIMD_INLINE \
Vec<T, W> OP(const Vec<T, W>& x, requires_arch<AVX>) noexcept \
{ \
    return avx::OP<T, W>::apply(x); \
} \
///

DEFINE_AVX_UNARY_OP(neg);
DEFINE_AVX_UNARY_OP(bitwise_not);

#define DEFINE_AVX_BINARY_CMP_OP(OP) \
template <typename T, size_t W> \
SIMD_INLINE \
VecBool<T, W> OP(const Vec<T, W>& lhs, const Vec<T, W>& rhs, requires_arch<AVX>) noexcept \
{ \
    return avx::OP<T, W>::apply(lhs, rhs); \
} \
///

DEFINE_AVX_BINARY_CMP_OP(eq);
DEFINE_AVX_BINARY_CMP_OP(ne);
DEFINE_AVX_BINARY_CMP_OP(gt);
DEFINE_AVX_BINARY_CMP_OP(ge);
DEFINE_AVX_BINARY_CMP_OP(lt);
DEFINE_AVX_BINARY_CMP_OP(le);

template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> broadcast(T val, requires_arch<AVX>) noexcept
{
    return avx::broadcast<T, W>::apply(val);
}

#undef DEFINE_AVX_BINARY_OP
#undef DEFINE_AVX_UNARY_OP
#undef DEFINE_AVX_BINARY_CMP_OP

} } // namespace simd::kernel
