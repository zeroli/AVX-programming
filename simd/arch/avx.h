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

#define DEFINE_AVX_UNARY_OP(OP) \
template <typename T, size_t W> \
SIMD_INLINE \
Vec<T, W> OP(const Vec<T, W>& x, requires_arch<AVX>) noexcept \
{ \
    return avx::OP<T, W>::apply(x); \
} \
///

DEFINE_AVX_UNARY_OP(neg);

template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> broadcast(T val, requires_arch<AVX>) noexcept
{
    return avx::broadcast<T, W>::apply(val);
}

#undef DEFINE_AVX_BINARY_OP
#undef DEFINE_AVX_UNARY_OP

} } // namespace simd::kernel
