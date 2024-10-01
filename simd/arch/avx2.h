#pragma once

namespace simd { namespace kernel { namespace avx2 {
#include "simd/arch/kernel_impl.h"
} } } // namespace simd::kernel::avx2

#include "simd/types/avx2_register.h"
#include "simd/types/traits.h"
#include "simd/arch/avx2/detail.h"
#include "simd/arch/avx2/algorithm.h"
#include "simd/arch/avx2/arithmetic.h"
#include "simd/arch/avx2/cast.h"
#include "simd/arch/avx2/compare.h"
#include "simd/arch/avx2/complex.h"
#include "simd/arch/avx2/logical.h"
#include "simd/arch/avx2/math.h"
#include "simd/arch/avx2/memory.h"
#include "simd/arch/avx2/trigo.h"

namespace simd { namespace kernel {

#define DEFINE_AVX2_BINARY_OP(OP) \
template <typename T, size_t W, \
  REQUIRES(std::is_integral<T>::value)> \
SIMD_INLINE \
Vec<T, W> OP(const Vec<T, W>& lhs, const Vec<T, W>& rhs, requires_arch<AVX2>) noexcept \
{ \
    return avx2::OP<T, W>::apply(lhs, rhs); \
} \
///

DEFINE_AVX2_BINARY_OP(add);
DEFINE_AVX2_BINARY_OP(sub);
DEFINE_AVX2_BINARY_OP(mul);
DEFINE_AVX2_BINARY_OP(div);
DEFINE_AVX2_BINARY_OP(mod);

DEFINE_AVX2_BINARY_OP(min);
DEFINE_AVX2_BINARY_OP(max);

#define DEFINE_AVX2_UNARY_OP(OP) \
template <typename T, size_t W, \
  REQUIRES(std::is_integral<T>::value)> \
SIMD_INLINE \
Vec<T, W> OP(const Vec<T, W>& x, requires_arch<AVX2>) noexcept \
{ \
    return avx2::OP<T, W>::apply(x); \
} \
///

DEFINE_AVX2_UNARY_OP(neg);

DEFINE_AVX2_UNARY_OP(abs);

#define DEFINE_AVX2_BINARY_CMP_OP(OP) \
template <typename T, size_t W, \
  REQUIRES(std::is_integral<T>::value)> \
SIMD_INLINE \
VecBool<T, W> OP(const Vec<T, W>& lhs, const Vec<T, W>& rhs, requires_arch<AVX2>) noexcept \
{ \
    return avx2::OP<T, W>::apply(lhs, rhs); \
} \
///

DEFINE_AVX2_BINARY_CMP_OP(eq);
DEFINE_AVX2_BINARY_CMP_OP(ne);
DEFINE_AVX2_BINARY_CMP_OP(gt);
DEFINE_AVX2_BINARY_CMP_OP(ge);
DEFINE_AVX2_BINARY_CMP_OP(lt);
DEFINE_AVX2_BINARY_CMP_OP(le);

template <typename T, size_t W,
  REQUIRES(std::is_integral<T>::value)>
SIMD_INLINE
bool all_of(const VecBool<T, W>& x, requires_arch<AVX2>) noexcept
{
    return avx2::all_of<T, W>::apply(x);
}

#undef DEFINE_AVX2_UNARY_OP
#undef DEFINE_AVX2_BINARY_OP
#undef DEFINE_AVX2_BINARY_CMP_OP
} }  // namespace simd::kernel
