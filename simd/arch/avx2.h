#pragma once

namespace simd {
namespace kernel {
namespace avx2 {
#include "simd/arch/kernel_impl.h"
}  // namespace avx2
}  // namespace kernel
}  // namespace simd

#include "simd/types/avx2_register.h"
#include "simd/arch/avx2/algorithm.h"
#include "simd/arch/avx2/arithmetic.h"
#include "simd/arch/avx2/cast.h"
#include "simd/arch/avx2/compare.h"
#include "simd/arch/avx2/complex.h"
#include "simd/arch/avx2/logical.h"
#include "simd/arch/avx2/math.h"
#include "simd/arch/avx2/memory.h"
#include "simd/arch/avx2/trigo.h"

namespace simd {
namespace kernel {

#define DEFINE_AVX2_BINARY_OP(OP) \
template <typename T, size_t W> \
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

#define DEFINE_AVX2_UNARY_OP(OP) \
template <typename T, size_t W> \
SIMD_INLINE \
Vec<T, W> OP(const Vec<T, W>& x, requires_arch<AVX2>) noexcept \
{ \
    return avx2::OP<T, W>::apply(x); \
} \
///

DEFINE_AVX2_UNARY_OP(neg);

#undef DEFINE_AVX2_BINARY_OP
#undef DEFINE_AVX2_UNARY_OP

}  // namespace kernel
}  // namespace simd
