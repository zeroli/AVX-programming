#pragma once

namespace simd {
namespace kernel {
namespace generic {
#include "simd/arch/kernel_impl.h"
}  // namespace generic
}  // namespace kernel
}  // namespace simd

#include "simd/arch/generic/algorithm.h"
#include "simd/arch/generic/arithmetic.h"
#include "simd/arch/generic/cast.h"
#include "simd/arch/generic/logical.h"
#include "simd/arch/generic/math.h"
#include "simd/arch/generic/memory.h"
#include "simd/arch/generic/trigo.h"

namespace simd {
namespace kernel {
#define DEFINE_GENERIC_UNARY_OP(OP) \
template <typename T, size_t W> \
Vec<T, W> OP(const Vec<T, W>& x, requires_arch<Generic>) noexcept \
{ \
    return generic::OP<T, W>::apply(x); \
} \
///

#define DEFINE_GENERIC_BINARY_OP(OP) \
template <typename T, size_t W> \
Vec<T, W> OP(const Vec<T, W>& lhs, const Vec<T, W>& rhs, requires_arch<Generic>) noexcept \
{ \
    return generic::OP<T, W>::apply(lhs, rhs); \
} \
///

DEFINE_GENERIC_UNARY_OP(sign);
DEFINE_GENERIC_UNARY_OP(bitofsign);

DEFINE_GENERIC_BINARY_OP(copysign);

template <typename T, size_t W>
bool none_of(const VecBool<T, W>& self, requires_arch<Generic>) noexcept
{
    return generic::none_of<T, W>::apply(self);
}

template <typename T, size_t W>
bool some_of(const VecBool<T, W>& self, requires_arch<Generic>) noexcept
{
    return generic::some_of<T, W>::apply(self);
}

#undef DEFINE_GENERIC_UNARY_OP
#undef DEFINE_GENERIC_BINARY_OP
}  // namespace kernel
}  // namespace simd
