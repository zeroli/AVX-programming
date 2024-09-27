#pragma once

#include "simd/api/detail.h"

namespace simd {
/// bitwise operations
DEFINE_API_BINARY_OP(bitwise_and);
DEFINE_API_BINARY_OP(bitwise_or);
DEFINE_API_BINARY_OP(bitwise_xor);
DEFINE_API_BINARY_OP(bitwise_andnot);
DEFINE_API_BINARY_OP(bitwise_lshift);
DEFINE_API_BINARY_OP(bitwise_rshift);

DEFINE_API_UNARY_OP(bitwise_not);

template <typename T, size_t W>
VecBool<T, W> bitwise_not(const VecBool<T, W>& x) noexcept
{
    using A = typename VecBool<T, W>::arch_t;
    return kernel::bitwise_not<T, W>(x, A{});
}

}  // namespace simd
