#pragma once

#include "simd/api/detail.h"

namespace simd {
/// arithmetic binary operations
DEFINE_API_BINARY_OP(add);
DEFINE_API_BINARY_OP(sub);
DEFINE_API_BINARY_OP(mul);
DEFINE_API_BINARY_OP(div);
DEFINE_API_BINARY_OP(mod);

DEFINE_API_UNARY_OP(neg);
}  // namespace simd
