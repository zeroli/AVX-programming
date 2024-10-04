#pragma once

#include "simd/api/detail.h"

#include <cmath>

namespace simd {

DEFINE_API_UNARY_OP(ceil);
DEFINE_API_UNARY_OP(floor);

DEFINE_API_UNARY_OP(abs);
DEFINE_API_UNARY_OP(sqrt);
/// Computes the natural logarithm of the vector x
DEFINE_API_UNARY_OP(log);

#if 0
/// Computes the natural exponential of the vector x
DEFINE_API_UNARY_OP(exp);

/// Computes the base 10 exponential of the vector x
DEFINE_API_UNARY_OP(exp10);

/// Computes the base 2 exponential of the vector x
DEFINE_API_UNARY_OP(exp2);

/// Computes the natural exponential of the vector x, minus one
DEFINE_API_UNARY_OP(expm1);

/// Computes the square root of the sum of the squares of the x and y
DEFINE_API_BINARY_OP(hypot);



/// Computes the base 2 logarithm of the vector x
DEFINE_API_UNARY_OP(log2);

/// Computes the base 10 logarithm of the vector x
DEFINE_API_UNARY_OP(log10);

/// computes the natural logarithm of one plus the vector x
DEFINE_API_UNARY_OP(log1p);

/// compute rounded average of 2 vectors per slot element
DEFINE_API_BINARY_OP(avgr);

/// compute average of 2 vectors per slot element
DEFINE_API_BINARY_OP(avg);
#endif

}  // namespace simd
