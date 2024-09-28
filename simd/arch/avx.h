#pragma once

namespace simd {
namespace kernel {
namespace avx {
#include "simd/arch/kernel_impl.h"
}  // namespace avx
}  // namespace kernel
}  // namespace simd

#include "simd/types/avx_register.h"
#include "simd/arch/avx/algorithm.h"
#include "simd/arch/avx/arithmetic.h"
#include "simd/arch/avx/cast.h"
#include "simd/arch/avx/compare.h"
#include "simd/arch/avx/complex.h"
#include "simd/arch/avx/logical.h"
#include "simd/arch/avx/math.h"
#include "simd/arch/avx/memory.h"
#include "simd/arch/avx/trigo.h"
