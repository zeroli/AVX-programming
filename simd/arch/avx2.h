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
