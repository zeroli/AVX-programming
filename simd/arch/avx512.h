#pragma once

namespace simd {
namespace kernel {
namespace avx512 {
#include "simd/arch/kernel_impl.h"
}  // namespace avx512
}  // namespace kernel
}  // namespace simd

#include "simd/types/avx512_register.h"
#include "simd/arch/avx512/algorithm.h"
#include "simd/arch/avx512/arithmetic.h"
#include "simd/arch/avx512/cast.h"
#include "simd/arch/avx512/compare.h"
#include "simd/arch/avx512/complex.h"
#include "simd/arch/avx512/logical.h"
#include "simd/arch/avx512/math.h"
#include "simd/arch/avx512/memory.h"
#include "simd/arch/avx512/trigo.h"
