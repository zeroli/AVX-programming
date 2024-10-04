#pragma once

#include "simd/arch/constants.h"
#include "simd/arch/generic_fwd.h"

#if SIMD_WITH_SSE
#include "simd/arch/sse.h"
#endif

#if SIMD_WITH_FMA3_SSE
#include "simd/arch/fma3_sse.h"
#endif

#if SIMD_WITH_AVX
#include "simd/arch/avx.h"
#endif

#if SIMD_WITH_FMA3_AVX
#include "simd/arch/fma3_avx.h"
#endif

#if SIMD_WITH_AVX2
#include "simd/arch/avx2.h"
#endif

#if SIMD_WITH_AVX512
#include "simd/arch/avx512.h"
#endif

/// this one must comes last
#include "simd/arch/generic.h"
