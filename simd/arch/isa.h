#pragma once

#if SIMD_WITH_SSE
#include "simd/arch/sse.h"
#endif

#if SIMD_WITH_AVX
#include "simd/arch/avx.h"
#endif

#if SIMD_WITH_AVX2
#include "simd/arch/avx2.h"
#endif

#if SIMD_WITH_AVX512
#include "simd/arch/avx512.h"
#endif

#include "simd/arch/generic.h"
