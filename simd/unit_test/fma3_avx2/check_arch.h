#pragma once


STATIC_CHECK_ARCH_ENABLED(FMA3_AVX2);
STATIC_CHECK_ARCH_ENABLED(FMA3_AVX);
STATIC_CHECK_ARCH_ENABLED(FMA3_SSE);
/// fma enables avx/sse internally
STATIC_CHECK_ARCH_ENABLED(AVX2);
STATIC_CHECK_ARCH_ENABLED(AVX);
STATIC_CHECK_ARCH_ENABLED(SSE);
