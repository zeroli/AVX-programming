#pragma once

STATIC_CHECK_ARCH_ENABLED(AVX512);
/// avx512 enables all below 3 arch
STATIC_CHECK_ARCH_ENABLED(AVX2);
STATIC_CHECK_ARCH_ENABLED(AVX);
STATIC_CHECK_ARCH_ENABLED(SSE);