#pragma once

STATIC_CHECK_ARCH_ENABLED(FMA3_SSE);
/// -mfma also enables AVX/SSE
STATIC_CHECK_ARCH_ENABLED(FMA3_AVX);
STATIC_CHECK_ARCH_ENABLED(AVX);
STATIC_CHECK_ARCH_ENABLED(SSE);
