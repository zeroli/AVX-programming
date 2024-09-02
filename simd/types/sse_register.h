#pragma once

#include "simd/types/generic_arch.h"

#if SIMD_WITH_SSE
#include <emmintrin.h>  // SSE2
#include <xmmintrin.h>  // SSE2
#include <pmmintrin.h>  // SSE3
#include <smmintrin.h>  // SSE4.1
#include <nmmintrin.h>  // SSE4.2
#include <tmmintrin.h>  // sSSE3
#endif

namespace simd {
/// all SSE instructions
struct SSE : generic
{
    static constexpr bool supported() noexcept { return SIMD_WITH_SSE; }
    static constexpr bool available() noexcept { return true; }
    static constexpr size_t alignment() noexcept { return 16; }
    static constexpr bool requires_alignment() noexcept { return true; }
    static constexpr const char* name() noexcept { return "SSE"; }
};
}  // namespace simd

#if SIMD_WITH_SSE
#include <complex>

namespace simd {
namespace types {
DECLARE_SIMD_REGISTER(signed char, SSE, __m128i);
DECLARE_SIMD_REGISTER(unsigned char, SSE, __m128i);
DECLARE_SIMD_REGISTER(char, SSE, __m128i);
DECLARE_SIMD_REGISTER(unsigned short, SSE, __m128i);
DECLARE_SIMD_REGISTER(short, SSE, __m128i);
DECLARE_SIMD_REGISTER(unsigned int, SSE, __m128i);
DECLARE_SIMD_REGISTER(int, SSE, __m128i);
DECLARE_SIMD_REGISTER(unsigned long int, SSE, __m128i);
DECLARE_SIMD_REGISTER(long int, SSE, __m128i);
DECLARE_SIMD_REGISTER(unsigned long long int, SSE, __m128i);
DECLARE_SIMD_REGISTER(long long int, SSE, __m128i);
DECLARE_SIMD_REGISTER(float, SSE, __m128);
DECLARE_SIMD_REGISTER(double, SSE, __m128d);
DECLARE_SIMD_REGISTER(std::complex<float>, SSE, __m128);
DECLARE_SIMD_REGISTER(std::complex<double>, SSE, __m128d);
}  // namespace types
}  // namespace simd
#endif  // SIMD_WITH_SSE
