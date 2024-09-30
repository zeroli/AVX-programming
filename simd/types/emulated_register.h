#pragma once

#include "simd/types/generic_arch.h"

namespace simd {
/// Emulated instructions
struct Emulated : Generic
{
    static constexpr bool supported() noexcept { return SIMD_WITH_EMULATED; }
    static constexpr bool available() noexcept { return true; }
    static constexpr size_t alignment() noexcept { return 8; }
    static constexpr bool requires_alignment() noexcept { return true; }
    static constexpr const char* name() noexcept { return "Emulated"; }
};
}  // namespace simd

#if SIMD_WITH_EMULATED
#include <complex>

namespace simd {
namespace types {
DECLARE_SIMD_REGISTER(signed char, Emulated, __m128i);
DECLARE_SIMD_REGISTER(unsigned char, Emulated, __m128i);
DECLARE_SIMD_REGISTER(char, Emulated, __m128i);
DECLARE_SIMD_REGISTER(unsigned short, Emulated, __m128i);
DECLARE_SIMD_REGISTER(short, Emulated, __m128i);
DECLARE_SIMD_REGISTER(unsigned int, Emulated, __m128i);
DECLARE_SIMD_REGISTER(int, Emulated, __m128i);
DECLARE_SIMD_REGISTER(unsigned long int, Emulated, __m128i);
DECLARE_SIMD_REGISTER(long int, Emulated, __m128i);
DECLARE_SIMD_REGISTER(unsigned long long int, Emulated, __m128i);
DECLARE_SIMD_REGISTER(long long int, Emulated, __m128i);
DECLARE_SIMD_REGISTER(float, Emulated, __m128);
DECLARE_SIMD_REGISTER(double, Emulated, __m128d);
DECLARE_SIMD_REGISTER(std::complex<float>, Emulated, __m128);
DECLARE_SIMD_REGISTER(std::complex<double>, Emulated, __m128d);
}  // namespace types
}  // namespace simd
#endif  // SIMD_WITH_SSE
