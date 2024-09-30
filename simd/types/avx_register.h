#pragma once

#include "simd/types/generic_arch.h"

namespace simd {
/// AVX instructions
struct AVX : Generic
{
    static constexpr bool supported() noexcept { return SIMD_WITH_AVX; }
    static constexpr bool available() noexcept { return true; }
    static constexpr size_t alignment() noexcept { return 32; }
    static constexpr bool requires_alignment() noexcept { return true; }
    static constexpr const char* name() noexcept { return "AVX"; }
};
}  // namespace simd

#if SIMD_WITH_AVX
#include <immintrin.h>

namespace simd {
namespace types {
DECLARE_SIMD_REGISTER(signed char, AVX, __m256i);
DECLARE_SIMD_REGISTER(unsigned char, AVX, __m256i);
DECLARE_SIMD_REGISTER(char, AVX, __m256i);
DECLARE_SIMD_REGISTER(unsigned short, AVX, __m256i);
DECLARE_SIMD_REGISTER(short, AVX, __m256i);
DECLARE_SIMD_REGISTER(unsigned int, AVX, __m256i);
DECLARE_SIMD_REGISTER(int, AVX, __m256i);
DECLARE_SIMD_REGISTER(unsigned long int, AVX, __m256i);
DECLARE_SIMD_REGISTER(long int, AVX, __m256i);
DECLARE_SIMD_REGISTER(unsigned long long int, AVX, __m256i);
DECLARE_SIMD_REGISTER(long long int, AVX, __m256i);
DECLARE_SIMD_REGISTER(float, AVX, __m256);
DECLARE_SIMD_REGISTER(double, AVX, __m256d);
DECLARE_SIMD_REGISTER(std::complex<float>, AVX, __m256);
DECLARE_SIMD_REGISTER(std::complex<double>, AVX, __m256d);
}  // namespace types
}  // namespace simd
#endif  // SIMD_WITH_AVX
