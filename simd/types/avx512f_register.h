#pragma once

#include "simd/types/generic_arch.h"

namespace simd {
/// AVX512F instructions
struct AVX512F : generic
{
    static constexpr bool supported() noexcept { return SIMD_WITH_AVX512F; }
    static constexpr bool available() noexcept { return true; }
    static constexpr size_t alignment() noexcept { return 64; }
    static constexpr bool requires_alignment() noexcept { return true; }
    static constexpr const char* name() noexcept { return "AVX512F"; }
};
}  // namespace simd

#if  SIMD_WITH_AVX512F
#include <immintrin.h>

namespace simd {
namespace types {
DECLARE_SIMD_REGISTER(signed char, AVX512F, __m512i);
DECLARE_SIMD_REGISTER(unsigned char, AVX512F, __m512i);
DECLARE_SIMD_REGISTER(char, AVX512F, __m512i);
DECLARE_SIMD_REGISTER(unsigned short, AVX512F, __m512i);
DECLARE_SIMD_REGISTER(short, AVX512F, __m512i);
DECLARE_SIMD_REGISTER(unsigned int, AVX512F, __m512i);
DECLARE_SIMD_REGISTER(int, AVX512F, __m512i);
DECLARE_SIMD_REGISTER(unsigned long int, AVX512F, __m512i);
DECLARE_SIMD_REGISTER(long int, AVX512F, __m512i);
DECLARE_SIMD_REGISTER(unsigned long long int, AVX512F, __m512i);
DECLARE_SIMD_REGISTER(long long int, AVX512F, __m512i);
DECLARE_SIMD_REGISTER(float, AVX512F, __m512);
DECLARE_SIMD_REGISTER(double, AVX512F, __m512d);
}  // namespace types
}  // namespace simd
#endif  // SIMD_WITH_AVX512F
