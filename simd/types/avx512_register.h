#pragma once

#include "simd/types/generic_arch.h"

namespace simd {
/// AVX512 instructions
struct AVX512 : Generic
{
    static constexpr bool supported() noexcept { return SIMD_WITH_AVX512; }
    static constexpr bool available() noexcept { return true; }
    static constexpr size_t alignment() noexcept { return 64; }
    static constexpr bool requires_alignment() noexcept { return true; }
    static constexpr const char* name() noexcept { return "AVX512"; }
};
}  // namespace simd

#if  SIMD_WITH_AVX512
#include <immintrin.h>
#include <complex>

namespace simd {
namespace types {
using avx512_reg_i = __m512i;
using avx512_reg_f = __m512;
using avx512_reg_d = __m512d;

template <typename T, typename Enable = void>
struct avx512_reg_traits;

template <typename T>
using avx512_reg_traits_t = typename avx512_reg_traits<T>::type;

template <typename T, typename Enable = void>
struct avx512_mask_traits;

template <typename T>
using avx512_mask_traits_t = typename avx512_mask_traits<T>::type;

/// bunch of avx512 mask types for different scalar type
/// basically: sizeof(T) * mask bits * 8 = 64 * 8 = 512
/// each bit indicates one element, in compact way
template <typename T>
struct avx512_mask_traits<T, ENABLE_IF(sizeof(T) == 1)> {
    using type = __mmask64;
};
template <typename T>
struct avx512_mask_traits<T, ENABLE_IF(sizeof(T) == 2)> {
    using type = __mmask32;
};
template <typename T>
struct avx512_mask_traits<T, ENABLE_IF(sizeof(T) == 4)> {
    using type = __mmask16;
};
template <typename T>
struct avx512_mask_traits<T, ENABLE_IF(sizeof(T) == 8)> {
    using type = __mmask8;
};
template <>
struct avx512_mask_traits<std::complex<float>> {
    /// complex<float>, 2 floats as unit, 2-bit of 16 bits for each unit
    using type = __mmask16;
};
template <>
struct avx512_mask_traits<std::complex<double>> {
    /// complex<double>, 2 doubles as unit, 2-bit of 8 bits for each unit
    using type = __mmask8;
};

#define DECLARE_SIMD_AVX512_REGISTER(SCALAR_TYPE, ISA, VECTOR_TYPE) \
template <> \
struct avx512_reg_traits<SCALAR_TYPE> { \
    using type = VECTOR_TYPE; \
}; \
DECLARE_SIMD_REGISTER(SCALAR_TYPE, ISA, VECTOR_TYPE) \
///###

DECLARE_SIMD_AVX512_REGISTER(int8_t,               AVX512, avx512_reg_i);
DECLARE_SIMD_AVX512_REGISTER(uint8_t,              AVX512, avx512_reg_i);
DECLARE_SIMD_AVX512_REGISTER(int16_t,              AVX512, avx512_reg_i);
DECLARE_SIMD_AVX512_REGISTER(uint16_t,             AVX512, avx512_reg_i);
DECLARE_SIMD_AVX512_REGISTER(int32_t,              AVX512, avx512_reg_i);
DECLARE_SIMD_AVX512_REGISTER(uint32_t,             AVX512, avx512_reg_i);
DECLARE_SIMD_AVX512_REGISTER(int64_t,              AVX512, avx512_reg_i);
DECLARE_SIMD_AVX512_REGISTER(uint64_t,             AVX512, avx512_reg_i);
DECLARE_SIMD_AVX512_REGISTER(float,                AVX512, avx512_reg_f);
DECLARE_SIMD_AVX512_REGISTER(double,               AVX512, avx512_reg_d);
DECLARE_SIMD_AVX512_REGISTER(std::complex<float>,  AVX512, avx512_reg_f);
DECLARE_SIMD_AVX512_REGISTER(std::complex<double>, AVX512, avx512_reg_d);
}  // namespace types
}  // namespace simd
#endif  // SIMD_WITH_AVX512
