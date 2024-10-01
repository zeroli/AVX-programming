#pragma once

#include "simd/types/generic_arch.h"

namespace simd {
/// AVX512F instructions
struct AVX512F : Generic
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

#define DECLARE_SIMD_AVX512_REGISTER(SCALAR_TYPE, ISA, VECTOR_TYPE) \
template <> \
struct avx512_reg_traits<SCALAR_TYPE> { \
    using type = VECTOR_TYPE; \
}; \
DECLARE_SIMD_REGISTER(SCALAR_TYPE, ISA, VECTOR_TYPE) \
///###

DECLARE_SIMD_AVX512_REGISTER(int8_t,               AVX512F, avx512_reg_i);
DECLARE_SIMD_AVX512_REGISTER(uint8_t,              AVX512F, avx512_reg_i);
DECLARE_SIMD_AVX512_REGISTER(int16_t,              AVX512F, avx512_reg_i);
DECLARE_SIMD_AVX512_REGISTER(uint16_t,             AVX512F, avx512_reg_i);
DECLARE_SIMD_AVX512_REGISTER(int32_t,              AVX512F, avx512_reg_i);
DECLARE_SIMD_AVX512_REGISTER(uint32_t,             AVX512F, avx512_reg_i);
DECLARE_SIMD_AVX512_REGISTER(int64_t,              AVX512F, avx512_reg_i);
DECLARE_SIMD_AVX512_REGISTER(uint64_t,             AVX512F, avx512_reg_i);
DECLARE_SIMD_AVX512_REGISTER(float,                AVX512F, avx512_reg_f);
DECLARE_SIMD_AVX512_REGISTER(double,               AVX512F, avx512_reg_d);
DECLARE_SIMD_AVX512_REGISTER(std::complex<float>,  AVX512F, avx512_reg_f);
DECLARE_SIMD_AVX512_REGISTER(std::complex<double>, AVX512F, avx512_reg_d);
}  // namespace types
}  // namespace simd
#endif  // SIMD_WITH_AVX512F
