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
#include <complex>

namespace simd {
namespace types {
using avx_reg_i = __m256i;
using avx_reg_f = __m256;
using avx_reg_d = __m256d;

template <typename T, typename Enable = void>
struct avx_reg_traits;

template <typename T>
using avx_reg_traits_t = typename avx_reg_traits<T>::type;

#define DECLARE_SIMD_AVX_REGISTER(SCALAR_TYPE, ISA, VECTOR_TYPE) \
template <> \
struct avx_reg_traits<SCALAR_TYPE> { \
    using type = VECTOR_TYPE; \
}; \
DECLARE_SIMD_REGISTER(SCALAR_TYPE, ISA, VECTOR_TYPE) \
///###

DECLARE_SIMD_AVX_REGISTER(int8_t,               AVX, avx_reg_i);
DECLARE_SIMD_AVX_REGISTER(uint8_t,              AVX, avx_reg_i);
DECLARE_SIMD_AVX_REGISTER(int16_t,              AVX, avx_reg_i);
DECLARE_SIMD_AVX_REGISTER(uint16_t,             AVX, avx_reg_i);
DECLARE_SIMD_AVX_REGISTER(int32_t,              AVX, avx_reg_i);
DECLARE_SIMD_AVX_REGISTER(uint32_t,             AVX, avx_reg_i);
DECLARE_SIMD_AVX_REGISTER(int64_t,              AVX, avx_reg_i);
DECLARE_SIMD_AVX_REGISTER(uint64_t,             AVX, avx_reg_i);
DECLARE_SIMD_AVX_REGISTER(float,                AVX, avx_reg_f);
DECLARE_SIMD_AVX_REGISTER(double,               AVX, avx_reg_d);
DECLARE_SIMD_AVX_REGISTER(std::complex<float>,  AVX, avx_reg_f);
DECLARE_SIMD_AVX_REGISTER(std::complex<double>, AVX, avx_reg_d);
}  // namespace types
}  // namespace simd
#endif  // SIMD_WITH_AVX
