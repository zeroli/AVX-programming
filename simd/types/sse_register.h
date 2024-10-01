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
struct SSE : Generic
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

using sse_reg_i = __m128i;
using sse_reg_f = __m128;
using sse_reg_d = __m128d;

template <typename T, typename Enable = void>
struct sse_reg_traits;

template <typename T>
using sse_reg_traits_t = typename sse_reg_traits<T>::type;

#define DECLARE_SIMD_SSE_REGISTER(SCALAR_TYPE, ISA, VECTOR_TYPE) \
template <> \
struct sse_reg_traits<SCALAR_TYPE> { \
    using type = VECTOR_TYPE; \
}; \
DECLARE_SIMD_REGISTER(SCALAR_TYPE, ISA, VECTOR_TYPE) \
///###

DECLARE_SIMD_SSE_REGISTER(int8_t,               SSE, sse_reg_i);
DECLARE_SIMD_SSE_REGISTER(uint8_t,              SSE, sse_reg_i);
DECLARE_SIMD_SSE_REGISTER(int16_t,              SSE, sse_reg_i);
DECLARE_SIMD_SSE_REGISTER(uint16_t,             SSE, sse_reg_i);
DECLARE_SIMD_SSE_REGISTER(int32_t,              SSE, sse_reg_i);
DECLARE_SIMD_SSE_REGISTER(uint32_t,             SSE, sse_reg_i);
DECLARE_SIMD_SSE_REGISTER(int64_t,              SSE, sse_reg_i);
DECLARE_SIMD_SSE_REGISTER(uint64_t,             SSE, sse_reg_i);
DECLARE_SIMD_SSE_REGISTER(float,                SSE, sse_reg_f);
DECLARE_SIMD_SSE_REGISTER(double,               SSE, sse_reg_d);
DECLARE_SIMD_SSE_REGISTER(std::complex<float>,  SSE, sse_reg_f);
DECLARE_SIMD_SSE_REGISTER(std::complex<double>, SSE, sse_reg_d);
}  // namespace types
}  // namespace simd
#endif  // SIMD_WITH_SSE
