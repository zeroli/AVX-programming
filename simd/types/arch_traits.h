#pragma once

#include "simd/types/all_registers.h"

#include <type_traits>
#include <cstddef>
#include <cstdint>
#include <complex>

namespace simd {
namespace types {
template <typename T, size_t W>
struct arch_traits;

template <typename T, size_t W>
using arch_traits_t = typename arch_traits<T, W>::arch_t;

/// 512bits
#if SIMD_WITH_AVX512F
#define DEFINE_ARCH_TRAITS_512_BITS(T) \
template <> \
struct arch_traits<T, 512/sizeof(T)/8> { \
    using arch_t = AVX512F; \
}; \
///

#elif SIMD_WITH_AVX2
#define DEFINE_ARCH_TRAITS_512_BITS(T) \
template <> \
struct arch_traits<T, 512/sizeof(T)/8> { \
    using arch_t = AVX2; \
}; \
///

#elif SIMD_WITH_AVX
#define DEFINE_ARCH_TRAITS_512_BITS(T) \
template <> \
struct arch_traits<T, 512/sizeof(T)/8> { \
    using arch_t = AVX; \
}; \
///

#elif SIMD_WITH_SSE
#define DEFINE_ARCH_TRAITS_512_BITS(T) \
template <> \
struct arch_traits<T, 512/sizeof(T)/8> { \
    using arch_t = SSE; \
}; \
///

#else
#define DEFINE_ARCH_TRAITS_512_BITS(T) \
template <> \
struct arch_traits<T, 512/sizeof(T)/8> { \
    using arch_t = generic; \
}; \
///
#endif

DEFINE_ARCH_TRAITS_512_BITS(int8_t);
DEFINE_ARCH_TRAITS_512_BITS(uint8_t);
DEFINE_ARCH_TRAITS_512_BITS(int16_t);
DEFINE_ARCH_TRAITS_512_BITS(uint16_t);
DEFINE_ARCH_TRAITS_512_BITS(int32_t);
DEFINE_ARCH_TRAITS_512_BITS(uint32_t);
DEFINE_ARCH_TRAITS_512_BITS(int64_t);
DEFINE_ARCH_TRAITS_512_BITS(uint64_t);
DEFINE_ARCH_TRAITS_512_BITS(float);
DEFINE_ARCH_TRAITS_512_BITS(double);
DEFINE_ARCH_TRAITS_512_BITS(std::complex<float>);
DEFINE_ARCH_TRAITS_512_BITS(std::complex<double>);

#undef DEFINE_ARCH_TRAITS_512_BITS

/// 256bits
#if SIMD_WITH_AVX512F || SIMD_WITH_AVX2 || SIMD_WITH_AVX
#define DEFINE_ARCH_TRAITS_512_BITS(T) \
template <> \
struct arch_traits<T, 256/sizeof(T)/8> { \
    using arch_t = AVX; \
}; \
///

#elif SIMD_WITH_SSE
#define DEFINE_ARCH_TRAITS_256_BITS(T) \
template <> \
struct arch_traits<T, 256/sizeof(T)/8> { \
    using arch_t = SSE; \
}; \
///

#else
#define DEFINE_ARCH_TRAITS_256_BITS(T) \
template <> \
struct arch_traits<T, 256/sizeof(T)/8> { \
    using arch_t = generic; \
}; \
///
#endif

DEFINE_ARCH_TRAITS_256_BITS(int8_t);
DEFINE_ARCH_TRAITS_256_BITS(uint8_t);
DEFINE_ARCH_TRAITS_256_BITS(int16_t);
DEFINE_ARCH_TRAITS_256_BITS(uint16_t);
DEFINE_ARCH_TRAITS_256_BITS(int32_t);
DEFINE_ARCH_TRAITS_256_BITS(uint32_t);
DEFINE_ARCH_TRAITS_256_BITS(int64_t);
DEFINE_ARCH_TRAITS_256_BITS(uint64_t);
DEFINE_ARCH_TRAITS_256_BITS(float);
DEFINE_ARCH_TRAITS_256_BITS(double);
DEFINE_ARCH_TRAITS_256_BITS(std::complex<float>);
DEFINE_ARCH_TRAITS_256_BITS(std::complex<double>);

#undef DEFINE_ARCH_TRAITS_256_BITS

/// 128bits
#if SIMD_WITH_AVX512F || SIMD_WITH_AVX2 || \
     SIMD_WITH_AVX || SIMD_WITH_SSE
#define DEFINE_ARCH_TRAITS_128_BITS(T) \
template <> \
struct arch_traits<T, 128/sizeof(T)/8> { \
    using arch_t = SSE; \
}; \
///

#else
#define DEFINE_ARCH_TRAITS_128_BITS(T) \
template <> \
struct arch_traits<T, 128/sizeof(T)/8> { \
    using arch_t = generic; \
}; \
///
#endif

DEFINE_ARCH_TRAITS_128_BITS(int8_t);
DEFINE_ARCH_TRAITS_128_BITS(uint8_t);
DEFINE_ARCH_TRAITS_128_BITS(int16_t);
DEFINE_ARCH_TRAITS_128_BITS(uint16_t);
DEFINE_ARCH_TRAITS_128_BITS(int32_t);
DEFINE_ARCH_TRAITS_128_BITS(uint32_t);
DEFINE_ARCH_TRAITS_128_BITS(int64_t);
DEFINE_ARCH_TRAITS_128_BITS(uint64_t);
DEFINE_ARCH_TRAITS_128_BITS(float);
DEFINE_ARCH_TRAITS_128_BITS(double);
DEFINE_ARCH_TRAITS_128_BITS(std::complex<float>);
DEFINE_ARCH_TRAITS_128_BITS(std::complex<double>);

#undef DEFINE_ARCH_TRAITS_128_BITS
}  // namespace types
}  // namespace simd
