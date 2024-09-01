#pragma once

#include "simd/types/all_registers.h"

#include <type_traits>
#include <cstddef>
#include <cstdint>

namespace simd {
namespace types {
template <typename T, size_t W>
struct arch_traits;

template <typename T, size_t W>
using arch_traits_t = typename arch_traits<T, W>::arch_t;

/// 512bits
template <>
struct arch_traits<int32_t, 16> {
#if SIMD_WITH_AVX512F
    using arch_t = AVX512F;
#elif SIMD_WITH_AVX2
    using arch_t = AVX2;
#elif SIMD_WITH_AVX
    using arch_t = AVX;
#elif SIMD_WITH_SSE
    using arch_t = SSE;
#endif
};
template <>
struct arch_traits<float, 16> {
#if SIMD_WITH_AVX512F
    using arch_t = AVX512F;
#elif SIMD_WITH_AVX2
    using arch_t = AVX2;
#elif SIMD_WITH_AVX
    using arch_t = AVX;
#elif SIMD_WITH_SSE
    using arch_t = SSE;
#endif
};
template <>
struct arch_traits<double, 8> {
#if SIMD_WITH_AVX512F
    using arch_t = AVX512F;
#elif SIMD_WITH_AVX2
    using arch_t = AVX2;
#elif SIMD_WITH_AVX
    using arch_t = AVX;
#elif SIMD_WITH_SSE
    using arch_t = SSE;
#endif
};

/// 256bits
template <>
struct arch_traits<int32_t, 8> {
#if SIMD_WITH_AVX
    using arch_t = AVX;
#elif SIMD_WITH_SSE
    using arch_t = SSE;
#endif
};

template <>
struct arch_traits<float, 8> {
#if SIMD_WITH_AVX
    using arch_t = AVX;
#elif SIMD_WITH_SSE
    using arch_t = SSE;
#endif
};
template <>
struct arch_traits<double, 4> {
#if SIMD_WITH_AVX
    using arch_t = AVX;
#elif SIMD_WITH_SSE
    using arch_t = SSE;
#endif
};

/// 128bits
template <>
struct arch_traits<int32_t, 4> {
#if SIMD_WITH_SSE
    using arch_t = SSE;
#endif
};
template <>
struct arch_traits<float, 4> {
#if SIMD_WITH_SSE
    using arch_t = SSE;
#endif
};
template <>
struct arch_traits<double, 2> {
#if SIMD_WITH_SSE
    using arch_t = SSE;
#endif
};
}  // namespace types
}  // namespace simd
