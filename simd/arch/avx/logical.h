#pragma once

#include "simd/types/avx_register.h"
#include "simd/types/vec.h"

namespace simd {
namespace kernel {
using namespace types;

template <typename Arch>
Vec<float, Arch> bitwise_and(const Vec<float, Arch>& self, const Vec<float, Arch>& other, requires_arch<AVX>) noexcept
{
    return _mm256_and_ps(self, other);
}

template <typename Arch>
Vec<double, Arch> bitwise_and(const Vec<double, Arch>& self, const Vec<double, Arch>& other, requires_arch<AVX>) noexcept
{
    return _mm256_and_pd(self, other);
}

template <typename Arch>
VecBool<float, Arch> bitwise_and(const VecBool<float, Arch>& self, const VecBool<float, Arch>& other, requires_arch<AVX>) noexcept
{
    return _mm256_and_ps(self, other);
}

template <typename Arch>
VecBool<double, Arch> bitwise_and(const VecBool<double, Arch>& self, const VecBool<double, Arch>& other, requires_arch<AVX>) noexcept
{
    return _mm256_and_pd(self, other);
}

template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
Vec<T, Arch> bitwise_and(const Vec<T, Arch>& self, const Vec<T, Arch>& other, requires_arch<AVX>) noexcept
{
    return detail::fwd_to_sse([](__m128i s, __m128i o) noexcept {
        return bitwise_and(Vec<T, SSE>(s), Vec<T, SSE>(o));
    }, self, other);
}

template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
VecBool<T, Arch> bitwise_and(const VecBool<T, Arch>& self, const VecBool<T, Arch>& other, requires_arch<AVX>) noexcept
{
    return detail::fwd_to_sse([](__m128i s, __m128i o) noexcept {
        return bitwise_and(Vec<T, SSE>(s), Vec<T, SSE>(o));
    }, self, other);
}

/// bitwise_andnot
template <typename Arch>
Vec<float, Arch> bitwise_andnot(const Vec<float, Arch>& self, const Vec<float, Arch>& other, requires_arch<AVX>) noexcept
{
    return _mm256_andnot_ps(self, other);
}

template <typename Arch>
Vec<double, Arch> bitwise_andnot(const Vec<double, Arch>& self, const Vec<double, Arch>& other, requires_arch<AVX>) noexcept
{
    return _mm256_andnot_pd(self, other);
}

template <typename Arch>
VecBool<float, Arch> bitwise_andnot(const VecBool<float, Arch>& self, const VecBool<float, Arch>& other, requires_arch<AVX>) noexcept
{
    return _mm256_andnot_ps(self, other);
}

template <typename Arch>
VecBool<double, Arch> bitwise_andnot(const VecBool<double, Arch>& self, const VecBool<double, Arch>& other, requires_arch<AVX>) noexcept
{
    return _mm256_andnot_pd(self, other);
}

template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
Vec<T, Arch> bitwise_andnot(const Vec<T, Arch>& self, const Vec<T, Arch>& other, requires_arch<AVX>) noexcept
{
    return detail::fwd_to_sse([](__m128i s, __m128i o) noexcept {
        return bitwise_andnot(Vec<T, SSE>(s), Vec<T, SSE>(o));
    }, self, other);
}

template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
VecBool<T, Arch> bitwise_andnot(const VecBool<T, Arch>& self, const VecBool<T, Arch>& other, requires_arch<AVX>) noexcept
{
    return detail::fwd_to_sse([](__m128i s, __m128i o) noexcept {
        return bitwise_andnot(Vec<T, SSE>(s), Vec<T, SSE>(o));
    }, self, other);
}

/// bitwise_not
template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
Vec<T, Arch> bitwise_not(const Vec<T, Arch>& self, requires_arch<AVX>) noexcept
{
    return detail::fwd_to_sse([](__m128i s) noexcept {
        return bitwise_not(Vec<T, SSE>(s), SSE{});
    }, self);
}

template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
VecBool<T, Arch> bitwise_not(const VecBool<T, Arch>& self, requires_arch<AVX>) noexcept
{
    return detail::fwd_to_sse([](__m128i s) noexcept {
        return bitwise_not(Vec<T, SSE>(s), SSE{});
    }, self, other);
}

/// bitwise_or
template <typename Arch>
Vec<float, Arch> bitwise_or(const Vec<float, Arch>& self, const Vec<float, Arch>& other, requires_arch<AVX>) noexcept
{
    return _mm256_or_ps(self, other);
}

template <typename Arch>
Vec<double, Arch> bitwise_or(const Vec<double, Arch>& self, const Vec<double, Arch>& other, requires_arch<AVX>) noexcept
{
    return _mm256_or_pd(self, other);
}

template <typename Arch>
VecBool<float, Arch> bitwise_or(const VecBool<float, Arch>& self, const VecBool<float, Arch>& other, requires_arch<AVX>) noexcept
{
    return _mm256_or_ps(self, other);
}

template <typename Arch>
VecBool<double, Arch> bitwise_or(const VecBool<double, Arch>& self, const VecBool<double, Arch>& other, requires_arch<AVX>) noexcept
{
    return _mm256_or_pd(self, other);
}

template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
Vec<T, Arch> bitwise_or(const Vec<T, Arch>& self, const Vec<T, Arch>& other, requires_arch<AVX>) noexcept
{
    return detail::fwd_to_sse([](__m128i s, __m128i o) noexcept {
        return bitwise_or(Vec<T, SSE>(s), Vec<T, SSE>(o));
    }, self, other);
}

template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
VecBool<T, Arch> bitwise_or(const VecBool<T, Arch>& self, const VecBool<T, Arch>& other, requires_arch<AVX>) noexcept
{
    return detail::fwd_to_sse([](__m128i s, __m128i o) noexcept {
        return bitwise_or(Vec<T, SSE>(s), Vec<T, SSE>(o));
    }, self, other);
}

/// bitwise_xor
template <typename Arch>
Vec<float, Arch> bitwise_xor(const Vec<float, Arch>& self, const Vec<float, Arch>& other, requires_arch<AVX>) noexcept
{
    return _mm256_xor_ps(self, other);
}

template <typename Arch>
Vec<double, Arch> bitwise_xor(const Vec<double, Arch>& self, const Vec<double, Arch>& other, requires_arch<AVX>) noexcept
{
    return _mm256_xor_pd(self, other);
}

template <typename Arch>
VecBool<float, Arch> bitwise_xor(const VecBool<float, Arch>& self, const VecBool<float, Arch>& other, requires_arch<AVX>) noexcept
{
    return _mm256_xor_ps(self, other);
}

template <typename Arch>
VecBool<double, Arch> bitwise_xor(const VecBool<double, Arch>& self, const VecBool<double, Arch>& other, requires_arch<AVX>) noexcept
{
    return _mm256_xor_pd(self, other);
}

template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
Vec<T, Arch> bitwise_xor(const Vec<T, Arch>& self, const Vec<T, Arch>& other, requires_arch<AVX>) noexcept
{
    return detail::fwd_to_sse([](__m128i s, __m128i o) noexcept {
        return bitwise_xor(Vec<T, SSE>(s), Vec<T, SSE>(o));
    }, self, other);
}

template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
VecBool<T, Arch> bitwise_xor(const VecBool<T, Arch>& self, const VecBool<T, Arch>& other, requires_arch<AVX>) noexcept
{
    return detail::fwd_to_sse([](__m128i s, __m128i o) noexcept {
        return bitwise_xor(Vec<T, SSE>(s), Vec<T, SSE>(o));
    }, self, other);
}

/// bitwise_not
template <typename Arch>
Vec<float, Arch> bitwise_not(const Vec<float, Arch>& self, requires_arch<AVX>) noexcept
{
    return _mm256_xor_ps(self, _m256_castsi256_ps(_mm256_set1_epi32(-1)));
}
template <typename Arch>
Vec<double, Arch> bitwise_not(const Vec<double, Arch>& self, requires_arch<AVX>) noexcept
{
    return _mm256_xor_pd(self, _m256_castsi256_pd(_mm256_set1_epi32(-1)));
}

template <typename Arch>
VecBool<float, Arch> bitwise_not(const VecBool<float, Arch>& self, requires_arch<AVX>) noexcept
{
    return _mm256_xor_ps(self, _m256_castsi256_ps(_mm256_set1_epi32(-1)));
}
template <typename Arch>
VecBool<double, Arch> bitwise_not(const VecBool<double, Arch>& self, requires_arch<AVX>) noexcept
{
    return _mm256_xor_pd(self, _m256_castsi256_pd(_mm256_set1_epi32(-1)));
}

template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
bool all(const VecBool<T, Arch>& self, requires_arch<AVX>) noexcept
{
    return _mm256_testc_si256(self, VecBool<T, Arch>(true)) != 0);
}

template <typename Arch>
bool all(const VecBool<float, Arch>& self, requires_arch<AVX>) noexcept
{
    return _mm256_testc_ps(self, VecBool<float, Arch>(true)) != 0);
}

template <typename Arch>
bool all(const VecBool<double, Arch>& self, requires_arch<AVX>) noexcept
{
    return _mm256_testc_pd(self, VecBool<double, Arch>(true)) != 0);
}

template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
bool any(const VecBool<T, Arch>& self, requires_arch<AVX>) noexcept
{
    return !_mm256_testz_si256(self, self);
}

template <typename Arch>
bool any(const VecBool<float, Arch>& self, requires_arch<AVX>) noexcept
{
    return !_mm256_testz_ps(self, self);
}

template <typename Arch>
bool any(const VecBool<double, Arch>& self, requires_arch<AVX>) noexcept
{
    return !_mm256_testz_pd(self, self);
}

}  // namespace kernel
}  // namespace simd
