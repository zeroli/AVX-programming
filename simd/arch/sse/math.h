#pragma once

#include "simd/types/sse_register.h"
#include "simd/types/vec.h"

#if 0
namespace simd {
namespace kernel {
using namespace types;

/// abs
template <typename Arch>
Vec<float, Arch> abs(const Vec<float, Arch>& self, requires_arch<SSE>) noexcept
{
    auto sign_mask = _mm_set1_ps(-0.f);  // -0.f = 1 << 31
    return _mm_andnot_ps(sign_mask, self);
}

template <typename Arch>
Vec<double, Arch> abs(const Vec<double, Arch>& self, requires_arch<SSE>) noexcept
{
    auto sign_mask = _mm_set1_pd(-0.f);  // -0.f = 1 << 31
    return _mm_andnot_pd(sign_mask, self);
}
template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
Vec<T, Arch> abs(const Vec<T, Arch>& self, requires_arch<SSE>) noexcept
{
    if (sizeof(T) == 1) {
        return _mm_abs_epi8(self);
    } else if (sizeof(T) == 2) {
        return _mm_abs_epi16(self);
    } else if (sizeof(T) == 4) {
        return _mm_abs_epi32(self);
    } else if (sizeof(T) == 8) {
        return _mm_abs_epi64(self);
    } else {
        assert(0 && "unsupported abs op for sizeof(T) in SSE arch");
        return {};
    }
}

/// max
template <typename Arch>
Vec<float, Arch> max(const Vec<float, Arch>& self, const Vec<float, Arch>& other, requires_arch<SSE>) noexcept
{
    return _mm_max_ps(self, other);
}
template <typename Arch>
Vec<double, Arch> max(const Vec<double, Arch>& self, const Vec<float, Arch>& other, requires_arch<SSE>) noexcept
{
    return _mm_max_pd(self, other);
}

template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
Vec<T, Arch> max(const Vec<T, Arch>& self, requires_arch<SSE>) noexcept
{
    return select(self > other, self, other);
}

/// min
template <typename Arch>
Vec<float, Arch> min(const Vec<float, Arch>& self, const Vec<float, Arch>& other, requires_arch<SSE>) noexcept
{
    return _mm_min_ps(self, other);
}
template <typename Arch>
Vec<double, Arch> min(const Vec<double, Arch>& self, const Vec<float, Arch>& other, requires_arch<SSE>) noexcept
{
    return _mm_min_pd(self, other);
}

template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
Vec<T, Arch> min(const Vec<T, Arch>& self, requires_arch<SSE>) noexcept
{
    return select(self <= other, self, other);
}

/// reciprocal
template <typename Arch>
Vec<float, Arch> reciprocal(const Vec<float, Arch>& self, requires_arch<SSE>) noexcept
{
    return _mm_rcp_ps(self);
}

/// rsqrt
template <typename Arch>
Vec<float, Arch> rsqrt(const Vec<float, Arch>& self, requires_arch<SSE>) noexcept
{
    return _mm_rsqrt_ps(self);
}
template <typename Arch>
Vec<double, Arch> rsqrt(const Vec<double, Arch>& self, requires_arch<SSE>) noexcept
{
    return _mm_cvtps_pd(_mm_rsqrt_ps(_mm_cvtpd_ps(self)));
}

/// sqrt
template <typename Arch>
Vec<float, Arch> sqrt(const Vec<float, Arch>& self, requires_arch<SSE>) noexcept
{
    return _mm_sqrt_ps(self);
}
template <typename Arch>
Vec<double, Arch> sqrt(const Vec<double, Arch>& self, requires_arch<SSE>) noexcept
{
    return _mm_sqrt_pd(self);
}


}  // namespace kernel
}  // namespace simd
#endif
