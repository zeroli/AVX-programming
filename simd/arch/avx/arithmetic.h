#pragma once

#include "simd/types/avx_register.h"
#include "simd/types/vec.h"

#include <cstddef>
#include <cstdint>

namespace simd {
namespace kernel {
using namespace types;

template <typename Arch>
Vec<float, Arch> abs(const Vec<float, Arch>& self, requires_arch<AVX>) noexcept
{
    __m256 sign_mask = _mm256_set1_ps(-0.f);  // -0.f = 1 << 31
    return _mm256_andnot_ps(sign_mask, self);
}

template <typename Arch>
Vec<double, Arch> abs(const Vec<double, Arch>& self, requires_arch<AVX>) noexcept
{
    __m256d sign_mask = _mm256_set1_ps(-0.f);  // -0.f = 1 << 31
    return _mm256_andnot_pd(sign_mask, self);
}

template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
Vec<T, Arch> add(const Vec<T, Arch>& self, const Vec<T, Arch>& other, requires_arch<AVX>) noexcept
{
    return detail::fwd_to_sse([](__m128i s, __m128i o) noexcept {
        return add(Vec<T, SSE>(s), Vec<T, SSE>(o));
    }, self, other);
}

template <typename Arch>
Vec<float, Arch> add(const Vec<float, Arch>& self, const Vec<float, Arch>& other, requires_arch<AVX>) noexcept
{
    return _mm256_add_ps(self, other);
}

template <typename Arch>
Vec<double, Arch> add(const Vec<double, Arch>& self, const Vec<double, Arch>& other, requires_arch<AVX>) noexcept
{
    return _mm256_add_pd(self, other);
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

template <typename Arch, typename TOut, typename TIn>
VecBool<TOut, Arch> vec_bool_cast(const VecBool<Tin, Arch>& self, requires_arch<AVX>) noexcept
{
    return { bitwise_cast<TOut>(Vec<TIn, Arch>(self.data)).data };
}

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

/// bitwise_lshift
template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
Vec<T, Arch> bitwise_lshift(const Vec<T, Arch>& self, int32_t other, requires_arch<AVX>) noexcept
{
    return detail::fwd_to_sse([](__m128i s, int32_t o) noexcept {
        return bitwise_lshift(Vec<T, SSE>(s), o, SSE{});
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

/// bitwise_rshift
template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
Vec<T, Arch> bitwise_rshift(const Vec<T, Arch>& self, int32_t other, requires_arch<AVX>) noexcept
{
    return detail::fwd_to_sse([](__m128i s, int32_t o) noexcept {
        return bitwise_rshift(Vec<T, SSE>(s), o, SSE{});
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

/// bitwise_cast
/// integer => float
template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
Vec<float, Arch> bitwise_cast(const Vec<T, Arch>& self, requires_arch<AVX>) noexcept
{
    return _mm256_castsi256_ps(self);
}
/// integer => double
template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
Vec<double, Arch> bitwise_cast(const Vec<T, Arch>& self, requires_arch<AVX>) noexcept
{
    return _mm256_castsi256_pd(self);
}

/// integer => integer
template <typename U, typename Arch, typename T,
    typename std::enable_if<
        std::is_integral<
            typename std::common_type<T, U>::type>::value>::type* = nullptr>
Vec<U, Arch> bitwise_cast(const Vec<T, Arch>& self, requires_arch<AVX>) noexcept
{
    return Vec<U, Arch>(self.data);
}

/// float => double
template <typename Arch>
Vec<double, Arch> bitwise_cast(const Vec<float, Arch>& self, requries_arch<AVX>) noexcept
{
    return _mm256_castps_pd(self);
}

/// float => integer
template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
Vec<T, Arch> bitwise_cast(const Vec<float, Arch>& self, requires_arch<AVX>) noexcept
{
    return _mm256_castps_si256(self);
}

/// double => float
template <typename Arch>
Vec<float, Arch> bitwise_cast(const Vec<double, Arch>& self, requries_arch<AVX>) noexcept
{
    return _mm256_castpd_ps(self);
}

/// double => integer
template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
Vec<T, Arch> bitwise_cast(const Vec<double, Arch>& self, requires_arch<AVX>) noexcept
{
    return _mm256_castpd_si256(self);
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

/// broadcast
template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
Vec<T, Arch> broadcast(T val, requires_arch<AVX>) noexcept
{
    if (sizeof(T) == 1) {
        return _mm256_set1_epi8(val);
    } else if (sizeof(T) == 2) {
        return _mm256_set1_epi16(val);
    } else if (sizeof(T) == 4) {
        return _mm256_set1_epi32(val);
    } else if (sizeof(T) == 8) {
        return _mm256_set1_epi64x(val);
    } else {
        assert(false && "unsupported sizeof(T) > 8");
        return {};
    }
}

template <typename Arch>
Vec<float, Arch> broadcast(float val, requries_arch<AVX>) noexcept
{
    return _mm256_set1_ps(val);
}

template <typename Arch>
Vec<double, Arch> broadcast(double val, requries_arch<AVX>) noexcept
{
    return _mm256_set1_pd(val);
}

/// ceil
template <typename Arch>
Vec<float, Arch> ceil(const Vec<float, Arch>& self, requries_arch<AVX>) noexcept
{
    return _mm256_ceil_ps(self);
}

template <typename Arch>
Vec<double, Arch> broadcast(const Vec<float, Arch>& self, requries_arch<AVX>) noexcept
{
    return _mm256_ceil_pd(val);
}

/// fast_cast
namespace detail {
template <typename Arch>
Vec<float, Arch> fast_cast(const Vec<int32_t, Arch>& self, requires_arch<AVX>) noexcept
{
    return _mm256_cvtepi32_ps(self);
}
template <typename Arch>
Vec<int32_t, Arch> fast_cast(const Vec<float, Arch>& self, requires_arch<AVX>) noexcept
{
    return _mm256_cvttps_epi32(self);
}

}  // namespace detail

/// decr_if
template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
Vec<T, Arch> decr_if(const Vec<T, Arch>& self, const VecBool<T, Arch>& mask, requires_arch<AVX>) noexcept
{
    return self - Vec<T, Arch>(mask.data);
}

/// div
template <typename Arch>
Vec<float, Arch> div(const Vec<float, Arch>& self, const Vec<float, Arch>& other, requries_arch<AVX>) noexcept
{
    return _mm256_div_ps(self, other);
}
template <typename Arch>
Vec<double, Arch> div(const Vec<double, Arch>& self, const Vec<double, Arch>& other, requries_arch<AVX>) noexcept
{
    return _mm256_div_pd(self, other);
}

/// eq
template <typename Arch>
VecBool<float, Arch> eq(const Vec<float, Arch>& self, const Vec<float, Arch>& other, requires_arch<AVX>) noexcept
{
    return _mm256_cmp_ps(self, other, _CMP_EQ_OQ);
}

template <typename Arch>
VecBool<double, Arch> eq(const Vec<double, Arch>& self, const Vec<float, Arch>& other, requires_arch<AVX>) noexcept
{
    return _mm256_cmp_pd(self, other, _CMP_EQ_OQ);
}

template <typename Arch>
VecBool<float, Arch> eq(const VecBool<float, Arch>& self, const VecBool<float, Arch>& other, requires_arch<AVX>) noexcept
{
    return ~(self != other);
}

template <typename Arch>
VecBool<double, Arch> eq(const VecBool<double, Arch>& self, const VecBool<float, Arch>& other, requires_arch<AVX>) noexcept
{
    return ~(self != other);
}

template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
VecBool<T, Arch> eq(const Vec<T, Arch>& self, const Vec<T, Arch>& other, requires_arch<AVX>) noexcept
{
    return detail::fwd_to_sse([](__m128i s, __m128i o) noexcept {
        return eq(Vec<T, SSE>(s), Vec<T, SSE>(o));
    }, self, other);
}

template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
VecBool<T, Arch> eq(const VecBool<T, Arch>& self, const VecBool<T, Arch>& other, requires_arch<AVX>) noexcept
{
    return ~(self != other);
}

/// floor
template <typename Arch>
Vec<float, Arch> floor(const Vec<float, Arch>& self, requries_arch<AVX>) noexcept
{
    return _mm256_floor_ps(self);
}
template <typename Arch>
Vec<double, Arch> floor(const Vec<double, Arch>& self, requries_arch<AVX>) noexcept
{
    return _mm256_floor_pd(self);
}

/// from_mask
template <typename Arch>
VecBool<float, Arch> from_mask(uint64_t mask, requires_arch<AVX>) noexcept
{
    alignas(Arch::alignment()) static const uint64_t lut32[] ={
        0x0000000000000000ul,
        0x00000000FFFFFFFFul,
        0xFFFFFFFF00000000ul,
        0xFFFFFFFFFFFFFFFFul,
    };

    assert(!(mask & ~0xFFul) && "inbound mask");
    return _mm256_castsi256_ps(_mm256_setr_epi64x(
                lut32[mask & 0x3],
                lut32[(mask >> 2) & 0x3],
                lut32[(mask >> 4) & 0x3],
                lut32[mask >> 6]
            ));
}

template <typename Arch>
VecBool<double, Arch> from_mask(uint64_t mask, requires_arch<AVX>) noexcept
{
    alignas(Arch::alignment()) static const uint64_t lut64[][4] = {
        { 0x0000000000000000ul, 0x0000000000000000ul, 0x0000000000000000ul, 0x0000000000000000ul },
        { 0xFFFFFFFFFFFFFFFFul, 0x0000000000000000ul, 0x0000000000000000ul, 0x0000000000000000ul },
        { 0x0000000000000000ul, 0xFFFFFFFFFFFFFFFFul, 0x0000000000000000ul, 0x0000000000000000ul },
        { 0xFFFFFFFFFFFFFFFFul, 0xFFFFFFFFFFFFFFFFul, 0x0000000000000000ul, 0x0000000000000000ul },
        { 0x0000000000000000ul, 0x0000000000000000ul, 0xFFFFFFFFFFFFFFFFul, 0x0000000000000000ul },
        { 0xFFFFFFFFFFFFFFFFul, 0x0000000000000000ul, 0xFFFFFFFFFFFFFFFFul, 0x0000000000000000ul },
        { 0x0000000000000000ul, 0xFFFFFFFFFFFFFFFFul, 0xFFFFFFFFFFFFFFFFul, 0x0000000000000000ul },
        { 0xFFFFFFFFFFFFFFFFul, 0xFFFFFFFFFFFFFFFFul, 0xFFFFFFFFFFFFFFFFul, 0x0000000000000000ul },
        { 0x0000000000000000ul, 0x0000000000000000ul, 0x0000000000000000ul, 0xFFFFFFFFFFFFFFFFul },
        { 0xFFFFFFFFFFFFFFFFul, 0x0000000000000000ul, 0x0000000000000000ul, 0xFFFFFFFFFFFFFFFFul },
        { 0x0000000000000000ul, 0xFFFFFFFFFFFFFFFFul, 0x0000000000000000ul, 0xFFFFFFFFFFFFFFFFul },
        { 0xFFFFFFFFFFFFFFFFul, 0xFFFFFFFFFFFFFFFFul, 0x0000000000000000ul, 0xFFFFFFFFFFFFFFFFul },
        { 0x0000000000000000ul, 0x0000000000000000ul, 0xFFFFFFFFFFFFFFFFul, 0xFFFFFFFFFFFFFFFFul },
        { 0xFFFFFFFFFFFFFFFFul, 0x0000000000000000ul, 0xFFFFFFFFFFFFFFFFul, 0xFFFFFFFFFFFFFFFFul },
        { 0x0000000000000000ul, 0xFFFFFFFFFFFFFFFFul, 0xFFFFFFFFFFFFFFFFul, 0xFFFFFFFFFFFFFFFFul },
        { 0xFFFFFFFFFFFFFFFFul, 0xFFFFFFFFFFFFFFFFul, 0xFFFFFFFFFFFFFFFFul, 0xFFFFFFFFFFFFFFFFul },
    };
    assert(!(mask & ~0xFul) && "inbound mask");
    return _mm256_castsi256_pd(_mm256_load_si256((const __m256i*)lut64[mask]));
}

template <typename Arch, typename T,
        typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
VecBool<T, Arch> from_mask(uint64_t mask, requires_arch<AVX>) noexcept
{
    alignas(Arch::alignment()) static const uint32_t lut32[] = {
        0x00000000,
        0x000000FF,
        0x0000FF00,
        0x0000FFFF,
        0x00FF0000,
        0x00FF00FF,
        0x00FFFF00,
        0x00FFFFFF,
        0xFF000000,
        0xFF0000FF,
        0xFF00FF00,
        0xFF00FFFF,
        0xFFFF0000,
        0xFFFF00FF,
        0xFFFFFF00,
        0xFFFFFFFF,
    };
    alignas(Arch::alignment()) static const uint64_t lut64[] = {
        0x0000000000000000ul,
        0x000000000000FFFFul,
        0x00000000FFFF0000ul,
        0x00000000FFFFFFFFul,
        0x0000FFFF00000000ul,
        0x0000FFFF0000FFFFul,
        0x0000FFFFFFFF0000ul,
        0x0000FFFFFFFFFFFFul,
        0xFFFF000000000000ul,
        0xFFFF00000000FFFFul,
        0xFFFF0000FFFF0000ul,
        0xFFFF0000FFFFFFFFul,
        0xFFFFFFFF00000000ul,
        0xFFFFFFFF0000FFFFul,
        0xFFFFFFFFFFFF0000ul,
        0xFFFFFFFFFFFFFFFFul,
    };
    if (sizeof(T) == 1) {
        assert(!(mask & ~0xFFFFFFFFul) && "inbound mask");
        return _mm256_setr_epi32(lut32[mask & 0xF], lut32[(mask >> 4) & 0xF],
                                    lut32[(mask >> 8) & 0xF], lut32[(mask >> 12) & 0xF],
                                    lut32[(mask >> 16) & 0xF], lut32[(mask >> 20) & 0xF],
                                    lut32[(mask >> 24) & 0xF], lut32[mask >> 28]);
    } else if(sizeof(T) == 2) {
        assert(!(mask & ~0xFFFFul) && "inbound mask");
        return _mm256_setr_epi64x(
                    lut64[mask & 0xF],
                    lut64[(mask >> 4) & 0xF],
                    lut64[(mask >> 8) & 0xF],
                    lut64[(mask >> 12) & 0xF]);
    } else if(sizeof(T) == 4) {
        return _mm256_castps_si256(from_mask(VecBool<float, Arch>{}, mask, AVX{}));
    } else if(sizeof(T) == 8) {
        return _mm256_castpd_si256(from_mask(VecBool<double, Arch>{}, mask, AVX{}));
    } else {
        assert(0 && "unsupported sizeof(T) > 8");
        return {};
    }
}
}  // namespace kernel
}  // namespace simd
