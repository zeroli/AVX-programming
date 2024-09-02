#pragma once

#include "simd/types/avx_register.h"
#include "simd/types/vec.h"

#include <cstddef>
#include <cstdint>

namespace simd {
namespace kernel {
using namespace types;

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

template <typename Arch, typename TOut, typename TIn>
VecBool<TOut, Arch> vec_bool_cast(const VecBool<Tin, Arch>& self, requires_arch<AVX>) noexcept
{
    return { bitwise_cast<TOut>(Vec<TIn, Arch>(self.data)).data };
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

/// bitwise_rshift
template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
Vec<T, Arch> bitwise_rshift(const Vec<T, Arch>& self, int32_t other, requires_arch<AVX>) noexcept
{
    return detail::fwd_to_sse([](__m128i s, int32_t o) noexcept {
        return bitwise_rshift(Vec<T, SSE>(s), o, SSE{});
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
Vec<double, Arch> bitwise_cast(const Vec<float, Arch>& self, requires_arch<AVX>) noexcept
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
Vec<float, Arch> bitwise_cast(const Vec<double, Arch>& self, requires_arch<AVX>) noexcept
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
Vec<float, Arch> broadcast(float val, requires_arch<AVX>) noexcept
{
    return _mm256_set1_ps(val);
}

template <typename Arch>
Vec<double, Arch> broadcast(double val, requires_arch<AVX>) noexcept
{
    return _mm256_set1_pd(val);
}

/// ceil
template <typename Arch>
Vec<float, Arch> ceil(const Vec<float, Arch>& self, requires_arch<AVX>) noexcept
{
    return _mm256_ceil_ps(self);
}

template <typename Arch>
Vec<double, Arch> broadcast(const Vec<float, Arch>& self, requires_arch<AVX>) noexcept
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
Vec<float, Arch> div(const Vec<float, Arch>& self, const Vec<float, Arch>& other, requires_arch<AVX>) noexcept
{
    return _mm256_div_ps(self, other);
}
template <typename Arch>
Vec<double, Arch> div(const Vec<double, Arch>& self, const Vec<double, Arch>& other, requires_arch<AVX>) noexcept
{
    return _mm256_div_pd(self, other);
}

/// floor
template <typename Arch>
Vec<float, Arch> floor(const Vec<float, Arch>& self, requires_arch<AVX>) noexcept
{
    return _mm256_floor_ps(self);
}
template <typename Arch>
Vec<double, Arch> floor(const Vec<double, Arch>& self, requires_arch<AVX>) noexcept
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

/// fnma
template <typename Arch>
Vec<float, Arch> fnma(const Vec<float, Arch>& x, const Vec<float, Arch>& y,
            const Vec<float, Arch>& z, requires_arch<AVX>) noexcept
{
    return _mm256_fnmadd_ps(x, y, z);
}

template <typename Arch>
Vec<double, Arch> fnma(const Vec<double, Arch>& x, const Vec<double, Arch>& y,
            const Vec<double, Arch>& z, requires_arch<AVX>) noexcept
{
    return _mm256_fnmadd_pd(x, y, z);
}

/// fnms
template <typename Arch>
Vec<float, Arch> fnms(const Vec<float, Arch>& x, const Vec<float, Arch>& y,
            const Vec<float, Arch>& z, requires_arch<AVX>) noexcept
{
    return _mm256_fnmsub_ps(x, y, z);
}

template <typename Arch>
Vec<double, Arch> fnms(const Vec<double, Arch>& x, const Vec<double, Arch>& y,
            const Vec<double, Arch>& z, requires_arch<AVX>) noexcept
{
    return _mm256_fnmsub_pd(x, y, z);
}

/// fma
template <typename Arch>
Vec<float, Arch> fma(const Vec<float, Arch>& x, const Vec<float, Arch>& y,
            const Vec<float, Arch>& z, requires_arch<AVX>) noexcept
{
    return _mm256_fmadd_ps(x, y, z);
}

template <typename Arch>
Vec<double, Arch> fma(const Vec<double, Arch>& x, const Vec<double, Arch>& y,
            const Vec<double, Arch>& z, requires_arch<AVX>) noexcept
{
    return _mm256_fmadd_pd(x, y, z);
}

/// fms
template <typename Arch>
Vec<float, Arch> fms(const Vec<float, Arch>& x, const Vec<float, Arch>& y,
            const Vec<float, Arch>& z, requires_arch<AVX>) noexcept
{
    return _mm256_fmsub_ps(x, y, z);
}

template <typename Arch>
Vec<double, Arch> fms(const Vec<double, Arch>& x, const Vec<double, Arch>& y,
            const Vec<double, Arch>& z, requires_arch<AVX>) noexcept
{
    return _mm256_fmsub_pd(x, y, z);
}


}  // namespace kernel
}  // namespace simd
