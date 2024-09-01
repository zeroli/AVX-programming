#pragma once

#include "simd/types/sse_register.h"
#include "simd/types/vec.h"

#include <limits>
#include <type_traits>
#include <cstddef>
#include <cstdint>

#if 0
namespace simd {
namespace kernel {
using namespace types;

/// add
template <typename Arch>
Vec<float, Arch> add(const Vec<float, Arch>& self, const Vec<float, Arch>& other, requires_arch<SSE>) noexcept
{
    return _mm_add_ps(self, other);
}

template <typename Arch>
Vec<double, Arch> add(const Vec<double, Arch>& self, const Vec<double, Arch>& other, requires_arch<SSE>) noexcept
{
    return _mm_add_pd(self, other);
}

template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
Vec<T, Arch> add(const Vec<T, Arch>& self, const Vec<T, Arch>& other, requires_arch<SSE>) noexcept
{
    if (sizeof(T) == 1) {
        return _mm_add_epi8(self, other);
    } else if (sizeof(T) == 2) {
        return _mm_add_epi16(self, other);
    } else if (sizeof(T) == 4) {
        return _mm_add_epi32(self, other);
    } else if (sizeof(T) == 8) {
        return _mm_add_epi64(self, other);
    } else {
        assert(false && "unsupported add op for sizeof(T) > 8 in SSE arch");
        return {};
    }
}

/// sub
template <typename Arch>
Vec<float, Arch> sub(const Vec<float, Arch>& self, const Vec<float, Arch>& other, requires_arch<SSE>) noexcept
{
    return _mm_sub_ps(self, other);
}

template <typename Arch>
Vec<double, Arch> sub(const Vec<double, Arch>& self, const Vec<double, Arch>& other, requires_arch<SSE>) noexcept
{
    return _mm_sub_pd(self, other);
}

template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
Vec<T, Arch> sub(const Vec<T, Arch>& self, const Vec<T, Arch>& other, requires_arch<SSE>) noexcept
{
    if (sizeof(T) == 1) {
        return _mm_sub_epi8(self, other);
    } else if (sizeof(T) == 2) {
        return _mm_sub_epi16(self, other);
    } else if (sizeof(T) == 4) {
        return _mm_sub_epi32(self, other);
    } else if (sizeof(T) == 8) {
        return _mm_sub_epi64(self, other);
    } else {
        assert(false && "unsupported sub op for sizeof(T) > 8 in SSE arch");
        return {};
    }
}

/// mul
template <typename Arch>
Vec<float, Arch> mul(const Vec<float, Arch>& self, const Vec<float, Arch>& other, requires_arch<SSE>) noexcept
{
    return _mm_mul_ps(self, other);
}

template <typename Arch>
Vec<double, Arch> mul(const Vec<double, Arch>& self, const Vec<float, Arch>& other, requires_arch<SSE>) noexcept
{
    return _mm_mul_pd(self, other);
}

template <typename Arch>
Vec<int16_t, Arch> mul(const Vec<int16_t, Arch>& self, const Vec<int16_t, Arch>& other, requires_arch<SSE>) noexcept
{
    return _mm_mullo_epi16(self, other);
}


/// div
template <typename Arch>
Vec<float, Arch> div(const Vec<float, Arch>& self, const Vec<float, Arch>& other, requires_arch<SSE>) noexcept
{
    return _mm_div_ps(self, other);
}

template <typename Arch>
Vec<double, Arch> div(const Vec<double, Arch>& self, const Vec<float, Arch>& other, requires_arch<SSE>) noexcept
{
    return _mm_div_pd(self, other);
}

/// neg
template <typename Arch>
Vec<float, Arch> neg(const Vec<float, Arch>& self, requires_arch<SSE>) noexcept
{
    return _mm_xor_ps(self,
                _mm_castsi128_ps(_mm_set1_epi32(0x80000000))
            );
}

template <typename Arch>
Vec<float, Arch> neg(const Vec<float, Arch>& self, requires_arch<SSE>) noexcept
{
    return _mm_xor_pd(self,
                _mm_castsi128_pd(
                    _mm_setr_epi32(0, 0x80000000, 0, 0x80000000))
            );
}

template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
Vec<T, Arch> neg(const Vec<T, Arch>& self, requires_arch<SSE>) noexcept
{
    return 0 - self;
}

/// fnma
template <typename Arch>
Vec<float, Arch> fnma(const Vec<float, Arch>& x, const Vec<float, Arch>& y,
        const Vec<float, Arch>& z, requires_arch<SSE>) noexcept
{
    return _mm_fnmadd_ps(x, y, z);
}

template <typename Arch>
Vec<double, Arch> fnma(const Vec<double, Arch>& x, const Vec<double, Arch>& y,
        const Vec<double, Arch>& z, requires_arch<SSE>) noexcept
{
    return _mm_fnmadd_pd(x, y, z);
}

/// fnms
template <typename Arch>
Vec<float, Arch> fnms(const Vec<float, Arch>& x, const Vec<float, Arch>& y,
        const Vec<float, Arch>& z, requires_arch<SSE>) noexcept
{
    return _mm_fnmsub_ps(x, y, z);
}

template <typename Arch>
Vec<double, Arch> fnms(const Vec<double, Arch>& x, const Vec<double, Arch>& y,
        const Vec<double, Arch>& z, requires_arch<SSE>) noexcept
{
    return _mm_fnmsub_pd(x, y, z);
}

/// fma
template <typename Arch>
Vec<float, Arch> fma(const Vec<float, Arch>& x, const Vec<float, Arch>& y,
        const Vec<float, Arch>& z, requires_arch<SSE>) noexcept
{
    return _mm_fmadd_ps(x, y, z);
}

template <typename Arch>
Vec<double, Arch> fma(const Vec<double, Arch>& x, const Vec<double, Arch>& y,
        const Vec<double, Arch>& z, requires_arch<SSE>) noexcept
{
    return _mm_fmadd_pd(x, y, z);
}

/// fms
template <typename Arch>
Vec<float, Arch> fma(const Vec<float, Arch>& x, const Vec<float, Arch>& y,
        const Vec<float, Arch>& z, requires_arch<SSE>) noexcept
{
    return _mm_fmsub_ps(x, y, z);
}

template <typename Arch>
Vec<double, Arch> fma(const Vec<double, Arch>& x, const Vec<double, Arch>& y,
        const Vec<double, Arch>& z, requires_arch<SSE>) noexcept
{
    return _mm_fmsub_pd(x, y, z);
}
}  // namespace kernel
}  // namespace simd
#endif
