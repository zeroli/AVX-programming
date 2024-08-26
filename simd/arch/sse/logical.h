#pragma once

#include "simd/types/sse_register.h"
#include "simd/types/vec.h"

#include <cstdint>
#include <cstddef>

namespace simd {
namespace kernel {
using namespace types;

/// eq
template <typename Arch>
VecBool<float, Arch> eq(const Vec<float, Arch>& self, const Vec<float, Arch>& other, requires_arch<SSE>) noexcept
{
    return _mm_cmpeq_ps(self, other);
}

template <typename Arch>
VecBool<float, Arch> eq(const VecBool<float, Arch>& self, const VecBool<float, Arch>& other, requires_arch<SSE>) noexcept
{
    return _mm_castsi128_ps(
                _mm_cmpeq_epi32(
                    _mm_castps_si128(self),
                    _mm_castps_si128(other)
                );
            );
}

template <typename Arch>
VecBool<double, Arch> eq(const Vec<double, Arch>& self, const Vec<double, Arch>& other, requires_arch<SSE>) noexcept
{
    return _mm_cmpeq_pd(self, other);
}

template <typename Arch>
VecBool<double, Arch> eq(const VecBool<double, Arch>& self, const VecBool<float, Arch>& other, requires_arch<SSE>) noexcept
{
    return _mm_castsi128_pd(
                _mm_cmpeq_epi32(
                    _mm_castpd_si128(self),
                    _mm_castpd_si128(other)
                );
            );
}

template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
VecBool<T, Arch> eq(const Vec<T, Arch>& self, const Vec<T, Arch>& other, requires_arch<SSE>) noexcept
{
    if (sizeof(T) == 1) {
        return _mm_cmpeq_epi8(self, other);
    } else if (sizeof(T) == 2) {
        return _mm_cmpeq_epi16(self, other);
    } else if (sizeof(T) == 4) {
        return _mm_cmpeq_epi32(self, other);
    } else if (sizeof(T) == 8) {
        return _mm_cmpeq_epi64(self, other);  // sse4.1
    } else {
        assert(0 && "unsupported sizeof(T) > 8");
        return {};
    }
}

template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
VecBool<T, Arch> eq(const VecBool<T, Arch>& self, const VecBool<T, Arch>& other, requires_arch<SSE>) noexcept
{
    return ~(self != other);
}

/// ne
template <typename Arch>
VecBool<float, Arch> ne(const Vec<float, Arch>& self, const Vec<float, Arch>& other, requires_arch<SSE>) noexcept
{
    return _mm_cmpneq_ps(self, other);
}

template <typename Arch>
VecBool<float, Arch> ne(const VecBool<float, Arch>& self, const VecBool<float, Arch>& other, requires_arch<SSE>) noexcept
{
    return _mm_xor_ps(self, other);
}

template <typename Arch>
VecBool<double, Arch> ne(const Vec<double, Arch>& self, const Vec<double, Arch>& other, requires_arch<SSE>) noexcept
{
    return _mm_cmpneq_pd(self, other);
}

template <typename Arch>
VecBool<double, Arch> ne(const VecBool<double, Arch>& self, const VecBool<double, Arch>& other, requires_arch<SSE>) noexcept
{
    return _mm_xor_pd(self, other);
}

template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
VecBool<T, Arch> ne(const Vec<T, Arch>& self, const Vec<T, Arch>& other, requires_arch<SSE>) noexcept
{
    return ~(self == other);
}

template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
VecBool<T, Arch> ne(const VecBool<T, Arch>& self, const VecBool<T, Arch>& other, requires_arch<SSE>) noexcept
{
    return _mm_castps_si128(
        _mm_xor_ps(_mm_castsi128_ps(self.data), _mm_castsi128_ps(other.data))
    );
}

/// ge
template <typename Arch>
VecBool<float, Arch> ge(const Vec<float, Arch>& self, const Vec<float, Arch>& other, requires_arch<SSE>) noexcept
{
    return _mm_cmpge_ps(self, other);
}
template <typename Arch>
VecBool<double, Arch> ge(const Vec<double, Arch>& self, const Vec<float, Arch>& other, requires_arch<SSE>) noexcept
{
    return _mm_cmpge_pd(self, other);
}

/// le
template <typename Arch>
VecBool<float, Arch> le(const Vec<float, Arch>& self, const Vec<float, Arch>& other, requires_arch<SSE>) noexcept
{
    return _mm_cmple_ps(self, other);
}
template <typename Arch>
VecBool<double, Arch> le(const Vec<double, Arch>& self, const Vec<float, Arch>& other, requires_arch<SSE>) noexcept
{
    return _mm_cmple_pd(self, other);
}

/// lt
template <typename Arch>
VecBool<float, Arch> le(const Vec<float, Arch>& self, const Vec<float, Arch>& other, requires_arch<SSE>) noexcept
{
    return _mm_cmplt_ps(self, other);
}
template <typename Arch>
VecBool<double, Arch> le(const Vec<double, Arch>& self, const Vec<float, Arch>& other, requires_arch<SSE>) noexcept
{
    return _mm_cmplt_pd(self, other);
}

template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
VecBool<T, Arch> lt(const Vec<T, Arch>& self, const Vec<T, Arch>& other, requires_arch<SSE>) noexcept
{
    if (std::is_signed<T>::value) {
        if (sizeof(T) == 1) {
            return _mm_cmplt_epi8(self, other);
        } else if (sizeof(T) == 2) {
            return _mm_cmplt_epi16(self, other);
        } else if (sizeof(T) == 4) {
            return _mm_cmplt_epi32(self, other);
        } else if (sizeof(T) == 8) {
            return _mm_cmpgt_epi64(other, self);  // sse4.2
        } else{
            assert(0 && "unsupported less than op for sizeof(T) > 8 in SSE arch");
            return {};
        }
    } else {
        if (sizeof(T) == 1) {
            return _mm_cmplt_epi8(
                        _mm_xor_si128(self, _mm_set1_epi8(std::numeric_limits<int8_t>::lowest())),
                        _mm_xor_si128(other, _mm_set1_epi8(std::numeric_limits<int8_t>::lowest()))
                    );
        } else if (sizeof(T) == 2) {
            return _mm_cmplt_epi16(
                        _mm_xor_si128(self, _mm_set1_epi16(std::numeric_limits<int16_t>::lowest())),
                        _mm_xor_si128(other, _mm_set1_epi16(std::numeric_limits<int16_t>::lowest()))
                    );
        } else if (sizeof(T) == 4) {
            return _mm_cmplt_epi32(
                        _mm_xor_si128(self, _mm_set1_epi32(std::numeric_limits<int32_t>::lowest())),
                        _mm_xor_si128(other, _mm_set1_epi32(std::numeric_limits<int32_t>::lowest()))
                    );
        } else if (sizeof(T) == 8) {
            auto xself = _mm_xor_si128(self, _mm_set1_epi64x(std::numeric_limits<int64_t>::lowest()));
            auto xother = _mm_xor_si128(other, _mm_set1_epi64x(std::numeric_limits<int64_t>::lowest()));
            return _mm_cmpgt_epi64(xother, xself);  // sse4.2
        } else {
            assert(0 && "unsupported less than op for sizeof(T) > 8 in SSE arch");
            return {};
        }
    }
}

/// all
template <typename Arch>
bool all(const VecBool<float, Arch>& self, requires_arch<SSE>) noexcept
{
    return _mm_movemask_ps(self) == 0x0F;
}

template <typename Arch>
bool all(const VecBool<double, Arch>& self, requires_arch<SSE>) noexcept
{
    return _mm_movemask_pd(self) == 0x0F;
}

template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
bool all(const VecBool<T, Arch>& self, requires_arch<SSE>) noexcept
{
    return _mm_movemask_epi8(self) == 0xFFFF;
}

/// any
template <typename Arch>
bool any(const VecBool<float, Arch>& self, requires_arch<SSE>) noexcept
{
    return _mm_movemask_ps(self) != 0;
}

template <typename Arch>
bool all(const VecBool<double, Arch>& self, requires_arch<SSE>) noexcept
{
    return _mm_movemask_pd(self) != 0;
}

template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
bool any(const Vec<T, Arch>& self, requires_arch<SSE>) noexcept
{
    return !_mm_testz_si128(self, self);  // sse4.1
}

}  // namespace kernel
}  // namespace simd
