#pragma once

#include "simd/types/sse_register.h"
#include "simd/types/vec.h"

#include <cstdint>
#include <cstddef>

namespace simd {
namespace kernel {
namespace impl {

using namespace types;

namespace detail {
struct and_functor {
    __m128i operator ()(const __m128i& x, const __m128i& y) const noexcept {
        return _mm_and_si128(x, y);
    }
    __m128 operator ()(const __m128& x, const __m128& y) const noexcept {
        return _mm_and_ps(x, y);
    }
    __m128d operator ()(const __m128d& x, const __m128d& y) const noexcept {
        return _mm_and_pd(x, y);
    }
};
struct or_functor {
    __m128i operator ()(const __m128i& x, const __m128i& y) const noexcept {
        return _mm_or_si128(x, y);
    }
    __m128 operator ()(const __m128& x, const __m128& y) const noexcept {
        return _mm_or_ps(x, y);
    }
    __m128d operator ()(const __m128d& x, const __m128d& y) const noexcept {
        return _mm_or_pd(x, y);
    }
};
struct xor_functor {
    __m128i operator ()(const __m128i& x, const __m128i& y) const noexcept {
        return _mm_xor_si128(x, y);
    }
    __m128 operator ()(const __m128& x, const __m128& y) const noexcept {
        return _mm_xor_ps(x, y);
    }
    __m128d operator ()(const __m128d& x, const __m128d& y) const noexcept {
        return _mm_xor_pd(x, y);
    }
};

template <typename T, size_t W, typename F>
struct bitwise_op
{
    static Vec<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        static_check_supported_type<T>();

        Vec<T, W> ret;
        constexpr int nregs = Vec<T, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = F()(lhs.reg(idx), rhs.reg(idx));
        }
        return ret;
    }
};
}  // namespace detail

template <typename T, size_t W>
struct bitwise_and<T, W, REQUIRE_INTEGRAL(T)> : detail::bitwise_op<T, W, detail::and_functor>
{
};

template <size_t W>
struct bitwise_and<float, W> : detail::bitwise_op<float, W, detail::and_functor>
{
};

template <size_t W>
struct bitwise_and<double, W> : detail::bitwise_op<double, W, detail::and_functor>
{
};

template <typename T, size_t W>
struct bitwise_or<T, W, REQUIRE_INTEGRAL(T)> : detail::bitwise_op<T, W, detail::or_functor>
{
};

template <size_t W>
struct bitwise_or<float, W> : detail::bitwise_op<float, W, detail::or_functor>
{
};

template <size_t W>
struct bitwise_or<double, W> : detail::bitwise_op<double, W, detail::or_functor>
{
};

template <typename T, size_t W>
struct bitwise_xor<T, W, REQUIRE_INTEGRAL(T)> : detail::bitwise_op<T, W, detail::xor_functor>
{
};

template <size_t W>
struct bitwise_xor<float, W> : detail::bitwise_op<float, W, detail::xor_functor>
{
};

template <size_t W>
struct bitwise_xor<double, W> : detail::bitwise_op<double, W, detail::xor_functor>
{
};


template <typename T, size_t W>
struct bitwise_not<T, W, REQUIRE_INTEGRAL(T)>
{
    static Vec<T, W> apply(const Vec<T, W>& lhs) noexcept
    {
        static_check_supported_type<T>();

        Vec<T, W> ret;
        constexpr int nregs = Vec<T, W>::n_regs();
        auto mask = _mm_set1_epi32(-1);
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_xor_si128(lhs.reg(idx), mask);
        }
        return ret;
    }
};

template <size_t W>
struct bitwise_not<float, W>
{
    static Vec<float, W> apply(const Vec<float, W>& lhs) noexcept
    {
        Vec<float, W> ret;
        constexpr int nregs = Vec<float, W>::n_regs();
        auto mask = _mm_castsi128_ps(_mm_set1_epi32(-1));
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_xor_ps(lhs.reg(idx), mask);
        }
        return ret;
    }
};

template <size_t W>
struct bitwise_not<double, W>
{
    static Vec<double, W> apply(const Vec<double, W>& lhs) noexcept
    {
        Vec<double, W> ret;
        constexpr int nregs = Vec<double, W>::n_regs();
        auto mask = _mm_castsi128_pd(_mm_set1_epi32(-1));
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_xor_pd(lhs.reg(idx), mask);
        }
        return ret;
    }
};

template <typename T, size_t W>
struct neg<T, W, REQUIRE_INTEGRAL(T)>
{
    static Vec<T, W> apply(const Vec<T, W>& self) noexcept
    {
        return kernel::impl::sub<T, W>::apply(Vec<T, W>(0), self);
    }
};

template <size_t W>
struct neg<float, W>
{
    static Vec<float, W> apply(const Vec<float, W>& self) noexcept
    {
        Vec<float, W> ret;
        constexpr int nregs = Vec<float, W>::n_regs();
        auto mask = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_xor_ps(self.reg(idx), mask);
        }
        return ret;
    }
};

template <size_t W>
struct neg<double, W>
{
    static Vec<double, W> apply(const Vec<double, W>& self) noexcept
    {
        Vec<double, W> ret;
        constexpr int nregs = Vec<double, W>::n_regs();
        auto mask = _mm_castsi128_pd(_mm_setr_epi32(0, 0x80000000, 0, 0x80000000));
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_xor_pd(self.reg(idx), mask);
        }
        return ret;
    }
};

#if 0
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
#endif

}  // namespace impl
}  // namespace kernel
}  // namespace simd
