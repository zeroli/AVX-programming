#pragma once

#include "simd/types/sse_register.h"
#include "simd/types/vec.h"

#include <cstdint>
#include <cstddef>

namespace simd {
namespace kernel {
namespace sse {
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
    SIMD_INLINE
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

namespace detail {
inline static __m128i bitwise_sra_int8(const __m128i& x, int32_t y)
{
    __m128i sign_mask = _mm_set1_epi16((0xFF00 >> y) & 0x00FF);
    __m128i cmp_is_negative = _mm_cmpgt_epi8(_mm_setzero_si128(), x);
    __m128i res = _mm_srai_epi16(x, y);
    return _mm_or_si128(_mm_and_si128(sign_mask, cmp_is_negative),
                        _mm_andnot_si128(sign_mask, res));
}
inline static __m128i bitwise_srl_int8(const __m128i& x, int32_t y)
{
    return _mm_and_si128(_mm_set1_epi8(0xFF >> y), _mm_srli_epi32(x, y));
}
inline static __m128i bitwise_sra_int64(const __m128i& x, int32_t y)
{
    return _mm_or_si128(
            _mm_srli_epi64(x, y),
            _mm_slli_epi64(
                _mm_srai_epi32(_mm_shuffle_epi32(x, _MM_SHUFFLE(3, 3, 1, 1)), 32),
                64 - y
            )
        );
}
inline static __m128i bitwise_srl_int64(const __m128i& x, int32_t y)
{
    return _mm_srli_epi64(x, y);
}
}  // namespace detail

template <typename T, size_t W>
struct bitwise_lshift<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& x, int32_t y) noexcept
    {
        static_check_supported_type<T>();

        Vec<T, W> ret;
        constexpr int nregs = Vec<T, W>::n_regs();
        SIMD_IF_CONSTEXPR(sizeof(T) == 1) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm_and_si128(_mm_set1_epi8(0xFF << y),
                                             _mm_slli_epi32(x.reg(idx), y));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 2) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm_slli_epi16(x.reg(idx), y);
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 4) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm_slli_epi32(x.reg(idx), y);
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 8) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm_slli_epi64(x.reg(idx), y);
            }
        }
        return ret;
    }
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& x, const Vec<T, W>& y) noexcept
    {
        // TODO:
        return x;
    }
};
template <typename T, size_t W>
struct bitwise_rshift<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& x, int32_t y) noexcept
    {
        static_check_supported_type<T>();

        Vec<T, W> ret;
        constexpr bool is_signed = std::is_signed<T>::value;
        constexpr int nregs = Vec<T, W>::n_regs();
        SIMD_IF_CONSTEXPR(sizeof(T) == 1) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = is_signed
                    ? detail::bitwise_sra_int8(x.reg(idx), y)
                    : detail::bitwise_srl_int8(x.reg(idx), y);
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 2) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = is_signed
                    ? _mm_srai_epi16(x.reg(idx), y)
                    : _mm_srli_epi16(x.reg(idx), y);
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 4) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = is_signed
                    ? _mm_srai_epi32(x.reg(idx), y)
                    : _mm_srli_epi32(x.reg(idx), y);
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 8) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = is_signed
                    ? detail::bitwise_sra_int64(x.reg(idx), y)
                    : detail::bitwise_srl_int64(x.reg(idx), y);
            }
        }
        return ret;
    }
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& x, const Vec<T, W>& y) noexcept
    {
        // TODO
        return x;
    }
};

template <typename T, size_t W>
struct bitwise_not<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& self) noexcept
    {
        static_check_supported_type<T>();

        Vec<T, W> ret;
        constexpr int nregs = Vec<T, W>::n_regs();
        auto mask = _mm_set1_epi32(-1);
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_xor_si128(self.reg(idx), mask);
        }
        return ret;
    }
    SIMD_INLINE
    static VecBool<T, W> apply(const VecBool<T, W>& self) noexcept
    {
        static_check_supported_type<T>();

        VecBool<T, W> ret;
        constexpr int nregs = VecBool<T, W>::n_regs();
        auto mask = _mm_set1_epi32(-1);
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_xor_si128(self.reg(idx), mask);
        }
        return ret;
    }
};

template <size_t W>
struct bitwise_not<float, W>
{
    SIMD_INLINE
    static Vec<float, W> apply(const Vec<float, W>& self) noexcept
    {
        Vec<float, W> ret;
        constexpr int nregs = Vec<float, W>::n_regs();
        auto mask = _mm_castsi128_ps(_mm_set1_epi32(-1));
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_xor_ps(self.reg(idx), mask);
        }
        return ret;
    }
    SIMD_INLINE
    static VecBool<float, W> apply(const VecBool<float, W>& self) noexcept
    {
        VecBool<float, W> ret;
        constexpr int nregs = VecBool<float, W>::n_regs();
        auto mask = _mm_castsi128_ps(_mm_set1_epi32(-1));
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_xor_ps(self.reg(idx), mask);
        }
        return ret;
    }
};

template <size_t W>
struct bitwise_not<double, W>
{
    SIMD_INLINE
    static Vec<double, W> apply(const Vec<double, W>& self) noexcept
    {
        Vec<double, W> ret;
        constexpr int nregs = Vec<double, W>::n_regs();
        auto mask = _mm_castsi128_pd(_mm_set1_epi32(-1));
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_xor_pd(self.reg(idx), mask);
        }
        return ret;
    }
    SIMD_INLINE
    static VecBool<double, W> apply(const VecBool<double, W>& self) noexcept
    {
        VecBool<double, W> ret;
        constexpr int nregs = VecBool<double, W>::n_regs();
        auto mask = _mm_castsi128_pd(_mm_set1_epi32(-1));
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_xor_pd(self.reg(idx), mask);
        }
        return ret;
    }
};
}  // namespace sse
}  // namespace kernel
}  // namespace simd
