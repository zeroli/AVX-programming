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
}  // namespace impl
}  // namespace kernel
}  // namespace simd
