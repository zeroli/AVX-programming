#pragma once

namespace simd { namespace kernel { namespace sse {

using namespace types;

namespace detail {
template <typename T, typename Enable = void>
struct abs_functor {
    SIMD_INLINE
    sse_reg_f operator ()(const sse_reg_f& x) const noexcept {
        return _mm_andnot_ps(detail::make_signmask<float>(), x);
    }
    SIMD_INLINE
    sse_reg_d operator ()(const sse_reg_d& x) const noexcept {
        return _mm_andnot_pd(detail::make_signmask<double>(), x);
    }
};

template <typename T>
struct abs_functor<T, REQUIRE_INTEGRAL_SIZE_1(T)> {
    SIMD_INLINE
    sse_reg_i operator ()(const sse_reg_i& x) const noexcept {
        return _mm_abs_epi8(x);
    }
};
template <typename T>
struct abs_functor<T, REQUIRE_INTEGRAL_SIZE_2(T)> {
    SIMD_INLINE
    sse_reg_i operator ()(const sse_reg_i& x) const noexcept {
        return _mm_abs_epi16(x);
    }
};

template <typename T>
struct abs_functor<T, REQUIRE_INTEGRAL_SIZE_4(T)> {
    SIMD_INLINE
    sse_reg_i operator ()(const sse_reg_i& x) const noexcept {
        return _mm_abs_epi32(x);
    }
};
template <typename T>
struct abs_functor<T, REQUIRE_INTEGRAL_SIZE_8(T)> {
    SIMD_INLINE
    sse_reg_i operator ()(const sse_reg_i& x) const noexcept {
        /// _mm_abs_epi64 is availabel since avx512 (AVX512VL)
        return _mm_castpd_si128(
                _mm_andnot_pd(
                    detail::make_signmask<double>(),
                    _mm_castsi128_pd(x)
                ));
    }
};

struct sqrt_functor {
    SIMD_INLINE
    sse_reg_f operator ()(const sse_reg_f& x) noexcept {
        return _mm_sqrt_ps(x);
    }
    SIMD_INLINE
    sse_reg_d operator ()(const sse_reg_d& x) noexcept {
        return _mm_sqrt_pd(x);
    }
};

struct ceil_functor {
    SIMD_INLINE
    sse_reg_f operator ()(const sse_reg_f& x) noexcept {
        return _mm_ceil_ps(x);
    }
    SIMD_INLINE
    sse_reg_d operator ()(const sse_reg_d& x) noexcept {
        return _mm_ceil_pd(x);
    }
};

struct floor_functor {
    SIMD_INLINE
    sse_reg_f operator ()(const sse_reg_f& x) noexcept {
        return _mm_floor_ps(x);
    }
    SIMD_INLINE
    sse_reg_d operator ()(const sse_reg_d& x) noexcept {
        return _mm_floor_pd(x);
    }
};

template <typename T, size_t W, typename F>
struct math_unary_op {
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& x) noexcept
    {
        Vec<T, W> ret;
        constexpr auto nregs = Vec<T, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = F()(x.reg(idx));
        }
        return ret;
    }
};
}  // namespace detail

/// abs
template <typename T, size_t W>
struct abs<T, W>
    : detail::math_unary_op<T, W, detail::abs_functor<T>>
{
};

/// sqrt
template <typename T, size_t W>
struct sqrt<T, W, REQUIRE_INTEGRAL(T)>
{
    /// non-supported sqrt for integral types
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& x) noexcept = delete;
};

template <typename T, size_t W>
struct sqrt<T, W, REQUIRE_FLOATING(T)>
    : detail::math_unary_op<T, W, detail::sqrt_functor>
{
};

/// ceil
template <typename T, size_t W>
struct ceil<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& x) noexcept
    {
        return x;
    }
};

template <typename T, size_t W>
struct ceil<T, W, REQUIRE_FLOATING(T)>
    : detail::math_unary_op<T, W, detail::ceil_functor>
{
};

/// ceil
template <typename T, size_t W>
struct floor<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& x) noexcept
    {
        return x;
    }
};

template <typename T, size_t W>
struct floor<T, W, REQUIRE_FLOATING(T)>
    : detail::math_unary_op<T, W, detail::floor_functor>
{
};

} } } // namespace simd::kernel::sse
