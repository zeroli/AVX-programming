#pragma once

namespace simd { namespace kernel { namespace sse {

using namespace types;

namespace detail {
template <typename T>
struct abs_functor {
    template <typename U = T, REQUIRES(IS_INT_SIZE_1(U))>
    SIMD_INLINE
    sse_reg_i operator ()(const sse_reg_i& x) const noexcept {
        return _mm_abs_epi8(x);
    }
    template <typename U = T, REQUIRES(IS_INT_SIZE_2(U))>
    SIMD_INLINE
    sse_reg_i operator ()(const sse_reg_i& x) const noexcept {
        return _mm_abs_epi16(x);
    }
    template <typename U = T, REQUIRES(IS_INT_SIZE_4(U))>
    SIMD_INLINE
    sse_reg_i operator ()(const sse_reg_i& x) const noexcept {
        return _mm_abs_epi32(x);
    }
    template <typename U = T, REQUIRES(IS_INT_SIZE_8(U))>
    SIMD_INLINE
    sse_reg_i operator ()(const sse_reg_i& x) const noexcept {
        /// _mm_abs_epi64 is availabel since avx512 (AVX512VL)
        return _mm_castpd_si128(
                _mm_andnot_pd(
                    detail::make_signmask<double>(),
                    _mm_castsi128_pd(x)
                ));
    }

    SIMD_INLINE
    sse_reg_f operator ()(const sse_reg_f& x) const noexcept {
        return _mm_andnot_ps(detail::make_signmask<float>(), x);
    }
    SIMD_INLINE
    sse_reg_d operator ()(const sse_reg_d& x) const noexcept {
        return _mm_andnot_pd(detail::make_signmask<double>(), x);
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

}  // namespace detail

/// abs
template <typename T, size_t W>
struct abs<T, W>
    : ops::arith_unary_op<T, W, detail::abs_functor<T>>
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
    : ops::arith_unary_op<T, W, detail::sqrt_functor>
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
    : ops::arith_unary_op<T, W, detail::ceil_functor>
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
    : ops::arith_unary_op<T, W, detail::floor_functor>
{
};

} } } // namespace simd::kernel::sse
