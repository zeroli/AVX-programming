#pragma once

namespace simd { namespace kernel { namespace sse {
using namespace types;

namespace detail {
template <typename T>
struct add_functor {
    template <typename U = T, REQUIRES(IS_INT_SIZE_1(U))>
    SIMD_INLINE
    sse_reg_i operator ()(const sse_reg_i& x, const sse_reg_i& y) noexcept {
        return _mm_add_epi8(x, y);
    }
    template <typename U = T, REQUIRES(IS_INT_SIZE_2(U))>
    SIMD_INLINE
    sse_reg_i operator ()(const sse_reg_i& x, const sse_reg_i& y) noexcept {
        return _mm_add_epi16(x, y);
    }
    template <typename U = T, REQUIRES(IS_INT_SIZE_4(U))>
    SIMD_INLINE
    sse_reg_i operator ()(const sse_reg_i& x, const sse_reg_i& y) noexcept {
        return _mm_add_epi32(x, y);
    }
    template <typename U = T, REQUIRES(IS_INT_SIZE_8(U))>
    SIMD_INLINE
    sse_reg_i operator ()(const sse_reg_i& x, const sse_reg_i& y) noexcept {
        return _mm_add_epi64(x, y);
    }

    SIMD_INLINE
    sse_reg_f operator ()(const sse_reg_f& x, const sse_reg_f& y) noexcept {
        return _mm_add_ps(x, y);
    }
    SIMD_INLINE
    sse_reg_d operator ()(const sse_reg_d& x, const sse_reg_d& y) noexcept {
        return _mm_add_pd(x, y);
    }
};

template <typename T>
struct sub_functor {
    template <typename U = T, REQUIRES(IS_INT_SIZE_1(U))>
    SIMD_INLINE
    sse_reg_i operator ()(const sse_reg_i& x, const sse_reg_i& y) noexcept {
        return _mm_sub_epi8(x, y);
    }
    template <typename U = T, REQUIRES(IS_INT_SIZE_2(U))>
    SIMD_INLINE
    sse_reg_i operator ()(const sse_reg_i& x, const sse_reg_i& y) noexcept {
        return _mm_sub_epi16(x, y);
    }
    template <typename U = T, REQUIRES(IS_INT_SIZE_4(U))>
    SIMD_INLINE
    sse_reg_i operator ()(const sse_reg_i& x, const sse_reg_i& y) noexcept {
        return _mm_sub_epi32(x, y);
    }
    template <typename U = T, REQUIRES(IS_INT_SIZE_8(U))>
    SIMD_INLINE
    sse_reg_i operator ()(const sse_reg_i& x, const sse_reg_i& y) noexcept {
        return _mm_sub_epi64(x, y);
    }

    SIMD_INLINE
    sse_reg_f operator ()(const sse_reg_f& x, const sse_reg_f& y) noexcept {
        return _mm_sub_ps(x, y);
    }
    SIMD_INLINE
    sse_reg_d operator ()(const sse_reg_d& x, const sse_reg_d& y) noexcept {
        return _mm_sub_pd(x, y);
    }
};

template <typename T>
struct mul_functor {
    sse_reg_i operator ()(const sse_reg_i& x, const sse_reg_i& y) noexcept = delete;

    SIMD_INLINE
    sse_reg_f operator ()(const sse_reg_f& x, const sse_reg_f& y) noexcept {
        return _mm_mul_ps(x, y);
    }
    SIMD_INLINE
    sse_reg_d operator ()(const sse_reg_d& x, const sse_reg_d& y) noexcept {
        return _mm_mul_pd(x, y);
    }
};

template <typename T>
struct div_functor {
    sse_reg_i operator ()(const sse_reg_i& x, const sse_reg_i& y) noexcept = delete;

    SIMD_INLINE
    sse_reg_f operator ()(const sse_reg_f& x, const sse_reg_f& y) noexcept {
        return _mm_div_ps(x, y);
    }
    SIMD_INLINE
    sse_reg_d operator ()(const sse_reg_d& x, const sse_reg_d& y) noexcept {
        return _mm_div_pd(x, y);
    }
};

template <typename T>
struct mod_functor {
    sse_reg_i operator ()(const sse_reg_i& x, const sse_reg_i& y) noexcept {
        // TODO:
        return x;
    }

    sse_reg_f operator ()(const sse_reg_f& x, const sse_reg_f& y) noexcept = delete;
    sse_reg_d operator ()(const sse_reg_d& x, const sse_reg_d& y) noexcept = delete;
};

template <typename T>
struct neg_functor {
    sse_reg_i operator ()(const sse_reg_i& x) noexcept = delete;

    SIMD_INLINE
    sse_reg_f operator ()(const sse_reg_f& x) noexcept {
        return _mm_xor_ps(detail::make_signmask<float>(), x);
    }
    SIMD_INLINE
    sse_reg_d operator ()(const sse_reg_d& x) noexcept {
        return _mm_xor_pd(detail::make_signmask<double>(), x);
    }
};

}  // namespace detail

/// add
template <typename T, size_t W>
struct add<T, W>
    : ops::arith_binary_op<T, W, detail::add_functor<T>>
{};

/// sub
template <typename T, size_t W>
struct sub<T, W>
    : ops::arith_binary_op<T, W, detail::sub_functor<T>>
{};

/// mul
template <typename T, size_t W>
struct mul<T, W>
    : ops::arith_binary_op<T, W, detail::mul_functor<T>>
{};

/// div
template <typename T, size_t W>
struct div<T, W>
    : ops::arith_binary_op<T, W, detail::div_functor<T>>
{};

/// mod for integral only (float/double, deleted)
template <typename T, size_t W>
struct mod<T, W>
    : ops::arith_binary_op<T, W, detail::mod_functor<T>>
{};

template <typename T, size_t W>
struct neg<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& x) noexcept
    {
        return sse::sub<T, W>::apply(Vec<T, W>(0), x);
    }
};

template <typename T, size_t W>
struct neg<T, W, REQUIRE_FLOATING(T)>
    : ops::arith_unary_op<T, W, detail::neg_functor<T>>
{};

} } } // namespace simd::kernel::sse
