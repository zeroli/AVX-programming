#pragma once

namespace simd { namespace kernel { namespace avx512 {
namespace detail {
using namespace types;

template <typename T>
struct add_functor {
    template <typename U = T, REQUIRES(IS_INT_SIZE_1(U))>
    SIMD_INLINE
    avx512_reg_i operator ()(const avx512_reg_i& x, const avx512_reg_i& y) noexcept {
        return _mm512_add_epi8(x, y);
    }
    template <typename U = T, REQUIRES(IS_INT_SIZE_2(U))>
    SIMD_INLINE
    avx512_reg_i operator ()(const avx512_reg_i& x, const avx512_reg_i& y) noexcept {
        return _mm512_add_epi16(x, y);
    }
    template <typename U = T, REQUIRES(IS_INT_SIZE_4(U))>
    SIMD_INLINE
    avx512_reg_i operator ()(const avx512_reg_i& x, const avx512_reg_i& y) noexcept {
        return _mm512_add_epi32(x, y);
    }
    template <typename U = T, REQUIRES(IS_INT_SIZE_8(U))>
    SIMD_INLINE
    avx512_reg_i operator ()(const avx512_reg_i& x, const avx512_reg_i& y) noexcept {
        return _mm512_add_epi64(x, y);
    }

    SIMD_INLINE
    avx512_reg_f operator ()(const avx512_reg_f& x, const avx512_reg_f& y) noexcept {
        return _mm512_add_ps(x, y);
    }
    SIMD_INLINE
    avx512_reg_d operator ()(const avx512_reg_d& x, const avx512_reg_d& y) noexcept {
        return _mm512_add_pd(x, y);
    }
};

template <typename T>
struct sub_functor {
    template <typename U = T, REQUIRES(IS_INT_SIZE_1(U))>
    SIMD_INLINE
    avx512_reg_i operator ()(const avx512_reg_i& x, const avx512_reg_i& y) noexcept {
        return _mm512_sub_epi8(x, y);
    }
    template <typename U = T, REQUIRES(IS_INT_SIZE_2(U))>
    SIMD_INLINE
    avx512_reg_i operator ()(const avx512_reg_i& x, const avx512_reg_i& y) noexcept {
        return _mm512_sub_epi16(x, y);
    }
    template <typename U = T, REQUIRES(IS_INT_SIZE_4(U))>
    SIMD_INLINE
    avx512_reg_i operator ()(const avx512_reg_i& x, const avx512_reg_i& y) noexcept {
        return _mm512_sub_epi32(x, y);
    }
    template <typename U = T, REQUIRES(IS_INT_SIZE_8(U))>
    SIMD_INLINE
    avx512_reg_i operator ()(const avx512_reg_i& x, const avx512_reg_i& y) noexcept {
        return _mm512_sub_epi64(x, y);
    }

    SIMD_INLINE
    avx512_reg_f operator ()(const avx512_reg_f& x, const avx512_reg_f& y) noexcept {
        return _mm512_sub_ps(x, y);
    }
    SIMD_INLINE
    avx512_reg_d operator ()(const avx512_reg_d& x, const avx512_reg_d& y) noexcept {
        return _mm512_sub_pd(x, y);
    }
};

template <typename T>
struct mul_functor {
    avx512_reg_i operator ()(const avx512_reg_i& x, const avx512_reg_i& y) noexcept = delete;

    SIMD_INLINE
    avx512_reg_f operator ()(const avx512_reg_f& x, const avx512_reg_f& y) noexcept {
        return _mm512_mul_ps(x, y);
    }
    SIMD_INLINE
    avx512_reg_d operator ()(const avx512_reg_d& x, const avx512_reg_d& y) noexcept {
        return _mm512_mul_pd(x, y);
    }
};

template <typename T>
struct div_functor {
    avx512_reg_i operator ()(const avx512_reg_i& x, const avx512_reg_i& y) noexcept = delete;

    SIMD_INLINE
    avx512_reg_f operator ()(const avx512_reg_f& x, const avx512_reg_f& y) noexcept {
        return _mm512_div_ps(x, y);
    }
    SIMD_INLINE
    avx512_reg_d operator ()(const avx512_reg_d& x, const avx512_reg_d& y) noexcept {
        return _mm512_div_pd(x, y);
    }
};

template <typename T>
struct mod_functor {
    avx512_reg_i operator ()(const avx512_reg_i& x, const avx512_reg_i& y) noexcept {
        // TODO:
        return x;
    }

    avx512_reg_f operator ()(const avx512_reg_f& x, const avx512_reg_f& y) noexcept = delete;
    avx512_reg_d operator ()(const avx512_reg_d& x, const avx512_reg_d& y) noexcept = delete;
};

template <typename T>
struct neg_functor {
    avx512_reg_i operator ()(const avx512_reg_i& x) noexcept = delete;

    SIMD_INLINE
    avx512_reg_f operator ()(const avx512_reg_f& x) noexcept {
        return _mm512_xor_ps(detail::make_signmask<float>(), x);
    }
    SIMD_INLINE
    avx512_reg_d operator ()(const avx512_reg_d& x) noexcept {
        return _mm512_xor_pd(detail::make_signmask<double>(), x);
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

} } } // namespace simd::kernel::avx512
