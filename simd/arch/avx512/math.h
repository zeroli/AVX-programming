#pragma once

namespace simd { namespace kernel { namespace avx512 {
using namespace types;

namespace detail {
template <typename T>
struct abs_functor {
    template <typename U = T, REQUIRES(IS_INT_SIZE_1(U))>
    avx512_reg_i operator ()(const avx512_reg_i& x) const noexcept {
        return _mm512_abs_epi8(x);
    }
    template <typename U = T, REQUIRES(IS_INT_SIZE_2(U))>
    avx512_reg_i operator ()(const avx512_reg_i& x) const noexcept {
        return _mm512_abs_epi16(x);
    }
    template <typename U = T, REQUIRES(IS_INT_SIZE_4(U))>
    avx512_reg_i operator ()(const avx512_reg_i& x) const noexcept {
        return _mm512_abs_epi32(x);
    }
    template <typename U = T, REQUIRES(IS_INT_SIZE_8(U))>
    avx512_reg_i operator ()(const avx512_reg_i& x) const noexcept {
        return _mm512_abs_epi64(x);
    }

    avx512_reg_f operator ()(const avx512_reg_f& x) const noexcept {
        return _mm512_abs_ps(x);
    }
    avx512_reg_d operator ()(const avx512_reg_d& x) const noexcept {
        return _mm512_abs_pd(x);
    }
};

}  // namespace detail

template <typename T, size_t W>
struct abs<T, W>
    : ops::arith_unary_op<T, W, detail::abs_functor<T>>
{
};

} } } // namespace simd::kernel::avx512
