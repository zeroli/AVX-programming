#pragma once

namespace simd { namespace kernel { namespace avx512 {
using namespace types;

namespace detail {
template <typename T, int CMP>
struct cmp_op_functor {
    template <typename U = T, REQUIRES(IS_INT_SIZE_1(U))>
    SIMD_INLINE
    avx512_mask_traits_t<T> operator ()(const avx512_reg_i& x, const avx512_reg_i& y) noexcept {
        return _mm512_cmp_epi8_mask(x, y, CMP);
    }
    template <typename U = T, REQUIRES(IS_INT_SIZE_2(U))>
    SIMD_INLINE
    avx512_mask_traits_t<T> operator ()(const avx512_reg_i& x, const avx512_reg_i& y) noexcept {
        return _mm512_cmp_epi16_mask(x, y, CMP);
    }
    template <typename U = T, REQUIRES(IS_INT_SIZE_4(U))>
    SIMD_INLINE
    avx512_mask_traits_t<T> operator ()(const avx512_reg_i& x, const avx512_reg_i& y) noexcept {
        return _mm512_cmp_epi32_mask(x, y, CMP);
    }
    template <typename U = T, REQUIRES(IS_INT_SIZE_8(U))>
    SIMD_INLINE
    avx512_mask_traits_t<T> operator ()(const avx512_reg_i& x, const avx512_reg_i& y) noexcept {
        return _mm512_cmp_epi64_mask(x, y, CMP);
    }

    SIMD_INLINE
    avx512_mask_traits_t<T> operator ()(const avx512_reg_f& x, const avx512_reg_f& y) noexcept {
        return _mm512_cmp_ps_mask(x, y, CMP);
    }
    SIMD_INLINE
    avx512_mask_traits_t<T> operator ()(const avx512_reg_d& x, const avx512_reg_d& y) noexcept {
        return _mm512_cmp_pd_mask(x, y, CMP);
    }
};

}  // namespace detail

/// eq
template <typename T, size_t W>
struct eq<T, W, REQUIRE_INTEGRAL(T)>
    : ops::cmp_binary_op<T, W, detail::cmp_op_functor<T, _MM_CMPINT_EQ>>
{};

template <typename T, size_t W>
struct eq<T, W, REQUIRE_FLOATING(T)>
    : ops::cmp_binary_op<T, W, detail::cmp_op_functor<T, _CMP_EQ_OQ>>
{};

/// ne
template <typename T, size_t W>
struct ne<T, W, REQUIRE_INTEGRAL(T)>
    : ops::cmp_binary_op<T, W, detail::cmp_op_functor<T, _MM_CMPINT_NE>>
{};

template <typename T, size_t W>
struct ne<T, W, REQUIRE_FLOATING(T)>
    : ops::cmp_binary_op<T, W, detail::cmp_op_functor<T, _CMP_NEQ_OQ>>
{};

/// lt
template <typename T, size_t W>
struct lt<T, W, REQUIRE_INTEGRAL(T)>
    : ops::cmp_binary_op<T, W, detail::cmp_op_functor<T, _MM_CMPINT_LT>>
{};

template <typename T, size_t W>
struct lt<T, W, REQUIRE_FLOATING(T)>
    : ops::cmp_binary_op<T, W, detail::cmp_op_functor<T, _CMP_LT_OQ>>
{};

/// le
template <typename T, size_t W>
struct le<T, W, REQUIRE_INTEGRAL(T)>
    : ops::cmp_binary_op<T, W, detail::cmp_op_functor<T, _MM_CMPINT_LE>>
{};

template <typename T, size_t W>
struct le<T, W, REQUIRE_FLOATING(T)>
    : ops::cmp_binary_op<T, W, detail::cmp_op_functor<T, _CMP_LE_OQ>>
{};

/// ge
template <typename T, size_t W>
struct ge<T, W, REQUIRE_INTEGRAL(T)>
    : ops::cmp_binary_op<T, W, detail::cmp_op_functor<T, _MM_CMPINT_GE>>
{};

template <typename T, size_t W>
struct ge<T, W, REQUIRE_FLOATING(T)>
    : ops::cmp_binary_op<T, W, detail::cmp_op_functor<T, _CMP_GE_OQ>>
{};

/// gt
template <typename T, size_t W>
struct gt<T, W, REQUIRE_INTEGRAL(T)>
    : ops::cmp_binary_op<T, W, detail::cmp_op_functor<T, _MM_CMPINT_GT>>
{};

template <typename T, size_t W>
struct gt<T, W, REQUIRE_FLOATING(T)>
    : ops::cmp_binary_op<T, W, detail::cmp_op_functor<T, _CMP_GT_OQ>>
{};

} } } // namespace simd::kernel::avx512
