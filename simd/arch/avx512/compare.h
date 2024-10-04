#pragma once

namespace simd { namespace kernel { namespace avx512 {
using namespace types;

/// eq
namespace detail {
template <typename T, int CMP>
avx512_mask_traits_t<T> cmp_epi_mask(const avx512_reg_i& x, const avx512_reg_i& y) noexcept;

template <typename T, int CMP, REQUIRES(IS_INT_SIZE_1(T))>
avx512_mask_traits_t<T> cmp_epi_mask(const avx512_reg_i& x, const avx512_reg_i& y) noexcept
{
    return _mm512_cmp_epi8_mask(x, y, CMP);
}

template <typename T, int CMP, REQUIRES(IS_INT_SIZE_2(T))>
avx512_mask_traits_t<T> cmp_epi_mask(const avx512_reg_i& x, const avx512_reg_i& y) noexcept
{
    return _mm512_cmp_epi16_mask(x, y, CMP);
}

template <typename T, int CMP, REQUIRES(IS_INT_SIZE_4(T))>
avx512_mask_traits_t<T> cmp_epi_mask(const avx512_reg_i& x, const avx512_reg_i& y) noexcept
{
    return _mm512_cmp_epi32_mask(x, y, CMP);
}

template <typename T, int CMP, REQUIRES(IS_INT_SIZE_8(T))>
avx512_mask_traits_t<T> cmp_epi_mask(const avx512_reg_i& x, const avx512_reg_i& y) noexcept
{
    return _mm512_cmp_epi64_mask(x, y, CMP);
}
}  // namespace detail

template <typename T, size_t W>
struct eq<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static VecBool<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        static_check_supported_type<T>();

        VecBool<T, W> ret;
        constexpr auto nregs = Vec<T, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = detail::cmp_epi_mask<T>(lhs.reg(idx), rhs.reg(idx));
        }
        return ret;
    }
    SIMD_INLINE
    static VecBool<T, W> apply(const VecBool<T, W>& lhs, const VecBool<T, W>& rhs) noexcept
    {
        static_check_supported_type<T>();

        VecBool<T, W> ret;
        constexpr auto nregs = VecBool<T, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            // TODO:
        }
        return ret;
    }
};

template <size_t W>
struct eq<float, W>
{
    SIMD_INLINE
    static VecBool<float, W> apply(const Vec<float, W>& lhs, const Vec<float, W>& rhs) noexcept
    {
        VecBool<float, W> ret;
        constexpr auto nregs = VecBool<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm512_cmp_ps_mask(lhs.reg(idx), rhs.reg(idx), _CMP_EQ_OQ);
        }
        return ret;
    }
    SIMD_INLINE
    static VecBool<float, W> apply(const VecBool<float, W>& lhs, const VecBool<float, W>& rhs) noexcept
    {
        VecBool<float, W> ret;
        constexpr auto nregs = VecBool<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            // TODO:
        }
    }
};

template <size_t W>
struct eq<double, W>
{
    SIMD_INLINE
    static VecBool<double, W> apply(const Vec<double, W>& lhs, const Vec<double, W>& rhs) noexcept
    {
        VecBool<double, W> ret;
        constexpr auto nregs = VecBool<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm512_cmp_pd_mask(lhs.reg(idx), rhs.reg(idx), _CMP_EQ_OQ);
        }
        return ret;
    }
    SIMD_INLINE
    static VecBool<double, W> apply(const VecBool<double, W>& lhs, const VecBool<double, W>& rhs) noexcept
    {
        VecBool<double, W> ret;
        constexpr auto nregs = VecBool<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            // TODO:
        }
    }
};

/// ne
template <typename T, size_t W>
struct ne<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static VecBool<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        return ~(lhs == rhs);
    }
    SIMD_INLINE
    static VecBool<T, W> apply(const VecBool<T, W>& lhs, const VecBool<T, W>& rhs) noexcept
    {
        return ~(lhs == rhs);
    }
};

template <size_t W>
struct ne<float, W>
{
    SIMD_INLINE
    static VecBool<float, W> apply(const Vec<float, W>& lhs, const Vec<float, W>& rhs) noexcept
    {
        VecBool<float, W> ret;
        constexpr auto nregs = VecBool<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm512_cmp_ps_mask(lhs.reg(idx), rhs.reg(idx), _CMP_NEQ_OQ);
        }
        return ret;
    }
    SIMD_INLINE
    static VecBool<float, W> apply(const VecBool<float, W>& lhs, const VecBool<float, W>& rhs) noexcept
    {
        VecBool<float, W> ret;
        constexpr auto nregs = VecBool<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            // TODO:
        }
    }
};

template <size_t W>
struct ne<double, W>
{
    SIMD_INLINE
    static VecBool<double, W> apply(const Vec<double, W>& lhs, const Vec<double, W>& rhs) noexcept
    {
        VecBool<double, W> ret;
        constexpr auto nregs = VecBool<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm512_cmp_pd_mask(lhs.reg(idx), rhs.reg(idx), _CMP_NEQ_OQ);
        }
        return ret;
    }
    SIMD_INLINE
    static VecBool<double, W> apply(const VecBool<double, W>& lhs, const VecBool<double, W>& rhs) noexcept
    {
        VecBool<double, W> ret;
        constexpr auto nregs = VecBool<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            // TODO:
        }
    }
};

/// ge
template <size_t W>
struct ge<float, W>
{
    SIMD_INLINE
    static VecBool<float, W> apply(const Vec<float, W>& lhs, const Vec<float, W>& rhs) noexcept
    {
        VecBool<float, W> ret;
        constexpr auto nregs = VecBool<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm512_cmp_ps_mask(lhs.reg(idx), rhs.reg(idx), _CMP_GE_OQ);
        }
        return ret;
    }
};

template <size_t W>
struct ge<double, W>
{
    SIMD_INLINE
    static VecBool<double, W> apply(const Vec<double, W>& lhs, const Vec<double, W>& rhs) noexcept
    {
        VecBool<double, W> ret;
        constexpr auto nregs = VecBool<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm512_cmp_pd_mask(lhs.reg(idx), rhs.reg(idx), _CMP_GE_OQ);
        }
        return ret;
    }
};

/// le
template <size_t W>
struct le<float, W>
{
    SIMD_INLINE
    static VecBool<float, W> apply(const Vec<float, W>& lhs, const Vec<float, W>& rhs) noexcept
    {
        VecBool<float, W> ret;
        constexpr auto nregs = VecBool<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm512_cmp_ps_mask(lhs.reg(idx), rhs.reg(idx), _CMP_LE_OQ);
        }
        return ret;
    }
};

template <size_t W>
struct le<double, W>
{
    SIMD_INLINE
    static VecBool<double, W> apply(const Vec<double, W>& lhs, const Vec<double, W>& rhs) noexcept
    {
        VecBool<double, W> ret;
        constexpr auto nregs = VecBool<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm512_cmp_pd_mask(lhs.reg(idx), rhs.reg(idx), _CMP_LE_OQ);
        }
        return ret;
    }
};

/// lt
template <size_t W>
struct lt<float, W>
{
    SIMD_INLINE
    static VecBool<float, W> apply(const Vec<float, W>& lhs, const Vec<float, W>& rhs) noexcept
    {
        VecBool<float, W> ret;
        constexpr auto nregs = VecBool<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm512_cmp_ps_mask(lhs.reg(idx), rhs.reg(idx), _CMP_LT_OQ);
        }
        return ret;
    }
};

template <size_t W>
struct lt<double, W>
{
    SIMD_INLINE
    static VecBool<double, W> apply(const Vec<double, W>& lhs, const Vec<double, W>& rhs) noexcept
    {
        VecBool<double, W> ret;
        constexpr auto nregs = VecBool<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm512_cmp_pd_mask(lhs.reg(idx), rhs.reg(idx), _CMP_LT_OQ);
        }
        return ret;
    }
};

/// gt
template <size_t W>
struct gt<float, W>
{
    SIMD_INLINE
    static VecBool<float, W> apply(const Vec<float, W>& lhs, const Vec<float, W>& rhs) noexcept
    {
        VecBool<float, W> ret;
        constexpr auto nregs = VecBool<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm512_cmp_ps_mask(lhs.reg(idx), rhs.reg(idx), _CMP_GT_OQ);
        }
        return ret;
    }
};

template <size_t W>
struct gt<double, W>
{
    SIMD_INLINE
    static VecBool<double, W> apply(const Vec<double, W>& lhs, const Vec<double, W>& rhs) noexcept
    {
        VecBool<double, W> ret;
        constexpr auto nregs = VecBool<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm512_cmp_pd_mask(lhs.reg(idx), rhs.reg(idx), _CMP_GT_OQ);
        }
        return ret;
    }
};

/// lt for integral
/// a < b
template <typename T, size_t W>
struct lt<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static VecBool<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        static_check_supported_type<T>();

        VecBool<T, W> ret;
        constexpr auto nregs = Vec<T, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = detail::cmp_epi_mask<T, _MM_CMPINT_LT>(lhs.reg(idx), rhs.reg(idx));
        }
        return ret;
    }
};

/// le for integral
/// a <= b => ~(b < a)
template <typename T, size_t W>
struct le<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static VecBool<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        auto ret = ~(avx512::lt<T, W>::apply(rhs, lhs));
        return ret;
    }
};

/// gt for integral
/// a > b => b < a
template <typename T, size_t W>
struct gt<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static VecBool<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        auto ret = avx512::lt<T, W>::apply(rhs, lhs);
        return ret;
    }
};

/// ge for integral
/// a >= b => ~(a < b)
template <typename T, size_t W>
struct ge<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static VecBool<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        auto ret = ~(avx512::lt<T, W>::apply(lhs, rhs));
        return ret;
    }
};

} } } // namespace simd::kernel::avx512
