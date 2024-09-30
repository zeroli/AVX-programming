#pragma once

namespace simd { namespace kernel { namespace avx {
using namespace types;

namespace detail {
struct sse_cmp_eq {
    template <typename VO, typename VI>
    SIMD_INLINE
    static VO apply(const VI& x, const VI& y) noexcept {
        return kernel::eq(x, y, SSE{});
    }
};

struct sse_cmp_lt {
    template <typename VO, typename VI>
    SIMD_INLINE
    static VO apply(const VI& x, const VI& y) noexcept {
        return kernel::lt(x, y, SSE{});
    }
};

}  // namespace detail

/// eq
template <typename T, size_t W>
struct eq<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static VecBool<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        static_check_supported_type<T>();

        VecBool<T, W> ret;
        constexpr auto nregs = Vec<T, W>::n_regs();
        constexpr auto reg_lanes = Vec<T, W>::reg_lanes();
        using sse_vec_t = Vec<T, reg_lanes/2>;
        using sse_vecbool_t = VecBool<T, reg_lanes/2>;
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = detail::forward_sse_op<detail::sse_cmp_eq, sse_vecbool_t, sse_vec_t>
                                (lhs.reg(idx), rhs.reg(idx));
        }
        return ret;
    }
    SIMD_INLINE
    static VecBool<T, W> apply(const VecBool<T, W>& lhs, const VecBool<T, W>& rhs) noexcept
    {
        static_check_supported_type<T>();

        VecBool<T, W> ret;
        constexpr auto nregs = VecBool<T, W>::n_regs();
        constexpr auto reg_lanes = VecBool<T, W>::reg_lanes();
        using sse_vecbool_t = VecBool<T, reg_lanes/2>;
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = detail::forward_sse_op<detail::sse_cmp_eq, sse_vecbool_t>
                                (lhs.reg(idx), rhs.reg(idx));
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
            ret.reg(idx) = _mm256_cmp_ps(lhs.reg(idx), rhs.reg(idx), _CMP_EQ_OQ);
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
            ret.reg(idx) = _mm256_castsi256_ps(
                                _mm256_cmpeq_epi64(
                                    _mm256_castps_si256(lhs.reg(idx)),
                                    _mm256_castps_si256(rhs.reg(idx))
                                )
                            );
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
            ret.reg(idx) = _mm256_cmp_pd(lhs.reg(idx), rhs.reg(idx), _CMP_EQ_OQ);
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
            ret.reg(idx) = _mm256_castsi256_pd(
                                _mm256_cmpeq_epi64(
                                    _mm256_castpd_si256(lhs.reg(idx)),
                                    _mm256_castpd_si256(rhs.reg(idx))
                                )
                            );
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
            ret.reg(idx) = _mm256_cmp_ps(lhs.reg(idx), rhs.reg(idx), _CMP_NEQ_OQ);
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
            ret.reg(idx) = _mm256_xor_ps(lhs.reg(idx), rhs.reg(idx));
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
            ret.reg(idx) = _mm256_cmp_pd(lhs.reg(idx), rhs.reg(idx), _CMP_NEQ_OQ);
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
            ret.reg(idx) = _mm256_xor_pd(lhs.reg(idx), rhs.reg(idx));
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
            ret.reg(idx) = _mm256_cmp_ps(lhs.reg(idx), rhs.reg(idx), _CMP_GE_OQ);
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
            ret.reg(idx) = _mm256_cmp_pd(lhs.reg(idx), rhs.reg(idx), _CMP_GE_OQ);
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
            ret.reg(idx) = _mm256_cmp_ps(lhs.reg(idx), rhs.reg(idx), _CMP_LE_OQ);
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
            ret.reg(idx) = _mm256_cmp_pd(lhs.reg(idx), rhs.reg(idx), _CMP_LE_OQ);
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
            ret.reg(idx) = _mm256_cmp_ps(lhs.reg(idx), rhs.reg(idx), _CMP_LT_OQ);
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
            ret.reg(idx) = _mm256_cmp_pd(lhs.reg(idx), rhs.reg(idx), _CMP_LT_OQ);
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
            ret.reg(idx) = _mm256_cmp_ps(lhs.reg(idx), rhs.reg(idx), _CMP_GT_OQ);
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
            ret.reg(idx) = _mm256_cmp_pd(lhs.reg(idx), rhs.reg(idx), _CMP_GT_OQ);
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
        constexpr auto reg_lanes = Vec<T, W>::reg_lanes();
        using sse_vec_t = Vec<T, reg_lanes/2>;
        using sse_vecbool_t = VecBool<T, reg_lanes/2>;
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = detail::forward_sse_op<detail::sse_cmp_lt, sse_vecbool_t, sse_vec_t>
                                (lhs.reg(idx), rhs.reg(idx));
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
        auto ret = ~(avx::lt<T, W>::apply(rhs, lhs));
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
        auto ret = avx::lt<T, W>::apply(rhs, lhs);
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
        auto ret = ~(avx::lt<T, W>::apply(lhs, rhs));
        return ret;
    }
};

} } } // namespace simd::kernel::avx
