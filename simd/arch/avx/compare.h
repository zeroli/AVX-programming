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

template <typename T>
struct cmp_eq_functor {
    template <typename U = T, REQUIRES(std::is_integral<U>::value)>
    SIMD_INLINE
    avx_reg_i operator ()(const avx_reg_i& x, const avx_reg_i& y) noexcept {
        using sse_vbool_t = VecBool<T, 128/8/sizeof(T)>;
        using sse_vec_t = Vec<T, 128/8/sizeof(T)>;
        return detail::forward_sse_op<detail::sse_cmp_eq, sse_vbool_t, sse_vec_t>
                                (x, y);
    }

    template <typename U = T, REQUIRES(std::is_integral<U>::value)>
    SIMD_INLINE
    avx_reg_i operator ()(const avx_reg_i& x, const avx_reg_i& y, int) noexcept {
        using sse_vbool_t = VecBool<T, 128/8/sizeof(T)>;
        return detail::forward_sse_op<detail::sse_cmp_eq, sse_vbool_t>
                                (x, y);
    }

    SIMD_INLINE
    avx_reg_f operator ()(const avx_reg_f& x, const avx_reg_f& y) noexcept {
        return _mm256_cmp_ps(x, y, _CMP_EQ_OQ);
    }
    SIMD_INLINE
    avx_reg_d operator ()(const avx_reg_d& x, const avx_reg_d& y) noexcept {
        return _mm256_cmp_pd(x, y, _CMP_EQ_OQ);
    }

    SIMD_INLINE
    avx_reg_f operator ()(const avx_reg_f& x, const avx_reg_f& y, int) noexcept {
        return _mm256_castsi256_ps(
                    _mm256_cmpeq_epi32(
                        _mm256_castps_si256(x),
                        _mm256_castps_si256(y)
                    )
                );
    }
    SIMD_INLINE
    avx_reg_d operator ()(const avx_reg_d& x, const avx_reg_d& y, int) noexcept {
        return _mm256_castsi256_pd(
                    _mm256_cmpeq_epi64(
                        _mm256_castpd_si256(x),
                        _mm256_castpd_si256(y)
                    )
                );
    }
};

template <typename T>
struct cmp_ne_functor {
    SIMD_INLINE
    avx_reg_f operator ()(const avx_reg_f& x, const avx_reg_f& y) noexcept {
        return _mm256_cmp_ps(x, y, _CMP_NEQ_OQ);
    }
    SIMD_INLINE
    avx_reg_d operator ()(const avx_reg_d& x, const avx_reg_d& y) noexcept {
        return _mm256_cmp_pd(x, y, _CMP_NEQ_OQ);
    }

    SIMD_INLINE
    avx_reg_f operator ()(const avx_reg_f& x, const avx_reg_f& y, int) noexcept {
        return _mm256_xor_ps(x, y);
    }
    SIMD_INLINE
    avx_reg_d operator ()(const avx_reg_d& x, const avx_reg_d& y, int) noexcept {
        return _mm256_xor_pd(x, y);
    }
};

template <typename T>
struct cmp_lt_functor {
    SIMD_INLINE
    avx_reg_f operator ()(const avx_reg_f& x, const avx_reg_f& y) noexcept {
        return _mm256_cmp_ps(x, y, _CMP_LT_OQ);
    }
    SIMD_INLINE
    avx_reg_d operator ()(const avx_reg_d& x, const avx_reg_d& y) noexcept {
        return _mm256_cmp_pd(x, y, _CMP_LT_OQ);
    }
};

template <typename T>
struct cmp_le_functor {
    SIMD_INLINE
    avx_reg_f operator ()(const avx_reg_f& x, const avx_reg_f& y) noexcept {
        return _mm256_cmp_ps(x, y, _CMP_LE_OQ);
    }
    SIMD_INLINE
    avx_reg_d operator ()(const avx_reg_d& x, const avx_reg_d& y) noexcept {
        return _mm256_cmp_pd(x, y, _CMP_LE_OQ);
    }
};

template <typename T>
struct cmp_gt_functor {
    SIMD_INLINE
    avx_reg_f operator ()(const avx_reg_f& x, const avx_reg_f& y) noexcept {
        return _mm256_cmp_ps(x, y, _CMP_GT_OQ);
    }
    SIMD_INLINE
    avx_reg_d operator ()(const avx_reg_d& x, const avx_reg_d& y) noexcept {
        return _mm256_cmp_pd(x, y, _CMP_GT_OQ);
    }
};

template <typename T>
struct cmp_ge_functor {
    SIMD_INLINE
    avx_reg_f operator ()(const avx_reg_f& x, const avx_reg_f& y) noexcept {
        return _mm256_cmp_ps(x, y, _CMP_GE_OQ);
    }
    SIMD_INLINE
    avx_reg_d operator ()(const avx_reg_d& x, const avx_reg_d& y) noexcept {
        return _mm256_cmp_pd(x, y, _CMP_GE_OQ);
    }
};
}  // namespace detail

/// eq
template <typename T, size_t W>
struct eq<T, W>
    : ops::cmp_binary_op<T, W, detail::cmp_eq_functor<T>>
{};

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

template <typename T, size_t W>
struct ne<T, W, REQUIRE_FLOATING(T)>
    : ops::cmp_binary_op<T, W, detail::cmp_ne_functor<T>>
{};

/// ge
template <typename T, size_t W>
struct ge<T, W, REQUIRE_FLOATING(T)>
    : ops::cmp_binary_op<T, W, detail::cmp_ge_functor<T>>
{};

/// le
template <typename T, size_t W>
struct le<T, W, REQUIRE_FLOATING(T)>
    : ops::cmp_binary_op<T, W, detail::cmp_le_functor<T>>
{};

/// lt
template <typename T, size_t W>
struct lt<T, W, REQUIRE_FLOATING(T)>
    : ops::cmp_binary_op<T, W, detail::cmp_lt_functor<T>>
{};

/// gt
template <typename T, size_t W>
struct gt<T, W, REQUIRE_FLOATING(T)>
    : ops::cmp_binary_op<T, W, detail::cmp_gt_functor<T>>
{};

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
        using sse_vbool_t = VecBool<T, reg_lanes/2>;
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = detail::forward_sse_op<detail::sse_cmp_lt, sse_vbool_t, sse_vec_t>
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
