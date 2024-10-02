#pragma once

namespace simd { namespace kernel { namespace avx {
using namespace types;

namespace detail {
struct sse_bitwise_and {
    template <typename VO, typename VI>
    static VO apply(const VI& x, const VI& y) noexcept {
        return kernel::bitwise_and(x, y, SSE{});
    }
};
struct sse_bitwise_or {
    template <typename VO, typename VI>
    static VO apply(const VI& x, const VI& y) noexcept {
        return kernel::bitwise_or(x, y, SSE{});
    }
};

struct sse_bitwise_xor {
    template <typename VO, typename VI>
    static VO apply(const VI& x, const VI& y) noexcept {
        return kernel::bitwise_xor(x, y, SSE{});
    }
};

struct and_functor {
    avx_reg_i operator ()(const avx_reg_i& x, const avx_reg_i& y) const noexcept {
        using sse_vec_t = Vec<int32_t, 4>;
        return detail::forward_sse_op<detail::sse_bitwise_and, sse_vec_t>(x, y);
    }
    avx_reg_f operator ()(const avx_reg_f& x, const avx_reg_f& y) const noexcept {
        return _mm256_and_ps(x, y);
    }
    avx_reg_d operator ()(const avx_reg_d& x, const avx_reg_d& y) const noexcept {
        return _mm256_and_pd(x, y);
    }
};
struct or_functor {
    avx_reg_i operator ()(const avx_reg_i& x, const avx_reg_i& y) const noexcept {
        using sse_vec_t = Vec<int32_t, 4>;
        return detail::forward_sse_op<detail::sse_bitwise_or, sse_vec_t>(x, y);
    }
    avx_reg_f operator ()(const avx_reg_f& x, const avx_reg_f& y) const noexcept {
        return _mm256_or_ps(x, y);
    }
    avx_reg_d operator ()(const avx_reg_d& x, const avx_reg_d& y) const noexcept {
        return _mm256_or_pd(x, y);
    }
};
struct xor_functor {
    avx_reg_i operator ()(const avx_reg_i& x, const avx_reg_i& y) const noexcept {
        using sse_vec_t = Vec<int32_t, 4>;
        return detail::forward_sse_op<detail::sse_bitwise_xor, sse_vec_t>(x, y);
    }
    avx_reg_f operator ()(const avx_reg_f& x, const avx_reg_f& y) const noexcept {
        return _mm256_xor_ps(x, y);
    }
    avx_reg_d operator ()(const avx_reg_d& x, const avx_reg_d& y) const noexcept {
        return _mm256_xor_pd(x, y);
    }
};

template <typename T, size_t W, typename F>
struct bitwise_op
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        Vec<T, W> ret;
        constexpr auto nregs = Vec<T, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = F()(lhs.reg(idx), rhs.reg(idx));
        }
        return ret;
    }
};
}  // namespace detail

template <typename T, size_t W>
struct bitwise_and<T, W> : detail::bitwise_op<T, W, detail::and_functor>
{
};

template <typename T, size_t W>
struct bitwise_or<T, W> : detail::bitwise_op<T, W, detail::or_functor>
{
};

template <typename T, size_t W>
struct bitwise_xor<T, W> : detail::bitwise_op<T, W, detail::xor_functor>
{
};

namespace detail {
struct sse_bitwise_lshift {
    template <typename VO, typename VI>
    static VO apply(const VI& lhs, int32_t rhs) noexcept {
        return kernel::bitwise_lshift(lhs, rhs, SSE{});
    }
    template <typename VO, typename VI>
    static VO apply(const VI& lhs, const VI& rhs) noexcept {
        return kernel::bitwise_lshift(lhs, rhs, SSE{});
    }
};

struct sse_bitwise_rshift {
    template <typename VO, typename VI>
    static VO apply(const VI& lhs, int32_t rhs) noexcept {
        return kernel::bitwise_rshift(lhs, rhs, SSE{});
    }
    template <typename VO, typename VI>
    static VO apply(const VI& lhs, const VI& rhs) noexcept {
        return kernel::bitwise_rshift(lhs, rhs, SSE{});
    }
};
}  // namespace detail

template <typename T, size_t W>
struct bitwise_lshift<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& lhs, int32_t rhs) noexcept
    {
        static_check_supported_type<T>();

        Vec<T, W> ret;
        constexpr auto nregs = Vec<T, W>::n_regs();
        constexpr auto reg_lanes = Vec<T, W>::reg_lanes();
        using sse_vec_t = Vec<T, reg_lanes/2>;
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = detail::forward_sse_op<detail::sse_bitwise_lshift, sse_vec_t>
                                (lhs.reg(idx), rhs);
        }
        return ret;
    }
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        static_check_supported_type<T>();

        Vec<T, W> ret;
        constexpr auto nregs = Vec<T, W>::n_regs();
        constexpr auto reg_lanes = Vec<T, W>::reg_lanes();
        using sse_vec_t = Vec<T, reg_lanes/2>;
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = detail::forward_sse_op<detail::sse_bitwise_lshift, sse_vec_t>
                                (lhs.reg(idx), rhs.reg(idx));
        }
        return ret;
    }
};
template <typename T, size_t W>
struct bitwise_rshift<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& lhs, int32_t rhs) noexcept
    {
        static_check_supported_type<T>();

        Vec<T, W> ret;
        constexpr auto nregs = Vec<T, W>::n_regs();
        constexpr auto reg_lanes = Vec<T, W>::reg_lanes();
        using sse_vec_t = Vec<T, reg_lanes/2>;
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = detail::forward_sse_op<detail::sse_bitwise_rshift, sse_vec_t>
                                (lhs.reg(idx), rhs);
        }
        return ret;
    }
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        static_check_supported_type<T>();

        Vec<T, W> ret;
        constexpr auto nregs = Vec<T, W>::n_regs();
        constexpr auto reg_lanes = Vec<T, W>::reg_lanes();
        using sse_vec_t = Vec<T, reg_lanes/2>;
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = detail::forward_sse_op<detail::sse_bitwise_rshift, sse_vec_t>
                                (lhs.reg(idx), rhs.reg(idx));
        }
        return ret;
    }
};

/// bitwise_not
namespace detail {
struct sse_bitwise_not {
    template <typename VO, typename VI>
    SIMD_INLINE
    static VO apply(const VI& x) noexcept {
        return kernel::bitwise_not(x, SSE{});
    }
};
}  // namespace detail
template <typename T, size_t W>
struct bitwise_not<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& x) noexcept
    {
        static_check_supported_type<T>();

        Vec<T, W> ret;
        constexpr auto nregs = Vec<T, W>::n_regs();
        constexpr auto reg_lanes = Vec<T, W>::reg_lanes();
        using sse_vec_t = Vec<T, reg_lanes/2>;
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = detail::forward_sse_op<detail::sse_bitwise_not, sse_vec_t>
                                (x.reg(idx));
        }
        return ret;
    }
    SIMD_INLINE
    static VecBool<T, W> apply(const VecBool<T, W>& x) noexcept
    {
        static_check_supported_type<T>();

        VecBool<T, W> ret;
        constexpr auto nregs = VecBool<T, W>::n_regs();
        constexpr auto reg_lanes = Vec<T, W>::reg_lanes();
        using sse_vbool_t = VecBool<T, reg_lanes/2>;
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = detail::forward_sse_op<detail::sse_bitwise_not, sse_vbool_t>
                                (x.reg(idx));
        }
        return ret;
    }
};

template <size_t W>
struct bitwise_not<float, W>
{
    SIMD_INLINE
    static Vec<float, W> apply(const Vec<float, W>& x) noexcept
    {
        Vec<float, W> ret;
        constexpr auto nregs = Vec<float, W>::n_regs();
        auto mask = detail::make_mask<float>();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm256_xor_ps(x.reg(idx), mask);
        }
        return ret;
    }
    SIMD_INLINE
    static VecBool<float, W> apply(const VecBool<float, W>& x) noexcept
    {
        VecBool<float, W> ret;
        constexpr auto nregs = VecBool<float, W>::n_regs();
        auto mask = detail::make_mask<float>();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm256_xor_ps(x.reg(idx), mask);
        }
        return ret;
    }
};

template <size_t W>
struct bitwise_not<double, W>
{
    SIMD_INLINE
    static Vec<double, W> apply(const Vec<double, W>& x) noexcept
    {
        Vec<double, W> ret;
        constexpr auto nregs = Vec<double, W>::n_regs();
        auto mask = detail::make_mask<double>();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm256_xor_pd(x.reg(idx), mask);
        }
        return ret;
    }
    SIMD_INLINE
    static VecBool<double, W> apply(const VecBool<double, W>& x) noexcept
    {
        VecBool<double, W> ret;
        constexpr auto nregs = VecBool<double, W>::n_regs();
        auto mask = detail::make_mask<double>();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm256_xor_pd(x.reg(idx), mask);
        }
        return ret;
    }
};

/// bitwise_andnot
namespace detail {
struct sse_bitwise_andnot {
    template <typename VO, typename VI1, typename VI2>
    static VO apply(const VI1& lhs, const VI2& rhs) noexcept {
        return kernel::bitwise_andnot(lhs, rhs, SSE{});
    }
};
}  // namespace detail
template <typename T, size_t W>
struct bitwise_andnot<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const VecBool<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        static_check_supported_type<T>();

        Vec<T, W> ret;
        constexpr auto nregs = Vec<T, W>::n_regs();
        constexpr auto reg_lanes = Vec<T, W>::reg_lanes();
        using sse_vec_t = Vec<T, reg_lanes/2>;
        using sse_vbool_t = VecBool<T, reg_lanes/2>;
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = detail::forward_sse_op2<detail::sse_bitwise_andnot, sse_vec_t, sse_vbool_t, sse_vec_t>
                                (lhs.reg(idx), rhs.reg(idx));
        }
        return ret;
    }
};

template <size_t W>
struct bitwise_andnot<float, W>
{
    SIMD_INLINE
    static Vec<float, W> apply(const VecBool<float, W>& lhs, const Vec<float, W>& rhs) noexcept
    {
        Vec<float, W> ret;
        constexpr auto nregs = Vec<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm256_andnot_ps(lhs.reg(idx), rhs.reg(idx));
        }
        return ret;
    }
};

template <size_t W>
struct bitwise_andnot<double, W>
{
    SIMD_INLINE
    static Vec<double, W> apply(const VecBool<double, W>& lhs, const Vec<double, W>& rhs) noexcept
    {
        Vec<double, W> ret;
        constexpr auto nregs = Vec<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm256_andnot_pd(lhs.reg(idx), rhs.reg(idx));
        }
        return ret;
    }
};

} } } // namespace simd::kernel::avx
