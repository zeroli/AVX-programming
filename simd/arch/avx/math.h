#pragma once

namespace simd { namespace kernel { namespace avx {
using namespace types;
/// abs
namespace detail {
struct sse_abs {
    template <typename VO, typename VI>
    SIMD_INLINE
    static VO apply(const VI& x) noexcept {
        return kernel::abs(x, SSE{});
    }
};
}  // namespace detail

template <typename T, size_t W>
struct abs<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& x) noexcept
    {
        static_check_supported_type<T, 8>();

        Vec<T, W> ret;
        constexpr int nregs = Vec<T, W>::n_regs();
        constexpr auto reg_lanes = Vec<T, W>::reg_lanes();
        using sse_vec_t = Vec<T, reg_lanes/2>;
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = detail::forward_sse_op<detail::sse_abs, sse_vec_t>(x.reg(idx));
        }
        return ret;
    }
};

template <size_t W>
struct abs<float, W>
{
    SIMD_INLINE
    static Vec<float, W> apply(const Vec<float, W>& x) noexcept
    {
        Vec<float, W> ret;
        auto sign_mask = detail::make_signmask<float>();
        constexpr int nregs = Vec<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm256_andnot_ps(sign_mask, x.reg(idx));
        }
        return ret;
    }
};

template <size_t W>
struct abs<double, W>
{
    SIMD_INLINE
    static Vec<double, W> apply(const Vec<double, W>& x) noexcept
    {
        Vec<double, W> ret;
        auto sign_mask = detail::make_signmask<double>();
        constexpr int nregs = Vec<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm256_andnot_pd(sign_mask, x.reg(idx));
        }
        return ret;
    }
};

/// sqrt
template <typename T, size_t W>
struct sqrt<T, W, REQUIRE_INTEGRAL(T)>
{
    /// non-supported sqrt for integral types
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& x) noexcept = delete;
};

template <size_t W>
struct sqrt<float, W>
{
    SIMD_INLINE
    static Vec<float, W> apply(const Vec<float, W>& x) noexcept
    {
        Vec<float, W> ret;
        constexpr int nregs = Vec<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm256_sqrt_ps(x.reg(idx));
        }
        return ret;
    }
};

template <size_t W>
struct sqrt<double, W>
{
    SIMD_INLINE
    static Vec<double, W> apply(const Vec<double, W>& x) noexcept
    {
        Vec<double, W> ret;
        constexpr int nregs = Vec<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm256_sqrt_pd(x.reg(idx));
        }
        return ret;
    }
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

template <size_t W>
struct ceil<float, W>
{
    SIMD_INLINE
    static Vec<float, W> apply(const Vec<float, W>& x) noexcept
    {
        Vec<float, W> ret;
        constexpr int nregs = Vec<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm256_ceil_ps(x.reg(idx));
        }
        return ret;
    }
};

template <size_t W>
struct ceil<double, W>
{
    SIMD_INLINE
    static Vec<double, W> apply(const Vec<double, W>& x) noexcept
    {
        Vec<double, W> ret;
        constexpr int nregs = Vec<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm256_ceil_pd(x.reg(idx));
        }
        return ret;
    }
};

/// floor
template <typename T, size_t W>
struct floor<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& x) noexcept
    {
        return x;
    }
};

template <size_t W>
struct floor<float, W>
{
    SIMD_INLINE
    static Vec<float, W> apply(const Vec<float, W>& x) noexcept
    {
        Vec<float, W> ret;
        constexpr int nregs = Vec<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm256_floor_ps(x.reg(idx));
        }
        return ret;
    }
};

template <size_t W>
struct floor<double, W>
{
    SIMD_INLINE
    static Vec<double, W> apply(const Vec<double, W>& x) noexcept
    {
        Vec<double, W> ret;
        constexpr int nregs = Vec<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm256_floor_pd(x.reg(idx));
        }
        return ret;
    }
};
} } } // namespace simd::kernel::avx
