#pragma once

namespace simd { namespace kernel { namespace avx {
using namespace types;

/// add
namespace detail {
template <typename T, size_t W>
avx_reg_i forward_sse_add(const avx_reg_i& lhs, const avx_reg_i& rhs) noexcept
{
    using A = SSE;
    using V = Vec<T, W>;
    static_assert(V::n_regs() == 1);

    sse_reg_i l_low, l_high, r_low, r_high;
    detail::split(lhs, l_low, l_high);
    detail::split(rhs, r_low, r_high);
    auto sum_low = kernel::add(V(l_low), V(r_low), A{});
    auto sum_high = kernel::add(V(l_low), V(r_low), A{});
    return detail::merge(sum_low.reg(), sum_high.reg());
}
}  // namespace detail
template <typename T, size_t W>
struct add<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        static_check_supported_type<T>();

        Vec<T, W> ret;
        constexpr auto nregs = Vec<T, W>::n_regs();
        constexpr auto reg_lanes = Vec<T, W>::reg_lanes();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = detail::forward_sse_add<T, reg_lanes/2>(lhs.reg(idx), rhs.reg(idx));
        }
        return ret;
    }
};

template <size_t W>
struct add<float, W>
{
    SIMD_INLINE
    static Vec<float, W> apply(const Vec<float, W>& lhs, const Vec<float, W>& rhs) noexcept
    {
        Vec<float, W> ret;
        constexpr auto nregs = Vec<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm256_add_ps(lhs.reg(idx), rhs.reg(idx));
        }
        return ret;
    }
};

template <size_t W>
struct add<double, W>
{
    SIMD_INLINE
    static Vec<double, W> apply(const Vec<double, W>& lhs, const Vec<double, W>& rhs) noexcept
    {
        Vec<double, W> ret;
        constexpr auto nregs = Vec<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm256_add_pd(lhs.reg(idx), rhs.reg(idx));
        }
        return ret;
    }
};

/// sub
template <typename T, size_t W>
struct sub<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        static_check_supported_type<T>();

        Vec<T, W> ret;
        constexpr auto nregs = Vec<T, W>::n_regs();
        SIMD_IF_CONSTEXPR(sizeof(T) == 1) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm_sub_epi8(lhs.reg(idx), rhs.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 2) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm_sub_epi16(lhs.reg(idx), rhs.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 4) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm_sub_epi32(lhs.reg(idx), rhs.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 8) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm_sub_epi64(lhs.reg(idx), rhs.reg(idx));
            }
        }
        return ret;
    }
};

template <size_t W>
struct sub<float, W>
{
    SIMD_INLINE
    static Vec<float, W> apply(const Vec<float, W>& lhs, const Vec<float, W>& rhs) noexcept
    {
        Vec<float, W> ret;
        constexpr auto nregs = Vec<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm256_sub_ps(lhs.reg(idx), rhs.reg(idx));
        }
        return ret;
    }
};

template <size_t W>
struct sub<double, W>
{
    SIMD_INLINE
    static Vec<double, W> apply(const Vec<double, W>& lhs, const Vec<double, W>& rhs) noexcept
    {
        Vec<double, W> ret;
        constexpr auto nregs = Vec<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm256_sub_pd(lhs.reg(idx), rhs.reg(idx));
        }
        return ret;
    }
};

/// mul
template <size_t W>
struct mul<float, W>
{
    SIMD_INLINE
    static Vec<float, W> apply(const Vec<float, W>& lhs, const Vec<float, W>& rhs) noexcept
    {
        Vec<float, W> ret;
        constexpr auto nregs = Vec<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm256_mul_ps(lhs.reg(idx), rhs.reg(idx));
        }
        return ret;
    }
};

template <size_t W>
struct mul<double, W>
{
    SIMD_INLINE
    static Vec<double, W> apply(const Vec<double, W>& lhs, const Vec<double, W>& rhs) noexcept
    {
        Vec<double, W> ret;
        constexpr auto nregs = Vec<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm256_mul_pd(lhs.reg(idx), rhs.reg(idx));
        }
        return ret;
    }
};

/// div
template <size_t W>
struct div<float, W>
{
    SIMD_INLINE
    static Vec<float, W> apply(const Vec<float, W>& lhs, const Vec<float, W>& rhs) noexcept
    {
        Vec<float, W> ret;
        constexpr auto nregs = Vec<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm256_div_ps(lhs.reg(idx), rhs.reg(idx));
        }
        return ret;
    }
};

template <size_t W>
struct div<double, W>
{
    SIMD_INLINE
    static Vec<double, W> apply(const Vec<double, W>& lhs, const Vec<double, W>& rhs) noexcept
    {
        Vec<double, W> ret;
        constexpr auto nregs = Vec<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm256_div_pd(lhs.reg(idx), rhs.reg(idx));
        }
        return ret;
    }
};

/// mod for integral only (float/double, deleted)
template <typename T, size_t W>
struct mod<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept {
        return {};  // TODO
    }
};

template <typename T, size_t W>
struct neg<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& x) noexcept
    {
        return avx::sub<T, W>::apply(Vec<T, W>(0), x);
    }
};

template <size_t W>
struct neg<float, W>
{
    SIMD_INLINE
    static Vec<float, W> apply(const Vec<float, W>& x) noexcept
    {
        Vec<float, W> ret;
        constexpr auto nregs = Vec<float, W>::n_regs();
        auto mask = _mm256_castsi256_ps(_mm256_set1_epi32(bits::one_zeros<int32_t>()));
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm256_xor_ps(x.reg(idx), mask);
        }
        return ret;
    }
};

template <size_t W>
struct neg<double, W>
{
    SIMD_INLINE
    static Vec<double, W> apply(const Vec<double, W>& x) noexcept
    {
        Vec<double, W> ret;
        constexpr auto nregs = Vec<double, W>::n_regs();
        auto mask = _mm256_castsi256_pd(_mm256_set1_epi64x(bits::one_zeros<int64_t>()));
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm256_xor_pd(x.reg(idx), mask);
        }
        return ret;
    }
};

} } } // namespace simd::kernel::avx
