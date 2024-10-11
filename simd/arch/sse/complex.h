#pragma once

#include <complex>

namespace simd { namespace kernel { namespace sse {
using namespace types;

namespace detail {
template <typename T>
struct cf_mul_functor {
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
struct cf_div_functor {
    SIMD_INLINE
    sse_reg_f operator ()(const sse_reg_f& x, const sse_reg_f& y) noexcept {
        return _mm_div_ps(x, y);
    }
    SIMD_INLINE
    sse_reg_d operator ()(const sse_reg_d& x, const sse_reg_d& y) noexcept {
        return _mm_div_pd(x, y);
    }
};

}  // namespace detail

/// add/sub share same code as float/double, per element slot

/// mul
template <typename T, size_t W>
struct mul<std::complex<T>, W>
    : ops::arith_binary_op<std::complex<T>, W, detail::cf_mul_functor<std::complex<T>>>
{};

/// div
template <typename T, size_t W>
struct div<std::complex<T>, W>
    : ops::arith_binary_op<std::complex<T>, W, detail::cf_div_functor<std::complex<T>>>
{};

template <typename T, size_t W>
struct neg<std::complex<T>, W, REQUIRE_FLOATING(T)>
    : ops::arith_unary_op<std::complex<T>, W, detail::neg_functor<std::complex<T>>>
{};

template <size_t W>
struct broadcast<cf32_t, W>
{
    SIMD_INLINE
    static Vec<cf32_t, W> apply(const cf32_t& val) noexcept
    {
        Vec<cf32_t, W> ret;
        constexpr int nregs = Vec<cf32_t, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_set1_ps(val.real());
        }
        return ret;
    }
};

template <size_t W>
struct broadcast<cf64_t, W>
{
    SIMD_INLINE
    static Vec<cf64_t, W> apply(const cf64_t& val) noexcept
    {
        Vec<cf64_t, W> ret;
        constexpr int nregs = Vec<cf64_t, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_set1_pd(val.real());
        }
        return ret;
    }
};

} } } // namespace simd::kernel::sse
