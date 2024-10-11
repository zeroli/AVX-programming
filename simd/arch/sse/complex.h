#pragma once

#include <complex>

namespace simd { namespace kernel { namespace sse {
using namespace types;

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
