#pragma once

#include "simd/arch/kernel_impl.h"
#include "simd/types/sse_register.h"
#include "simd/types/traits.h"

namespace simd {
namespace kernel {
namespace impl {
using namespace types;

template <size_t W>
struct broadcast<float, W>
{
    static Vec<float, W> apply(float val) noexcept
    {
        Vec<float, W> ret;
        constexpr int nregs = Vec<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_set1_ps(val);
        }
        return ret;
    }
};

template <size_t W>
struct broadcast<double, W>
{
    static Vec<double, W> apply(double val) noexcept
    {
        Vec<double, W> ret;
        constexpr int nregs = Vec<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < W; idx++) {
            ret.reg(idx) = _mm_set1_pd(val);
        }
        return ret;
    }
};

template <typename T, size_t W>
struct broadcast<T, W, REQUIRE_INTEGRAL(T)>
{
    static Vec<T, W> apply(T val) noexcept
    {
        Vec<T, W> ret;
        constexpr int nregs = Vec<T, W>::n_regs();
        SIMD_IF_CONSTEXPR(sizeof(T) == 1) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm_set1_epi8(val);
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 2) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm_set1_epi16(val);
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 4) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm_set1_epi32(val);
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 8) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm_set1_epi32(val);
            }
        } else {
            assert(0 && "unsupported arch/op combination");
        }
        return ret;
    }
};
}  // namespace impl
}  // namespace kernel
}  // namespace simd