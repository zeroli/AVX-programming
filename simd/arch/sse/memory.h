#pragma once

#include "simd/types/sse_register.h"
#include "simd/types/traits.h"

namespace simd {
namespace kernel {
using namespace types;

template <size_t W>
Vec<float, W> broadcast(float val, requires_arch<SSE>) noexcept
{
    Vec<float, W> ret;
    constexpr int nregs = Vec<float, W>::n_regs();
    #pragma unroll
    for (auto idx = 0; idx < nregs; idx++) {
        ret.reg(idx) = _mm_set1_ps(val);
    }
    return ret;
}
template <size_t W>
Vec<double, W> broadcast(double val, requires_arch<SSE>) noexcept
{
    Vec<double, W> ret;
    constexpr int nregs = Vec<double, W>::n_regs();
    #pragma unroll
    for (auto idx = 0; idx < W; idx++) {
        ret.reg(idx) = _mm_set1_pd(val);
    }
    return ret;
}

template <size_t W, typename T, REQUIRE_INTEGRAL(T)>
Vec<T, W> broadcast(T val, requires_arch<SSE>) noexcept
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
}  // namespace kernel
}  // namespace simd
