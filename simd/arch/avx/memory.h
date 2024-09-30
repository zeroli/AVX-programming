#pragma once

#include "simd/types/avx_register.h"
#include "simd/types/traits.h"

#include <limits>
#include <type_traits>
#include <cstddef>
#include <cstdint>

namespace simd {
namespace kernel {
namespace avx {
using namespace types;

/// broadcast
template <typename T, size_t W>
struct broadcast<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(T val) noexcept
    {
        static_check_supported_type<T, 8>();

        Vec<T, W> ret;
        constexpr auto nregs = Vec<T, W>::n_regs();
        SIMD_IF_CONSTEXPR(sizeof(T) == 1) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm256_set1_epi8(val);
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 2) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm256_set1_epi16(val);
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 4) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm256_set1_epi32(val);
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 8) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm256_set1_epi64x(val);
            }
        }
        return ret;
    }
};

template <size_t W>
struct broadcast<float, W>
{
    SIMD_INLINE
    static Vec<float, W> apply(float val) noexcept
    {
        Vec<float, W> ret;
        constexpr auto nregs = Vec<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm256_set1_ps(val);
        }
        return ret;
    }
};

template <size_t W>
struct broadcast<double, W>
{
    SIMD_INLINE
    static Vec<double, W> apply(double val) noexcept
    {
        Vec<double, W> ret;
        constexpr auto nregs = Vec<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm256_set1_pd(val);
        }
        return ret;
    }
};
}  // namespace avx
}  // namespace kernel
}  // namespace simd
