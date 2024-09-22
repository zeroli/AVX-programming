#pragma once

#include "simd/types/sse_register.h"
#include "simd/types/vec.h"

namespace simd {
namespace kernel {
namespace sse {

using namespace types;

/// abs
template <typename T, size_t W>
struct abs<T, W, REQUIRE_INTEGRAL(T)>
{
    static Vec<T, W> apply(const Vec<T, W>& x) noexcept
    {
        static_check_supported_type<T, 4>();

        Vec<T, W> ret;
        constexpr int nregs = Vec<T, W>::n_regs();
        SIMD_IF_CONSTEXPR(sizeof(T) == 1) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm_abs_epi8(x.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 2) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm_abs_epi16(x.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 4) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm_abs_epi32(x.reg(idx));
            }
        // _mm_abs_epi64 is provided in AVX512F
        // } else SIMD_IF_CONSTEXPR(sizeof(T) == 8) {
        //     #pragma unroll
        //     for (auto idx = 0; idx < nregs; idx++) {
        //         ret.reg(idx) = _mm_abs_epi64(x.reg(idx));
        //     }
        }
        return ret;
    }
};

template <size_t W>
struct abs<float, W>
{
    static Vec<float, W> apply(const Vec<float, W>& x) noexcept
    {
        Vec<float, W> ret;
        auto sign_mask = _mm_set1_ps(-0.f);  // -0.f = 1 << 31
        constexpr int nregs = Vec<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_andnot_ps(sign_mask, x.reg(idx));
        }
        return ret;
    }
};

template <size_t W>
struct abs<double, W>
{
    static Vec<double, W> apply(const Vec<double, W>& x) noexcept
    {
        Vec<double, W> ret;
        auto sign_mask = _mm_set1_pd(-0.f);  // -0.f = 1 << 63
        constexpr int nregs = Vec<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_andnot_pd(sign_mask, x.reg(idx));
        }
        return ret;
    }
};

/// sqrt
template <typename T, size_t W>
struct sqrt<T, W, REQUIRE_INTEGRAL(T)>
{
    /// non-supported sqrt for integral types
    static Vec<T, W> apply(const Vec<T, W>& x) noexcept = delete;
};

template <size_t W>
struct sqrt<float, W>
{
    static Vec<float, W> apply(const Vec<float, W>& x) noexcept
    {
        Vec<float, W> ret;
        constexpr int nregs = Vec<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_sqrt_ps(x.reg(idx));
        }
        return ret;
    }
};

template <size_t W>
struct sqrt<double, W>
{
    static Vec<double, W> apply(const Vec<double, W>& x) noexcept
    {
        Vec<double, W> ret;
        constexpr int nregs = Vec<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_sqrt_pd(x.reg(idx));
        }
        return ret;
    }
};

/// ceil
template <typename T, size_t W>
struct ceil<T, W, REQUIRE_INTEGRAL(T)>
{
    static Vec<T, W> apply(const Vec<T, W>& x) noexcept
    {
        return x;
    }
};

template <size_t W>
struct ceil<float, W>
{
    static Vec<float, W> apply(const Vec<float, W>& x) noexcept
    {
        Vec<float, W> ret;
        constexpr int nregs = Vec<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_ceil_ps(x.reg(idx));
        }
        return ret;
    }
};

template <size_t W>
struct ceil<double, W>
{
    static Vec<double, W> apply(const Vec<double, W>& x) noexcept
    {
        Vec<double, W> ret;
        constexpr int nregs = Vec<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_ceil_pd(x.reg(idx));
        }
        return ret;
    }
};

/// ceil
template <typename T, size_t W>
struct floor<T, W, REQUIRE_INTEGRAL(T)>
{
    static Vec<T, W> apply(const Vec<T, W>& x) noexcept
    {
        return x;
    }
};

template <size_t W>
struct floor<float, W>
{
    static Vec<float, W> apply(const Vec<float, W>& x) noexcept
    {
        Vec<float, W> ret;
        constexpr int nregs = Vec<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_floor_ps(x.reg(idx));
        }
        return ret;
    }
};

template <size_t W>
struct floor<double, W>
{
    static Vec<double, W> apply(const Vec<double, W>& x) noexcept
    {
        Vec<double, W> ret;
        constexpr int nregs = Vec<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_floor_pd(x.reg(idx));
        }
        return ret;
    }
};

#if 0

/// reciprocal
template <typename Arch>
Vec<float, Arch> reciprocal(const Vec<float, Arch>& x, requires_arch<SSE>) noexcept
{
    return _mm_rcp_ps(x);
}

/// rsqrt
template <typename Arch>
Vec<float, Arch> rsqrt(const Vec<float, Arch>& x, requires_arch<SSE>) noexcept
{
    return _mm_rsqrt_ps(x);
}
template <typename Arch>
Vec<double, Arch> rsqrt(const Vec<double, Arch>& x, requires_arch<SSE>) noexcept
{
    return _mm_cvtps_pd(_mm_rsqrt_ps(_mm_cvtpd_ps(x)));
}

#endif

}  // namespace sse
}  // namespace kernel
}  // namespace simd
