#pragma once

#include "simd/types/sse_register.h"
#include "simd/types/traits.h"

namespace simd {
namespace kernel {
namespace sse {

using namespace types;

template <typename T, size_t W>
struct broadcast<T, W, REQUIRE_INTEGRAL(T)>
{
    static Vec<T, W> apply(T val) noexcept
    {
        static_check_supported_type<T, 8>();

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
                ret.reg(idx) = _mm_set1_epi64x(val);
            }
        }
        return ret;
    }
};

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
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_set1_pd(val);
        }
        return ret;
    }
};

template <typename T, size_t W>
struct set<T, W, REQUIRE_INTEGRAL(T)>
{
    static Vec<T, W> apply(T v0, T v1) noexcept
    {
        Vec<T, W> ret;
        ret.reg(0) = _mm_set_epi64x(v1, v0);
        return ret;
    }
    static Vec<T, W> apply(T v0, T v1, T v2, T v3) noexcept
    {
        Vec<T, W> ret;
        SIMD_IF_CONSTEXPR(sizeof(T) == 4) {
            ret.reg(0) = _mm_set_epi32(v3, v2, v1, v0);
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 8) {
            ret.reg(0) = _mm_set_epi64x(v1, v0);
            ret.reg(1) = _mm_set_epi64x(v3, v2);
        } else {
            assert(0);
        }
        return ret;
    }
    static Vec<T, W> apply(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7) noexcept
    {
        Vec<T, W> ret;
        SIMD_IF_CONSTEXPR(sizeof(T) == 2) {
            ret.reg(0) = _mm_set_epi16(v7, v6, v5, v4, v3, v2, v1, v0);
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 4) {
            ret.reg(0) = _mm_set_epi32(v3, v2, v1, v0);
            ret.reg(1) = _mm_set_epi32(v7, v6, v5, v4);
        } else {
            assert(0);
        }
        return ret;
    }
    static Vec<T, W> apply(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7, T v8, T v9, T v10, T v11, T v12, T v13, T v14, T v15) noexcept
    {
        Vec<T, W> ret;
        SIMD_IF_CONSTEXPR(sizeof(T) == 1) {
            ret.reg(0) = _mm_set_epi8(v15, v14, v13, v12, v11, v10, v9, v8, v7, v6, v5, v4, v3, v2, v1, v0);
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 2) {
            ret.reg(0) = _mm_set_epi16(v7, v6, v5, v4, v3, v2, v1, v0);
            ret.reg(0) = _mm_set_epi16(v15, v14, v13, v12, v11, v10, v9, v8);
        } else {
            assert(0);
        }
        return ret;
    }
};

template <size_t W>
struct set<float, W>
{
    static Vec<float, W> apply(float v0, float v1, float v2, float v3) noexcept
    {
        return _mm_set_ps(v3, v2, v1, v0);
    }
    static Vec<float, W> apply(float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7) noexcept
    {
        Vec<float, W> ret;
        constexpr int nregs = Vec<float, W>::n_regs();
        ret.reg(0) = _mm_set_ps(v3, v2, v1, v0);
        ret.reg(1) = _mm_set_ps(v7, v6, v5, v4);
        return ret;
    }
    static Vec<float, W> apply(float v0, float v1, float v2,  float v3,  float v4,   float v5,   float v6,   float v7,
                                            float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15) noexcept
    {
        Vec<float, W> ret;
        constexpr int nregs = Vec<float, W>::n_regs();
        ret.reg(0) = _mm_set_ps(v3, v2, v1, v0);
        ret.reg(1) = _mm_set_ps(v7, v6, v5, v4);
        ret.reg(2) = _mm_set_ps(v11, v10, v9, v8);
        ret.reg(3) = _mm_set_ps(v15, v14, v13, v12);
        return ret;
    }
};

template <size_t W>
struct set<double, W>
{
    static Vec<double, W> apply(double v0, double v1) noexcept
    {
        return _mm_set_pd(v1, v0);
    }
    static Vec<double, W> apply(double v0, double v1, double v2, double v3) noexcept
    {
        Vec<double, W> ret;
        constexpr int nregs = Vec<double, W>::n_regs();
        ret.reg(0) = _mm_set_pd(v1, v0);
        ret.reg(1) = _mm_set_pd(v3, v2);
        return ret;
    }
    static Vec<double, W> apply(double v0, double v1, double v2, double v3, float v4, float v5, float v6, float v7) noexcept
    {
        Vec<double, W> ret;
        constexpr int nregs = Vec<double, W>::n_regs();
        ret.reg(0) = _mm_set_pd(v1, v0);
        ret.reg(1) = _mm_set_pd(v3, v2);
        ret.reg(2) = _mm_set_pd(v5, v4);
        ret.reg(3) = _mm_set_pd(v7, v6);
        return ret;
    }
};

/// load_aligned
template <typename T, size_t W>
struct load_aligned<T, W, REQUIRE_INTEGRAL(T)>
{
    static Vec<T, W> apply(const T* mem) noexcept
    {
        static_check_supported_type<T, 8>();

        Vec<T, W> ret;
        constexpr auto nregs = Vec<T, W>::n_regs();
        constexpr auto reg_lanes = Vec<T, W>::reg_lanes();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_load_si128((const __m128i*)(mem + idx * reg_lanes));
        }
        return ret;
    }
};

template <size_t W>
struct load_aligned<float, W>
{
    static Vec<float, W> apply(const float* mem) noexcept
    {
        Vec<float, W> ret;
        constexpr auto nregs = Vec<float, W>::n_regs();
        constexpr auto reg_lanes = Vec<float, W>::reg_lanes();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_load_ps(mem + idx * reg_lanes);
        }
        return ret;
    }
};

template <size_t W>
struct load_aligned<double, W>
{
    static Vec<double, W> apply(const double* mem) noexcept
    {
        Vec<double, W> ret;
        constexpr auto nregs = Vec<double, W>::n_regs();
        constexpr auto reg_lanes = Vec<double, W>::reg_lanes();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_load_pd(mem + idx * reg_lanes);
        }
        return ret;
    }
};

/// load_unaligned
template <typename T, size_t W>
struct load_unaligned<T, W, REQUIRE_INTEGRAL(T)>
{
    static Vec<T, W> apply(const T* mem) noexcept
    {
        static_check_supported_type<T, 8>();

        Vec<T, W> ret;
        constexpr auto nregs = Vec<T, W>::n_regs();
        constexpr auto reg_lanes = Vec<T, W>::reg_lanes();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_loadu_si128((const __m128i*)(mem + idx * reg_lanes));
        }
        return ret;
    }
};

template <size_t W>
struct load_unaligned<float, W>
{
    static Vec<float, W> apply(const float* mem) noexcept
    {
        Vec<float, W> ret;
        constexpr auto nregs = Vec<float, W>::n_regs();
        constexpr auto reg_lanes = Vec<float, W>::reg_lanes();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_loadu_ps(mem + idx * reg_lanes);
        }
        return ret;
    }
};

template <size_t W>
struct load_unaligned<double, W>
{
    static Vec<double, W> apply(const double* mem) noexcept
    {
        Vec<double, W> ret;
        constexpr auto nregs = Vec<double, W>::n_regs();
        constexpr auto reg_lanes = Vec<double, W>::reg_lanes();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_loadu_pd(mem + idx * reg_lanes);
        }
        return ret;
    }
};

/// store_aligned
template <typename T, size_t W>
struct store_aligned<T, W, REQUIRE_INTEGRAL(T)>
{
    static void apply(T* mem, const Vec<T, W>& x) noexcept
    {
        static_check_supported_type<T, 8>();

        constexpr auto nregs = Vec<T, W>::n_regs();
        constexpr auto reg_lanes = Vec<T, W>::reg_lanes();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            _mm_store_si128((__m128i*)(mem + idx * reg_lanes), x.reg(idx));
        }
    }
};

template <size_t W>
struct store_aligned<float, W>
{
    static void apply(float* mem, const Vec<float, W>& x) noexcept
    {
        constexpr auto nregs = Vec<float, W>::n_regs();
        constexpr auto reg_lanes = Vec<float, W>::reg_lanes();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            _mm_store_ps(mem + idx * reg_lanes, x.reg(idx));
        }
    }
};

template <size_t W>
struct store_aligned<double, W>
{
    static void apply(double* mem, const Vec<double, W>& x) noexcept
    {
        constexpr auto nregs = Vec<double, W>::n_regs();
        constexpr auto reg_lanes = Vec<double, W>::reg_lanes();

        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            _mm_store_pd(mem + idx * reg_lanes, x.reg(idx));
        }
    }
};

/// store_unaligned
template <typename T, size_t W>
struct store_unaligned<T, W, REQUIRE_INTEGRAL(T)>
{
    static void apply(T* mem, const Vec<T, W>& x) noexcept
    {
        static_check_supported_type<T, 8>();

        constexpr auto nregs = Vec<T, W>::n_regs();
        constexpr auto reg_lanes = Vec<T, W>::reg_lanes();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            _mm_storeu_si128((__m128i*)(mem + idx * reg_lanes), x.reg(idx));
        }
    }
};

template <size_t W>
struct store_unaligned<float, W>
{
    static void apply(float* mem, const Vec<float, W>& x) noexcept
    {
        constexpr auto nregs = Vec<float, W>::n_regs();
        constexpr auto reg_lanes = Vec<float, W>::reg_lanes();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            _mm_storeu_ps(mem + idx * reg_lanes, x.reg(idx));
        }
    }
};

template <size_t W>
struct store_unaligned<double, W>
{
    static void apply(double* mem, const Vec<double, W>& x) noexcept
    {
        constexpr auto nregs = Vec<double, W>::n_regs();
        constexpr auto reg_lanes = Vec<double, W>::reg_lanes();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            _mm_storeu_pd(mem + idx * reg_lanes, x.reg(idx));
        }
    }
};

}  // namespace sse
}  // namespace kernel
}  // namespace simd
