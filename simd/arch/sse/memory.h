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

/// load_aligned
template <typename T, size_t W>
struct load_aligned<T, W, REQUIRE_INTEGRAL(T)>
{
    static Vec<T, W> apply(const T* mem) noexcept
    {
        static_check_supported_type<T, 8>();

        Vec<T, W> ret;
        constexpr int nregs = Vec<T, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_load_si128((const __m128i*)(mem + idx * W));
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
        constexpr int nregs = Vec<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_load_ps(mem + idx * W);
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
        constexpr int nregs = Vec<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_load_pd(mem + idx * W);
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
        constexpr int nregs = Vec<T, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_loadu_si128((const __m128i*)(mem + idx * W));
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
        constexpr int nregs = Vec<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_loadu_ps(mem + idx * W);
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
        constexpr int nregs = Vec<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_loadu_pd(mem + idx * W);
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

        constexpr int nregs = Vec<T, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            _mm_store_si128((__m128i*)(mem + idx * W), x.reg(idx));
        }
    }
};

template <size_t W>
struct store_aligned<float, W>
{
    static void apply(float* mem, const Vec<float, W>& x) noexcept
    {
        constexpr int nregs = Vec<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            _mm_store_ps(mem + idx * W, x.reg(idx));
        }
    }
};

template <size_t W>
struct store_aligned<double, W>
{
    static void apply(double* mem, const Vec<double, W>& x) noexcept
    {
        constexpr int nregs = Vec<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            _mm_store_pd(mem + idx * W, x.reg(idx));
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

        constexpr int nregs = Vec<T, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            _mm_storeu_si128((__m128i*)(mem + idx * W), x.reg(idx));
        }
    }
};

template <size_t W>
struct store_unaligned<float, W>
{
    static void apply(float* mem, const Vec<float, W>& x) noexcept
    {
        constexpr int nregs = Vec<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            _mm_storeu_ps(mem + idx * W, x.reg(idx));
        }
    }
};

template <size_t W>
struct store_unaligned<double, W>
{
    static void apply(double* mem, const Vec<double, W>& x) noexcept
    {
        constexpr int nregs = Vec<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            _mm_storeu_pd(mem + idx * W, x.reg(idx));
        }
    }
};

}  // namespace sse
}  // namespace kernel
}  // namespace simd
