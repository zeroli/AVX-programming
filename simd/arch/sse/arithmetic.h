#pragma once

#include "simd/types/sse_register.h"
#include "simd/types/traits.h"

#include <limits>
#include <type_traits>
#include <cstddef>
#include <cstdint>

namespace simd {
namespace kernel {
namespace sse {
using namespace types;

/// add
template <typename T, size_t W>
struct add<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        static_check_supported_type<T>();

        Vec<T, W> ret;
        constexpr int nregs = Vec<T, W>::n_regs();
        SIMD_IF_CONSTEXPR(sizeof(T) == 1) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm_add_epi8(lhs.reg(idx), rhs.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 2) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm_add_epi16(lhs.reg(idx), rhs.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 4) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm_add_epi32(lhs.reg(idx), rhs.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 8) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm_add_epi64(lhs.reg(idx), rhs.reg(idx));
            }
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
        constexpr int nregs = Vec<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_add_ps(lhs.reg(idx), rhs.reg(idx));
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
        constexpr int nregs = Vec<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_add_pd(lhs.reg(idx), rhs.reg(idx));
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
        constexpr int nregs = Vec<T, W>::n_regs();
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
        constexpr int nregs = Vec<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_sub_ps(lhs.reg(idx), rhs.reg(idx));
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
        constexpr int nregs = Vec<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_sub_pd(lhs.reg(idx), rhs.reg(idx));
        }
        return ret;
    }
};

/// mul
template <typename T, size_t W>
struct mul<T, W, REQUIRE_INTEGRAL(T)>
{
    // TODO
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept = delete;
};

template <size_t W>
struct mul<float, W>
{
    SIMD_INLINE
    static Vec<float, W> apply(const Vec<float, W>& lhs, const Vec<float, W>& rhs) noexcept
    {
        Vec<float, W> ret;
        constexpr int nregs = Vec<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_mul_ps(lhs.reg(idx), rhs.reg(idx));
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
        constexpr int nregs = Vec<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_mul_pd(lhs.reg(idx), rhs.reg(idx));
        }
        return ret;
    }
};

/// div
template <typename T, size_t W>
struct div<T, W, REQUIRE_INTEGRAL(T)>
{
    // TODO
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept = delete;
};

template <size_t W>
struct div<float, W>
{
    SIMD_INLINE
    static Vec<float, W> apply(const Vec<float, W>& lhs, const Vec<float, W>& rhs) noexcept
    {
        Vec<float, W> ret;
        constexpr int nregs = Vec<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_div_ps(lhs.reg(idx), rhs.reg(idx));
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
        constexpr int nregs = Vec<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_div_pd(lhs.reg(idx), rhs.reg(idx));
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

template <size_t W>
struct mod<float, W>
{
    SIMD_INLINE
    static Vec<float, W> apply(const Vec<float, W>& lhs, const Vec<float, W>& rhs) noexcept = delete;
};

template <size_t W>
struct mod<double, W>
{
    SIMD_INLINE
    static Vec<double, W> apply(const Vec<double, W>& lhs, const Vec<double, W>& rhs) noexcept = delete;
};

template <typename T, size_t W>
struct neg<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& self) noexcept
    {
        return sse::sub<T, W>::apply(Vec<T, W>(0), self);
    }
};

template <size_t W>
struct neg<float, W>
{
    SIMD_INLINE
    static Vec<float, W> apply(const Vec<float, W>& self) noexcept
    {
        Vec<float, W> ret;
        constexpr int nregs = Vec<float, W>::n_regs();
        auto mask = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_xor_ps(self.reg(idx), mask);
        }
        return ret;
    }
};

template <size_t W>
struct neg<double, W>
{
    SIMD_INLINE
    static Vec<double, W> apply(const Vec<double, W>& self) noexcept
    {
        Vec<double, W> ret;
        constexpr int nregs = Vec<double, W>::n_regs();
        auto mask = _mm_castsi128_pd(_mm_setr_epi32(0, 0x80000000, 0, 0x80000000));
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_xor_pd(self.reg(idx), mask);
        }
        return ret;
    }
};

}  // namespace sse
}  // namespace kernel
}  // namespace simd
