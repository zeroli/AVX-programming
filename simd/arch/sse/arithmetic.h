#pragma once

#include "simd/arch/kernel_impl.h"
#include "simd/types/sse_register.h"
#include "simd/types/traits.h"

#include <limits>
#include <type_traits>
#include <cstddef>
#include <cstdint>

namespace simd {
namespace kernel {
namespace impl {
using namespace types;

/// add
template <typename T, size_t W>
struct add<T, W, REQUIRE_INTEGRAL(T)>
{
    static Vec<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
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
        } else {
            assert(0 && "unsupported arch/op combination");
        }
        return ret;
    }
};

template <size_t W>
struct add<float, W>
{
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
    static Vec<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
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
        } else {
            assert(0 && "unsupported arch/op combination");
        }
        return ret;
    }
};

template <size_t W>
struct sub<float, W>
{
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
    static Vec<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
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
        } else {
            assert(0 && "unsupported arch/op combination");
        }
        return ret;
    }
};

template <size_t W>
struct mul<float, W>
{
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
template <size_t W>
struct div<float, W>
{
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

#if 0
template <typename Arch>
Vec<int16_t, Arch> mul(const Vec<int16_t, Arch>& lhs, const Vec<int16_t, Arch>& rhs, requires_arch<SSE>) noexcept
{
    return _mm_mullo_epi16(lhs, rhs);
}

/// neg
template <typename Arch>
Vec<float, Arch> neg(const Vec<float, Arch>& lhs, requires_arch<SSE>) noexcept
{
    return _mm_xor_ps(lhs,
                _mm_castsi128_ps(_mm_set1_epi32(0x80000000))
            );
}

template <typename Arch>
Vec<float, Arch> neg(const Vec<float, Arch>& lhs, requires_arch<SSE>) noexcept
{
    return _mm_xor_pd(lhs,
                _mm_castsi128_pd(
                    _mm_setr_epi32(0, 0x80000000, 0, 0x80000000))
            );
}

template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
Vec<T, Arch> neg(const Vec<T, Arch>& lhs, requires_arch<SSE>) noexcept
{
    return 0 - lhs;
}

/// fnma
template <typename Arch>
Vec<float, Arch> fnma(const Vec<float, Arch>& x, const Vec<float, Arch>& y,
        const Vec<float, Arch>& z, requires_arch<SSE>) noexcept
{
    return _mm_fnmadd_ps(x, y, z);
}

template <typename Arch>
Vec<double, Arch> fnma(const Vec<double, Arch>& x, const Vec<double, Arch>& y,
        const Vec<double, Arch>& z, requires_arch<SSE>) noexcept
{
    return _mm_fnmadd_pd(x, y, z);
}

/// fnms
template <typename Arch>
Vec<float, Arch> fnms(const Vec<float, Arch>& x, const Vec<float, Arch>& y,
        const Vec<float, Arch>& z, requires_arch<SSE>) noexcept
{
    return _mm_fnmsub_ps(x, y, z);
}

template <typename Arch>
Vec<double, Arch> fnms(const Vec<double, Arch>& x, const Vec<double, Arch>& y,
        const Vec<double, Arch>& z, requires_arch<SSE>) noexcept
{
    return _mm_fnmsub_pd(x, y, z);
}

/// fma
template <typename Arch>
Vec<float, Arch> fma(const Vec<float, Arch>& x, const Vec<float, Arch>& y,
        const Vec<float, Arch>& z, requires_arch<SSE>) noexcept
{
    return _mm_fmadd_ps(x, y, z);
}

template <typename Arch>
Vec<double, Arch> fma(const Vec<double, Arch>& x, const Vec<double, Arch>& y,
        const Vec<double, Arch>& z, requires_arch<SSE>) noexcept
{
    return _mm_fmadd_pd(x, y, z);
}

/// fms
template <typename Arch>
Vec<float, Arch> fma(const Vec<float, Arch>& x, const Vec<float, Arch>& y,
        const Vec<float, Arch>& z, requires_arch<SSE>) noexcept
{
    return _mm_fmsub_ps(x, y, z);
}

template <typename Arch>
Vec<double, Arch> fma(const Vec<double, Arch>& x, const Vec<double, Arch>& y,
        const Vec<double, Arch>& z, requires_arch<SSE>) noexcept
{
    return _mm_fmsub_pd(x, y, z);
}
#endif
}  // namespace impl
}  // namespace kernel
}  // namespace simd
