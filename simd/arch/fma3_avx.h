#pragma once

namespace simd { namespace kernel { namespace fma3_avx {
#include "simd/arch/kernel_impl.h"
} } } // namespace simd::kernel::sse

#include "simd/types/fma3_avx_register.h"
#include "simd/types/traits.h"

namespace simd { namespace kernel { namespace fma3_avx {
using namespace types;

namespace detail {
SIMD_INLINE
static avx_reg_f fmadd(const avx_reg_f& x, const avx_reg_f& y, const avx_reg_f& z) noexcept
{
    return _mm256_fmadd_ps(x, y, z);
}
SIMD_INLINE
static avx_reg_d fmadd(const avx_reg_d& x, const avx_reg_d& y, const avx_reg_d& z) noexcept
{
    return _mm256_fmadd_pd(x, y, z);
}

SIMD_INLINE
static avx_reg_f fmsub(const avx_reg_f& x, const avx_reg_f& y, const avx_reg_f& z) noexcept
{
    return _mm256_fmsub_ps(x, y, z);
}
SIMD_INLINE
static avx_reg_d fmsub(const avx_reg_d& x, const avx_reg_d& y, const avx_reg_d& z) noexcept
{
    return _mm256_fmsub_pd(x, y, z);
}

SIMD_INLINE
static avx_reg_f fnmadd(const avx_reg_f& x, const avx_reg_f& y, const avx_reg_f& z) noexcept
{
    return _mm256_fnmadd_ps(x, y, z);
}
SIMD_INLINE
static avx_reg_d fnmadd(const avx_reg_d& x, const avx_reg_d& y, const avx_reg_d& z) noexcept
{
    return _mm256_fnmadd_pd(x, y, z);
}

SIMD_INLINE
static avx_reg_f fnmsub(const avx_reg_f& x, const avx_reg_f& y, const avx_reg_f& z) noexcept
{
    return _mm256_fnmsub_ps(x, y, z);
}
SIMD_INLINE
static avx_reg_d fnmsub(const avx_reg_d& x, const avx_reg_d& y, const avx_reg_d& z) noexcept
{
    return _mm256_fnmsub_pd(x, y, z);
}

SIMD_INLINE
static avx_reg_f fmaddsub(const avx_reg_f& x, const avx_reg_f& y, const avx_reg_f& z) noexcept
{
    return _mm256_fmaddsub_ps(x, y, z);
}
SIMD_INLINE
static avx_reg_d fmaddsub(const avx_reg_d& x, const avx_reg_d& y, const avx_reg_d& z) noexcept
{
    return _mm256_fmaddsub_pd(x, y, z);
}

SIMD_INLINE
static avx_reg_f fmsubadd(const avx_reg_f& x, const avx_reg_f& y, const avx_reg_f& z) noexcept
{
    return _mm256_fmsubadd_ps(x, y, z);
}
SIMD_INLINE
static avx_reg_d fmsubadd(const avx_reg_d& x, const avx_reg_d& y, const avx_reg_d& z) noexcept
{
    return _mm256_fmsubadd_pd(x, y, z);
}
}  // namespace detail

/// fmadd
template <typename T, size_t W>
struct fmadd<T, W, REQUIRE_FLOATING(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& x, const Vec<T, W>& y, const Vec<T, W>& z) noexcept
    {
        Vec<T, W> ret;
        constexpr auto nregs = Vec<T, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = detail::fmadd(x.reg(idx), y.reg(idx), z.reg(idx));
        }
        return ret;
    }
};

/// fmsub
template <typename T, size_t W>
struct fmsub<T, W, REQUIRE_FLOATING(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& x, const Vec<T, W>& y, const Vec<T, W>& z) noexcept
    {
        Vec<T, W> ret;
        constexpr auto nregs = Vec<T, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = detail::fmsub(x.reg(idx), y.reg(idx), z.reg(idx));
        }
        return ret;
    }
};

/// fnmadd
template <typename T, size_t W>
struct fnmadd<T, W, REQUIRE_FLOATING(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& x, const Vec<T, W>& y, const Vec<T, W>& z) noexcept
    {
        Vec<T, W> ret;
        constexpr auto nregs = Vec<T, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = detail::fnmadd(x.reg(idx), y.reg(idx), z.reg(idx));
        }
        return ret;
    }
};

/// fnmsub
template <typename T, size_t W>
struct fnmsub<T, W, REQUIRE_FLOATING(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& x, const Vec<T, W>& y, const Vec<T, W>& z) noexcept
    {
        Vec<T, W> ret;
        constexpr auto nregs = Vec<T, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = detail::fnmsub(x.reg(idx), y.reg(idx), z.reg(idx));
        }
        return ret;
    }
};

/// fmaddsub
template <typename T, size_t W>
struct fmaddsub<T, W, REQUIRE_FLOATING(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& x, const Vec<T, W>& y, const Vec<T, W>& z) noexcept
    {
        Vec<T, W> ret;
        constexpr auto nregs = Vec<T, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = detail::fmaddsub(x.reg(idx), y.reg(idx), z.reg(idx));
        }
        return ret;
    }
};

/// fmsubadd
template <typename T, size_t W>
struct fmsubadd<T, W, REQUIRE_FLOATING(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& x, const Vec<T, W>& y, const Vec<T, W>& z) noexcept
    {
        Vec<T, W> ret;
        constexpr auto nregs = Vec<T, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = detail::fmsubadd(x.reg(idx), y.reg(idx), z.reg(idx));
        }
        return ret;
    }
};

} } } // namespace simd::kernel::fma3_avx

namespace simd { namespace kernel {

template <typename T, size_t W>
Vec<T, W> fmadd(const Vec<T, W>& x, const Vec<T, W>& y, const Vec<T, W>& z, requires_arch<FMA3_AVX>) noexcept
{
    return fma3_avx::fmadd<T, W>::apply(x, y, z);
}

template <typename T, size_t W>
Vec<T, W> fmsub(const Vec<T, W>& x, const Vec<T, W>& y, const Vec<T, W>& z, requires_arch<FMA3_AVX>) noexcept
{
    return fma3_avx::fmsub<T, W>::apply(x, y, z);
}

template <typename T, size_t W>
Vec<T, W> fnmadd(const Vec<T, W>& x, const Vec<T, W>& y, const Vec<T, W>& z, requires_arch<FMA3_AVX>) noexcept
{
    return fma3_avx::fnmadd<T, W>::apply(x, y, z);
}

template <typename T, size_t W>
Vec<T, W> fnmsub(const Vec<T, W>& x, const Vec<T, W>& y, const Vec<T, W>& z, requires_arch<FMA3_AVX>) noexcept
{
    return fma3_avx::fnmsub<T, W>::apply(x, y, z);
}

template <typename T, size_t W>
Vec<T, W> fmaddsub(const Vec<T, W>& x, const Vec<T, W>& y, const Vec<T, W>& z, requires_arch<FMA3_AVX>) noexcept
{
    return fma3_avx::fmaddsub<T, W>::apply(x, y, z);
}

template <typename T, size_t W>
Vec<T, W> fmsubadd(const Vec<T, W>& x, const Vec<T, W>& y, const Vec<T, W>& z, requires_arch<FMA3_AVX>) noexcept
{
    return fma3_avx::fmsubadd<T, W>::apply(x, y, z);
}
} }  // namespace simd::kernel
