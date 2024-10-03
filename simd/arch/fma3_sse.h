#pragma once

namespace simd { namespace kernel { namespace fma3_sse {
#include "simd/arch/kernel_impl.h"
} } } // namespace simd::kernel::sse

#include "simd/types/fma3_sse_register.h"
#include "simd/types/traits.h"

namespace simd { namespace kernel { namespace fma3_sse {
using namespace types;

namespace detail {
SIMD_INLINE
static sse_reg_f fmadd(const sse_reg_f& x, const sse_reg_f& y, const sse_reg_f& z) noexcept
{
    return _mm_fmadd_ps(x, y, z);
}
SIMD_INLINE
static sse_reg_d fmadd(const sse_reg_d& x, const sse_reg_d& y, const sse_reg_d& z) noexcept
{
    return _mm_fmadd_pd(x, y, z);
}

SIMD_INLINE
static sse_reg_f fmsub(const sse_reg_f& x, const sse_reg_f& y, const sse_reg_f& z) noexcept
{
    return _mm_fmsub_ps(x, y, z);
}
SIMD_INLINE
static sse_reg_d fmsub(const sse_reg_d& x, const sse_reg_d& y, const sse_reg_d& z) noexcept
{
    return _mm_fmsub_pd(x, y, z);
}

SIMD_INLINE
static sse_reg_f fnmadd(const sse_reg_f& x, const sse_reg_f& y, const sse_reg_f& z) noexcept
{
    return _mm_fnmadd_ps(x, y, z);
}
SIMD_INLINE
static sse_reg_d fnmadd(const sse_reg_d& x, const sse_reg_d& y, const sse_reg_d& z) noexcept
{
    return _mm_fnmadd_pd(x, y, z);
}

SIMD_INLINE
static sse_reg_f fnmsub(const sse_reg_f& x, const sse_reg_f& y, const sse_reg_f& z) noexcept
{
    return _mm_fnmsub_ps(x, y, z);
}
SIMD_INLINE
static sse_reg_d fnmsub(const sse_reg_d& x, const sse_reg_d& y, const sse_reg_d& z) noexcept
{
    return _mm_fnmsub_pd(x, y, z);
}

SIMD_INLINE
static sse_reg_f fmaddsub(const sse_reg_f& x, const sse_reg_f& y, const sse_reg_f& z) noexcept
{
    return _mm_fmaddsub_ps(x, y, z);
}
SIMD_INLINE
static sse_reg_d fmaddsub(const sse_reg_d& x, const sse_reg_d& y, const sse_reg_d& z) noexcept
{
    return _mm_fmaddsub_pd(x, y, z);
}

SIMD_INLINE
static sse_reg_f fmsubadd(const sse_reg_f& x, const sse_reg_f& y, const sse_reg_f& z) noexcept
{
    return _mm_fmsubadd_ps(x, y, z);
}
SIMD_INLINE
static sse_reg_d fmsubadd(const sse_reg_d& x, const sse_reg_d& y, const sse_reg_d& z) noexcept
{
    return _mm_fmsubadd_pd(x, y, z);
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

} } } // namespace simd::kernel::fma3_sse

namespace simd { namespace kernel {

template <typename T, size_t W>
Vec<T, W> fmadd(const Vec<T, W>& x, const Vec<T, W>& y, const Vec<T, W>& z, requires_arch<FMA3<SSE>>) noexcept
{
    return fma3_sse::fmadd<T, W>::apply(x, y, z);
}

template <typename T, size_t W>
Vec<T, W> fmsub(const Vec<T, W>& x, const Vec<T, W>& y, const Vec<T, W>& z, requires_arch<FMA3<SSE>>) noexcept
{
    return fma3_sse::fmsub<T, W>::apply(x, y, z);
}

template <typename T, size_t W>
Vec<T, W> fnmadd(const Vec<T, W>& x, const Vec<T, W>& y, const Vec<T, W>& z, requires_arch<FMA3<SSE>>) noexcept
{
    return fma3_sse::fnmadd<T, W>::apply(x, y, z);
}

template <typename T, size_t W>
Vec<T, W> fnmsub(const Vec<T, W>& x, const Vec<T, W>& y, const Vec<T, W>& z, requires_arch<FMA3<SSE>>) noexcept
{
    return fma3_sse::fnmsub<T, W>::apply(x, y, z);
}

template <typename T, size_t W>
Vec<T, W> fmaddsub(const Vec<T, W>& x, const Vec<T, W>& y, const Vec<T, W>& z, requires_arch<FMA3<SSE>>) noexcept
{
    return fma3_sse::fmaddsub<T, W>::apply(x, y, z);
}

template <typename T, size_t W>
Vec<T, W> fmsubadd(const Vec<T, W>& x, const Vec<T, W>& y, const Vec<T, W>& z, requires_arch<FMA3<SSE>>) noexcept
{
    return fma3_sse::fmsubadd<T, W>::apply(x, y, z);
}
} }  // namespace simd::kernel
