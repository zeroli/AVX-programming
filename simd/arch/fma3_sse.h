#pragma once

namespace simd { namespace kernel { namespace fma3_sse {
#include "simd/arch/kernel_impl.h"
} } } // namespace simd::kernel::sse

#include "simd/types/fma3_sse_register.h"
#include "simd/types/traits.h"

namespace simd { namespace kernel { namespace fma3_sse {
using namespace types;

namespace detail {
struct fmadd_functor {
    SIMD_INLINE
    sse_reg_f operator ()(const sse_reg_f& x, const sse_reg_f& y, const sse_reg_f& z) noexcept {
        return _mm_fmadd_ps(x, y, z);
    }
    SIMD_INLINE
    sse_reg_d operator ()(const sse_reg_d& x, const sse_reg_d& y, const sse_reg_d& z) noexcept {
        return _mm_fmadd_pd(x, y, z);
    }
};

struct fmsub_functor {
    SIMD_INLINE
    sse_reg_f operator ()(const sse_reg_f& x, const sse_reg_f& y, const sse_reg_f& z) noexcept {
        return _mm_fmsub_ps(x, y, z);
    }
    SIMD_INLINE
    sse_reg_d operator ()(const sse_reg_d& x, const sse_reg_d& y, const sse_reg_d& z) noexcept {
        return _mm_fmsub_pd(x, y, z);
    }
};

struct fnmadd_functor {
    SIMD_INLINE
    sse_reg_f operator ()(const sse_reg_f& x, const sse_reg_f& y, const sse_reg_f& z) noexcept {
        return _mm_fnmadd_ps(x, y, z);
    }
    SIMD_INLINE
    sse_reg_d operator ()(const sse_reg_d& x, const sse_reg_d& y, const sse_reg_d& z) noexcept {
        return _mm_fnmadd_pd(x, y, z);
    }
};

struct fnmsub_functor {
    SIMD_INLINE
    sse_reg_f operator ()(const sse_reg_f& x, const sse_reg_f& y, const sse_reg_f& z) noexcept {
        return _mm_fnmsub_ps(x, y, z);
    }
    SIMD_INLINE
    sse_reg_d operator ()(const sse_reg_d& x, const sse_reg_d& y, const sse_reg_d& z) noexcept {
        return _mm_fnmsub_pd(x, y, z);
    }
};

struct fmaddsub_functor {
    SIMD_INLINE
    sse_reg_f operator ()(const sse_reg_f& x, const sse_reg_f& y, const sse_reg_f& z) noexcept {
        return _mm_fmaddsub_ps(x, y, z);
    }
    SIMD_INLINE
    sse_reg_d operator ()(const sse_reg_d& x, const sse_reg_d& y, const sse_reg_d& z) noexcept {
        return _mm_fmaddsub_pd(x, y, z);
    }
};

struct fmsubadd_functor {
    SIMD_INLINE
    sse_reg_f operator ()(const sse_reg_f& x, const sse_reg_f& y, const sse_reg_f& z) noexcept {
        return _mm_fmsubadd_ps(x, y, z);
    }
    SIMD_INLINE
    sse_reg_d operator ()(const sse_reg_d& x, const sse_reg_d& y, const sse_reg_d& z) noexcept {
        return _mm_fmsubadd_pd(x, y, z);
    }
};
}  // namespace detail

/// fmadd
template <typename T, size_t W>
struct fmadd<T, W>
    : ops::arith_ternary_op<T, W, detail::fmadd_functor>
{};

/// fmsub
template <typename T, size_t W>
struct fmsub<T, W>
    : ops::arith_ternary_op<T, W, detail::fmsub_functor>
{};

/// fnmadd
template <typename T, size_t W>
struct fnmadd<T, W>
    : ops::arith_ternary_op<T, W, detail::fnmadd_functor>
{};

/// fnmsub
template <typename T, size_t W>
struct fnmsub<T, W>
    : ops::arith_ternary_op<T, W, detail::fnmsub_functor>
{};

/// fmaddsub
template <typename T, size_t W>
struct fmaddsub<T, W>
    : ops::arith_ternary_op<T, W, detail::fmaddsub_functor>
{};

/// fmsubadd
template <typename T, size_t W>
struct fmsubadd<T, W>
    : ops::arith_ternary_op<T, W, detail::fmsubadd_functor>
{};

} } } // namespace simd::kernel::fma3_sse

namespace simd { namespace kernel {

template <typename T, size_t W>
Vec<T, W> fmadd(const Vec<T, W>& x, const Vec<T, W>& y, const Vec<T, W>& z, requires_arch<FMA3_SSE>) noexcept
{
    return fma3_sse::fmadd<T, W>::apply(x, y, z);
}

template <typename T, size_t W>
Vec<T, W> fmsub(const Vec<T, W>& x, const Vec<T, W>& y, const Vec<T, W>& z, requires_arch<FMA3_SSE>) noexcept
{
    return fma3_sse::fmsub<T, W>::apply(x, y, z);
}

template <typename T, size_t W>
Vec<T, W> fnmadd(const Vec<T, W>& x, const Vec<T, W>& y, const Vec<T, W>& z, requires_arch<FMA3_SSE>) noexcept
{
    return fma3_sse::fnmadd<T, W>::apply(x, y, z);
}

template <typename T, size_t W>
Vec<T, W> fnmsub(const Vec<T, W>& x, const Vec<T, W>& y, const Vec<T, W>& z, requires_arch<FMA3_SSE>) noexcept
{
    return fma3_sse::fnmsub<T, W>::apply(x, y, z);
}

template <typename T, size_t W>
Vec<T, W> fmaddsub(const Vec<T, W>& x, const Vec<T, W>& y, const Vec<T, W>& z, requires_arch<FMA3_SSE>) noexcept
{
    return fma3_sse::fmaddsub<T, W>::apply(x, y, z);
}

template <typename T, size_t W>
Vec<T, W> fmsubadd(const Vec<T, W>& x, const Vec<T, W>& y, const Vec<T, W>& z, requires_arch<FMA3_SSE>) noexcept
{
    return fma3_sse::fmsubadd<T, W>::apply(x, y, z);
}
} }  // namespace simd::kernel
