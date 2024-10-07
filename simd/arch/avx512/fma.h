#pragma once

namespace simd { namespace kernel { namespace avx512 {
namespace detail {
using namespace types;

struct fmadd_functor {
    SIMD_INLINE
    avx512_reg_f operator ()(const avx512_reg_f& x, const avx512_reg_f& y, const avx512_reg_f& z) noexcept {
        return _mm512_fmadd_ps(x, y, z);
    }
    SIMD_INLINE
    avx512_reg_d operator ()(const avx512_reg_d& x, const avx512_reg_d& y, const avx512_reg_d& z) noexcept {
        return _mm512_fmadd_pd(x, y, z);
    }
};

struct fmsub_functor {
    SIMD_INLINE
    avx512_reg_f operator ()(const avx512_reg_f& x, const avx512_reg_f& y, const avx512_reg_f& z) noexcept {
        return _mm512_fmsub_ps(x, y, z);
    }
    SIMD_INLINE
    avx512_reg_d operator ()(const avx512_reg_d& x, const avx512_reg_d& y, const avx512_reg_d& z) noexcept {
        return _mm512_fmsub_pd(x, y, z);
    }
};

struct fnmadd_functor {
    SIMD_INLINE
    avx512_reg_f operator ()(const avx512_reg_f& x, const avx512_reg_f& y, const avx512_reg_f& z) noexcept {
        return _mm512_fnmadd_ps(x, y, z);
    }
    SIMD_INLINE
    avx512_reg_d operator ()(const avx512_reg_d& x, const avx512_reg_d& y, const avx512_reg_d& z) noexcept {
        return _mm512_fnmadd_pd(x, y, z);
    }
};

struct fnmsub_functor {
    SIMD_INLINE
    avx512_reg_f operator ()(const avx512_reg_f& x, const avx512_reg_f& y, const avx512_reg_f& z) noexcept {
        return _mm512_fnmsub_ps(x, y, z);
    }
    SIMD_INLINE
    avx512_reg_d operator ()(const avx512_reg_d& x, const avx512_reg_d& y, const avx512_reg_d& z) noexcept {
        return _mm512_fnmsub_pd(x, y, z);
    }
};

struct fmaddsub_functor {
    SIMD_INLINE
    avx512_reg_f operator ()(const avx512_reg_f& x, const avx512_reg_f& y, const avx512_reg_f& z) noexcept {
        return _mm512_fmaddsub_ps(x, y, z);
    }
    SIMD_INLINE
    avx512_reg_d operator ()(const avx512_reg_d& x, const avx512_reg_d& y, const avx512_reg_d& z) noexcept {
        return _mm512_fmaddsub_pd(x, y, z);
    }
};

struct fmsubadd_functor {
    SIMD_INLINE
    avx512_reg_f operator ()(const avx512_reg_f& x, const avx512_reg_f& y, const avx512_reg_f& z) noexcept {
        return _mm512_fmsubadd_ps(x, y, z);
    }
    SIMD_INLINE
    avx512_reg_d operator ()(const avx512_reg_d& x, const avx512_reg_d& y, const avx512_reg_d& z) noexcept {
        return _mm512_fmsubadd_pd(x, y, z);
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

} } } // namespace simd::kernel::avx512
