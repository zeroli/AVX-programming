#pragma once

namespace simd { namespace kernel { namespace avx {
using namespace types;

namespace detail {
struct sse_add {
    template <typename VO, typename VI>
    SIMD_INLINE
    static VO apply(const VI& x, const VI& y) noexcept {
        return kernel::add(x, y, SSE{});
    }
};

struct sse_sub {
    template <typename VO, typename VI>
    SIMD_INLINE
    static VO apply(const VI& x, const VI& y) noexcept {
        return kernel::sub(x, y, SSE{});
    }
};

struct sse_mul {
    template <typename VO, typename VI>
    SIMD_INLINE
    static VO apply(const VI& x, const VI& y) noexcept {
        return kernel::mul(x, y, SSE{});
    }
};

struct sse_div {
    template <typename VO, typename VI>
    SIMD_INLINE
    static VO apply(const VI& x, const VI& y) noexcept {
        return kernel::div(x, y, SSE{});
    }
};

}  // namespace detail

namespace detail {
template <typename T>
struct add_functor {
    template <typename U = T, REQUIRES(std::is_integral<U>::value)>
    SIMD_INLINE
    avx_reg_i operator ()(const avx_reg_i& x, const avx_reg_i& y) noexcept {
        using sse_vec_t = Vec<T, 128/8/sizeof(T)>;
        return detail::forward_sse_op<detail::sse_add, sse_vec_t>(x, y);
    }

    SIMD_INLINE
    avx_reg_f operator ()(const avx_reg_f& x, const avx_reg_f& y) noexcept {
        return _mm256_add_ps(x, y);
    }
    SIMD_INLINE
    avx_reg_d operator ()(const avx_reg_d& x, const avx_reg_d& y) noexcept {
        return _mm256_add_pd(x, y);
    }
};

template <typename T>
struct sub_functor {
    template <typename U = T, REQUIRES(std::is_integral<U>::value)>
    SIMD_INLINE
    avx_reg_i operator ()(const avx_reg_i& x, const avx_reg_i& y) noexcept {
        using sse_vec_t = Vec<T, 128/8/sizeof(T)>;
        return detail::forward_sse_op<detail::sse_sub, sse_vec_t>(x, y);
    }

    SIMD_INLINE
    avx_reg_f operator ()(const avx_reg_f& x, const avx_reg_f& y) noexcept {
        return _mm256_sub_ps(x, y);
    }
    SIMD_INLINE
    avx_reg_d operator ()(const avx_reg_d& x, const avx_reg_d& y) noexcept {
        return _mm256_sub_pd(x, y);
    }
};

template <typename T>
struct mul_functor {
    avx_reg_i operator ()(const avx_reg_i& x, const avx_reg_i& y) noexcept = delete;

    SIMD_INLINE
    avx_reg_f operator ()(const avx_reg_f& x, const avx_reg_f& y) noexcept {
        return _mm256_mul_ps(x, y);
    }
    SIMD_INLINE
    avx_reg_d operator ()(const avx_reg_d& x, const avx_reg_d& y) noexcept {
        return _mm256_mul_pd(x, y);
    }
};

template <typename T>
struct div_functor {
    avx_reg_i operator ()(const avx_reg_i& x, const avx_reg_i& y) noexcept = delete;

    SIMD_INLINE
    avx_reg_f operator ()(const avx_reg_f& x, const avx_reg_f& y) noexcept {
        return _mm256_div_ps(x, y);
    }
    SIMD_INLINE
    avx_reg_d operator ()(const avx_reg_d& x, const avx_reg_d& y) noexcept {
        return _mm256_div_pd(x, y);
    }
};

template <typename T>
struct mod_functor {
    avx_reg_i operator ()(const avx_reg_i& x, const avx_reg_i& y) noexcept {
        // TODO:
        return x;
    }

    avx_reg_f operator ()(const avx_reg_f& x, const avx_reg_f& y) noexcept = delete;
    avx_reg_d operator ()(const avx_reg_d& x, const avx_reg_d& y) noexcept = delete;
};

template <typename T>
struct neg_functor {
    avx_reg_i operator ()(const avx_reg_i& x) noexcept = delete;

    SIMD_INLINE
    avx_reg_f operator ()(const avx_reg_f& x) noexcept {
        return _mm256_xor_ps(detail::make_signmask<float>(), x);
    }
    SIMD_INLINE
    avx_reg_d operator ()(const avx_reg_d& x) noexcept {
        return _mm256_xor_pd(detail::make_signmask<double>(), x);
    }
};

}  // namespace detail

/// add
template <typename T, size_t W>
struct add<T, W>
    : ops::arith_binary_op<T, W, detail::add_functor<T>>
{};

/// sub
template <typename T, size_t W>
struct sub<T, W>
    : ops::arith_binary_op<T, W, detail::sub_functor<T>>
{};

/// mul
template <typename T, size_t W>
struct mul<T, W>
    : ops::arith_binary_op<T, W, detail::mul_functor<T>>
{};

/// div
template <typename T, size_t W>
struct div<T, W>
    : ops::arith_binary_op<T, W, detail::div_functor<T>>
{};

/// mod for integral only (float/double, deleted)
template <typename T, size_t W>
struct mod<T, W>
    : ops::arith_binary_op<T, W, detail::mod_functor<T>>
{};

template <typename T, size_t W>
struct neg<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& x) noexcept
    {
        return avx::sub<T, W>::apply(Vec<T, W>(0), x);
    }
};

template <typename T, size_t W>
struct neg<T, W, REQUIRE_FLOATING(T)>
    : ops::arith_unary_op<T, W, detail::neg_functor<T>>
{};

} } } // namespace simd::kernel::avx
