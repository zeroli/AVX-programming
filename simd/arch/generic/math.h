#pragma once

#include <limits>
#include <cmath>

namespace simd { namespace kernel { namespace generic {
using namespace types;

/// sign
template <typename T, size_t W>
struct sign<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& x) noexcept
    {
        static_check_supported_type<T, 8>();

        using A = typename Vec<T, W>::arch_t;
        using vec_t = Vec<T, W>;
        // +1 for positive, -1 for negative, 0 for zero
        vec_t ret = kernel::select(x > 0, vec_t(1), vec_t(0), A{})
                  - kernel::select(x < 0, vec_t(1), vec_t(0), A{});
        return ret;
    }
};

template <typename T, size_t W>
struct sign<T, W, REQUIRE_FLOATING(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& x) noexcept
    {
        using A = typename Vec<T, W>::arch_t;
        using vec_t = Vec<T, W>;
        // +1 for positive, -1 for negative, 0 for zero
        vec_t ret = kernel::select(x > 0, vec_t(1), vec_t(0), A{})
                  - kernel::select(x < 0, vec_t(1), vec_t(0), A{});
        return ret;
    }
};

/// bitofsign
template <typename T, size_t W>
struct bitofsign<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& x) noexcept
    {
        static_check_supported_type<T, 8>();

        using vec_t = Vec<T, W>;
        if (std::is_unsigned<T>::value) {
            return vec_t(0);
        } else {
            return x & vec_t(bits::signmask<T>());
        }
    }
};

template <typename T, size_t W>
struct bitofsign<T, W, REQUIRE_FLOATING(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& x) noexcept
    {
        using vec_t = Vec<T, W>;
        vec_t ret = x & constants::signmask<vec_t>();
        return ret;
    }
};

/// copysign
template <typename T, size_t W>
struct copysign<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept = delete;
};

template <typename T, size_t W>
struct copysign<T, W, REQUIRE_FLOATING(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        using A = typename Vec<T, W>::arch_t;

        using vec_t = Vec<T, W>;
        vec_t ret = kernel::abs(lhs, A{}) | generic::bitofsign<T, W>::apply(rhs);
        return ret;
    }
};

namespace detail {
template <typename T, size_t W, typename Enable = void>
struct abs_functor;

template <typename T, size_t W>
struct abs_functor<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    Vec<T, W> operator ()(const Vec<T, W>& x) const noexcept {
        Vec<T, W> ret;
        detail::apply(ret, x, [](T a) {
            return std::abs(a);
        });
        return ret;
    }
};
template <typename T, size_t W>
struct abs_functor<T, W, REQUIRE_FLOATING(T)>
{
    SIMD_INLINE
    Vec<T, W> operator ()(const Vec<T, W>& x) const noexcept {
        Vec<T, W> ret;
        detail::apply(ret, x, [](T a) {
            return std::fabs(a);
        });
        return ret;
    }
};

template <typename T, size_t W>
struct sqrt_functor {
    SIMD_INLINE
    Vec<T, W> operator ()(const Vec<T, W>& x) const noexcept {
        Vec<T, W> ret;
        detail::apply(ret, x, [](T a) {
            return std::sqrt(a);
        });
        return ret;
    }
};

template <typename T, size_t W>
struct log_functor {
    SIMD_INLINE
    Vec<T, W> operator ()(const Vec<T, W>& x) const noexcept {
        Vec<T, W> ret;
        detail::apply(ret, x, [](T a) {
            return std::log(a);
        });
        return ret;
    }
};
template <typename T, size_t W, typename F>
struct math_unary_op {
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& x) noexcept
    {
        return F()(x);
    }
};

}  // namespace detail
template <typename T, size_t W>
struct abs<T, W>
    : detail::math_unary_op<T, W, detail::abs_functor<T, W>>
{};

template <typename T, size_t W>
struct sqrt<T, W>
    : detail::math_unary_op<T, W, detail::sqrt_functor<T, W>>
{};

template <typename T, size_t W>
struct log<T, W>
    : detail::math_unary_op<T, W, detail::log_functor<T, W>>
{};

} } } // namespace simd::kernel::generic
