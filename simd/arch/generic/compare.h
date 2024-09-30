#pragma once

#include <cmath>

namespace simd { namespace kernel { namespace generic {
using namespace types;

/// eq
template <typename T, size_t W>
struct eq<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static VecBool<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        static_check_supported_type<T>();

        VecBool<T, W> ret;
        detail::apply(ret, lhs, rhs, [](T x, T y) {
            return bits::extend<T>(x == y);
        });
        return ret;
    }
    SIMD_INLINE
    static VecBool<T, W> apply(const VecBool<T, W>& lhs, const VecBool<T, W>& rhs) noexcept
    {
        static_check_supported_type<T>();

        VecBool<T, W> ret;
        detail::apply(ret, lhs, rhs, [](T x, T y) {
            return bits::extend<T>(x == y);
        });
        return ret;
    }
};

template <typename T, size_t W>
struct eq<T, W, REQUIRE_FLOATING(T)>
{
    SIMD_INLINE
    static VecBool<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        VecBool<T, W> ret;
        detail::apply(ret, lhs, rhs, [](T x, T y) {
            return bits::extend<T>(std::abs(x - y) <= 1e-9);
        });
        return ret;
    }
    SIMD_INLINE
    static VecBool<T, W> apply(const VecBool<T, W>& lhs, const VecBool<T, W>& rhs) noexcept
    {
        VecBool<T, W> ret;
        detail::apply(ret, lhs, rhs, [](T x, T y) {
            return bits::extend<T>(x == y);  // TODO:
        });
        return ret;
    }
};

/// ne
template <typename T, size_t W>
struct ne<T, W>
{
    SIMD_INLINE
    static VecBool<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        return ~(lhs == rhs);
    }
    SIMD_INLINE
    static VecBool<T, W> apply(const VecBool<T, W>& lhs, const VecBool<T, W>& rhs) noexcept
    {
        return ~(lhs == rhs);
    }
};

/// ge
template <typename T, size_t W>
struct ge<T, W, REQUIRE_FLOATING(T)>
{
    SIMD_INLINE
    static VecBool<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        VecBool<T, W> ret;
        detail::apply(ret, lhs, rhs, [](T x, T y) {
            return bits::extend<T>(!(x < y));
        });
        return ret;
    }
};

/// le
template <typename T, size_t W>
struct le<T, W, REQUIRE_FLOATING(T)>
{
    SIMD_INLINE
    static VecBool<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        VecBool<T, W> ret;
        detail::apply(ret, lhs, rhs, [](T x, T y) {
            return bits::extend<T>(!(x > y));
        });
        return ret;
    }
};

/// lt
template <typename T, size_t W>
struct lt<T, W, REQUIRE_FLOATING(T)>
{
    SIMD_INLINE
    static VecBool<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        VecBool<T, W> ret;
        detail::apply(ret, lhs, rhs, [](T x, T y) {
            return bits::extend<T>(x < y);
        });
        return ret;
    }
};

/// gt
template <typename T, size_t W>
struct gt<T, W, REQUIRE_FLOATING(T)>
{
    SIMD_INLINE
    static VecBool<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        VecBool<T, W> ret;
        detail::apply(ret, lhs, rhs, [](T x, T y) {
            return bits::extend<T>(x > y);
        });
        return ret;
    }
};

template <typename T, size_t W>
struct lt<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static VecBool<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        static_check_supported_type<T>();

        VecBool<T, W> ret;
        detail::apply(ret, lhs, rhs, [](T x, T y) {
            return bits::extend<T>(x < y);
        });
        return ret;
    }
};

// a <= b
template <typename T, size_t W>
struct le<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static VecBool<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        VecBool<T, W> ret;
        detail::apply(ret, lhs, rhs, [](T x, T y) {
            return bits::extend<T>(x <= y);
        });
        return ret;
    }
};

template <typename T, size_t W>
struct gt<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static VecBool<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        static_check_supported_type<T>();

        VecBool<T, W> ret;
        detail::apply(ret, lhs, rhs, [](T x, T y) {
            return bits::extend<T>(x > y);
        });
        return ret;
    }
};

// a >= b
template <typename T, size_t W>
struct ge<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static VecBool<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        VecBool<T, W> ret;
        detail::apply(ret, lhs, rhs, [](T x, T y) {
            return bits::extend<T>(x >= y);
        });
        return ret;
    }
};

} } } // namespace simd::kernel::generic
