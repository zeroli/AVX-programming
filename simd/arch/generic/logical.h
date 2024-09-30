#pragma once

namespace simd { namespace kernel { namespace generic {
using namespace types;

template <typename T, size_t W>
struct bitwise_and<T, W>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        Vec<T, W> ret;
        detail::apply(ret, lhs, rhs, [](T x, T y) {
            return bits::bitwise_and(x, y);
        });
        return ret;
    }
};

template <typename T, size_t W>
struct bitwise_or<T, W>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        Vec<T, W> ret;
        detail::apply(ret, lhs, rhs, [](T x, T y) {
            return bits::bitwise_or(x, y);
        });
        return ret;
    }
};

template <typename T, size_t W>
struct bitwise_xor<T, W>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        Vec<T, W> ret;
        detail::apply(ret, lhs, rhs, [](T x, T y) {
            return bits::bitwise_xor(x, y);
        });
        return ret;
    }
};

template <typename T, size_t W>
struct bitwise_lshift<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& lhs, int32_t y) noexcept
    {
        static_check_supported_type<T>();

        Vec<T, W> ret;
        detail::apply(ret, lhs, [y](T x) {
            return bits::bitwise_lshift(x, y);
        });
        return ret;
    }
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        Vec<T, W> ret;
        detail::apply(ret, lhs, rhs, [](T x, T y) {
            return bits::bitwise_lshift(x, y);
        });
        return ret;
    }
};
template <typename T, size_t W>
struct bitwise_rshift<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& lhs, int32_t y) noexcept
    {
        static_check_supported_type<T>();

        Vec<T, W> ret;
        detail::apply(ret, lhs, [y](T x) {
            return bits::bitwise_rshift(x, y);
        });
        return ret;
    }
    SIMD_INLINE
    static Vec<T, W> apply(const VecBool<T, W>& lhs, const VecBool<T, W>& rhs) noexcept
    {
        Vec<T, W> ret;
        detail::apply(ret, lhs, rhs, [](T x, T y) {
            return bits::bitwise_rshift(x, y);
        });
        return ret;
    }
};

/// bitwise_not
template <typename T, size_t W>
struct bitwise_not<T, W>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& self) noexcept
    {
        Vec<T, W> ret;
        detail::apply(ret, self, [](T x) {
            return bits::bitwise_not(x);
        });
        return ret;
    }
    SIMD_INLINE
    static VecBool<T, W> apply(const VecBool<T, W>& self) noexcept
    {
        VecBool<T, W> ret;
        detail::apply(ret, self, [](T x) {
            return bits::bitwise_not(x);
        });
        return ret;
    }
};

/// bitwise_andnot
template <typename T, size_t W>
struct bitwise_andnot<T, W>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        Vec<T, W> ret;
        detail::apply(ret, lhs, rhs, [](T x, T y) {
            return bits::bitwise_andnot(x, y);
        });
        return ret;
    }
    SIMD_INLINE
    static Vec<T, W> apply(const VecBool<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        Vec<T, W> ret;
        detail::apply(ret, lhs, rhs, [](T x, T y) {
            return bits::bitwise_andnot(x, y);
        });
        return ret;
    }
};
} } } // namespace simd::kernel::generic
