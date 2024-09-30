#pragma once



namespace simd { namespace kernel { namespace generic {
using namespace types;

/// add
template <typename T, size_t W>
struct add<T, W>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        static_check_supported_type<T>();

        Vec<T, W> ret = detail::apply(lhs, rhs, [](T x, T y) {
            return x + y;
        });
        return ret;
    }
};

/// sub
template <typename T, size_t W>
struct sub<T, W>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        static_check_supported_type<T>();

        Vec<T, W> ret = detail::apply(lhs, rhs, [](T x, T y) {
            return x - y;
        });
        return ret;
    }
};

/// mul
template <typename T, size_t W>
struct mul<T, W>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        Vec<T, W> ret = detail::apply(lhs, rhs, [](T x, T y) {
            return x * y;
        });
        return ret;
    }
};

/// div
template <typename T, size_t W>
struct div<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        Vec<T, W> ret = detail::apply(lhs, rhs, [](T x, T y) {
            return x / y;
        });
        return ret;
    }
};

/// mod for integral only (float/double, deleted)
template <typename T, size_t W>
struct mod<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        Vec<T, W> ret = detail::apply(lhs, rhs, [](T x, T y) {
            return x % y;
        });
        return ret;
    }
};

template <typename T, size_t W>
struct mod<T, W, REQUIRE_FLOATING(T)>
{
    SIMD_INLINE
    static Vec<float, W> apply(const Vec<float, W>& lhs, const Vec<float, W>& rhs) noexcept = delete;
};

template <typename T, size_t W>
struct neg<T, W>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& x) noexcept
    {
        using A = typename Vec<T, W>::arch_t;
        return kernel::sub<T, W>(Vec<T, W>(0), x, A{});
    }
};

} } } // namespace simd::kernel::generic
