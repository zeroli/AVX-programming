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

        Vec<T, W> ret;
        detail::apply(ret, lhs, rhs, [](T x, T y) {
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

        Vec<T, W> ret;
        detail::apply(ret, lhs, rhs, [](T x, T y) {
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
        Vec<T, W> ret;
        detail::apply(ret, lhs, rhs, [](T x, T y) {
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
        Vec<T, W> ret;
        detail::apply(ret, lhs, rhs, [](T x, T y) {
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
        Vec<T, W> ret;
        detail::apply(ret, lhs, rhs, [](T x, T y) {
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

template <typename T, size_t W>
struct fmadd<T, W, REQUIRE_FLOATING(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& x, const Vec<T, W>& y, const Vec<T, W>& z) noexcept
    {
        return x * y + z;
    }
};

template <typename T, size_t W>
struct fmsub<T, W, REQUIRE_FLOATING(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& x, const Vec<T, W>& y, const Vec<T, W>& z) noexcept
    {
        return x * y - z;
    }
};

template <typename T, size_t W>
struct fnmadd<T, W, REQUIRE_FLOATING(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& x, const Vec<T, W>& y, const Vec<T, W>& z) noexcept
    {
        return -(x * y) + z;
    }
};

template <typename T, size_t W>
struct fnmsub<T, W, REQUIRE_FLOATING(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& x, const Vec<T, W>& y, const Vec<T, W>& z) noexcept
    {
        return -(x * y) - z;
    }
};

namespace detail {
/// template parameter `P`
/// true: +1 first in even position 0, and then -1, ...
/// false: -1 first in even position 0, and then +1, ...
template <typename T, size_t W, bool P>
struct mask_interleaved_lut;

template <typename T, bool P>
struct mask_interleaved_lut<T, 2, P> {
    static const T* get() {
        alignas(sizeof(T)*2)
        static T mask_lut_p[] = { +1, -1 };

        alignas(sizeof(T)*2)
        static T mask_lut_n[] = { -1, +1 };
        return P ? mask_lut_p : mask_lut_n;
    }
};

template <typename T, bool P>
struct mask_interleaved_lut<T, 4, P> {
    static const T* get() {
        alignas(sizeof(T)*4)
        static T mask_lut_p[] = { +1, -1, +1, -1 };

        alignas(sizeof(T)*4)
        static T mask_lut_n[] = { -1, +1, -1, +1 };
        return P ? mask_lut_p : mask_lut_n;
    }
};

template <typename T, bool P>
struct mask_interleaved_lut<T, 8, P> {
    static const T* get() {
        alignas(sizeof(T)*8)
        static T mask_lut_p[] = { +1, -1, +1, -1, +1, -1, +1, -1 };

        alignas(sizeof(T)*8)
        static T mask_lut_n[] = { -1, +1, -1, +1, -1, +1, -1, +1 };
        return P ? mask_lut_p : mask_lut_n;
    }
};

template <typename T, bool P>
struct mask_interleaved_lut<T, 16, P> {
    static const T* get() {
        alignas(sizeof(T)*16)
        static T mask_lut_p[] = { +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1 };

        alignas(sizeof(T)*16)
        static T mask_lut_n[] = { -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1 };
        return P ? mask_lut_p : mask_lut_n;
    }
};

template <typename T, size_t W, bool P>
SIMD_INLINE
static Vec<T, W> make_interleaved_mask()
{
    return Vec<T, W>::load_aligned(mask_interleaved_lut<T, W, P>::get());
}

}  // namespace detail

template <typename T, size_t W>
struct fmaddsub<T, W, REQUIRE_FLOATING(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& x, const Vec<T, W>& y, const Vec<T, W>& z) noexcept
    {
        Vec<T, W> scale = detail::make_interleaved_mask<T, W, 0>();
        return x * y + scale * z;
    }
};

template <typename T, size_t W>
struct fmsubadd<T, W, REQUIRE_FLOATING(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& x, const Vec<T, W>& y, const Vec<T, W>& z) noexcept
    {
        Vec<T, W> scale = detail::make_interleaved_mask<T, W, 1>();
        return x * y + scale * z;
    }
};
} } } // namespace simd::kernel::generic
