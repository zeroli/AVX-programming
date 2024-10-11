#pragma once

#include <complex>

namespace simd { namespace kernel { namespace generic {
using namespace types;

/// add
template <typename T, size_t W>
struct add<std::complex<T>, W>
{
    using value_type = std::complex<T>;

    SIMD_INLINE
    static Vec<value_type, W> apply(const Vec<value_type, W>& lhs, const Vec<value_type, W>& rhs) noexcept
    {
        Vec<value_type, W> ret(
                lhs.real() + rhs.real(),
                lhs.imag() + rhs.imag());
        return ret;
    }
};

/// sub
template <typename T, size_t W>
struct sub<std::complex<T>, W>
{
    using value_type = std::complex<T>;

    SIMD_INLINE
    static Vec<value_type, W> apply(const Vec<value_type, W>& lhs, const Vec<value_type, W>& rhs) noexcept
    {
        Vec<value_type, W> ret(
                lhs.real() - rhs.real(),
                lhs.imag() - rhs.imag());
        return ret;
    }
};

/// mul
template <typename T, size_t W>
struct mul<std::complex<T>, W>
{
    using value_type = std::complex<T>;

    SIMD_INLINE
    static Vec<value_type, W> apply(const Vec<value_type, W>& lhs, const Vec<value_type, W>& rhs) noexcept
    {
        // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        auto&& a = lhs.real();
        auto&& b = lhs.imag();
        auto&& c = rhs.real();
        auto&& d = rhs.imag();
        Vec<value_type, W> ret(
                (a * c - b * d),
                (a * d + b * c));
        return ret;
    }
};

/// div
template <typename T, size_t W>
struct div<std::complex<T>, W>
{
    using value_type = std::complex<T>;

    SIMD_INLINE
    static Vec<value_type, W> apply(const Vec<value_type, W>& lhs, const Vec<value_type, W>& rhs) noexcept
    {
        /*
            (a + bi)   (a + bi) * (c - di)   (ac + bd) + (bc - ad)i
            -------- = ------------------- = ----------------------
            (c + di)   (c + di) * (c - di)         (cc + dd)
        */
        auto&& a = lhs.real();
        auto&& b = lhs.imag();
        auto&& c = rhs.real();
        auto&& d = rhs.imag();
        auto&& cc_dd = (c * c + d * d);
        Vec<value_type, W> ret(
            (a * c + b * d) / cc_dd,
            (b * c - a * d) / cc_dd);
        return ret;
    }
};
} } } // namespace simd::kernel::generic
