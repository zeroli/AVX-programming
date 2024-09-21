#pragma once

namespace simd {
template <typename T, size_t W>
class Vec;

namespace types {
template <typename T, size_t W>
struct integral_only_ops
{
    Vec<T, W>& operator %=(const Vec<T, W>& rhs) noexcept;
    Vec<T, W>& operator >>=(int32_t rhs) noexcept;
    Vec<T, W>& operator >>=(const Vec<T, W>& rhs) noexcept;
    Vec<T, W>& operator <<=(int32_t rhs) noexcept;
    Vec<T, W>& operator <<=(const Vec<T, W>& rhs) noexcept;

    /// Shorthand for simd::mod()
    friend Vec<T, W> operator %(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        return Vec<T, W>(lhs) %= rhs;
    }

    /// Shorthand for simd::bitwise_rshift()
    friend Vec<T, W> operator >>(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        return Vec<T, W>(lhs) >>= rhs;
    }

    /// Shorthand for simd::bitwise_lshift()
    friend Vec<T, W> operator <<(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        return Vec<T, W>(lhs) <<= rhs;
    }

    /// Shorthand for simd::bitwise_rshift()
    friend Vec<T, W> operator >>(const Vec<T, W>& lhs, int32_t rhs) noexcept
    {
        return Vec<T, W>(lhs) >>= rhs;
    }

    /// Shorthand for simd::bitwise_lshift()
    friend Vec<T, W> operator <<(const Vec<T, W>& lhs, int32_t rhs) noexcept
    {
        return Vec<T, W>(lhs) <<= rhs;
    }

protected:
    Vec<T, W>& ref_vec() {
        return static_cast<Vec<T, W>&>(*this);
    }
};
/// no these operations for float/double
template <size_t W>
struct integral_only_ops<float, W>
{
};
template <size_t W>
struct integral_only_ops<double, W>
{
};

}  // namespace types
}  // namespace simd