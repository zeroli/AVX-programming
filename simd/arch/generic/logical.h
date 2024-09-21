#pragma once

#include "simd/arch/generic/detail.h"

namespace simd {
namespace kernel {
using namespace types;

template <typename Arch, typename T>
VecBool<T, Arch> from_mask(const VecBool<T, Arch>&, uint64_t mask, requires<Generic>) noexcept
{
    alignas(Arch::alignment()) bool buffer[VecBool<T, Arch>::size()];
    for (auto i = 0u; i < VecBool<T, Arch>::size(); i++) {
        buffer[i] = mask & (1ull << i);
    }
    return VecBool<T, Arch>::load_aligned(buffer);
}

template <typename Arch, typename T>
VecBool<T, Arch> ge(const Vec<T, Arch>& self, const Vec<T, Arch>& other, requires<Generic>) noexcept
{
    return other <= self;
}

template <typename Arch, typename T>
VecBool<T, Arch> gt(const Vec<T, Arch>& self, const Vec<T, Arch>& other, requires<Generic>) noexcept
{
    return other < self;
}

template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
VecBool<T, Arch> isinf(const Vec<T, Arch>& self, requires<Generic>) noexcept
{
    return VecBool<T, Arch>(false);
}

template <typename Arch>
VecBool<float, Arch> isinf(const Vec<float, Arch>& self, requires<Generic>) noexcept
{
    return abs(self) == std::numeric_limits<float>::infinity();
}

template <typename Arch>
VecBool<double, Arch> isinf(const Vec<double, Arch>& self, requires<Generic>) noexcept
{
    return abs(self) == std::numeric_limits<double>::infinity();
}

template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
VecBool<T, Arch> isfinite(const Vec<T, Arch>& self, requires<Generic>) noexcept
{
    return VecBool<T, Arch>(true);
}

template <typename Arch>
VecBool<float, Arch> isfinite(const Vec<float, Arch>& self, requires<Generic>) noexcept
{
    return (self - self) == 0.f;
}

template <typename Arch>
VecBool<double, Arch> isfinite(const Vec<double, Arch>& self, requires<Generic>) noexcept
{
    return (self - self) == 0.;
}

template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
VecBool<T, Arch> isnan(const Vec<T, Arch>& self, requires<Generic>) noexcept
{
    return VecBool<T, Arch>(false);
}

template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
VecBool<T, Arch> le(const Vec<T, Arch>& self, const Vec<T, Arch>& other, requires<Generic>) noexcept
{
    return (self < other) || (self == other);
}

template <typename Arch, typename T>
VecBool<T, Arch> ne(const Vec<T, Arch>& self, const Vec<T, Arch>& other, requires<Generic>) noexcept
{
    return !(other == self);
}

template <typename Arch, typename T>
VecBool<T, Arch> logical_and(const Vec<T, Arch>& self, const Vec<T, Arch>& other, requires<Generic>) noexcept
{
    return detail::apply([](T x, T y) noexcept {
        return x && y;
    }, self, other);
}

template <typename Arch, typename T>
VecBool<T, Arch> logical_or(const Vec<T, Arch>& self, const Vec<T, Arch>& other, requires<Generic>) noexcept
{
    return detail::apply([](T x, T y) noexcept {
        return x || y;
    }, self, other);
}

template <typename Arch, typename T>
VecBool<T, Arch> to_mask(const VecBool<T, Arch>&,  requires<Generic>) noexcept
{
    alignas(Arch::alignment()) bool buffer[VecBool<T, Arch>::size()];
    self.store_aligned(buffer);
    uint64_t res = 0;
    for (auto i = 0u; i < VecBool<T, Arch>::size(); i++) {
        if (buffer[i]) {
            res |= 1ul << i;
        }
    }
    return res;
}

}  // namespace kernel
}  // namespace simd
