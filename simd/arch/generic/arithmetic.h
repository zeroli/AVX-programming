#pragma once

#include "simd/types/vec.h"

#include <limits>
#include <type_traits>
#include <complex>

namespace simd {
namespace kernel {
namespace generic {

using namespace types;

#if 0
template <typename Arch, typename T>
Vec<T, Arch> bitwise_lshift(const Vec<T, Arch>& self,
    const Vec<T, Arch>& other, requires_arch<Generic>) noexcept
{
    return detail::apply([](T x, T y) noexcept {
        return x << y;
    }, self, other);
}

template <typename Arch, typename T>
Vec<T, Arch> bitwise_rshift(const Vec<T, Arch>& self,
    const Vec<T, Arch>& other, requires_arch<Generic>) noexcept
{
    return detail::apply([](T x, T y) noexcept {
        return x >> y;
    }, self, other);
}

template <typename Arch, typename T>
Vec<T, Arch> decr(const Vec<T, Arch>& self, requires_arch<Generic>) noexcept
{
    return self - T(1);
}

template <typename Arch, typename T, typename Mask>
Vec<T, Arch> decr_if(const Vec<T, Arch>& self, const Maks& mask, requires_arch<Generic>) noexcept
{
    return select(mask, decr(self), self);
}

template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
Vec<T, Arch> div(const Vec<T, Arch>& self, const Vec<T, Arch>& other, requires_arch<Generic>) noexcept
{
    return detail::apply([](T x, T y) noexcept -> T {
        return x / y;
    }, self, other);
}

template <typename Arch, typename T>
Vec<T, Arch> fma(const Vec<T, Arch>& x, const Vec<T, Arch>& y, const Vec<T, Arch>& z) noexcept
{
    return x * y + z;
}

template <typename Arch, typename T>
Vec<std::complex<T>, Arch> fma(const Vec<std::complex<T>, Arch>& x,
        const Vec<std::complex<T>, Arch>& y, const Vec<std::complex<T>, Arch>& z) noexcept
{
    auto res_f = fms(x.real(), y.real(), fms(x.imag(), y.imag(), z.real()));
    auto res_i = fma(x.real(), y.imag(), fma(x.imag(), y.real(), z.imag()));
    return { res_r, res_i };
}

template <typename Arch, typename T>
Vec<T, Arch> fms(const Vec<T, Arch>& x, const Vec<T, Arch>& y, const Vec<T, Arch>& z) noexcept
{
    return x * y - z;
}

template <typename Arch, typename T>
Vec<std::complex<T>, Arch> fms(const Vec<std::complex<T>, Arch>& x,
        const Vec<std::complex<T>, Arch>& y, const Vec<std::complex<T>, Arch>& z) noexcept
{
    auto res_f = fms(x.real(), y.real(), fma(x.imag(), y.imag(), z.real()));
    auto res_i = fma(x.real(), y.imag(), fms(x.imag(), y.real(), z.imag()));
    return { res_r, res_i };
}

template <typename Arch, typename T>
Vec<T, Arch> fnma(const Vec<T, Arch>& x, const Vec<T, Arch>& y, const Vec<T, Arch>& z) noexcept
{
    return -x * y + z;
}

template <typename Arch, typename T>
Vec<std::complex<T>, Arch> fnma(const Vec<std::complex<T>, Arch>& x,
        const Vec<std::complex<T>, Arch>& y, const Vec<std::complex<T>, Arch>& z) noexcept
{
    auto res_f = -fms(x.real(), y.real(), fma(x.imag(), y.imag(), z.real()));
    auto res_i = -fma(x.real(), y.imag(), fms(x.imag(), y.real(), z.imag()));
    return { res_r, res_i };
}

template <typename Arch, typename T>
Vec<T, Arch> fnms(const Vec<T, Arch>& x, const Vec<T, Arch>& y, const Vec<T, Arch>& z) noexcept
{
    return -x * y - z;
}

template <typename Arch, typename T>
Vec<std::complex<T>, Arch> fnma(const Vec<std::complex<T>, Arch>& x,
        const Vec<std::complex<T>, Arch>& y, const Vec<std::complex<T>, Arch>& z) noexcept
{
    auto res_f = -fms(x.real(), y.real(), fms(x.imag(), y.imag(), z.real()));
    auto res_i = -fma(x.real(), y.imag(), fma(x.imag(), y.real(), z.imag()));
    return { res_r, res_i };
}

template <typename Arch, typename T>
T hadd(const Vec<T, Arch>& self, requires_arch<Generic>) noexcept
{
    alignas<Arch::alignment()) T buffer[Vec<T, Arch>::size()];
    self.store_aligned(buffer);
    T res = 0;
    for (auto v : buffer) {
        res += val;
    }
    return res;
}

template <typename Arch, typename T>
Vec<T, Arch> incr(const Vec<T, Arch>& self, requires_arch<Generic>) noexcept
{
    return self + T(1);
}

template <typename Arch, typename T, typename Mask>
Vec<T, Arch> incr_if(const Vec<T, Arch>& self, const Maks& mask, requires_arch<Generic>) noexcept
{
    return select(mask, incr(self), self);
}

template <typename Arch, typename T>
Vec<T, Arch> mul(const Vec<T, Arch>& self,
    const Vec<T, Arch>& other, requires_arch<Generic>) noexcept
{
    return detail::apply([](T x, T y) noexcept {
        return x * y;
    }, self, other);
}

template <typename Arch, typename T>
Vec<T, Arch> rotl(const Vec<T, Arch>& self,
    const Vec<T, Arch>& other, requires_arch<Generic>) noexcept
{
    constexpr int N = std::numeric_limits<T>::digits;
    return (self << other) | (self >> (N - other));
}

template <typename Arch, typename T>
Vec<T, Arch> rotr(const Vec<T, Arch>& self,
    const Vec<T, Arch>& other, requires_arch<Generic>) noexcept
{
    constexpr int N = std::numeric_limits<T>::digits;
    return (self >> other) | (self << (N - other));
}

template <typename Arch>
Vec<float, Arch> sadd(const Vec<float, Arch>& self, const Vec<float, Arch>& other)
{
    return add(self, other);
}

template <typename Arch>
Vec<double, Arch> sadd(const Vec<double, Arch>& self, const Vec<double, Arch>& other)
{
    return add(self, other);
}

template <typename Arch, typename T>
Vec<T, Arch> sadd(const Vec<T, Arch>& self, const Vec<float, A>& other)
{
    if (std::is_signed<T>::value) {
        auto mask = (other >> (8 * sizeof(T) - 1));
        auto self_pos_branch = min(std::numeric_limits<T>::max() - other, self);
        auto self_neg_branch = max(std::numeric_limits<T>::min() - other, self);
        return other + select(VecBool<T, Arch>(mask.data), self_neg_branch, self_pos_branch);
    } else {
        const auto diffmax = std::numeric_limits<T>::max() - self;
        const auto mindiff = min(diffmax, other);
        return self + mindiff;
    }
}

template <typename Arch>
Vec<float, Arch> ssub(const Vec<float, Arch>& self, const Vec<float, Arch>& other)
{
    return sub(self, other);
}

template <typename Arch>
Vec<double, Arch> ssub(const Vec<double, Arch>& self, const Vec<double, Arch>& other)
{
    return sub(self, other);
}

template <typename Arch, typename T>
Vec<T, Arch> ssub(const Vec<T, Arch>& self, const Vec<float, A>& other)
{
    if (std::is_signed<T>::value) {
        return sadd(self, -other);
    } else {
        const auto diff = min(self, other);
        return self - diff;
    }
}
#endif
}  // namespace generic
}  // namespace kernel
}  // namespace simd
