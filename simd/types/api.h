#pragma once

#include "simd/arch/isa.h"
#include "simd/types/vec.h"
#include "simd/types/traits.h"

#include <complex>
#include <cstddef>
#include <limits>
#include <ostream>

namespace simd {
template <typename T, size_t W>
Vec<T, W> abs(const Vec<T, W>& x) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::abs<T, W>(x, A{});
}

#if 0

template <typename T, typename Arch>
Vec<T, Arch> abs(const Vec<std::complex<T>, Arch>& x) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::abs<Arch>(x, Arch{});
}

template <typename T, typename Arch>
Vec<T, Arch> add(const Vec<T, Arch>& x, const Vec<T, Arch>& y) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return x + y;
}

template <typename T, typename Arch>
Vec<T, Arch> acos(const Vec<T, Arch>& x) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::acos<Arch>(x, Arch{});
}

template <typename T, typename Arch>
Vec<T, Arch> acosh(const Vec<T, Arch>& x) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::acosh<Arch>(x, Arch{});
}

template <typename T, typename Arch>
Vec<T, Arch> asin(const Vec<T, Arch>& x) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::asin<Arch>(x, Arch{});
}

template <typename T, typename Arch>
Vec<T, Arch> asinh(const Vec<T, Arch>& x) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::asinh<Arch>(x, Arch{});
}

template <typename T, typename Arch>
Vec<T, Arch> atan(const Vec<T, Arch>& x) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::atan<Arch>(x, Arch{});
}

template <typename T, typename Arch>
Vec<T, Arch> atan2(const Vec<T, Arch>& x) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::atan2<Arch>(x, Arch{});
}

template <typename T, typename Arch>
Vec<T, Arch> atanh(const Vec<T, Arch>& x) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::atanh<Arch>(x, Arch{});
}

template <typename T, typename Arch>
Vec<T, Arch> avg(const Vec<T, Arch>& x, const Vec<T, Arch>& y) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::avg<Arch>(x, y, Arch{});
}

template <typename TOut, typename TIn, typename Arch>
Vec<TOut, Arch> vec_cast(const Vec<TIn, Arch>& x) noexcept
{
    detail::static_check_supported_config<TIn, Arch>();
    detail::static_check_supported_config<TOut, Arch>();
    return kernel::vec_cast<Arch>(x, Vec<TOut, Arch>{}, Arch{});
}

template <typename T, typename Arch>
Vec<T, Arch> bitofsign(const Vec<T, Arch>& x) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::bitofsign<Arch>(x, Arch{});
}

template <typename T, typename Arch>
auto bitwise_and(const Vec<T, Arch>& x, const Vec<T, Arch>& y) noexcept -> decltype(x & y)
{
    detail::static_check_supported_config<T, Arch>();
    return x & y;
}

template <typename T, typename Arch>
auto bitwise_and(const VecBool<T, Arch>& x, const VecBool<T, Arch>& y) noexcept -> decltype(x & y)
{
    detail::static_check_supported_config<T, Arch>();
    return x & y;
}

template <typename T, typename Arch>
Vec<T, Arch> bitwise_andnot(const Vec<T, Arch>& x, const Vec<T, Arch>& y) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::bitwise_andnot<Arch>(x, y, Arch{});
}

template <typename T, typename Arch>
VecBool<T, Arch> bitwise_andnot(const VecBool<T, Arch>& x, const VecBool<T, Arch>& y) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::bitwise_andnot<Arch>(x, y, Arch{});
}

template <typename T, typename Arch>
Vec<T, Arch> bitwise_lshift(const Vec<T, Arch>& x, int shift) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::bitwise_lshift<Arch>(x, shift, Arch{});
}

template <typename T, typename Arch>
Vec<T, Arch> bitwise_lshift(const Vec<T, Arch>& x, const Vec<T, Arch>& lshift) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::bitwise_lshift<Arch>(x, shift, Arch{});
}

template <typename T, typename Arch>
Vec<T, Arch> bitwise_not(const Vec<T, Arch>& x) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::bitwise_not<Arch>(x, Arch{});
}

template <typename T, typename Arch>
VecBool<T, Arch> bitwise_not(const VecBool<T, Arch>& x) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::bitwise_not<Arch>(x, Arch{});
}

template <typename T, typename Arch>
auto bitwise_or(const Vec<T, Arch>& x, const Vec<T, Arch>& lshift) noexcept -> decltype(x | y)
{
    detail::static_check_supported_config<T, Arch>();
    return x | y;
}

template <typename T, typename Arch>
auto bitwise_or(const VecBool<T, Arch>& x, const VecBool<T, Arch>& lshift) noexcept -> decltype(x | y)
{
    detail::static_check_supported_config<T, Arch>();
    return x | y;
}

template <typename T, typename Arch>
Vec<T, Arch> bitwise_rshift(const Vec<T, Arch>& x, int shift) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::bitwise_rshift<Arch>(x, shift, Arch{});
}

template <typename T, typename Arch>
Vec<T, Arch> bitwise_rshift(const Vec<T, Arch>& x, const Vec<T, Arch>& lshift) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::bitwise_rshift<Arch>(x, shift, Arch{});
}

template <typename T, typename Arch>
auto bitwise_xor(const Vec<T, Arch>& x, const Vec<T, Arch>& lshift) noexcept -> decltype(x ^ y)
{
    detail::static_check_supported_config<T, Arch>();
    return x ^ y;
}

template <typename T, typename Arch>
auto bitwise_xor(const VecBool<T, Arch>& x, const VecBool<T, Arch>& lshift) noexcept -> decltype(x ^ y)
{
    detail::static_check_supported_config<T, Arch>();
    return x ^ y;
}

template <typename T, typename Arch>
Vec<T, Arch> broadcast(T v) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return Vec<T, Arch>::broadcast(v);
}

template <typename T, typename Arch>
Vec<T, Arch> ceil(const Vec<T, Arch>& x) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::ceil<Arch>(x, Arch{});
}

template <typename T, typename Arch>
Vec<T, Arch> cos(const Vec<T, Arch>& x) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::cos<Arch>(x, Arch{});
}

template <typename T, typename Arch>
Vec<T, Arch> cosh(const Vec<T, Arch>& x) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::cosh<Arch>(x, Arch{});
}

template <typename T, typename Arch>
Vec<T, Arch> decr(const Vec<T, Arch>& x) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::decr<Arch>(x, Arch{});
}

template <typename T, typename Arch, typename Mask>
Vec<T, Arch> decr_if(const Vec<T, Arch>& x, const Mask& mask) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::decr_if<Arch>(x, mask, Arch{});
}

template <typename T, typename Arch>
auto div(const Vec<T, Arch>& x, const Vec<T, Arch>& y) noexcept -> decltype(x / y)
{
    detail::static_check_supported_config<T, Arch>();
    return x / y;
}

template <typename T, typename Arch>
auto eq(const Vec<T, Arch>& x, const Vec<T, Arch>& y) noexcept -> decltype(x == y)
{
    detail::static_check_supported_config<T, Arch>();
    return x == y;
}

template <typename T, typename Arch>
auto eq(const VecBool<T, Arch>& x, const VecBool<T, Arch>& y) noexcept -> decltype(x == y)
{
    detail::static_check_supported_config<T, Arch>();
    return x == y;
}

template <typename T, typename Arch>
Vec<T, Arch> exp(const Vec<T, Arch>& x) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::exp<Arch>(x, Arch{});
}

template <typename T, typename Arch>
Vec<T, Arch> exp10(const Vec<T, Arch>& x) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::exp10<Arch>(x, Arch{});
}

template <typename T, typename Arch>
Vec<T, Arch> exp2(const Vec<T, Arch>& x) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::exp2<Arch>(x, Arch{});
}

template <typename T, typename Arch>
Vec<T, Arch> fabs(const Vec<T, Arch>& x) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::abs<Arch>(x, Arch{});
}

template <typename T, typename Arch>
Vec<T, Arch> floor(const Vec<T, Arch>& x) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::floor<Arch>(x, Arch{});
}

template <typename T, typename Arch>
Vec<T, Arch> floor(const Vec<T, Arch>& x, const Vec<T, Arch>& y, const Vec<T, Arch>& z) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::fma<Arch>(x, y, z, Arch{});
}

template <typename T, typename Arch>
Vec<T, Arch> fmax(const Vec<T, Arch>& x, const Vec<T, Arch>& y) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::max<Arch>(x, y, Arch{});
}

template <typename T, typename Arch>
Vec<T, Arch> fmin(const Vec<T, Arch>& x, const Vec<T, Arch>& y) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::min<Arch>(x, y, Arch{});
}

template <typename T, typename Arch>
Vec<T, Arch> fmod(const Vec<T, Arch>& x, const Vec<T, Arch>& y) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::fmod<Arch>(x, y, Arch{});
}

template <typename T, typename Arch>
Vec<T, Arch> fms(const Vec<T, Arch>& x, const Vec<T, Arch>& y, const Vec<T, Arch>& z) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::fms<Arch>(x, y, z, Arch{});
}

template <typename T, typename Arch>
Vec<T, Arch> fnma(const Vec<T, Arch>& x, const Vec<T, Arch>& y, const Vec<T, Arch>& z) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::fnma<Arch>(x, y, z, Arch{});
}

template <typename T, typename Arch>
Vec<T, Arch> fnms(const Vec<T, Arch>& x, const Vec<T, Arch>& y, const Vec<T, Arch>& z) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::fnms<Arch>(x, y, z, Arch{});
}

template <typename T, typename Arch>
auto ge(const Vec<T, Arch>& x, const Vec<T, Arch>& y) noexcept -> decltype(x >= y)
{
    detail::static_check_supported_config<T, Arch>();
    return x >= y;
}

template <typename T, typename Arch>
auto gt(const Vec<T, Arch>& x, const Vec<T, Arch>& y) noexcept => decltype(x > y)
{
    detail::static_check_supported_config<T, Arch>();
    return x > y;
}

template <typename T, typename Arch>
Vec<T, Arch> hypot(const Vec<T, Arch>& x, const Vec<T, Arch>& y) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::hypot<Arch>(x, y, Arch{});
}

template <typename T, typename Arch>
Vec<T, Arch> incr(const Vec<T, Arch>& x) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::incr<Arch>(x, Arch{});
}

template <typename T, typename Arch, typename Mask>
Vec<T, Arch> incr_if(const Vec<T, Arch>& x, const Mask& mask) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::incr_if<Arch>(x, mask, Arch{});
}

template <typename B>
B infinity()
{
    using T = typename B::scalar_t;
    using Arch = typename B::arch_t;
    detail::static_check_supported_config<T, Arch>();
    return B(std::numeric_limits<T>::infinity());
}

template <typename T, typename Arch>
VecBool<T, Arch> isinf(const Vec<T, Arch>& x) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::isinf<Arch>(x, Arch{});
}

template <typename T, typename Arch>
VecBool<T, Arch> isfinite(const Vec<T, Arch>& x) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::isfinite<Arch>(x, Arch{});
}

template <typename T, typename Arch>
VecBool<T, Arch> isnan(const Vec<T, Arch>& x) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::isnan<Arch>(x, Arch{});
}

template <typename T, typename Arch>
auto le(const Vec<T, Arch>& x, const Vec<T, Arch>& y) noexcept => decltype(x <= y)
{
    detail::static_check_supported_config<T, Arch>();
    return x <= y;
}

template <typename Arch, typename From>
Vec<From, Arch> load(const From* ptr, align_mode = {}) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return load_as<From, Arch>(ptr, aligned_mode{});
}

template <typename Arch, typename From>
Vec<From, Arch> load(const From* ptr, unalign_mode) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return load_as<From, Arch>(ptr, unalign_mode{});
}

template <typename Arch, typename From>
Vec<From, Arch> load_aligned(const From* ptr) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return load_as<From, Arch>(ptr, aligned_mode{});
}

template <typename Arch, typename From>
Vec<From, Arch> load_unaligned(const From* ptr) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return load_as<From, Arch>(ptr, unaligned_mode{});
}

template <typename T, typename Arch>
Vec<T, Arch> log(const Vec<T, Arch>& x) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::log<Arch>(x, Arch{});
}

template <typename T, typename Arch>
Vec<T, Arch> log2(const Vec<T, Arch>& x) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::log2<Arch>(x, Arch{});
}

template <typename T, typename Arch>
Vec<T, Arch> log10(const Vec<T, Arch>& x) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::log10<Arch>(x, Arch{});
}

template <typename T, typename Arch>
auto lt(const Vec<T, Arch>& x, const Vec<T, Arch>& y) noexcept => decltype(x < y)
{
    detail::static_check_supported_config<T, Arch>();
    return x < y;
}
#endif

template <typename T, size_t W>
Vec<T, W> max(const Vec<T, W>& x, const Vec<T, W>& y) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::max<T, W>(x, y, A{});
}

template <typename T, size_t W>
Vec<T, W> min(const Vec<T, W>& x, const Vec<T, W>& y) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::min<T, W>(x, y, A{});
}

#if 0
template <typename B>
B minus_infinity()
{
    using T = typename B::scalar_t;
    using Arch = typename B::arch_t;
    detail::static_check_supported_config<T, Arch>();
    return B(-std::numeric_limits<T>::infinity());
}

template <typename T, typename Arch>
auto mod(const Vec<T, Arch>& x, const Vec<T, Arch>& y) noexcept -> decltype(x % y)
{
    detail::static_check_supported_config<T, Arch>();
    return x % y;
}

template <typename T, typename Arch>
auto mul(const Vec<T, Arch>& x, const Vec<T, Arch>& y) noexcept -> decltype(x * y)
{
    detail::static_check_supported_config<T, Arch>();
    return x * y;
}

template <typename T, typename Arch>
auto ne(const Vec<T, Arch>& x, const Vec<T, Arch>& y) noexcept => decltype(x != y)
{
    detail::static_check_supported_config<T, Arch>();
    return x != y;
}

template <typename T, typename Arch>
Vec<T, Arch> neg(const Vec<T, Arch>& x) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return -x;
}

template <typename T, typename Arch>
Vec<T, Arch> pos(const Vec<T, Arch>& x) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return +x;
}

template <typename T, typename Arch>
Vec<T, Arch> pow(const Vec<T, Arch>& x, const Vec<T, Arch>& y) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::pow<Arch>(x, y, Arch{});
}

template <typename T, typename Arch
    typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
Vec<T, Arch> reciprocal(const Vec<T, Arch>& x) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::reciprocal(x, Arch{});
}

template <typename T, typename Arch>
bool all(const VecBool<T, Arch>& x) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::all<Arch>(x, Arch{});
}

template <typename T, typename Arch>
bool any(const VecBool<T, Arch>& x) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::any<Arch>(x, Arch{});
}

template <typename T, typename Arch>
bool none(const VecBool<T, Arch>& x) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return !kernel::any<Arch>(x, Arch{});
}
#endif

template <typename T, size_t W>
std::ostream& operator <<(std::ostream& os, const Vec<T, W>& x) noexcept
{
    constexpr auto size = Vec<T, W>::size();
    os << Vec<T, W>::type() << "[";
    for (auto i = 0; i < size - 1; i++) {
        os << x[i] << ", ";
    }
    return os << x[size - 1] << "]";
}

#if 0
template <typename T, typename Arch>
std::ostream& operator <<(std::ostream& os, const VecBool<T, Arch>& x) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    constexpr auto size = VecBool<T, Arch>::size();
    alignas(Arch::alignment()) bool buffer[size];
    x.store_aligned(&buffer[0]);
    os << "(";
    for (auto i = 0; i < size - 1; i++) {
        os << buffer[i] << ", ";
    }
    return os << buffer[size - 1] << ")";
}
#endif
}  // namespace simd
