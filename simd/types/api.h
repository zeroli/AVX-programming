#pragma once

#include "simd/arch/isa.h"
#include "simd/types/vec.h"
#include "simd/types/traits.h"

#include <complex>
#include <cstddef>
#include <limits>
#include <ostream>

namespace simd {
#define DEFINE_API_BINARY_OP(OP) \
template <typename T, size_t W> \
Vec<T, W> OP(const Vec<T, W>& x, const Vec<T, W>& y) noexcept \
{ \
    using A = typename Vec<T, W>::arch_t; \
    return kernel::OP<T, W>(x, y, A{}); \
} \
template <typename T, size_t W> \
Vec<T, W> OP(const Vec<T, W>& x, T y) noexcept \
{ \
    return OP(x, Vec<T, W>(y)); \
} \
template <typename T, size_t W> \
Vec<T, W> OP(T x, const Vec<T, W>& y) noexcept \
{ \
    return OP(Vec<T, W>(x), y); \
} \
///

/// arithmetic binary operations
DEFINE_API_BINARY_OP(add);
DEFINE_API_BINARY_OP(sub);
DEFINE_API_BINARY_OP(mul);
DEFINE_API_BINARY_OP(div);
DEFINE_API_BINARY_OP(mod);

/// bitwise operations
DEFINE_API_BINARY_OP(bitwise_and);
DEFINE_API_BINARY_OP(bitwise_or);
DEFINE_API_BINARY_OP(bitwise_xor);
DEFINE_API_BINARY_OP(bitwise_andnot);
DEFINE_API_BINARY_OP(bitwise_lshift);
DEFINE_API_BINARY_OP(bitwise_rshift);

DEFINE_API_BINARY_OP(logical_and);
DEFINE_API_BINARY_OP(logical_or);

DEFINE_API_BINARY_OP(eq);
DEFINE_API_BINARY_OP(ne);
DEFINE_API_BINARY_OP(gt);
DEFINE_API_BINARY_OP(ge);
DEFINE_API_BINARY_OP(lt);
DEFINE_API_BINARY_OP(le);

#undef DEFINE_API_BINARY_OP

/// math operations
#define DEFINE_API_UNARY_OP(OP) \
template <typename T, size_t W> \
Vec<T, W> OP(const Vec<T, W>& x) noexcept \
{ \
    using A = typename Vec<T, W>::arch_t; \
    return kernel::OP<T, W>(x, A{}); \
} \
///

DEFINE_API_UNARY_OP(bitwise_not);
DEFINE_API_UNARY_OP(neg);

DEFINE_API_UNARY_OP(abs);
DEFINE_API_UNARY_OP(sqrt);

#undef DEFINE_API_UNARY_OP

template <typename T, size_t W>
VecBool<T, W> bitwise_not(const VecBool<T, W>& x) noexcept
{
    using A = typename VecBool<T, W>::arch_t;
    return kernel::bitwise_not<T, W>(x, A{});
}

/// static_cast each element of T to U, with same width
template <typename U, typename T, size_t W>
Vec<U, W> cast(const Vec<T, W>& x) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::cast<U>(x, A{});
}

template <typename T, size_t W>
bool all(const VecBool<T, W>& x) noexcept
{
    using A = typename VecBool<T, W>::arch_t;
    return kernel::all<T, W>(x, A{});
}

template <typename T, size_t W>
bool any(const VecBool<T, W>& x) noexcept
{
    using A = typename VecBool<T, W>::arch_t;
    return kernel::any<T, W>(x, A{});
}

template <typename T, size_t W>
Vec<T, W> broadcast(T v) noexcept
{
    using A = typename VecBool<T, W>::arch_t;
    return Vec<T, W>::broadcast(v, A{});
}

template <typename T, size_t W>
Vec<T, W> load_aligned(const T* mem) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::load_aligned<T, W>(mem, A{});
}

template <typename T, size_t W>
Vec<T, W> load_unaligned(const T* mem) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::load_unaligned<T, W>(mem, A{});
}

template <typename T, size_t W>
Vec<T, W> load(const T* mem, aligned_mode) noexcept
{
    return load_aligned(mem);
}

template <typename T, size_t W>
Vec<T, W> load(const T* mem, unaligned_mode) noexcept
{
    return load_unaligned(mem);
}

template <typename T, size_t W>
void store_aligned(T* mem, const Vec<T, W>& x) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    kernel::store_aligned<T, W>(mem, x, A{});
}

template <typename T, size_t W>
void store_unaligned(T* mem, const Vec<T, W>& x) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    kernel::store_unaligned<T, W>(mem, x, A{});
}

template <typename T, size_t W>
void store(T* mem, const Vec<T, W>& x, aligned_mode) noexcept
{
    store_aligned(mem, x);
}

template <typename T, size_t W>
void store(T* mem, const Vec<T, W>& x, unaligned_mode) noexcept
{
    store_unaligned(mem, x);
}

template <typename T, size_t W>
uint64_t to_mask(const VecBool<T, W>& x) noexcept
{
    using A = typename VecBool<T, W>::arch_t;
    return to_mask(x, A{});
}

template <typename T, size_t W>
VecBool<T, W> from_mask(uint64_t x) noexcept
{
    using A = typename VecBool<T, W>::arch_t;
    return from_mask<T, W>(x, A{});
}

#if 0

template <typename T, typename Arch>
Vec<T, Arch> abs(const Vec<std::complex<T>, Arch>& x) noexcept
{
    detail::static_check_supported_config<T, Arch>();
    return kernel::abs<Arch>(x, Arch{});
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
