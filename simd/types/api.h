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

DEFINE_API_BINARY_OP(eq);
DEFINE_API_BINARY_OP(ne);
DEFINE_API_BINARY_OP(gt);
DEFINE_API_BINARY_OP(ge);
DEFINE_API_BINARY_OP(lt);
DEFINE_API_BINARY_OP(le);

DEFINE_API_BINARY_OP(copysign);

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
DEFINE_API_UNARY_OP(sign);
DEFINE_API_UNARY_OP(bitofsign);

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
Vec<T, W> select(const VecBool<T, W>& cond, const Vec<T, W>& x, const Vec<T, W>& y) noexcept
{
    using A = typename VecBool<T, W>::arch_t;
    return kernel::select<T, W>(cond, x, y, A{});
}

template <typename T, size_t W>
Vec<T, W> broadcast(T v) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return Vec<T, W>::broadcast(v, A{});
}

template <typename T, size_t W>
Vec<T, W> setzero() noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::setzero<T, W>(A{});
}

template <size_t W, typename T>
Vec<T, W> load(const T* mem) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::load_aligned<T, W>(mem, A{});
}

template <size_t W, typename T>
Vec<T, W> loadu(const T* mem) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::load_unaligned<T, W>(mem, A{});
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

template <typename T, size_t W>
Vec<T, W> ceil(const Vec<T, W>& x) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::ceil<T, W>(x, A{});
}

template <typename T, size_t W>
Vec<T, W> floor(const Vec<T, W>& x) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::floor<T, W>(x, A{});
}

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

template <typename T, size_t W>
std::ostream& operator <<(std::ostream& os, const VecBool<T, W>& x) noexcept
{
    using A = typename VecBool<T, W>::arch_t;
    constexpr auto size = VecBool<T, W>::size();
    alignas(A::alignment()) bool buffer[size];
    x.store_aligned(&buffer[0]);
    os << VecBool<T, W>::type() << "[";
    for (auto i = 0; i < size - 1; i++) {
        os << (buffer[i] ? 'T' : 'F') << ", ";
    }
    return os << (buffer[size - 1] ? 'T' : 'F') << "]";
}
}  // namespace simd
