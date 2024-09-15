#pragma once

#include "simd/arch/isa.h"

namespace simd {
template <typename T, size_t W>
Vec<T, W>::Vec(T val) noexcept
    : self_t(kernel::broadcast<T, W>(val, A{}))
{
}

#if 0  // TODO:
template <typename T, size_t W>
template <typename... Ts>
Vec<T, W>::Vec(T val0, T val1, Ts... vals) noexcept
    : self_t(kernel::set<A>(Vec{}, A{}, val0, val1, static_cast<T>(vals)...))
{
    static_assert(sizeof...(Ts) + 2 == size(),
        "the constructor requires as many arguments as vector elements");
}
template <typename T, size_t W>
Vec<T, W>::Vec(vec_mask_t b) noexcept
    : self_t(kernel::from_bool(b, A{}))
{
}
#endif

template <typename T, size_t W>
template <typename... Regs>
Vec<T, W>::Vec(register_t arg, Regs... others) noexcept
    : base_t({arg, others...})
{
}

template <typename T, size_t W>
template <typename U>
void Vec<T, W>::store_aligned(U* mem) const noexcept
{
    using A = typename Vec<T, W>::arch_t;
    assert(is_aligned(mem, A::alignment())
        && "store location is not properly aligned");
    kernel::store_aligned<T, W>((T*)mem, *this, A{});
}

template <typename T, size_t W>
template <typename U>
void Vec<T, W>::store_unaligned(U* mem) const noexcept
{
    using A = typename Vec<T, W>::arch_t;
    kernel::store_unaligned<T, W>((T*)mem, *this, A{});
}

template <typename T, size_t W>
template <typename U>
Vec<T, W> Vec<T, W>::load_aligned(const U* mem) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    assert(is_aligned(mem, A::alignment())
        && "loaded location is not properly aligned");
    return kernel::load_aligned<T, W>((const T*)mem, A{});
}

template <typename T, size_t W>
template <typename U>
Vec<T, W> Vec<T, W>::load_unaligned(const U* mem) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::load_unaligned<T, W>((const T*)mem, A{});
}

template <typename T, size_t W>
Vec<T, W> Vec<T, W>::operator ~() const noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::bitwise_not<T, W>(*this, A{});
}

template <typename T, size_t W>
Vec<T, W> Vec<T, W>::operator -() const noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::neg<T, W>(*this, A{});
}

/// VecBool
template <typename T, size_t W>
VecBool<T, W> VecBool<T, W>::operator ==(const VecBool<T, W>& other) const noexcept
{
    using A = typename VecBool<T, W>::arch_t;
    return kernel::eq<T, W>(*this, other, A{});
}
template <typename T, size_t W>
VecBool<T, W> VecBool<T, W>::operator !=(const VecBool<T, W>& other) const noexcept
{
    using A = typename VecBool<T, W>::arch_t;
    return kernel::ne<T, W>(*this, other, A{});
}
template <typename T, size_t W>
VecBool<T, W> VecBool<T, W>::operator ~() const noexcept
{
    using A = typename VecBool<T, W>::arch_t;
    return kernel::bitwise_not<T, W>(*this, A{});
}

template <typename T, size_t W>
VecBool<T, W> VecBool<T, W>::operator &(const VecBool& other) const noexcept
{
    using A = typename VecBool<T, W>::arch_t;
    return kernel::bitwise_and<A>(*this, other, A{});
}
template <typename T, size_t W>
VecBool<T, W> VecBool<T, W>::operator |(const VecBool<T, W>& other) const noexcept
{
    using A = typename VecBool<T, W>::arch_t;
    return kernel::bitwise_or<T, W>(*this, other, A{});
}
template <typename T, size_t W>
VecBool<T, W> VecBool<T, W>::operator ^(const VecBool<T, W>& other) const noexcept
{
    using A = typename VecBool<T, W>::arch_t;
    return kernel::bitwise_xor<T, W>(*this, other, A{});
}
}  // namespace simd
