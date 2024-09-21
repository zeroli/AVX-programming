#pragma once

#include "simd/arch/isa.h"

namespace simd {
namespace types {
template <typename T, size_t W>
Vec<T, W>& integral_only_ops<T, W>::operator %=(const Vec<T, W>& rhs) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return ref_vec() = kernel::mod<T, W>(ref_vec(), rhs, A{});
}

template <typename T, size_t W>
Vec<T, W>& integral_only_ops<T, W>::operator >>=(int32_t rhs) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return ref_vec() = kernel::bitwise_rshift<T, W>(ref_vec(), rhs, A{});
}

template <typename T, size_t W>
Vec<T, W>& integral_only_ops<T, W>::operator >>=(const Vec<T, W>& rhs) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return ref_vec() = kernel::bitwise_rshift<T, W>(ref_vec(), rhs, A{});
}

template <typename T, size_t W>
Vec<T, W>& integral_only_ops<T, W>::operator <<=(int32_t rhs) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return ref_vec() = kernel::bitwise_lshift<T, W>(ref_vec(), rhs, A{});
}

template <typename T, size_t W>
Vec<T, W>& integral_only_ops<T, W>::operator <<=(const Vec<T, W>& rhs) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return ref_vec() = kernel::bitwise_lshift<T, W>(ref_vec(), rhs, A{});
}
}  // namespace types

template <typename T, size_t W>
Vec<T, W>::Vec() noexcept
{
}

template <typename T, size_t W>
Vec<T, W>::Vec(T val) noexcept
    : self_t(kernel::broadcast<T, W>(val, A{}))
{
}

template <typename T, size_t W>
template <typename... Ts>
Vec<T, W>::Vec(T val0, T val1, Ts... vals) noexcept
    : self_t(kernel::set<T, W>(val0, val1, static_cast<T>(vals)..., A{}))
{
    static_assert(sizeof...(Ts) + 2 == W,
        "the constructor requires as many arguments as vector elements");
}

template <typename T, size_t W>
Vec<T, W>::Vec(const vec_bool_t& b) noexcept
{
    constexpr int nregs = Vec<T, W>::n_regs();
    #pragma unroll
    for (auto idx = 0; idx < nregs; idx++) {
        this->reg(idx) = b.reg(idx);
    }
}

template <typename T, size_t W>
template <typename... Regs>
Vec<T, W>::Vec(register_t arg, Regs... others) noexcept
    : base_t({arg, others...})
{
    static_assert(sizeof...(Regs) + 1 <= self_t::n_regs(),
        "the constructor requires not-beyond number of registers");
}

template <typename T, size_t W>
template <size_t... Ws>
Vec<T, W>::Vec(const Vec<T, Ws>&... vecs) noexcept
    : self_t(vecs.reg()...)
{
    // TODO: validation
}

template <typename T, size_t W>
template <typename U>
void Vec<T, W>::store_aligned(U* mem) const noexcept
{
    assert(is_aligned(mem, A::alignment())
        && "store location is not properly aligned");
    kernel::store_aligned<T, W>((T*)mem, *this, A{});
}

template <typename T, size_t W>
template <typename U>
void Vec<T, W>::store_unaligned(U* mem) const noexcept
{
    kernel::store_unaligned<T, W>((T*)mem, *this, A{});
}

template <typename T, size_t W>
template <typename U>
Vec<T, W> Vec<T, W>::load_aligned(const U* mem) noexcept
{
    assert(is_aligned(mem, A::alignment())
        && "loaded location is not properly aligned");
    return kernel::load_aligned<T, W>((const T*)mem, A{});
}

template <typename T, size_t W>
template <typename U>
Vec<T, W> Vec<T, W>::load_unaligned(const U* mem) noexcept
{
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
template <typename... Regs>
VecBool<T, W>::VecBool(register_t arg, Regs... others) noexcept
    : base_t({arg, others...})
{
}

template <typename T, size_t W>
template <typename... V>
typename VecBool<T, W>::register_t
VecBool<T, W>::make_register(detail::index_sequence<>, V... v) noexcept
{
    return kernel::set<T, W>(
        static_cast<T>(v ? bits::ones<T>() : bits::zeros<T>())..., A{}).reg(0);
}

template <typename T, size_t W>
VecBool<T, W>::VecBool(bool val) noexcept
{
    auto regval = make_register(detail::make_index_sequence<size() - 1>(), val);
    constexpr int nregs = VecBool<T, W>::n_regs();
    #pragma unroll
    for (auto idx = 0; idx < nregs; idx++) {
        this->reg(idx) = regval;
    }
}

template <typename T, size_t W>
template <typename... Ts>
VecBool<T, W>::VecBool(bool val0, bool val1, Ts... vals) noexcept
{
    static_assert(sizeof...(Ts) + 2 == W,
        "constructor requires as many as arguments as vector elements");
    auto vec = kernel::set<T, W>(
                    val0 ? bits::ones<T>() : bits::zeros<T>(),
                    val1 ? bits::ones<T>() : bits::zeros<T>(),
     static_cast<T>(vals ? bits::ones<T>() : bits::zeros<T>())..., A{});
    constexpr int nregs = self_t::n_regs();
    #pragma unroll
    for (auto idx = 0; idx < nregs; idx++) {
        this->reg(idx) = vec.reg(idx);
    }
}

template <typename T, size_t W>
void VecBool<T, W>::store_aligned(bool* mem) const noexcept
{
    #pragma unroll
    for (auto i = 0; i < size(); i++) {
        mem[i] = bits::at_msb(this->get(i));
    }
}

template <typename T, size_t W>
void VecBool<T, W>::store_unaligned(bool* mem) const noexcept
{
    store_aligned(mem);
}

template <typename T, size_t W>
VecBool<T, W> VecBool<T, W>::load_aligned(const bool* mem) noexcept
{
    Vec<T, W> vec;
    #pragma unroll
    for (auto i = 0; i < size(); i++) {
        vec[i] = mem[i] ? bits::ones<T>() : bits::zeros<T>();
    }
    VecBool<T, W> ret;
    constexpr int nregs = self_t::n_regs();
    #pragma unroll
    for (auto idx = 0; idx < nregs; idx++) {
        ret.reg(idx) = vec.reg(idx);
    }
    return ret;
}

template <typename T, size_t W>
VecBool<T, W> VecBool<T, W>::load_unaligned(const bool* mem) noexcept
{
    return load_aligned(mem);
}

template <typename T, size_t W>
VecBool<T, W> VecBool<T, W>::operator ==(const VecBool<T, W>& other) const noexcept
{
    return kernel::eq<T, W>(*this, other, A{});
}
template <typename T, size_t W>
VecBool<T, W> VecBool<T, W>::operator !=(const VecBool<T, W>& other) const noexcept
{
    return kernel::ne<T, W>(*this, other, A{});
}
template <typename T, size_t W>
VecBool<T, W> VecBool<T, W>::operator ~() const noexcept
{
    return kernel::bitwise_not<T, W>(*this, A{});
}

template <typename T, size_t W>
VecBool<T, W> VecBool<T, W>::operator &(const VecBool& other) const noexcept
{
    return kernel::bitwise_and<A>(*this, other, A{});
}
template <typename T, size_t W>
VecBool<T, W> VecBool<T, W>::operator |(const VecBool<T, W>& other) const noexcept
{
    return kernel::bitwise_or<T, W>(*this, other, A{});
}
template <typename T, size_t W>
VecBool<T, W> VecBool<T, W>::operator ^(const VecBool<T, W>& other) const noexcept
{
    return kernel::bitwise_xor<T, W>(*this, other, A{});
}
}  // namespace simd
