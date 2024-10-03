#pragma once

#include "simd/arch/isa.h"

namespace simd {
namespace types {
template <typename T, size_t W>
SIMD_INLINE
Vec<T, W>& integral_only_ops<T, W>::operator %=(const Vec<T, W>& rhs) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return ref_vec() = kernel::mod<T, W>(ref_vec(), rhs, A{});
}

template <typename T, size_t W>
SIMD_INLINE
Vec<T, W>& integral_only_ops<T, W>::operator >>=(int32_t rhs) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return ref_vec() = kernel::bitwise_rshift<T, W>(ref_vec(), rhs, A{});
}

template <typename T, size_t W>
SIMD_INLINE
Vec<T, W>& integral_only_ops<T, W>::operator >>=(const Vec<T, W>& rhs) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return ref_vec() = kernel::bitwise_rshift<T, W>(ref_vec(), rhs, A{});
}

template <typename T, size_t W>
SIMD_INLINE
Vec<T, W>& integral_only_ops<T, W>::operator <<=(int32_t rhs) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return ref_vec() = kernel::bitwise_lshift<T, W>(ref_vec(), rhs, A{});
}

template <typename T, size_t W>
SIMD_INLINE
Vec<T, W>& integral_only_ops<T, W>::operator <<=(const Vec<T, W>& rhs) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return ref_vec() = kernel::bitwise_lshift<T, W>(ref_vec(), rhs, A{});
}
}  // namespace types

template <typename T, size_t W>
SIMD_INLINE
Vec<T, W>::Vec(T val) noexcept
    : self_t(kernel::broadcast<T, W>(val, A{}))
{
}

template <typename T, size_t W>
template <typename... Ts>
SIMD_INLINE
Vec<T, W>::Vec(T val0, T val1, Ts... vals) noexcept
    : self_t(kernel::set<T, W>(A{}, val0, val1, static_cast<T>(vals)...))
{
    static_assert(sizeof...(Ts) + 2 == W,
        "the constructor requires as many arguments as vector elements");
}

template <typename T, size_t W>
SIMD_INLINE
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
SIMD_INLINE
Vec<T, W>::Vec(const register_t& arg, Regs&&... others) noexcept
    : base_t({arg, others...})
{
    static_assert(sizeof...(Regs) + 1 <= self_t::n_regs(),
        "the constructor requires not-beyond number of registers");
}

template <typename T, size_t W>
template <size_t... Ws>
SIMD_INLINE
Vec<T, W>::Vec(const Vec<T, Ws>&... vecs) noexcept
    : self_t(vecs.reg()...)
{
    // TODO: validation
}

template <typename T, size_t W>
template <typename G>
SIMD_INLINE
void Vec<T, W>::gen_values(G&& generator) noexcept
{
    alignas(A::alignment()) T buf[W];
    #pragma unroll
    for (int i = 0; i < W; i++) {
        buf[i] = (T)generator(i);
    }
    load(buf);
}

template <typename T, size_t W>
SIMD_INLINE
void Vec<T, W>::clear() noexcept
{
    *this = kernel::setzero<T, W>(A{});
}

template <typename T, size_t W>
template <typename U>
SIMD_INLINE
void Vec<T, W>::store_aligned(U* mem) const noexcept
{
    assert(is_aligned(mem, A::alignment())
        && "store location is not properly aligned");
    kernel::store_aligned<T, W>((T*)mem, *this, A{});
}

template <typename T, size_t W>
template <typename U>
SIMD_INLINE
void Vec<T, W>::store_unaligned(U* mem) const noexcept
{
    kernel::store_unaligned<T, W>((T*)mem, *this, A{});
}

template <typename T, size_t W>
template <typename U>
SIMD_INLINE
Vec<T, W> Vec<T, W>::load_aligned(const U* mem) noexcept
{
    assert(is_aligned(mem, A::alignment())
        && "loaded location is not properly aligned");
    return kernel::load_aligned<T, W>((const T*)mem, A{});
}

template <typename T, size_t W>
template <typename U>
SIMD_INLINE
Vec<T, W> Vec<T, W>::load_unaligned(const U* mem) noexcept
{
    return kernel::load_unaligned<T, W>((const T*)mem, A{});
}

template <typename T, size_t W>
template <typename U, typename V>
SIMD_INLINE
Vec<T, W> Vec<T, W>::gather(const U* src, const Vec<V, W>& index) noexcept
{
    static_assert(std::is_convertible<U, T>::value,
        "Cannot convert from src type to scalar type T");
    return kernel::gather<T, W>(src, index, A{});
}

template <typename T, size_t W>
template <typename U, typename V>
SIMD_INLINE
void Vec<T, W>::scatter(U* dst, const Vec<V, W>& index) const noexcept
{
    static_assert(std::is_convertible<T, U>::Vec,
        "Cannot convert from this type T to dst type");
    return kernel::scatter<T, W>(*this, dst, index, A{});
}

template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> Vec<T, W>::operator ~() const noexcept
{
    return kernel::bitwise_not<T, W>(*this, A{});
}

template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> Vec<T, W>::operator -() const noexcept
{
    return kernel::neg<T, W>(*this, A{});
}

/// VecBool
template <typename T, size_t W>
template <typename... Regs>
SIMD_INLINE
VecBool<T, W>::VecBool(register_t arg, Regs... others) noexcept
    : base_t({arg, others...})
{
}

template <typename T, size_t W>
template <typename... V>
SIMD_INLINE
typename VecBool<T, W>::register_t
VecBool<T, W>::make_register(detail::index_sequence<>, V... v) noexcept
{
    return kernel::set<T, self_t::reg_lanes()>(A{},
        static_cast<T>(v ? bits::ones<T>() : bits::zeros<T>())...).reg(0);
}

template <typename T, size_t W>
SIMD_INLINE
VecBool<T, W>::VecBool(bool val) noexcept
{
    auto regval = make_register(detail::make_index_sequence<self_t::reg_lanes() - 1>(), val);
    constexpr auto nregs = VecBool<T, W>::n_regs();
    #pragma unroll
    for (auto idx = 0; idx < nregs; idx++) {
        this->reg(idx) = regval;
    }
}

template <typename T, size_t W>
template <typename... Ts>
SIMD_INLINE
VecBool<T, W>::VecBool(bool val0, bool val1, Ts... vals) noexcept
{
    static_assert(sizeof...(Ts) + 2 == W,
        "constructor requires as many as arguments as vector elements");
    auto vec = kernel::set<T, W>(A{},
                    val0 ? bits::ones<T>() : bits::zeros<T>(),
                    val1 ? bits::ones<T>() : bits::zeros<T>(),
     static_cast<T>(vals ? bits::ones<T>() : bits::zeros<T>())...);
    constexpr int nregs = self_t::n_regs();
    #pragma unroll
    for (auto idx = 0; idx < nregs; idx++) {
        this->reg(idx) = vec.reg(idx);
    }
}

template <typename T, size_t W>
SIMD_INLINE
void VecBool<T, W>::store_aligned(bool* mem) const noexcept
{
    #pragma unroll
    for (auto i = 0; i < size(); i++) {
        mem[i] = bits::at_msb(this->get(i));
    }
}

template <typename T, size_t W>
SIMD_INLINE
void VecBool<T, W>::store_unaligned(bool* mem) const noexcept
{
    store_aligned(mem);
}

template <typename T, size_t W>
SIMD_INLINE
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
SIMD_INLINE
VecBool<T, W> VecBool<T, W>::load_unaligned(const bool* mem) noexcept
{
    return load_aligned(mem);
}

template <typename T, size_t W>
SIMD_INLINE
uint64_t VecBool<T, W>::to_mask() const noexcept
{
    return kernel::to_mask(*this, A{});
}

template <typename T, size_t W>
SIMD_INLINE
VecBool<T, W> VecBool<T, W>::from_mask(uint64_t mask) noexcept
{
    return kernel::from_mask<T, W>(mask, A{});
}

template <typename T, size_t W>
SIMD_INLINE
VecBool<T, W> VecBool<T, W>::operator ==(const VecBool<T, W>& other) const noexcept
{
    return kernel::eq<T, W>(*this, other, A{});
}
template <typename T, size_t W>
SIMD_INLINE
VecBool<T, W> VecBool<T, W>::operator !=(const VecBool<T, W>& other) const noexcept
{
    return kernel::ne<T, W>(*this, other, A{});
}
template <typename T, size_t W>
SIMD_INLINE
VecBool<T, W> VecBool<T, W>::operator ~() const noexcept
{
    return kernel::bitwise_not<T, W>(*this, A{});
}

template <typename T, size_t W>
SIMD_INLINE
VecBool<T, W> VecBool<T, W>::operator &(const VecBool& other) const noexcept
{
    return kernel::bitwise_and<A>(*this, other, A{});
}
template <typename T, size_t W>
SIMD_INLINE
VecBool<T, W> VecBool<T, W>::operator |(const VecBool<T, W>& other) const noexcept
{
    return kernel::bitwise_or<T, W>(*this, other, A{});
}
template <typename T, size_t W>
SIMD_INLINE
VecBool<T, W> VecBool<T, W>::operator ^(const VecBool<T, W>& other) const noexcept
{
    return kernel::bitwise_xor<T, W>(*this, other, A{});
}
}  // namespace simd
