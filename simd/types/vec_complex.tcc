#pragma once

#include "simd/arch/isa.h"

namespace simd {
template <typename T, size_t W>
SIMD_INLINE
Vec<std::complex<T>, W>::Vec(const value_type& val) noexcept
    : real_(val.real()), imag_(val.imag())
{
}

template <typename T, size_t W>
SIMD_INLINE
Vec<std::complex<T>, W>::Vec(const real_vec_t& real, const imag_vec_t& imag) noexcept
    : real_(real), imag_(imag)
{
}

template <typename T, size_t W>
SIMD_INLINE
Vec<std::complex<T>, W>::Vec(const real_vec_t& real) noexcept
    : real_(real), imag_(0)
{
}

template <typename T, size_t W>
SIMD_INLINE
Vec<std::complex<T>, W>::Vec(const T& val) noexcept
    : self_t(value_type(val), A{})
{
}

template <typename T, size_t W>
template <typename... Ts>
SIMD_INLINE
Vec<std::complex<T>, W>::Vec(const value_type& val0, const value_type& val1, Ts... vals) noexcept
    //: self_t(kernel::set<value_type, W>(A{}, val0, val1, static_cast<value_type>(vals)...))
{
    static_assert(sizeof...(Ts) + 2 == W,
        "the constructor requires as many arguments as vector elements");
}

template <typename T, size_t W>
template <typename... Regs>
SIMD_INLINE
Vec<std::complex<T>, W>::Vec(const register_t& arg, Regs&&... others) noexcept
{
    assert(0 && "TODO");
    static_assert(sizeof...(Regs) + 1 <= self_t::n_regs(),
        "the constructor requires not-beyond number of registers");
}

template <typename T, size_t W>
template <typename G>
SIMD_INLINE
void Vec<std::complex<T>, W>::gen_values(G&& generator) noexcept
{
    alignas(A::alignment()) value_type buf[W];
    #pragma unroll
    for (int i = 0; i < W; i++) {
        buf[i] = (T)generator(i);
    }
    load(buf);
}

template <typename T, size_t W>
SIMD_INLINE
Vec<std::complex<T>, W>::Vec(const vec_bool_t& b) noexcept
{
    // TODO:
}

/// set all elements to (0,0)
template <typename T, size_t W>
SIMD_INLINE
void Vec<std::complex<T>, W>::clear() noexcept
{
    *this = kernel::setzero<value_type, W>(A{});
}

template <typename T, size_t W>
SIMD_INLINE /*static*/
Vec<std::complex<T>, W> Vec<std::complex<T>, W>::load_aligned(const T* real, const T* imag) noexcept
{
    Vec<std::complex<T>, W> ret;
    real ? ret.real().load(real) : ret.real().clear();
    imag ? ret.imag().load(imag) : ret.imag().clear();
    return ret;
}

template <typename T, size_t W>
SIMD_INLINE /*static*/
Vec<std::complex<T>, W> Vec<std::complex<T>, W>::load_unaligned(const T* real, const T* imag) noexcept
{
    Vec<std::complex<T>, W> ret;
    real ? ret.real().loadu(real) : ret.real().clear();
    imag ? ret.imag().loadu(imag) : ret.imag().clear();
    return ret;
}

template <typename T, size_t W>
template <typename U>
SIMD_INLINE /*static*/
Vec<std::complex<T>, W> Vec<std::complex<T>, W>::load(const U* mem, aligned_mode) noexcept
{
    assert(is_aligned(mem, alignment())
        && "loaded location is not properly aligned");
    return kernel::load_aligned<value_type, W>((const value_type*)mem, A{});
}

template <typename T, size_t W>
template <typename U>
SIMD_INLINE /*static*/
Vec<std::complex<T>, W> Vec<std::complex<T>, W>::load(const U* mem, unaligned_mode) noexcept
{
    return kernel::load_unaligned<value_type, W>((const value_type*)mem, A{});
}

template <typename T, size_t W>
SIMD_INLINE
void Vec<std::complex<T>, W>::store_aligned(T* real, T* imag) const noexcept
{
    real ? real().store_aligned(real) : (void)0;
    imag ? imag().store_aligned(imag) : (void)0;
}

template <typename T, size_t W>
SIMD_INLINE
void Vec<std::complex<T>, W>::store_unaligned(T* real, T* imag) const noexcept
{
    real ? real().store_unaligned(real) : (void)0;
    imag ? imag().store_unaligned(imag) : (void)0;
}

template <typename T, size_t W>
template <typename U>
SIMD_INLINE
void Vec<std::complex<T>, W>::store(U* mem, aligned_mode) const noexcept
{
    assert(is_aligned(mem, alignment())
        && "store location is not properly aligned");
    kernel::store_aligned<value_type, W>((value_type*)mem, *this, A{});
}

template <typename T, size_t W>
template <typename U>
SIMD_INLINE
void Vec<std::complex<T>, W>::store(U* mem, unaligned_mode) const noexcept
{
    kernel::store_unaligned<value_type, W>((value_type*)mem, *this, A{});
}

template <typename T, size_t W>
SIMD_INLINE
Vec<std::complex<T>, W> Vec<std::complex<T>, W>::operator -() const noexcept
{
    return kernel::neg<value_type, W>(*this, A{});
}

}  // namespace simd
