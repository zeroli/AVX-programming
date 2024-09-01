#pragma once

#include "simd/arch/isa.h"

namespace simd {
template <typename T, size_t W>
Vec<T, W>::Vec(T val) noexcept
    : self_t(kernel::broadcast<W>(val, A{}))
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

}  // namespace simd
