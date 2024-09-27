#pragma once

#include "simd/api/detail.h"

namespace simd {
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
