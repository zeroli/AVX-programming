#pragma once

#include "simd/types/vec.h"

#include <type_traits>
#include <cstddef>

namespace simd {
namespace kernel {
namespace detail {
template <typename F, typename Arch, typename T>
Vec<T, Arch> apply(F&& func, const Vec<T, Arch>& self, const Vec<T, Arch>& other) noexcept
{
    constexpr size_t size = Vec<T, Arch>::size();
    alignas(Arch::alignment()) T self_buffer[size];
    alignas(Arch::alignment()) T other_buffer[size];
    self.store_aligned(&self_buffer[0]);
    other.store_aligned(&other_buffer[0]);
    for (auto i = 0u; i < size; i++) {
        self_buffer[i] = func(self_buffer[i], other_buffer[i]);
    }
    return Vec<T, Arch>::load_aligned(self_buffer);
}

template <typename U, typename F, typename Arch, typename T>
Vec<T, Arch> apply_transform(F&& func, const Vec<T, Arch>& self) noexcept
{
    static_assert(Vec<T, Arch>::size() == Vec<U, Arch>::size(),
        "source and destination sizes must match");
    constexpr size_t size = Vec<T, Arch>::size();
    alignas(Arch::alignment()) T self_buffer[size];
    alignas(Arch::alignment()) T other_buffer[size];
    self.store_aligned(&self_buffer[0]);
    for (auto i = 0u; i < src_size; i++) {
        other_buffer[i] = func(self_buffer[i]);
    }
    return Vec<T, Arch>::load_aligned(other_buffer);
}
}  // namespace detail
}  // namespace kernel
}  // namespace simd
