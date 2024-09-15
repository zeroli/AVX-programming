#pragma once

#include <cstddef>

namespace simd {
struct aligned_mode { };
struct unaligned_mode { };

inline bool is_aligned(const void* ptr, size_t alignment)
{
    return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
}

template <size_t Alignment>
inline bool is_aligned(void* ptr)
{
    return is_aligned(ptr, Alignment);
}

template <typename Arch>
inline bool is_aligned(const void* ptr)
{
    return is_aligned(ptr, Arch::alignment());
}
}  // namespace simd
