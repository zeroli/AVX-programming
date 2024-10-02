#pragma once

#include "simd/api/detail.h"

namespace simd {

template <typename T, size_t W>
Vec<T, W> broadcast(T v) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::broadcast<T, W>(v, A{});
}

template <typename T, size_t W>
Vec<T, W> setzero() noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::setzero<T, W>(A{});
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

template <size_t W, typename T>
Vec<T, W> load(const T* mem) noexcept
{
    return load_aligned<T, W>(mem);
}

template <size_t W, typename T>
Vec<T, W> loadu(const T* mem) noexcept
{
    return load_unaligned<T, W>(mem);
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
void store(T* mem, const Vec<T, W>& x) noexcept
{
    store_aligned<T, W>(mem, x);
}

template <typename T, size_t W>
void storeu(T* mem, const Vec<T, W>& x) noexcept
{
    store_unaligned<T, W>(mem, x);
}

/// set values sequentially from lower to higher
/// vec[0] = v0, vec[1] = v1, vec[2] = v2, ...
/// NOTE: the order is opposite from sse/avx intrinsic: set
template <typename T, size_t W, typename... Ts>
Vec<T, W> set(T v0, T v1, Ts... vals) noexcept
{
    static_assert(sizeof...(Ts) + 2 == W);
    using A = typename Vec<T, W>::arch_t;
    return kernel::set<T, W>(A{}, v0, v1, static_cast<T>(vals)...);
}

template <typename T, size_t W, typename U, typename V>
Vec<T, W> gather(const U* mem, const Vec<V, W>& index) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::gather<T, W, U, V>(mem, index, A{});
}

template <typename T, size_t W, typename V>
Vec<T, W> gather(const T* mem, const Vec<V, W>& index) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::gather<T, W, T, V>(mem, index, A{});
}

template <typename T, size_t W, typename U, typename V>
void scatter(const Vec<T, W>& x, U* mem, const Vec<V, W>& index) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::scatter<T, W, U, V>(x, mem, index, A{});
}

template <typename T, size_t W>
uint64_t to_mask(const VecBool<T, W>& x) noexcept
{
    using A = typename VecBool<T, W>::arch_t;
    return kernel::to_mask(x, A{});
}

template <typename T, size_t W>
VecBool<T, W> from_mask(uint64_t x) noexcept
{
    using A = typename VecBool<T, W>::arch_t;
    return kernel::from_mask<T, W>(x, A{});
}

}  // namespace simd
