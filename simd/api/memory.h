#pragma once

namespace simd {

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

}  // namespace simd
