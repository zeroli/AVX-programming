#pragma once

#include "simd/api/detail.h"

namespace simd {
DEFINE_API_BINARY_OP(max);
DEFINE_API_BINARY_OP(min);

DEFINE_API_BINARY_OP(copysign);

/// Computes an estimate of the inverse square root of the vector x
/// this doesn't return the same result as the equivalent scalar operations,
/// trading accuracy for speed
//DEFINE_API_UNARY_OP(rsqrt);
DEFINE_API_UNARY_OP(sign);
DEFINE_API_UNARY_OP(bitofsign);

template <typename T, size_t W>
bool all_of(const VecBool<T, W>& x) noexcept
{
    using A = typename VecBool<T, W>::arch_t;
    return kernel::all_of<T, W>(x, A{});
}

template <typename T, size_t W>
bool any_of(const VecBool<T, W>& x) noexcept
{
    using A = typename VecBool<T, W>::arch_t;
    return kernel::any_of<T, W>(x, A{});
}

template <typename T, size_t W>
bool none_of(const VecBool<T, W>& x) noexcept
{
    using A = typename VecBool<T, W>::arch_t;
    return kernel::none_of<T, W>(x, A{});
}

template <typename T, size_t W>
bool some_of(const VecBool<T, W>& x) noexcept
{
    using A = typename VecBool<T, W>::arch_t;
    return kernel::some_of<T, W>(x, A{});
}

template <typename T, size_t W>
int popcount(const VecBool<T, W>& x) noexcept
{
    using A = typename VecBool<T, W>::arch_t;
    return kernel::popcount<T, W>(x, A{});
}

template <typename T, size_t W>
int find_first_set(const VecBool<T, W>& x) noexcept
{
    using A = typename VecBool<T, W>::arch_t;
    return kernel::find_first_set<T, W>(x, A{});
}

template <typename T, size_t W>
int find_last_set(const VecBool<T, W>& x) noexcept
{
    using A = typename VecBool<T, W>::arch_t;
    return kernel::find_last_set<T, W>(x, A{});
}

template <typename T, size_t W>
Vec<T, W> select(const VecBool<T, W>& cond, const Vec<T, W>& x, const Vec<T, W>& y) noexcept
{
    using A = typename VecBool<T, W>::arch_t;
    return kernel::select<T, W>(cond, x, y, A{});
}

/// reduction
template <typename T, size_t W, typename F>
T reduce(F&& f, const Vec<T, W>& x) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::reduce<T, W, F>(std::forward<F>(f), x, A{});
}

template <typename T, size_t W>
T reduce_sum(const Vec<T, W>& x) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::reduce_sum<T, W>(x, A{});
}

template <typename T, size_t W>
T reduce_max(const Vec<T, W>& x) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::reduce_max<T, W>(x, A{});
}

template <typename T, size_t W>
T reduce_min(const Vec<T, W>& x) noexcept
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::reduce_min<T, W>(x, A{});
}

/// permute
// template <typename T, size_t W, typename U>
// Vec<T, W> permute(const Vec<T, W>& x, const Vec<U, W>& index) noexcept
// {
//     using A = typename Vec<T, W>::arch_t;
//     return kernel::permute<T, W>(x, index, A{});
// }

}  // namespace simd
