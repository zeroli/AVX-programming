#pragma once

#define DEFINE_API_BINARY_OP(OP) \
template <typename T, size_t W> \
Vec<T, W> OP(const Vec<T, W>& x, const Vec<T, W>& y) noexcept \
{ \
    using A = typename Vec<T, W>::arch_t; \
    return kernel::OP<T, W>(x, y, A{}); \
} \
template <typename T, size_t W> \
Vec<T, W> OP(const Vec<T, W>& x, T y) noexcept \
{ \
    return OP(x, Vec<T, W>(y)); \
} \
template <typename T, size_t W> \
Vec<T, W> OP(T x, const Vec<T, W>& y) noexcept \
{ \
    return OP(Vec<T, W>(x), y); \
} \
///

#define DEFINE_API_UNARY_OP(OP) \
template <typename T, size_t W> \
Vec<T, W> OP(const Vec<T, W>& x) noexcept \
{ \
    using A = typename Vec<T, W>::arch_t; \
    return kernel::OP<T, W>(x, A{}); \
} \
///
