#pragma

#include "simd/arch/isa.h"
#include "simd/types/vec.h"

namespace simd {
namespace ops {
template <typename T, size_t W>
Vec<T, W> add(const Vec<T, W>& lhs, const Vec<T, W>& rhs)
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::add<T, W>(lhs, rhs, A{});
}

template <typename T, size_t W>
Vec<T, W> sub(const Vec<T, W>& lhs, const Vec<T, W>& rhs)
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::sub<T, W>(lhs, rhs, A{});
}

#if 0
template <typename T, size_t W>
Vec<T, W> mul(const Vec<T, W>& lhs, const Vec<T, W>& rhs)
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::mul<W>(lhs, rhs, A{});
}
template <typename T, size_t W>
Vec<T, W> div(const Vec<T, W>& lhs, const Vec<T, W>& rhs)
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::div<W>(lhs, rhs, A{});
}
template <typename T, size_t W>
Vec<T, W> bitwise_and(const Vec<T, W>& lhs, const Vec<T, W>& rhs)
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::bitwise_and<W>(lhs, rhs, A{});
}
template <typename T, size_t W>
Vec<T, W> bitwise_or(const Vec<T, W>& lhs, const Vec<T, W>& rhs)
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::bitwise_or<W>(lhs, rhs, A{});
}
template <typename T, size_t W>
Vec<T, W> bitwise_xor(const Vec<T, W>& lhs, const Vec<T, W>& rhs)
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::bitwise_xor<W>(lhs, rhs, A{});
}
template <typename T, size_t W>
Vec<T, W> logical_and(const Vec<T, W>& lhs, const Vec<T, W>& rhs)
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::logical_and<W>(lhs, rhs, A{});
}
template <typename T, size_t W>
Vec<T, W> logical_or(const Vec<T, W>& lhs, const Vec<T, W>& rhs)
{
    using A = typename Vec<T, W>::arch_t;
    return kernel::logical_or<W>(lhs, rhs, A{});
}
#endif
}  // namespace ops
}  // namespace simd
