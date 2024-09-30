#pragma once

#include <algorithm>
#include <numeric>

namespace simd { namespace kernel { namespace generic {
using namespace types;

template <typename T, size_t W>
struct all_of<T, W>
{
    SIMD_INLINE
    static bool apply(const VecBool<T, W>& x) noexcept
    {
        bool ret = true;
        constexpr auto nregs = VecBool<T, W>::n_regs();
        #pragma unroll
        for (auto i = 0; i < W; i++) {
            ret = ret && (true == bits::at_msb(x[i]));
        }
        return ret;
    }
};

template <typename T, size_t W>
struct any_of<T, W>
{
    SIMD_INLINE
    static bool apply(const VecBool<T, W>& x) noexcept
    {
        bool ret = false;
        constexpr auto nregs = VecBool<T, W>::n_regs();
        #pragma unroll
        for (auto i = 0; i < W; i++) {
            ret = ret || (true == bits::at_msb(x[i]));
        }
        return ret;
    }
};

/// none_of (!any_of)
template <typename T, size_t W>
struct none_of<T, W>
{
    SIMD_INLINE
    static bool apply(const VecBool<T, W>& x) noexcept
    {
        using A = typename VecBool<T, W>::arch_t;
        bool ret = !kernel::any_of<T, W>(x, A{});
        return ret;
    }
};

/// some_of (any_of && !all_of)
template <typename T, size_t W>
struct some_of<T, W>
{
    SIMD_INLINE
    static bool apply(const VecBool<T, W>& x) noexcept
    {
        using A = typename VecBool<T, W>::arch_t;
        bool ret = kernel::any_of<T, W>(x, A{})
               && !kernel::all_of<T, W>(x, A{});
        return ret;
    }
};

template <typename T, size_t W>
struct hadd<T, W>
{
    SIMD_INLINE
    static T apply(const Vec<T, W>& x) noexcept
    {
        T ret{};
        constexpr auto nregs = Vec<T, W>::n_regs();
        #pragma unroll
        for (auto i = 0u; i < W; i++) {
            ret += x[i];
        }
        return ret;
    }
};
} } } // namespace simd::kernel::generic
