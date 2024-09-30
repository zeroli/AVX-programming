#pragma once

#include "simd/arch/generic/detail.h"
#include "simd/types/generic_arch.h"
#include "simd/types/traits.h"

#include <limits>
#include <type_traits>
#include <complex>

namespace simd {
namespace kernel {
namespace generic {

using namespace types;

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
}  // namespace generic
}  // namespace kernel
}  // namespace simd
