#pragma once

#include "simd/types/vec.h"

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
    static bool apply(const VecBool<T, W>& x) noexcept
    {
        using A = typename VecBool<T, W>::arch_t;
        bool ret = kernel::any_of<T, W>(x, A{})
               && !kernel::all_of<T, W>(x, A{});
        return ret;
    }
};
}  // namespace generic
}  // namespace kernel
}  // namespace simd
