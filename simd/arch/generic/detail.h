#pragma once

#include <algorithm>
#include <numeric>
#include <type_traits>
#include <cstddef>

namespace simd { namespace kernel { namespace generic {
namespace detail {
template <typename VT, typename UT, typename F>
void apply(UT& y, const VT& x, F&& f) noexcept
{
    std::transform(std::begin(x), std::end(x), std::begin(y), f);
}

template <typename VT, typename UT, typename F>
void apply(UT& z, const VT& x, const VT& y, F&& f) noexcept
{
    std::transform(std::begin(x), std::end(x), std::begin(y), std::begin(z), f);
}

}  // namespace detail
} } } // namespace simd::kernel::generic
