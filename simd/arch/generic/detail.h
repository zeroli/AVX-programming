#pragma once

#include <algorithm>
#include <numeric>
#include <type_traits>
#include <cstddef>

namespace simd {
namespace kernel {
namespace generic {

namespace detail {
template <typename T, size_t W, typename F>
Vec<T, W> apply(const Vec<T, W>& x, F&& f) noexcept
{
    Vec<T, W> ret;
    std::transform(std::begin(x), std::end(x), std::begin(ret), f);
    return ret;
}

template <typename T, size_t W, typename F>
Vec<T, W> apply(const Vec<T, W>& x, const Vec<T, W>& y, F&& f) noexcept
{
    Vec<T, W> ret;
    std::transform(std::begin(x), std::end(x), std::begin(y), std::begin(ret), f);
    return ret;
}
}  // namespace detail
}  // namespace generic
}  // namespace kernel
}  // namespace simd
