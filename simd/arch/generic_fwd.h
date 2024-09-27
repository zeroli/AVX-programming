#pragma once

namespace simd {
namespace kernel {
template <typename T, size_t W>
T hadd(const Vec<T, W>& x, requires_arch<Generic>) noexcept;

}  // namespace kernel
}  // namespace simd
