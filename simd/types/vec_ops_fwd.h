#pragma once

namespace simd {
template <typename T, size_t W>
class Vec;
template <typename T, size_t W>
class VecBool;

// These functions are forwarded declared here so that they can be used
// by friend functions with Vec<T, W>. Their implementation must appear
// only once the kernel implementations have been included.
namespace ops {
template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> add(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept;

template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> sub(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept;

template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> mul(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept;

template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> div(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept;

template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> bitwise_and(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept;

template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> bitwise_or(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept;

template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> bitwise_xor(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept;

template <typename T, size_t W>
SIMD_INLINE
VecBool<T, W> eq(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept;

template <typename T, size_t W>
SIMD_INLINE
VecBool<T, W> ne(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept;

template <typename T, size_t W>
SIMD_INLINE
VecBool<T, W> gt(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept;

template <typename T, size_t W>
SIMD_INLINE
VecBool<T, W> ge(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept;

template <typename T, size_t W>
SIMD_INLINE
VecBool<T, W> lt(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept;

template <typename T, size_t W>
SIMD_INLINE
VecBool<T, W> le(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept;

}  // namepace ops
}  // namespace simd
