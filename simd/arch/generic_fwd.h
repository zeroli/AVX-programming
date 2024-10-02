#pragma once

namespace simd {
namespace kernel {
#define DECLARE_GENERIC_UNARY_OP(OP) \
template <typename T, size_t W> \
SIMD_INLINE \
Vec<T, W> OP(const Vec<T, W>& lhs, requires_arch<Generic>) noexcept; \
template <typename T, size_t W> \
SIMD_INLINE \
VecBool<T, W> OP(const VecBool<T, W>& lhs, requires_arch<Generic>) noexcept \
///###

#define DECLARE_GENERIC_BINARY_OP(OP) \
template <typename T, size_t W> \
SIMD_INLINE \
Vec<T, W> OP(const Vec<T, W>& lhs, const Vec<T, W>& rhs, requires_arch<Generic>) noexcept \
///###

#define DECLARE_GENERIC_BINARY_CMP_OP(OP) \
template <typename T, size_t W> \
SIMD_INLINE \
VecBool<T, W> OP(const Vec<T, W>& lhs, const Vec<T, W>& rhs, requires_arch<Generic>) noexcept \
///###

DECLARE_GENERIC_UNARY_OP(sign);
DECLARE_GENERIC_UNARY_OP(bitofsign);

DECLARE_GENERIC_BINARY_OP(add);
DECLARE_GENERIC_BINARY_OP(sub);
DECLARE_GENERIC_BINARY_OP(mul);
DECLARE_GENERIC_BINARY_OP(div);
DECLARE_GENERIC_BINARY_OP(mod);

DECLARE_GENERIC_BINARY_OP(bitwise_and);
DECLARE_GENERIC_BINARY_OP(bitwise_or);
DECLARE_GENERIC_BINARY_OP(bitwise_xor);
DECLARE_GENERIC_BINARY_OP(bitwise_lshift);
DECLARE_GENERIC_BINARY_OP(bitwise_rshift);

DECLARE_GENERIC_BINARY_CMP_OP(eq);
DECLARE_GENERIC_BINARY_CMP_OP(ne);
DECLARE_GENERIC_BINARY_CMP_OP(gt);
DECLARE_GENERIC_BINARY_CMP_OP(ge);
DECLARE_GENERIC_BINARY_CMP_OP(lt);
DECLARE_GENERIC_BINARY_CMP_OP(le);

template <typename T, size_t W>
Vec<T, W> fmadd(const Vec<T, W>& x, const Vec<T, W>& y, const Vec<T, W>& z) noexcept;

template <typename T, size_t W>
Vec<T, W> fmsub(const Vec<T, W>& x, const Vec<T, W>& y, const Vec<T, W>& z) noexcept;

template <typename T, size_t W>
Vec<T, W> fnmadd(const Vec<T, W>& x, const Vec<T, W>& y, const Vec<T, W>& z) noexcept;

template <typename T, size_t W>
Vec<T, W> fnmsub(const Vec<T, W>& x, const Vec<T, W>& y, const Vec<T, W>& z) noexcept;

template <typename T, size_t W>
SIMD_INLINE
VecBool<T, W> bitwise_not(const VecBool<T, W>& lhs, requires_arch<Generic>) noexcept;

template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> bitwise_andnot(const VecBool<T, W>& lhs, const Vec<T, W>& rhs, requires_arch<Generic>) noexcept;

template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> bitwise_lshift(const Vec<T, W>& lhs, int32_t rhs, requires_arch<Generic>) noexcept;

template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> bitwise_rshift(const Vec<T, W>& lhs, int32_t rhs, requires_arch<Generic>) noexcept;

template <typename T, size_t W>
SIMD_INLINE
T hadd(const Vec<T, W>& x, requires_arch<Generic>) noexcept;

template <typename T, size_t W>
SIMD_INLINE
bool all_of(const VecBool<T, W>& x, requires_arch<Generic>) noexcept;

template <typename T, size_t W>
SIMD_INLINE
bool any_of(const VecBool<T, W>& x, requires_arch<Generic>) noexcept;

template <typename T, size_t W>
SIMD_INLINE
bool some_of(const VecBool<T, W>& x, requires_arch<Generic>) noexcept;

#undef DECLARE_GENERIC_UNARY_OP
#undef DECLARE_GENERIC_BINARY_OP
#undef DECLARE_GENERIC_BINARY_CMP_OP

}  // namespace kernel
}  // namespace simd
