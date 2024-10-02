#pragma once

namespace simd { namespace kernel { namespace avx {
#include "simd/arch/kernel_impl.h"
} } } // namespace simd::kernel::avx

#include "simd/types/avx_register.h"
#include "simd/types/traits.h"
#include "simd/arch/avx/detail.h"
#include "simd/arch/avx/algorithm.h"
#include "simd/arch/avx/arithmetic.h"
#include "simd/arch/avx/cast.h"
#include "simd/arch/avx/compare.h"
#include "simd/arch/avx/complex.h"
#include "simd/arch/avx/logical.h"
#include "simd/arch/avx/math.h"
#include "simd/arch/avx/memory.h"
#include "simd/arch/avx/trigo.h"

namespace simd { namespace kernel {

#define DEFINE_AVX_BINARY_OP(OP) \
template <typename T, size_t W> \
SIMD_INLINE \
Vec<T, W> OP(const Vec<T, W>& lhs, const Vec<T, W>& rhs, requires_arch<AVX>) noexcept \
{ \
    return avx::OP<T, W>::apply(lhs, rhs); \
} \
///

DEFINE_AVX_BINARY_OP(add);
DEFINE_AVX_BINARY_OP(sub);
DEFINE_AVX_BINARY_OP(mul);
DEFINE_AVX_BINARY_OP(div);
DEFINE_AVX_BINARY_OP(mod);

DEFINE_AVX_BINARY_OP(bitwise_and);
DEFINE_AVX_BINARY_OP(bitwise_or);
DEFINE_AVX_BINARY_OP(bitwise_xor);
DEFINE_AVX_BINARY_OP(bitwise_lshift);
DEFINE_AVX_BINARY_OP(bitwise_rshift);

template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> bitwise_andnot(const VecBool<T, W>& lhs, const Vec<T, W>& rhs, requires_arch<AVX>) noexcept
{
    return avx::bitwise_andnot<T, W>::apply(lhs, rhs);
}

template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> bitwise_lshift(const Vec<T, W>& lhs, int32_t rhs, requires_arch<AVX>) noexcept
{
    return avx::bitwise_lshift<T, W>::apply(lhs, rhs);
}
template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> bitwise_rshift(const Vec<T, W>& lhs, int32_t rhs, requires_arch<AVX>) noexcept
{
    return avx::bitwise_rshift<T, W>::apply(lhs, rhs);
}

#define DEFINE_AVX_UNARY_OP(OP) \
template <typename T, size_t W> \
SIMD_INLINE \
Vec<T, W> OP(const Vec<T, W>& x, requires_arch<AVX>) noexcept \
{ \
    return avx::OP<T, W>::apply(x); \
} \
///

DEFINE_AVX_UNARY_OP(neg);
DEFINE_AVX_UNARY_OP(bitwise_not);

#define DEFINE_AVX_BINARY_CMP_OP(OP) \
template <typename T, size_t W> \
SIMD_INLINE \
VecBool<T, W> OP(const Vec<T, W>& lhs, const Vec<T, W>& rhs, requires_arch<AVX>) noexcept \
{ \
    return avx::OP<T, W>::apply(lhs, rhs); \
} \
///

DEFINE_AVX_BINARY_CMP_OP(eq);
DEFINE_AVX_BINARY_CMP_OP(ne);
DEFINE_AVX_BINARY_CMP_OP(gt);
DEFINE_AVX_BINARY_CMP_OP(ge);
DEFINE_AVX_BINARY_CMP_OP(lt);
DEFINE_AVX_BINARY_CMP_OP(le);

template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> broadcast(T val, requires_arch<AVX>) noexcept
{
    return avx::broadcast<T, W>::apply(val);
}

template <typename T, size_t W>
SIMD_INLINE
bool all_of(const VecBool<T, W>& x, requires_arch<AVX>) noexcept
{
    return avx::all_of<T, W>::apply(x);
}

template <typename T, size_t W>
SIMD_INLINE
bool any_of(const VecBool<T, W>& x, requires_arch<AVX>) noexcept
{
    return avx::any_of<T, W>::apply(x);
}

template <typename T, size_t W>
SIMD_INLINE
int popcount(const VecBool<T, W>& x, requires_arch<AVX>) noexcept
{
    return avx::popcount<T, W>::apply(x);
}

template <typename T, size_t W>
SIMD_INLINE
int find_first_set(const VecBool<T, W>& x, requires_arch<AVX>) noexcept
{
    return avx::find_first_set<T, W>::apply(x);
}

template <typename T, size_t W>
SIMD_INLINE
int find_last_set(const VecBool<T, W>& x, requires_arch<AVX>) noexcept
{
    return avx::find_last_set<T, W>::apply(x);
}

template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> setzero(requires_arch<AVX>) noexcept
{
    return avx::setzero<T, W>::apply();
}

template <typename T, size_t W, typename... Ts,
    REQUIRES((!std::is_same<T, bool>::value))>
SIMD_INLINE
Vec<T, W> set(requires_arch<AVX>, T v0, T v1, Ts... vals) noexcept
{
    static_assert(sizeof...(Ts) + 2 == W);
    return avx::set<T, W>::apply(v0, v1, static_cast<T>(vals)...);
}

template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> load_aligned(const T* mem, requires_arch<AVX>) noexcept
{
    return avx::load_aligned<T, W>::apply(mem);
}

template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> load_unaligned(const T* mem, requires_arch<AVX>) noexcept
{
    return avx::load_unaligned<T, W>::apply(mem);
}

template <typename T, size_t W>
SIMD_INLINE
void store_aligned(T* mem, const Vec<T, W>& x, requires_arch<AVX>) noexcept
{
    avx::store_aligned<T, W>::apply(mem, x);
}

template <typename T, size_t W>
SIMD_INLINE
void store_unaligned(T* mem, const Vec<T, W>& x, requires_arch<AVX>) noexcept
{
    avx::store_unaligned<T, W>::apply(mem, x);
}

template <typename T, size_t W, typename U, typename V>
SIMD_INLINE
Vec<T, W> gather(const U* mem, const Vec<V, W>& index, requires_arch<AVX>) noexcept
{
    return avx::gather<T, W, U, V>::apply(mem, index);
}

template <typename T, size_t W, typename U, typename V>
SIMD_INLINE
void scatter(const Vec<T, W>& x, U* mem, const Vec<V, W>& index, requires_arch<AVX>) noexcept
{
    return avx::scatter<T, W, U, V>::apply(x, mem, index);
}

template <typename T, size_t W>
SIMD_INLINE
uint64_t to_mask(const VecBool<T, W>& x, requires_arch<AVX>) noexcept
{
    return avx::to_mask<T, W>::apply(x);
}

template <typename T, size_t W>
SIMD_INLINE
VecBool<T, W> from_mask(uint64_t x, requires_arch<AVX>) noexcept
{
    return avx::from_mask<T, W>::apply(x);
}

#undef DEFINE_AVX_BINARY_OP
#undef DEFINE_AVX_UNARY_OP
#undef DEFINE_AVX_BINARY_CMP_OP

} } // namespace simd::kernel
