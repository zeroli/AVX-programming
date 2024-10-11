#pragma once

namespace simd { namespace kernel { namespace sse {
#include "simd/arch/kernel_impl.h"
} } } // namespace simd::kernel::sse

#include "simd/types/sse_register.h"
#include "simd/types/traits.h"
#include "simd/arch/sse/detail.h"
#include "simd/arch/sse/algorithm.h"
#include "simd/arch/sse/arithmetic.h"
#include "simd/arch/sse/cast.h"
#include "simd/arch/sse/compare.h"
#include "simd/arch/sse/logical.h"
#include "simd/arch/sse/math.h"
#include "simd/arch/sse/memory.h"
#include "simd/arch/sse/trigo.h"
#include "simd/arch/sse/complex.h"

namespace simd { namespace kernel {
#define DEFINE_SSE_BINARY_OP(OP) \
template <typename T, size_t W> \
SIMD_INLINE \
Vec<T, W> OP(const Vec<T, W>& lhs, const Vec<T, W>& rhs, requires_arch<SSE>) noexcept \
{ \
    return sse::OP<T, W>::apply(lhs, rhs); \
} \
///

DEFINE_SSE_BINARY_OP(add);
DEFINE_SSE_BINARY_OP(sub);
DEFINE_SSE_BINARY_OP(mul);
DEFINE_SSE_BINARY_OP(div);
DEFINE_SSE_BINARY_OP(mod);

DEFINE_SSE_BINARY_OP(bitwise_and);
DEFINE_SSE_BINARY_OP(bitwise_or);
DEFINE_SSE_BINARY_OP(bitwise_xor);
DEFINE_SSE_BINARY_OP(bitwise_lshift);
DEFINE_SSE_BINARY_OP(bitwise_rshift);

template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> bitwise_andnot(const VecBool<T, W>& lhs, const Vec<T, W>& rhs, requires_arch<SSE>) noexcept
{
    return sse::bitwise_andnot<T, W>::apply(lhs, rhs);
}

template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> bitwise_lshift(const Vec<T, W>& lhs, int32_t rhs, requires_arch<SSE>) noexcept
{
    return sse::bitwise_lshift<T, W>::apply(lhs, rhs);
}
template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> bitwise_rshift(const Vec<T, W>& lhs, int32_t rhs, requires_arch<SSE>) noexcept
{
    return sse::bitwise_rshift<T, W>::apply(lhs, rhs);
}

#define DEFINE_SSE_BINARY_CMP_OP(OP) \
template <typename T, size_t W> \
SIMD_INLINE \
VecBool<T, W> OP(const Vec<T, W>& lhs, const Vec<T, W>& rhs, requires_arch<SSE>) noexcept \
{ \
    return sse::OP<T, W>::apply(lhs, rhs); \
} \
///

DEFINE_SSE_BINARY_CMP_OP(eq);
DEFINE_SSE_BINARY_CMP_OP(ne);
DEFINE_SSE_BINARY_CMP_OP(gt);
DEFINE_SSE_BINARY_CMP_OP(ge);
DEFINE_SSE_BINARY_CMP_OP(lt);
DEFINE_SSE_BINARY_CMP_OP(le);

DEFINE_SSE_BINARY_OP(max);
DEFINE_SSE_BINARY_OP(min);

#define DEFINE_SSE_UNARY_OP(OP) \
template <typename T, size_t W> \
SIMD_INLINE \
Vec<T, W> OP(const Vec<T, W>& x, requires_arch<SSE>) noexcept \
{ \
    return sse::OP<T, W>::apply(x); \
} \
///

#define DEFINE_SSE_MATH_UNARY_OP(OP) \
template <typename T, size_t W> \
SIMD_INLINE \
Vec<T, W> OP(const Vec<T, W>& x, requires_arch<SSE>) noexcept \
{ \
    return sse::OP<T, W>::apply(x); \
} \
///

DEFINE_SSE_UNARY_OP(bitwise_not);
DEFINE_SSE_UNARY_OP(neg);

DEFINE_SSE_MATH_UNARY_OP(abs);
DEFINE_SSE_MATH_UNARY_OP(sqrt);
DEFINE_SSE_MATH_UNARY_OP(ceil);
DEFINE_SSE_MATH_UNARY_OP(floor);

template <typename T, size_t W>
SIMD_INLINE
VecBool<T, W> bitwise_not(const VecBool<T, W>& x, requires_arch<SSE>) noexcept
{
    return sse::bitwise_not<T, W>::apply(x);
}

template <typename T, size_t W>
SIMD_INLINE
bool all_of(const VecBool<T, W>& x, requires_arch<SSE>) noexcept
{
    return sse::all_of<T, W>::apply(x);
}

template <typename T, size_t W>
SIMD_INLINE
bool any_of(const VecBool<T, W>& x, requires_arch<SSE>) noexcept
{
    return sse::any_of<T, W>::apply(x);
}

template <typename T, size_t W>
SIMD_INLINE
int popcount(const VecBool<T, W>& x, requires_arch<SSE>) noexcept
{
    return sse::popcount<T, W>::apply(x);
}

template <typename T, size_t W>
SIMD_INLINE
int find_first_set(const VecBool<T, W>& x, requires_arch<SSE>) noexcept
{
    return sse::find_first_set<T, W>::apply(x);
}

template <typename T, size_t W>
SIMD_INLINE
int find_last_set(const VecBool<T, W>& x, requires_arch<SSE>) noexcept
{
    return sse::find_last_set<T, W>::apply(x);
}

template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> broadcast(T val, requires_arch<SSE>) noexcept
{
    return sse::broadcast<T, W>::apply(val);
}

template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> setzero(requires_arch<SSE>) noexcept
{
    return sse::setzero<T, W>::apply();
}

template <typename T, size_t W, typename... Ts,
    REQUIRES((!std::is_same<T, bool>::value))>
SIMD_INLINE
Vec<T, W> set(requires_arch<SSE>, T v0, T v1, Ts... vals) noexcept
{
    static_assert(sizeof...(Ts) + 2 == W);
    return sse::set<T, W>::apply(v0, v1, static_cast<T>(vals)...);
}

template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> load_aligned(const T* mem, requires_arch<SSE>) noexcept
{
    return sse::load_aligned<T, W>::apply(mem);
}

template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> load_unaligned(const T* mem, requires_arch<SSE>) noexcept
{
    return sse::load_unaligned<T, W>::apply(mem);
}

template <typename T, size_t W>
SIMD_INLINE
void store_aligned(T* mem, const Vec<T, W>& x, requires_arch<SSE>) noexcept
{
    sse::store_aligned<T, W>::apply(mem, x);
}

template <typename T, size_t W>
SIMD_INLINE
void store_unaligned(T* mem, const Vec<T, W>& x, requires_arch<SSE>) noexcept
{
    sse::store_unaligned<T, W>::apply(mem, x);
}

template <typename T, size_t W, typename U, typename V>
SIMD_INLINE
Vec<T, W> gather(const U* mem, const Vec<V, W>& index, requires_arch<SSE>) noexcept
{
    return sse::gather<T, W, U, V>::apply(mem, index);
}

template <typename T, size_t W, typename U, typename V>
SIMD_INLINE
void scatter(const Vec<T, W>& x, U* mem, const Vec<V, W>& index, requires_arch<SSE>) noexcept
{
    return sse::scatter<T, W, U, V>::apply(x, mem, index);
}

template <typename T, size_t W>
SIMD_INLINE
uint64_t to_mask(const VecBool<T, W>& x, requires_arch<SSE>) noexcept
{
    return sse::to_mask<T, W>::apply(x);
}

template <typename T, size_t W>
SIMD_INLINE
VecBool<T, W> from_mask(uint64_t x, requires_arch<SSE>) noexcept
{
    return sse::from_mask<T, W>::apply(x);
}

template <typename U, typename T, size_t W>
SIMD_INLINE
Vec<U, W> cast(const Vec<T, W>& x, requires_arch<SSE>) noexcept
{
    return sse::cast<U, T, W>::apply(x);
}

template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> select(const VecBool<T, W>& cond, const Vec<T, W>& lhs, const Vec<T, W>& rhs, requires_arch<SSE>) noexcept
{
    return sse::select<T, W>::apply(cond, lhs, rhs);
}

/// reduction
template <typename T, size_t W, typename F>
SIMD_INLINE
T reduce(F&& f, const Vec<T, W>& x, requires_arch<SSE>) noexcept
{
    return sse::reduce<T, W, F>::apply(std::forward<F>(f), x);
}

template <typename T, size_t W>
SIMD_INLINE
T reduce_sum(const Vec<T, W>& x, requires_arch<SSE>) noexcept
{
    return sse::reduce_sum<T, W>::apply(x);
}

template <typename T, size_t W>
SIMD_INLINE
T reduce_max(const Vec<T, W>& x, requires_arch<SSE>) noexcept
{
    return sse::reduce_max<T, W>::apply(x);
}

template <typename T, size_t W>
SIMD_INLINE
T reduce_min(const Vec<T, W>& x, requires_arch<SSE>) noexcept
{
    return sse::reduce_min<T, W>::apply(x);
}

#undef DEFINE_SSE_UNARY_OP
#undef DEFINE_SSE_BINARY_OP
#undef DEFINE_SSE_BINARY_CMP_OP

} } // namespace simd::kernel
