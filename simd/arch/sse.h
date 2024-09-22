#pragma once

namespace simd {
namespace kernel {
namespace sse {
#include "simd/arch/kernel_impl.h"
}  // namespace sse
}  // namespace kernel
}  // namespace simd

#include "simd/types/sse_register.h"
#include "simd/arch/sse/algorithm.h"
#include "simd/arch/sse/arithmetic.h"
#include "simd/arch/sse/cast.h"
#include "simd/arch/sse/compare.h"
#include "simd/arch/sse/complex.h"
#include "simd/arch/sse/logical.h"
#include "simd/arch/sse/math.h"
#include "simd/arch/sse/memory.h"
#include "simd/arch/sse/trigo.h"

namespace simd {
namespace kernel {

#define DEFINE_SSE_BINARY_OP(OP) \
template <typename T, size_t W> \
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
DEFINE_SSE_BINARY_OP(bitwise_andnot);
DEFINE_SSE_BINARY_OP(bitwise_lshift);
DEFINE_SSE_BINARY_OP(bitwise_rshift);

template <typename T, size_t W>
Vec<T, W> bitwise_lshift(const Vec<T, W>& lhs, int32_t rhs, requires_arch<SSE>) noexcept
{
    return sse::bitwise_lshift<T, W>::apply(lhs, rhs);
}
template <typename T, size_t W>
Vec<T, W> bitwise_rshift(const Vec<T, W>& lhs, int32_t rhs, requires_arch<SSE>) noexcept
{
    return sse::bitwise_rshift<T, W>::apply(lhs, rhs);
}

DEFINE_SSE_BINARY_OP(logical_and);
DEFINE_SSE_BINARY_OP(logical_or);

#define DEFINE_SSE_BINARY_COMP_OP(OP) \
template <typename T, size_t W> \
VecBool<T, W> OP(const Vec<T, W>& lhs, const Vec<T, W>& rhs, requires_arch<SSE>) noexcept \
{ \
    return sse::OP<T, W>::apply(lhs, rhs); \
} \
///

DEFINE_SSE_BINARY_COMP_OP(eq);
DEFINE_SSE_BINARY_COMP_OP(ne);
DEFINE_SSE_BINARY_COMP_OP(gt);
DEFINE_SSE_BINARY_COMP_OP(ge);
DEFINE_SSE_BINARY_COMP_OP(lt);
DEFINE_SSE_BINARY_COMP_OP(le);

DEFINE_SSE_BINARY_OP(max);
DEFINE_SSE_BINARY_OP(min);

#define DEFINE_SSE_UNARY_OP(OP) \
template <typename T, size_t W> \
Vec<T, W> OP(const Vec<T, W>& self, requires_arch<SSE>) noexcept \
{ \
    return sse::OP<T, W>::apply(self); \
} \
///

DEFINE_SSE_UNARY_OP(bitwise_not);
DEFINE_SSE_UNARY_OP(neg);

DEFINE_SSE_UNARY_OP(abs);
DEFINE_SSE_UNARY_OP(sqrt);
DEFINE_SSE_UNARY_OP(ceil);
DEFINE_SSE_UNARY_OP(floor);

template <typename T, size_t W>
VecBool<T, W> bitwise_not(const VecBool<T, W>& self, requires_arch<SSE>) noexcept
{
    return sse::bitwise_not<T, W>::apply(self);
}

template <typename T, size_t W>
bool all(const VecBool<T, W>& self, requires_arch<SSE>) noexcept
{
    return sse::all<T, W>::apply(self);
}

template <typename T, size_t W>
bool any(const VecBool<T, W>& self, requires_arch<SSE>) noexcept
{
    return sse::any<T, W>::apply(self);
}

template <typename T, size_t W>
Vec<T, W> broadcast(T val, requires_arch<SSE>) noexcept
{
    return sse::broadcast<T, W>::apply(val);
}

template <typename T, size_t W,
    REQUIRES((!std::is_same<T, bool>::value))>
Vec<T, W> set(T v0, T v1, requires_arch<SSE>) noexcept
{
    return sse::set<T, W>::apply(v0, v1);
}

template <typename T, size_t W,
    REQUIRES((!std::is_same<T, bool>::value))>
Vec<T, W> set(T v0, T v1, T v2, T v3, requires_arch<SSE>) noexcept
{
    return sse::set<T, W>::apply(v0, v1, v2, v3);
}

template <typename T, size_t W,
    REQUIRES((!std::is_same<T, bool>::value))>
Vec<T, W> set(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7, requires_arch<SSE>) noexcept
{
    return sse::set<T, W>::apply(v0, v1, v2, v3, v4, v5, v6, v7);
}

template <typename T, size_t W,
    REQUIRES((!std::is_same<T, bool>::value))>
Vec<T, W> set(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7,
                    T v8, T v9, T v10, T v11, T v12, T v13, T v14, T v15, requires_arch<SSE>) noexcept
{
    return sse::set<T, W>::apply(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15);
}

template <typename T, size_t W>
Vec<T, W> load_aligned(const T* mem, requires_arch<SSE>) noexcept
{
    return sse::load_aligned<T, W>::apply(mem);
}

template <typename T, size_t W>
Vec<T, W> load_unaligned(const T* mem, requires_arch<SSE>) noexcept
{
    return sse::load_unaligned<T, W>::apply(mem);
}

template <typename T, size_t W>
void store_aligned(T* mem, const Vec<T, W>& x, requires_arch<SSE>) noexcept
{
    sse::store_aligned<T, W>::apply(mem, x);
}

template <typename T, size_t W>
void store_unaligned(T* mem, const Vec<T, W>& x, requires_arch<SSE>) noexcept
{
    sse::store_unaligned<T, W>::apply(mem, x);
}

template <typename T, size_t W>
uint64_t to_mask(const VecBool<T, W>& x, requires_arch<SSE>) noexcept
{
    return sse::to_mask<T, W>::apply(x);
}

template <typename T, size_t W>
VecBool<T, W> from_mask(uint64_t x, requires_arch<SSE>) noexcept
{
    return sse::from_mask<T, W>::apply(x);
}

template <typename U, typename T, size_t W>
Vec<U, W> cast(const Vec<T, W>& x, requires_arch<SSE>) noexcept
{
    return sse::cast<U, T, W>::apply(x);
}

template <typename T, size_t W>
Vec<T, W> select(const VecBool<T, W>& cond, const Vec<T, W>& lhs, const Vec<T, W>& rhs, requires_arch<SSE>) noexcept
{
    return sse::select<T, W>::apply(cond, lhs, rhs);
}

#undef DEFINE_SSE_UNARY_OP
#undef DEFINE_SSE_BINARY_OP
#undef DEFINE_SSE_BINARY_COMP_OP
}  // namespace kernel
}  // namespace simd
