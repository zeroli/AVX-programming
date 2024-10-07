#pragma once

namespace simd { namespace kernel { namespace avx512 {
#include "simd/arch/kernel_impl.h"
} } } // namespace simd::kernel::avx512

#include "simd/types/avx512_register.h"
#include "simd/types/traits.h"
#include "simd/arch/avx512/detail.h"
#include "simd/arch/avx512/algorithm.h"
#include "simd/arch/avx512/arithmetic.h"
#include "simd/arch/avx512/fma.h"
#include "simd/arch/avx512/cast.h"
#include "simd/arch/avx512/compare.h"
#include "simd/arch/avx512/complex.h"
#include "simd/arch/avx512/logical.h"
#include "simd/arch/avx512/math.h"
#include "simd/arch/avx512/memory.h"
#include "simd/arch/avx512/trigo.h"

namespace simd { namespace kernel {

#define DEFINE_AVX512_BINARY_OP(OP) \
template <typename T, size_t W> \
SIMD_INLINE \
Vec<T, W> OP(const Vec<T, W>& lhs, const Vec<T, W>& rhs, requires_arch<AVX512>) noexcept \
{ \
    return avx512::OP<T, W>::apply(lhs, rhs); \
} \
///

DEFINE_AVX512_BINARY_OP(add);
DEFINE_AVX512_BINARY_OP(sub);
DEFINE_AVX512_BINARY_OP(mul);
DEFINE_AVX512_BINARY_OP(div);
DEFINE_AVX512_BINARY_OP(mod);

DEFINE_AVX512_BINARY_OP(bitwise_and);
DEFINE_AVX512_BINARY_OP(bitwise_or);
DEFINE_AVX512_BINARY_OP(bitwise_xor);
DEFINE_AVX512_BINARY_OP(bitwise_lshift);
DEFINE_AVX512_BINARY_OP(bitwise_rshift);

template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> bitwise_andnot(const VecBool<T, W>& lhs, const Vec<T, W>& rhs, requires_arch<AVX512>) noexcept
{
    return avx512::bitwise_andnot<T, W>::apply(lhs, rhs);
}

template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> bitwise_lshift(const Vec<T, W>& lhs, int32_t rhs, requires_arch<AVX512>) noexcept
{
    return avx512::bitwise_lshift<T, W>::apply(lhs, rhs);
}
template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> bitwise_rshift(const Vec<T, W>& lhs, int32_t rhs, requires_arch<AVX512>) noexcept
{
    return avx512::bitwise_rshift<T, W>::apply(lhs, rhs);
}

#define DEFINE_AVX512_UNARY_OP(OP) \
template <typename T, size_t W> \
SIMD_INLINE \
Vec<T, W> OP(const Vec<T, W>& x, requires_arch<AVX512>) noexcept \
{ \
    return avx512::OP<T, W>::apply(x); \
} \
///

DEFINE_AVX512_UNARY_OP(neg);
DEFINE_AVX512_UNARY_OP(bitwise_not);

#define DEFINE_AVX512_BINARY_CMP_OP(OP) \
template <typename T, size_t W> \
SIMD_INLINE \
VecBool<T, W> OP(const Vec<T, W>& lhs, const Vec<T, W>& rhs, requires_arch<AVX512>) noexcept \
{ \
    return avx512::OP<T, W>::apply(lhs, rhs); \
} \
///

DEFINE_AVX512_BINARY_CMP_OP(eq);
DEFINE_AVX512_BINARY_CMP_OP(ne);
DEFINE_AVX512_BINARY_CMP_OP(gt);
DEFINE_AVX512_BINARY_CMP_OP(ge);
DEFINE_AVX512_BINARY_CMP_OP(lt);
DEFINE_AVX512_BINARY_CMP_OP(le);

template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> broadcast(T val, requires_arch<AVX512>) noexcept
{
    return avx512::broadcast<T, W>::apply(val);
}

template <typename T, size_t W>
SIMD_INLINE
bool all_of(const VecBool<T, W>& x, requires_arch<AVX512>) noexcept
{
    return avx512::all_of<T, W>::apply(x);
}

template <typename T, size_t W>
SIMD_INLINE
bool any_of(const VecBool<T, W>& x, requires_arch<AVX512>) noexcept
{
    return avx512::any_of<T, W>::apply(x);
}

template <typename T, size_t W>
SIMD_INLINE
int popcount(const VecBool<T, W>& x, requires_arch<AVX512>) noexcept
{
    return avx512::popcount<T, W>::apply(x);
}

template <typename T, size_t W>
SIMD_INLINE
int find_first_set(const VecBool<T, W>& x, requires_arch<AVX512>) noexcept
{
    return avx512::find_first_set<T, W>::apply(x);
}

template <typename T, size_t W>
SIMD_INLINE
int find_last_set(const VecBool<T, W>& x, requires_arch<AVX512>) noexcept
{
    return avx512::find_last_set<T, W>::apply(x);
}

template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> setzero(requires_arch<AVX512>) noexcept
{
    return avx512::setzero<T, W>::apply();
}

template <typename T, size_t W, typename... Ts,
    REQUIRES((!std::is_same<T, bool>::value))>
SIMD_INLINE
Vec<T, W> set(requires_arch<AVX512>, T v0, T v1, Ts... vals) noexcept
{
    static_assert(sizeof...(Ts) + 2 == W);
    return avx512::set<T, W>::apply(v0, v1, static_cast<T>(vals)...);
}

template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> load_aligned(const T* mem, requires_arch<AVX512>) noexcept
{
    return avx512::load_aligned<T, W>::apply(mem);
}

template <typename T, size_t W>
SIMD_INLINE
Vec<T, W> load_unaligned(const T* mem, requires_arch<AVX512>) noexcept
{
    return avx512::load_unaligned<T, W>::apply(mem);
}

template <typename T, size_t W>
SIMD_INLINE
void store_aligned(T* mem, const Vec<T, W>& x, requires_arch<AVX512>) noexcept
{
    avx512::store_aligned<T, W>::apply(mem, x);
}

template <typename T, size_t W>
SIMD_INLINE
void store_unaligned(T* mem, const Vec<T, W>& x, requires_arch<AVX512>) noexcept
{
    avx512::store_unaligned<T, W>::apply(mem, x);
}

template <typename T, size_t W, typename U, typename V>
SIMD_INLINE
Vec<T, W> gather(const U* mem, const Vec<V, W>& index, requires_arch<AVX512>) noexcept
{
    return avx512::gather<T, W, U, V>::apply(mem, index);
}

template <typename T, size_t W, typename U, typename V>
SIMD_INLINE
void scatter(const Vec<T, W>& x, U* mem, const Vec<V, W>& index, requires_arch<AVX512>) noexcept
{
    return avx512::scatter<T, W, U, V>::apply(x, mem, index);
}

template <typename T, size_t W>
SIMD_INLINE
uint64_t to_mask(const VecBool<T, W>& x, requires_arch<AVX512>) noexcept
{
    return avx512::to_mask<T, W>::apply(x);
}

template <typename T, size_t W>
SIMD_INLINE
VecBool<T, W> from_mask(uint64_t x, requires_arch<AVX512>) noexcept
{
    return avx512::from_mask<T, W>::apply(x);
}

template <typename T, size_t W>
Vec<T, W> fmadd(const Vec<T, W>& x, const Vec<T, W>& y, const Vec<T, W>& z, requires_arch<AVX512>) noexcept
{
    return avx512::fmadd<T, W>::apply(x, y, z);
}

template <typename T, size_t W>
Vec<T, W> fmsub(const Vec<T, W>& x, const Vec<T, W>& y, const Vec<T, W>& z, requires_arch<AVX512>) noexcept
{
    return avx512::fmsub<T, W>::apply(x, y, z);
}

template <typename T, size_t W>
Vec<T, W> fnmadd(const Vec<T, W>& x, const Vec<T, W>& y, const Vec<T, W>& z, requires_arch<AVX512>) noexcept
{
    return avx512::fnmadd<T, W>::apply(x, y, z);
}

template <typename T, size_t W>
Vec<T, W> fnmsub(const Vec<T, W>& x, const Vec<T, W>& y, const Vec<T, W>& z, requires_arch<AVX512>) noexcept
{
    return avx512::fnmsub<T, W>::apply(x, y, z);
}

template <typename T, size_t W>
Vec<T, W> fmaddsub(const Vec<T, W>& x, const Vec<T, W>& y, const Vec<T, W>& z, requires_arch<AVX512>) noexcept
{
    return avx512::fmaddsub<T, W>::apply(x, y, z);
}

template <typename T, size_t W>
Vec<T, W> fmsubadd(const Vec<T, W>& x, const Vec<T, W>& y, const Vec<T, W>& z, requires_arch<AVX512>) noexcept
{
    return avx512::fmsubadd<T, W>::apply(x, y, z);
}

#undef DEFINE_AVX512_BINARY_OP
#undef DEFINE_AVX512_UNARY_OP
#undef DEFINE_AVX512_BINARY_CMP_OP
} }  // namespace simd::kernel
