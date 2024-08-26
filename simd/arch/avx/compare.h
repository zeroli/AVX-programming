#pragma once

#include "simd/types/avx_register.h"
#include "simd/types/vec.h"

namespace simd {
namespace kernel {
using namespace types;

/// eq
template <typename Arch>
VecBool<float, Arch> eq(const Vec<float, Arch>& self, const Vec<float, Arch>& other, requires_arch<AVX>) noexcept
{
    return _mm256_cmp_ps(self, other, _CMP_EQ_OQ);
}

template <typename Arch>
VecBool<double, Arch> eq(const Vec<double, Arch>& self, const Vec<float, Arch>& other, requires_arch<AVX>) noexcept
{
    return _mm256_cmp_pd(self, other, _CMP_EQ_OQ);
}

template <typename Arch>
VecBool<float, Arch> eq(const VecBool<float, Arch>& self, const VecBool<float, Arch>& other, requires_arch<AVX>) noexcept
{
    return ~(self != other);
}

template <typename Arch>
VecBool<double, Arch> eq(const VecBool<double, Arch>& self, const VecBool<float, Arch>& other, requires_arch<AVX>) noexcept
{
    return ~(self != other);
}

template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
VecBool<T, Arch> eq(const Vec<T, Arch>& self, const Vec<T, Arch>& other, requires_arch<AVX>) noexcept
{
    return detail::fwd_to_sse([](__m128i s, __m128i o) noexcept {
        return eq(Vec<T, SSE>(s), Vec<T, SSE>(o));
    }, self, other);
}

template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
VecBool<T, Arch> eq(const VecBool<T, Arch>& self, const VecBool<T, Arch>& other, requires_arch<AVX>) noexcept
{
    return ~(self != other);
}

}  // namespace kernel
}  // namespace simd
