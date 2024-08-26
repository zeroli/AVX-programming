#pragma once

#include "simd/types/avx_register.h"
#include "simd/types/vec.h"

namespace simd {
namespace kernel {
using namespace types;

/// abs
template <typename Arch>
Vec<float, Arch> abs(const Vec<float, Arch>& self, requires_arch<AVX>) noexcept
{
    __m256 sign_mask = _mm256_set1_ps(-0.f);  // -0.f = 1 << 31
    return _mm256_andnot_ps(sign_mask, self);
}

template <typename Arch>
Vec<double, Arch> abs(const Vec<double, Arch>& self, requires_arch<AVX>) noexcept
{
    __m256d sign_mask = _mm256_set1_ps(-0.f);  // -0.f = 1 << 31
    return _mm256_andnot_pd(sign_mask, self);
}


/// avg

/// avgr

/// bitofsign

/// cbrt

/// clip

/// copysign

/// erf

/// erfc

/// estrin

/// exp

/// exp10

/// exp2

/// expm1

/// polar

/// fdim

/// fmod

/// frexp

/// horner

/// hypot

/// ipow

/// ldexp

/// lgamma

/// log

/// log2

/// log10

/// log1p

/// mod

/// pow

/// reciprocal

/// reduce_add

/// reduce_max

/// reduce_min

/// remainder

/// select

/// sign

/// signnz

/// sqrt

/// tgamma


}  // namespace kernel
}  // namespace simd
