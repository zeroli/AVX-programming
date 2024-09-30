#pragma once

namespace simd { namespace kernel { namespace avx {
using namespace types;

} } } // namespace simd::kernel::avx

#if 0
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
#endif
