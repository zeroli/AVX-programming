#pragma once

namespace simd { namespace kernel { namespace sse {
namespace detail {
using namespace types;

template <typename T>
SIMD_INLINE
sse_reg_traits_t<T> make_mask() {
    static sse_reg_traits_t<T> reg = _mm_set1_epi32(-1);
    return reg;
}
template <>
SIMD_INLINE
sse_reg_f make_mask<float>() {
    static sse_reg_f reg = _mm_castsi128_ps(_mm_set1_epi32(-1));
    return reg;
}

template <>
SIMD_INLINE
sse_reg_d make_mask<double>() {
    static sse_reg_d reg = _mm_castsi128_pd(_mm_set1_epi32(-1));
    return reg;
}

template <typename T>
SIMD_INLINE
sse_reg_traits_t<T> make_signmask();  // no implementation

template <>
SIMD_INLINE
sse_reg_f make_signmask<float>() {
    static sse_reg_f reg = _mm_set1_ps(-0.f);  // -0.f => 1 << 31
    return reg;
}

template <>
SIMD_INLINE
sse_reg_d make_signmask<double>() {
    static sse_reg_d reg = _mm_set1_pd(-0.f);  // -0.f => 1 << 63
    return reg;
}

}  // namespace detail
} } }  // namespace simd::kernel::sse
