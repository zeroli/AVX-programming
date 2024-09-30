#pragma once

namespace simd { namespace kernel { namespace avx {
namespace detail {
using namespace types;
/// split from one avx register into two sse registers
SIMD_INLINE
void split(const avx_reg_i& val, sse_reg_i& low, sse_reg_i& high) noexcept
{
    low = _mm256_castsi256_si128(val); // no latency
    high = _mm256_extractf128_si256(val, 1);
}
SIMD_INLINE
void split(const avx_reg_f& val, sse_reg_f& low, sse_reg_f& high) noexcept
{
    low  = _mm256_castps256_ps128(val); // no latency
    high = _mm256_extractf128_ps(val, 1);
}
SIMD_INLINE
void split(const avx_reg_d& val, sse_reg_d& low, sse_reg_d& high) noexcept
{
    low  = _mm256_castpd256_pd128(val); // no latency
    high = _mm256_extractf128_pd(val, 1);
}

/// merge from two sse registers to one avx register
SIMD_INLINE
avx_reg_i merge(const sse_reg_i& low, const sse_reg_i& high) noexcept
{
    return _mm256_insertf128_si256(_mm256_castsi128_si256(low), high, 1);
}
SIMD_INLINE
avx_reg_f merge(const sse_reg_f& low, const sse_reg_f& high) noexcept
{
    return _mm256_insertf128_ps(_mm256_castps128_ps256(low), high, 1);
}
SIMD_INLINE
avx_reg_d merge(const sse_reg_d& low, const sse_reg_d& high) noexcept
{
    return _mm256_insertf128_pd(_mm256_castpd128_pd256(low), high, 1);
}

}  // namespace detail
} } }  // namespace simd::kernel::avx
