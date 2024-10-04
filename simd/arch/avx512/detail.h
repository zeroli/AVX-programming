#pragma once

namespace simd { namespace kernel { namespace avx512 {
namespace detail {
using namespace types;

template <typename T>
SIMD_INLINE
avx512_reg_traits_t<T> make_mask() {
    static avx512_reg_traits_t<T> reg = _mm512_set1_epi32(-1);
    return reg;
}
template <>
SIMD_INLINE
avx512_reg_f make_mask<float>() {
    static avx512_reg_f reg = _mm512_castsi512_ps(_mm512_set1_epi32(-1));
    return reg;
}

template <>
SIMD_INLINE
avx512_reg_d make_mask<double>() {
    static avx512_reg_d reg = _mm512_castsi512_pd(_mm512_set1_epi32(-1));
    return reg;
}

template <typename T>
SIMD_INLINE
avx512_reg_traits_t<T> make_signmask();  // no implementation

template <>
SIMD_INLINE
avx512_reg_f make_signmask<float>() {
    static avx512_reg_f reg = _mm512_set1_ps(-0.f);  // -0.f => 1 << 31
    return reg;
}

template <>
SIMD_INLINE
avx512_reg_d make_signmask<double>() {
    static avx512_reg_d reg = _mm512_set1_pd(-0.f);  // -0.f => 1 << 63
    return reg;
}

/// split from one avx512 register into two avx registers
SIMD_INLINE
static void split_reg(const avx512_reg_i& val, avx_reg_i& low, avx_reg_i& high) noexcept
{
    low  = _mm512_castsi512_si256(val); // no latency
    high = _mm512_extracti64x4_epi64(val, 1);
}
SIMD_INLINE
static void split_reg(const avx512_reg_f& val, avx_reg_f& low, avx_reg_f& high) noexcept
{
    low  = _mm512_castps512_ps256(val); // no latency
    high = _mm512_extractf32x8_ps(val, 1);
}
SIMD_INLINE
static void split_reg(const avx512_reg_d& val, avx_reg_d& low, avx_reg_d& high) noexcept
{
    low  = _mm512_castpd512_pd256(val); // no latency
    high = _mm512_extractf64x4_pd(val, 1);
}

/// merge from two avx registers to one avx512 register
SIMD_INLINE
static avx512_reg_i merge_reg(const avx_reg_i& low, const avx_reg_i& high) noexcept
{
    return _mm512_inserti64x4(_mm512_castsi256_si512(low), high, 1);
}
SIMD_INLINE
static avx512_reg_f merge_reg(const avx_reg_f& low, const avx_reg_f& high) noexcept
{
    return _mm512_insertf32x8(_mm512_castps256_ps512(low), high, 1);
}
SIMD_INLINE
static avx512_reg_d merge_reg(const avx_reg_d& low, const avx_reg_d& high) noexcept
{
    return _mm512_insertf64x4(_mm512_castpd256_pd512(low), high, 1);
}

template <typename OP, typename VO, typename VI = VO>
SIMD_INLINE
static avx512_reg_i forward_avx_op(const avx512_reg_i& lhs, const avx512_reg_i& rhs) noexcept
{
    static_assert(VI::n_regs() == 1);

    avx_reg_i l_low, l_high, r_low, r_high;
    detail::split_reg(lhs, l_low, l_high);
    detail::split_reg(rhs, r_low, r_high);
    auto low_result  = OP::template apply<VO, VI>(VI(l_low),  VI(r_low));
    auto high_result = OP::template apply<VO, VI>(VI(l_high), VI(r_high));
    return detail::merge_reg(low_result.reg(), high_result.reg());
}

template <typename OP, typename VO, typename VI1 = VO, typename VI2 = VI1>
SIMD_INLINE
static avx512_reg_i forward_avx_op2(const avx512_reg_i& lhs, const avx512_reg_i& rhs) noexcept
{
    static_assert(VI1::n_regs() == 1);

    avx_reg_i l_low, l_high, r_low, r_high;
    detail::split_reg(lhs, l_low, l_high);
    detail::split_reg(rhs, r_low, r_high);
    auto low_result  = OP::template apply<VO, VI1, VI2>(VI1(l_low),  VI2(r_low));
    auto high_result = OP::template apply<VO, VI1, VI2>(VI1(l_high), VI2(r_high));
    return detail::merge_reg(low_result.reg(), high_result.reg());
}

template <typename OP, typename VO, typename VI = VO>
SIMD_INLINE
static avx512_reg_i forward_avx_op(const avx512_reg_i& lhs, int32_t rhs) noexcept
{
    static_assert(VI::n_regs() == 1);

    avx_reg_i l_low, l_high;
    detail::split_reg(lhs, l_low, l_high);
    auto low_result  = OP::template apply<VO, VI>(VI(l_low),  rhs);
    auto high_result = OP::template apply<VO, VI>(VI(l_high), rhs);
    return detail::merge_reg(low_result.reg(), high_result.reg());
}

template <typename OP, typename VO, typename VI = VO>
SIMD_INLINE
static avx512_reg_i forward_avx_op(const avx512_reg_i& lhs) noexcept
{
    static_assert(VI::n_regs() == 1);

    avx_reg_i l_low, l_high;
    detail::split_reg(lhs, l_low, l_high);
    auto low_result  = OP::template apply<VO, VI>(VI(l_low));
    auto high_result = OP::template apply<VO, VI>(VI(l_high));
    return detail::merge_reg(low_result.reg(), high_result.reg());
}

template <typename OP, typename VO, typename VI = VO>
SIMD_INLINE
static std::pair<VO, VO> forward_avx_op0(const avx512_reg_i& lhs) noexcept
{
    static_assert(VI::n_regs() == 1);

    avx_reg_i l_low, l_high;
    detail::split_reg(lhs, l_low, l_high);
    auto low_result  = OP::template apply<VO, VI>(VI(l_low));
    auto high_result = OP::template apply<VO, VI>(VI(l_high));
    return std::make_pair(low_result, high_result);
}

}  // namespace detail
} } }  // namespace simd::kernel::avx
