#pragma once

namespace simd { namespace kernel { namespace avx {
namespace detail {
using namespace types;

template <typename T>
SIMD_INLINE
avx_reg_traits_t<T> make_mask() {
    return _mm256_set1_epi32(-1);
}
template <>
SIMD_INLINE
avx_reg_traits_t<float> make_mask<float>() {
    return _mm256_castsi256_ps(_mm256_set1_epi32(-1));
}

template <>
SIMD_INLINE
avx_reg_traits_t<double> make_mask<double>() {
    return _mm256_castsi256_pd(_mm256_set1_epi32(-1));
}

template <typename T>
SIMD_INLINE
avx_reg_traits_t<T> make_signmask();  // no implementation

template <>
SIMD_INLINE
avx_reg_traits_t<float> make_signmask<float>() {
    return _mm256_set1_ps(-0.f);  // -0.f => 1 << 31
}

template <>
SIMD_INLINE
avx_reg_traits_t<double> make_signmask<double>() {
    return _mm256_set1_pd(-0.f);  // -0.f => 1 << 63
}

struct sse_min {
    template <typename VO, typename VI>
    static VO apply(const VI& x, const VI& y) noexcept {
        return kernel::min(x, y, SSE{});
    }
};
struct sse_max {
    template <typename VO, typename VI>
    static VO apply(const VI& x, const VI& y) noexcept {
        return kernel::max(x, y, SSE{});
    }
};

/// split from one avx register into two sse registers
SIMD_INLINE
void split_reg(const avx_reg_i& val, sse_reg_i& low, sse_reg_i& high) noexcept
{
    low = _mm256_castsi256_si128(val); // no latency
    high = _mm256_extractf128_si256(val, 1);
}
SIMD_INLINE
void split_reg(const avx_reg_f& val, sse_reg_f& low, sse_reg_f& high) noexcept
{
    low  = _mm256_castps256_ps128(val); // no latency
    high = _mm256_extractf128_ps(val, 1);
}
SIMD_INLINE
void split_reg(const avx_reg_d& val, sse_reg_d& low, sse_reg_d& high) noexcept
{
    low  = _mm256_castpd256_pd128(val); // no latency
    high = _mm256_extractf128_pd(val, 1);
}

/// merge from two sse registers to one avx register
SIMD_INLINE
avx_reg_i merge_reg(const sse_reg_i& low, const sse_reg_i& high) noexcept
{
    return _mm256_insertf128_si256(_mm256_castsi128_si256(low), high, 1);
}
SIMD_INLINE
avx_reg_f merge_reg(const sse_reg_f& low, const sse_reg_f& high) noexcept
{
    return _mm256_insertf128_ps(_mm256_castps128_ps256(low), high, 1);
}
SIMD_INLINE
avx_reg_d merge_reg(const sse_reg_d& low, const sse_reg_d& high) noexcept
{
    return _mm256_insertf128_pd(_mm256_castpd128_pd256(low), high, 1);
}

template <typename OP, typename VO, typename VI = VO>
SIMD_INLINE
avx_reg_i forward_sse_op(const avx_reg_i& lhs, const avx_reg_i& rhs) noexcept
{
    static_assert(VI::n_regs() == 1);

    sse_reg_i l_low, l_high, r_low, r_high;
    detail::split_reg(lhs, l_low, l_high);
    detail::split_reg(rhs, r_low, r_high);
    auto low_result  = OP::template apply<VO, VI>(VI(l_low),  VI(r_low));
    auto high_result = OP::template apply<VO, VI>(VI(l_high), VI(r_high));
    return detail::merge_reg(low_result.reg(), high_result.reg());
}

template <typename OP, typename VO, typename VI1 = VO, typename VI2 = VI1>
SIMD_INLINE
avx_reg_i forward_sse_op2(const avx_reg_i& lhs, const avx_reg_i& rhs) noexcept
{
    static_assert(VI1::n_regs() == 1);

    sse_reg_i l_low, l_high, r_low, r_high;
    detail::split_reg(lhs, l_low, l_high);
    detail::split_reg(rhs, r_low, r_high);
    auto low_result  = OP::template apply<VO, VI1, VI2>(VI1(l_low),  VI2(r_low));
    auto high_result = OP::template apply<VO, VI1, VI2>(VI1(l_high), VI2(r_high));
    return detail::merge_reg(low_result.reg(), high_result.reg());
}

template <typename OP, typename VO, typename VI = VO>
SIMD_INLINE
avx_reg_i forward_sse_op(const avx_reg_i& lhs, int32_t rhs) noexcept
{
    static_assert(VI::n_regs() == 1);

    sse_reg_i l_low, l_high;
    detail::split_reg(lhs, l_low, l_high);
    auto low_result  = OP::template apply<VO, VI>(VI(l_low),  rhs);
    auto high_result = OP::template apply<VO, VI>(VI(l_high), rhs);
    return detail::merge_reg(low_result.reg(), high_result.reg());
}

template <typename OP, typename VO, typename VI = VO>
SIMD_INLINE
avx_reg_i forward_sse_op(const avx_reg_i& lhs) noexcept
{
    static_assert(VI::n_regs() == 1);

    sse_reg_i l_low, l_high;
    detail::split_reg(lhs, l_low, l_high);
    auto low_result  = OP::template apply<VO, VI>(VI(l_low));
    auto high_result = OP::template apply<VO, VI>(VI(l_high));
    return detail::merge_reg(low_result.reg(), high_result.reg());
}

template <typename OP, typename VO, typename VI = VO>
SIMD_INLINE
std::pair<VO, VO> forward_sse_op0(const avx_reg_i& lhs) noexcept
{
    static_assert(VI::n_regs() == 1);

    sse_reg_i l_low, l_high;
    detail::split_reg(lhs, l_low, l_high);
    auto low_result  = OP::template apply<VO, VI>(VI(l_low));
    auto high_result = OP::template apply<VO, VI>(VI(l_high));
    return std::make_pair(low_result, high_result);
}

}  // namespace detail
} } }  // namespace simd::kernel::avx
