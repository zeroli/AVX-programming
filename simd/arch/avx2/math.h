#pragma once

namespace simd { namespace kernel { namespace avx2 {
using namespace types;
/// abs

namespace detail {
struct sse_abs {
    template <typename VO, typename VI>
    SIMD_INLINE
    static VO apply(const VI& x) noexcept {
        return kernel::abs(x, SSE{});
    }
};
}  // namespace detail

template <typename T, size_t W>
struct abs<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& x) noexcept
    {
        static_check_supported_type<T>();

        Vec<T, W> ret;
        constexpr int nregs = Vec<T, W>::n_regs();
        SIMD_IF_CONSTEXPR(sizeof(T) == 1) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm256_abs_epi8(x.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 2) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm256_abs_epi16(x.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 4) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm256_abs_epi32(x.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 8) {
            // _mm256_abs_epi64 is provided in AVX512F + AVX512VL
            constexpr auto reg_lanes = Vec<T, W>::reg_lanes();
            using sse_vec_t = Vec<T, reg_lanes/2>;
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = detail::forward_sse_op<detail::sse_abs, sse_vec_t>(x.reg(idx));
            }
        }
        return ret;
    }
};

} } } // namespace simd::kernel::avx2
