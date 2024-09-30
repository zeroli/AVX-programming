#pragma once

namespace simd { namespace kernel { namespace avx2 {
using namespace types;

namespace detail {
SIMD_INLINE
avx_reg_i avx_slli_epi8(const avx_reg_i& x, int32_t y) noexcept
{
    return _mm256_and_si256(_mm256_set1_epi8(0xFF << y),
                            _mm256_slli_epi32(x, y));
}
SIMD_INLINE
avx_reg_i avx_sllv_epi8(const avx_reg_i& x, const avx_reg_i& y) noexcept
{
    avx_reg_i ret;
    return ret;
}
SIMD_INLINE
avx_reg_i avx_sllv_epi16(const avx_reg_i& x, const avx_reg_i& y) noexcept
{
    // _mm256_sllv_epi16 provided by AVX512BW + AVX512VL
    avx_reg_i ret;
    return ret;
}

}  // namespace detail

template <typename T, size_t W>
struct bitwise_lshift<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& x, int32_t y) noexcept
    {
        static_check_supported_type<T>();

        Vec<T, W> ret;
        constexpr auto nregs = Vec<T, W>::n_regs();
        SIMD_IF_CONSTEXPR(sizeof(T) == 1) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = detail::avx_slli_epi8(x.reg(idx), y);
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 2) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm256_slli_epi16(x.reg(idx), y);
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 4) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm256_slli_epi32(x.reg(idx), y);
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 8) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm256_slli_epi64(x.reg(idx), y);
            }
        }
        return ret;
    }
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& x, const Vec<T, W>& y) noexcept
    {
        static_check_supported_type<T>();

        Vec<T, W> ret;
        constexpr auto nregs = Vec<T, W>::n_regs();
        SIMD_IF_CONSTEXPR(sizeof(T) == 1) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = detail::avx_sllv_epi8(x.reg(idx), y.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 2) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = detail::avx_sllv_epi16(x.reg(idx), y.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 4) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm256_sllv_epi32(x.reg(idx), y.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 8) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm256_sllv_epi64(x.reg(idx), y.reg(idx));
            }
        }
        return ret;
    }
};
template <typename T, size_t W>
struct bitwise_rshift<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& x, int32_t y) noexcept
    {
        static_check_supported_type<T>();

        Vec<T, W> ret;
        #if 0
        constexpr bool is_signed = std::is_signed<T>::value;
        constexpr int nregs = Vec<T, W>::n_regs();
        SIMD_IF_CONSTEXPR(sizeof(T) == 1) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = is_signed
                    ? detail::a_sra_int8(x.reg(idx), y)
                    : detail::bitwise_srl_int8(x.reg(idx), y);
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 2) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = is_signed
                    ? _mm_srai_epi16(x.reg(idx), y)
                    : _mm_srli_epi16(x.reg(idx), y);
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 4) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = is_signed
                    ? _mm_srai_epi32(x.reg(idx), y)
                    : _mm_srli_epi32(x.reg(idx), y);
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 8) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = is_signed
                    ? detail::bitwise_sra_int64(x.reg(idx), y)
                    : detail::bitwise_srl_int64(x.reg(idx), y);
            }
        }
        #endif
        return ret;
    }
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& x, const Vec<T, W>& y) noexcept
    {
        // TODO
        return x;
    }
};

/// bitwise_not
template <typename T, size_t W>
struct bitwise_not<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& x) noexcept
    {
        static_check_supported_type<T>();

        Vec<T, W> ret;
        constexpr int nregs = Vec<T, W>::n_regs();
        auto mask = avx::detail::make_mask_i();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm256_xor_si256(x.reg(idx), mask);
        }
        return ret;
    }
    SIMD_INLINE
    static VecBool<T, W> apply(const VecBool<T, W>& x) noexcept
    {
        static_check_supported_type<T>();

        VecBool<T, W> ret;
        constexpr int nregs = VecBool<T, W>::n_regs();
        auto mask = avx::detail::make_mask_i();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm256_xor_si256(x.reg(idx), mask);
        }
        return ret;
    }
};

} } } // namespace simd::kernel::avx2
