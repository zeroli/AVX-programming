#pragma once

namespace simd { namespace kernel { namespace avx2 {
using namespace types;

namespace detail {
SIMD_INLINE
avx_reg_i bitwise_slli_epi8(const avx_reg_i& x, int32_t y) noexcept
{
    return _mm256_and_si256(_mm256_set1_epi8(0xFF << y),
                            _mm256_slli_epi32(x, y));
}
SIMD_INLINE
avx_reg_i bitwise_sllv_epi8(const avx_reg_i& x, const avx_reg_i& y) noexcept
{
    /// FIXME: use efficient way to shift right by variable amount in 8bits
    /// dispatch to generic naive way as temporary
    Vec<int8_t, 32> vx(x), vy(y);
    avx_reg_i ret = kernel::bitwise_rshift(vx, vy, Generic{}).reg();
    return ret;
}
SIMD_INLINE
avx_reg_i bitwise_sllv_epi16(const avx_reg_i& x, const avx_reg_i& y) noexcept
{
    // _mm256_sllv_epi16 provided by AVX512BW + AVX512VL
    /// FIXME: use efficient way to shift right by variable amount in 8bits
    /// dispatch to generic naive way as temporary
    Vec<int16_t, 16> vx(x), vy(y);
    avx_reg_i ret = kernel::bitwise_rshift(vx, vy, Generic{}).reg();
    return ret;
}
}  // namespace detail

template <typename T, size_t W>
struct bitwise_lshift<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& lhs, int32_t rhs) noexcept
    {
        static_check_supported_type<T>();

        Vec<T, W> ret;
        constexpr auto nregs = Vec<T, W>::n_regs();
        SIMD_IF_CONSTEXPR(sizeof(T) == 1) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = detail::bitwise_slli_epi8(lhs.reg(idx), rhs);
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 2) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm256_slli_epi16(lhs.reg(idx), rhs);
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 4) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm256_slli_epi32(lhs.reg(idx), rhs);
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 8) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm256_slli_epi64(lhs.reg(idx), rhs);
            }
        }
        return ret;
    }
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        static_check_supported_type<T>();

        Vec<T, W> ret;
        constexpr auto nregs = Vec<T, W>::n_regs();
        SIMD_IF_CONSTEXPR(sizeof(T) == 1) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = detail::bitwise_sllv_epi8(lhs.reg(idx), rhs.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 2) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = detail::bitwise_sllv_epi16(lhs.reg(idx), rhs.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 4) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm256_sllv_epi32(lhs.reg(idx), rhs.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 8) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm256_sllv_epi64(lhs.reg(idx), rhs.reg(idx));
            }
        }
        return ret;
    }
};

namespace detail {
SIMD_INLINE
static avx_reg_i bitwise_sra_epi8(const avx_reg_i& x, int32_t y) noexcept
{
    return x;  // TODO
}

SIMD_INLINE
static avx_reg_i bitwise_srl_epi8(const avx_reg_i& x, int32_t y) noexcept
{
    return x;  // TODO
}

SIMD_INLINE
static avx_reg_i bitwise_sra_epi64(const avx_reg_i& x, int32_t y) noexcept
{
    return x;  // TODO
}

SIMD_INLINE
static avx_reg_i bitwise_srl_epi64(const avx_reg_i& x, int32_t y) noexcept
{
    return x;  // TODO
}
}  // namespace detail

template <typename T, size_t W>
struct bitwise_rshift<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& lhs, int32_t rhs) noexcept
    {
        static_check_supported_type<T>();

        Vec<T, W> ret;
        constexpr bool is_signed = std::is_signed<T>::value;
        constexpr auto nregs = Vec<T, W>::n_regs();
        SIMD_IF_CONSTEXPR(sizeof(T) == 1) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = is_signed
                    ? detail::bitwise_sra_epi8(lhs.reg(idx), rhs)
                    : detail::bitwise_srl_epi8(lhs.reg(idx), rhs);
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 2) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = is_signed
                    ? _mm256_sra_epi16(lhs.reg(idx), _mm256_set1_epi64x(rhs))
                    : _mm256_srl_epi16(lhs.reg(idx), _mm256_set1_epi64x(rhs));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 4) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = is_signed
                    ? _mm256_sra_epi32(lhs.reg(idx), _mm256_set1_epi64x(rhs))
                    : _mm256_srl_epi32(lhs.reg(idx), _mm256_set1_epi64x(rhs));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 8) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = is_signed
                    ? detail::bitwise_sra_epi64(lhs.reg(idx), rhs)
                    : detail::bitwise_srl_epi64(lhs.reg(idx), rhs);
            }
        }
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
        constexpr auto nregs = Vec<T, W>::n_regs();
        auto mask = avx::detail::make_mask<T>();
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
        constexpr auto nregs = VecBool<T, W>::n_regs();
        auto mask = avx::detail::make_mask<T>();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm256_xor_si256(x.reg(idx), mask);
        }
        return ret;
    }
};

} } } // namespace simd::kernel::avx2
