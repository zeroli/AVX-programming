#pragma once

namespace simd { namespace kernel { namespace avx2 {
using namespace types;

/// min
namespace detail {
/// _mm256_min_epi64/_mm256_min_epu64 are provided in AVX512F + AVX512VL
/// therefore, forward avx2 min op for 64bits to SSE ISA
SIMD_INLINE
static avx_reg_i algo_min_epi64(const avx_reg_i& x, const avx_reg_i& y) noexcept
{
    using sse_vec_t = Vec<int64_t, 2>;
    return detail::forward_sse_op<detail::sse_min, sse_vec_t>(x, y);
}
SIMD_INLINE
static avx_reg_i algo_min_epu64(const avx_reg_i& x, const avx_reg_i& y) noexcept
{
    using sse_vec_t = Vec<uint64_t, 2>;
    return detail::forward_sse_op<detail::sse_min, sse_vec_t>(x, y);
}
}  // namespace detail

template <typename T, size_t W>
struct min<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        static_check_supported_type<T>();

        Vec<T, W> ret;
        constexpr int nregs = Vec<T, W>::n_regs();
        constexpr bool is_signed = std::is_signed<T>::value;
        SIMD_IF_CONSTEXPR(sizeof(T) == 1) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = is_signed
                                ? _mm256_min_epi8(lhs.reg(idx), rhs.reg(idx))
                                : _mm256_min_epu8(lhs.reg(idx), rhs.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 2) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = is_signed
                                ? _mm256_min_epi16(lhs.reg(idx), rhs.reg(idx))
                                : _mm256_min_epu16(lhs.reg(idx), rhs.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 4) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = is_signed
                                ? _mm256_min_epi32(lhs.reg(idx), rhs.reg(idx))
                                : _mm256_min_epu32(lhs.reg(idx), rhs.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 8) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = is_signed
                                ? detail::algo_min_epi64(lhs.reg(idx), rhs.reg(idx))
                                : detail::algo_min_epu64(lhs.reg(idx), rhs.reg(idx));
            }
        }
        return ret;
    }
};

/// max
namespace detail {
/// _mm256_max_epi64/_mm256_max_epu64 are provided in AVX512F + AVX512VL
/// therefore, forward avx2 max op for 64bits to SSE ISA
SIMD_INLINE
static avx_reg_i algo_max_epi64(const avx_reg_i& x, const avx_reg_i& y) noexcept
{
    using sse_vec_t = Vec<int64_t, 2>;
    return detail::forward_sse_op<detail::sse_max, sse_vec_t>(x, y);
}
SIMD_INLINE
static avx_reg_i algo_max_epu64(const avx_reg_i& x, const avx_reg_i& y) noexcept
{
    using sse_vec_t = Vec<uint64_t, 2>;
    return detail::forward_sse_op<detail::sse_max, sse_vec_t>(x, y);
}
}  // namespace detail
template <typename T, size_t W>
struct max<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        static_check_supported_type<T>();

        Vec<T, W> ret;
        constexpr int nregs = Vec<T, W>::n_regs();
        constexpr bool is_signed = std::is_signed<T>::value;
        SIMD_IF_CONSTEXPR(sizeof(T) == 1) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = is_signed
                                ? _mm256_max_epi8(lhs.reg(idx), rhs.reg(idx))
                                : _mm256_max_epu8(lhs.reg(idx), rhs.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 2) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = is_signed
                                ? _mm256_max_epi16(lhs.reg(idx), rhs.reg(idx))
                                : _mm256_max_epu16(lhs.reg(idx), rhs.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 4) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = is_signed
                                ? _mm256_max_epi32(lhs.reg(idx), rhs.reg(idx))
                                : _mm256_max_epu32(lhs.reg(idx), rhs.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 8) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = is_signed
                                ? detail::algo_max_epi64(lhs.reg(idx), rhs.reg(idx))
                                : detail::algo_max_epu64(lhs.reg(idx), rhs.reg(idx));
            }
        }
        return ret;
    }
};

template <typename T, size_t W>
struct all_of<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static bool apply(const VecBool<T, W>& x) noexcept
    {
        static_check_supported_type<T>();

        bool ret = true;
        constexpr auto nregs = VecBool<T, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret = ret && (_mm256_movemask_epi8(x.reg(idx)) == 0xFFFFFFFF);
        }
        return ret;
    }
};

} } } // namespace simd::kernel::avx2
