#pragma once

namespace simd { namespace kernel { namespace avx2 {
using namespace types;
/// eq
template <typename T, size_t W>
struct eq<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static VecBool<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        static_check_supported_type<T>();

        VecBool<T, W> ret;
        constexpr auto nregs = VecBool<T, W>::n_regs();
        SIMD_IF_CONSTEXPR(sizeof(T) == 1) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm256_cmpeq_epi8(lhs.reg(idx), rhs.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 2) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm256_cmpeq_epi16(lhs.reg(idx), rhs.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 4) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm256_cmpeq_epi32(lhs.reg(idx), rhs.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 8) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm256_cmpeq_epi64(lhs.reg(idx), rhs.reg(idx));
            }
        }
        return ret;
    }
    SIMD_INLINE
    static VecBool<T, W> apply(const VecBool<T, W>& lhs, const VecBool<T, W>& rhs) noexcept
    {
        static_check_supported_type<T>();
        VecBool<T, W> ret;
        constexpr auto nregs = Vec<T, W>::n_regs();
        SIMD_IF_CONSTEXPR(sizeof(T) == 1) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm256_cmpeq_epi8(lhs.reg(idx), rhs.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 2) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm256_cmpeq_epi16(lhs.reg(idx), rhs.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 4) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm256_cmpeq_epi32(lhs.reg(idx), rhs.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 8) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm256_cmpeq_epi64(lhs.reg(idx), rhs.reg(idx));
            }
        }
        return ret;
    }
};

/// ne for integral
/// a != b => ~(a == b)
template <typename T, size_t W>
struct ne<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static VecBool<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        return ~(lhs == rhs);
    }
    SIMD_INLINE
    static VecBool<T, W> apply(const VecBool<T, W>& lhs, const VecBool<T, W>& rhs) noexcept
    {
        static_check_supported_type<T>();

        VecBool<T, W> ret;
        constexpr auto nregs = VecBool<T, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm256_xor_si256(lhs.reg(idx), rhs.reg(idx));
        }
    }
};

/// gt for integral
/// a > b
template <typename T, size_t W>
struct gt<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static VecBool<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        static_check_supported_type<T>();

        VecBool<T, W> ret;
        constexpr bool is_signed = std::is_signed<T>::value;
        constexpr auto nregs = VecBool<T, W>::n_regs();
        SIMD_IF_CONSTEXPR(sizeof(T) == 1) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = is_signed
                    ? _mm256_cmpgt_epi8(lhs.reg(idx), rhs.reg(idx))
                    : _mm256_cmpgt_epi8(
                        _mm256_xor_si256(lhs.reg(idx), _mm256_set1_epi8(std::numeric_limits<int8_t>::lowest())),
                        _mm256_xor_si256(rhs.reg(idx), _mm256_set1_epi8(std::numeric_limits<int8_t>::lowest()))
                    );
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 2) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = is_signed
                    ? _mm256_cmpgt_epi16(lhs.reg(idx), rhs.reg(idx))
                    : _mm256_cmpgt_epi16(
                        _mm256_xor_si256(lhs.reg(idx), _mm256_set1_epi16(std::numeric_limits<int16_t>::lowest())),
                        _mm256_xor_si256(rhs.reg(idx), _mm256_set1_epi16(std::numeric_limits<int16_t>::lowest()))
                    );
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 4) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = is_signed
                    ? _mm256_cmpgt_epi32(lhs.reg(idx), rhs.reg(idx))
                    : _mm256_cmpgt_epi32(
                        _mm256_xor_si256(lhs.reg(idx), _mm256_set1_epi32(std::numeric_limits<int32_t>::lowest())),
                        _mm256_xor_si256(rhs.reg(idx), _mm256_set1_epi32(std::numeric_limits<int32_t>::lowest()))
                    );
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 8) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = is_signed
                    ? _mm256_cmpgt_epi64(lhs.reg(idx), rhs.reg(idx))
                    : _mm256_cmpgt_epi64(
                        _mm256_xor_si256(lhs.reg(idx), _mm256_set1_epi64x(std::numeric_limits<int64_t>::lowest())),
                        _mm256_xor_si256(rhs.reg(idx), _mm256_set1_epi64x(std::numeric_limits<int64_t>::lowest()))
                    );
            }
        }
        return ret;
    }
};

/// lt for integral
/// a < b => b > a
template <typename T, size_t W>
struct lt<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static VecBool<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        auto ret = avx2::gt<T, W>::apply(rhs, lhs);
        return ret;
    }
};

/// le for integral
/// a <= b => ~(a > b)
template <typename T, size_t W>
struct le<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static VecBool<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        auto ret = ~(avx2::gt<T, W>::apply(lhs, rhs));
        return ret;
    }
};

/// ge for integral
/// a >= b => ~(b > a)
template <typename T, size_t W>
struct ge<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static VecBool<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        auto ret = ~(avx2::gt<T, W>::apply(rhs, lhs));
        return ret;
    }
};

} } } // namespace simd::kernel::avx2
