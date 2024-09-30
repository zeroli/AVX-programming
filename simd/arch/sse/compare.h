#pragma once

namespace simd { namespace kernel { namespace sse {
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
        constexpr int nregs = VecBool<T, W>::n_regs();
        SIMD_IF_CONSTEXPR(sizeof(T) == 1) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm_cmpeq_epi8(lhs.reg(idx), rhs.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 2) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm_cmpeq_epi16(lhs.reg(idx), rhs.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 4) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm_cmpeq_epi32(lhs.reg(idx), rhs.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 8) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm_cmpeq_epi64(lhs.reg(idx), rhs.reg(idx));
            }
        }
        return ret;
    }
    SIMD_INLINE
    static VecBool<T, W> apply(const VecBool<T, W>& lhs, const VecBool<T, W>& rhs) noexcept
    {
        static_check_supported_type<T>();
        VecBool<T, W> ret;
        constexpr int nregs = Vec<T, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_cmpeq_epi64(lhs.reg(idx), rhs.reg(idx));
        }
    }
};

template <size_t W>
struct eq<float, W>
{
    SIMD_INLINE
    static VecBool<float, W> apply(const Vec<float, W>& lhs, const Vec<float, W>& rhs) noexcept
    {
        VecBool<float, W> ret;
        constexpr int nregs = VecBool<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_cmpeq_ps(lhs.reg(idx), rhs.reg(idx));
        }
        return ret;
    }
    SIMD_INLINE
    static VecBool<float, W> apply(const VecBool<float, W>& lhs, const VecBool<float, W>& rhs) noexcept
    {
        VecBool<float, W> ret;
        constexpr int nregs = VecBool<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_castsi128_ps(
                                _mm_cmpeq_epi64(
                                    _mm_castps_si128(lhs.reg(idx)),
                                    _mm_castps_si128(rhs.reg(idx))
                                )
                            );
        }
    }
};

template <size_t W>
struct eq<double, W>
{
    SIMD_INLINE
    static VecBool<double, W> apply(const Vec<double, W>& lhs, const Vec<double, W>& rhs) noexcept
    {
        VecBool<double, W> ret;
        constexpr int nregs = VecBool<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_cmpeq_pd(lhs.reg(idx), rhs.reg(idx));
        }
        return ret;
    }
    SIMD_INLINE
    static VecBool<double, W> apply(const VecBool<double, W>& lhs, const VecBool<double, W>& rhs) noexcept
    {
        VecBool<double, W> ret;
        constexpr int nregs = VecBool<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_castsi128_pd(
                                _mm_cmpeq_epi64(
                                    _mm_castpd_si128(lhs.reg(idx)),
                                    _mm_castpd_si128(rhs.reg(idx))
                                )
                            );
        }
    }
};

/// ne
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
        constexpr int nregs = VecBool<T, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_xor_si128(lhs.reg(idx), rhs.reg(idx));
        }
    }
};

template <size_t W>
struct ne<float, W>
{
    SIMD_INLINE
    static VecBool<float, W> apply(const Vec<float, W>& lhs, const Vec<float, W>& rhs) noexcept
    {
        VecBool<float, W> ret;
        constexpr int nregs = VecBool<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_cmpneq_ps(lhs.reg(idx), rhs.reg(idx));
        }
        return ret;
    }
    SIMD_INLINE
    static VecBool<float, W> apply(const VecBool<float, W>& lhs, const VecBool<float, W>& rhs) noexcept
    {
        VecBool<float, W> ret;
        constexpr int nregs = VecBool<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_xor_ps(lhs.reg(idx), rhs.reg(idx));
        }
    }
};

template <size_t W>
struct ne<double, W>
{
    SIMD_INLINE
    static VecBool<double, W> apply(const Vec<double, W>& lhs, const Vec<double, W>& rhs) noexcept
    {
        VecBool<double, W> ret;
        constexpr int nregs = VecBool<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_cmpneq_pd(lhs.reg(idx), rhs.reg(idx));
        }
        return ret;
    }
    SIMD_INLINE
    static VecBool<double, W> apply(const VecBool<double, W>& lhs, const VecBool<double, W>& rhs) noexcept
    {
        VecBool<double, W> ret;
        constexpr int nregs = VecBool<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_xor_pd(lhs.reg(idx), rhs.reg(idx));
        }
    }
};

/// ge
template <size_t W>
struct ge<float, W>
{
    SIMD_INLINE
    static VecBool<float, W> apply(const Vec<float, W>& lhs, const Vec<float, W>& rhs) noexcept
    {
        VecBool<float, W> ret;
        constexpr int nregs = VecBool<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_cmpge_ps(lhs.reg(idx), rhs.reg(idx));
        }
        return ret;
    }
};

template <size_t W>
struct ge<double, W>
{
    SIMD_INLINE
    static VecBool<double, W> apply(const Vec<double, W>& lhs, const Vec<double, W>& rhs) noexcept
    {
        VecBool<double, W> ret;
        constexpr int nregs = VecBool<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_cmpge_pd(lhs.reg(idx), rhs.reg(idx));
        }
        return ret;
    }
};

/// le
template <size_t W>
struct le<float, W>
{
    SIMD_INLINE
    static VecBool<float, W> apply(const Vec<float, W>& lhs, const Vec<float, W>& rhs) noexcept
    {
        VecBool<float, W> ret;
        constexpr int nregs = VecBool<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_cmple_ps(lhs.reg(idx), rhs.reg(idx));
        }
        return ret;
    }
};

template <size_t W>
struct le<double, W>
{
    SIMD_INLINE
    static VecBool<double, W> apply(const Vec<double, W>& lhs, const Vec<double, W>& rhs) noexcept
    {
        VecBool<double, W> ret;
        constexpr int nregs = VecBool<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_cmple_pd(lhs.reg(idx), rhs.reg(idx));
        }
        return ret;
    }
};

/// lt
template <size_t W>
struct lt<float, W>
{
    SIMD_INLINE
    static VecBool<float, W> apply(const Vec<float, W>& lhs, const Vec<float, W>& rhs) noexcept
    {
        VecBool<float, W> ret;
        constexpr int nregs = VecBool<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_cmplt_ps(lhs.reg(idx), rhs.reg(idx));
        }
        return ret;
    }
};

template <size_t W>
struct lt<double, W>
{
    SIMD_INLINE
    static VecBool<double, W> apply(const Vec<double, W>& lhs, const Vec<double, W>& rhs) noexcept
    {
        VecBool<double, W> ret;
        constexpr int nregs = VecBool<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_cmplt_pd(lhs.reg(idx), rhs.reg(idx));
        }
        return ret;
    }
};

/// gt
template <size_t W>
struct gt<float, W>
{
    SIMD_INLINE
    static VecBool<float, W> apply(const Vec<float, W>& lhs, const Vec<float, W>& rhs) noexcept
    {
        VecBool<float, W> ret;
        constexpr int nregs = VecBool<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_cmpgt_ps(lhs.reg(idx), rhs.reg(idx));
        }
        return ret;
    }
};

template <size_t W>
struct gt<double, W>
{
    SIMD_INLINE
    static VecBool<double, W> apply(const Vec<double, W>& lhs, const Vec<double, W>& rhs) noexcept
    {
        VecBool<double, W> ret;
        constexpr int nregs = VecBool<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_cmpgt_pd(lhs.reg(idx), rhs.reg(idx));
        }
        return ret;
    }
};

template <typename T, size_t W>
struct lt<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static VecBool<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        static_check_supported_type<T>();

        VecBool<T, W> ret;
        constexpr bool is_signed = std::is_signed<T>::value;
        constexpr int nregs = VecBool<T, W>::n_regs();
        SIMD_IF_CONSTEXPR(sizeof(T) == 1) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = is_signed
                    ? _mm_cmplt_epi8(lhs.reg(idx), rhs.reg(idx))
                    : _mm_cmplt_epi8(
                        _mm_xor_si128(lhs.reg(idx), _mm_set1_epi8(std::numeric_limits<int8_t>::lowest())),
                        _mm_xor_si128(rhs.reg(idx), _mm_set1_epi8(std::numeric_limits<int8_t>::lowest()))
                    );
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 2) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = is_signed
                    ? _mm_cmplt_epi16(lhs.reg(idx), rhs.reg(idx))
                    : _mm_cmplt_epi16(
                        _mm_xor_si128(lhs.reg(idx), _mm_set1_epi16(std::numeric_limits<int16_t>::lowest())),
                        _mm_xor_si128(rhs.reg(idx), _mm_set1_epi16(std::numeric_limits<int16_t>::lowest()))
                    );
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 4) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = is_signed
                    ? _mm_cmplt_epi32(lhs.reg(idx), rhs.reg(idx))
                    : _mm_cmplt_epi32(
                        _mm_xor_si128(lhs.reg(idx), _mm_set1_epi32(std::numeric_limits<int32_t>::lowest())),
                        _mm_xor_si128(rhs.reg(idx), _mm_set1_epi32(std::numeric_limits<int32_t>::lowest()))
                    );
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 8) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = is_signed
                    ? _mm_cmpgt_epi64(rhs.reg(idx), lhs.reg(idx))
                    : _mm_cmpgt_epi64(
                        _mm_xor_si128(rhs.reg(idx), _mm_set1_epi64x(std::numeric_limits<int64_t>::lowest())),
                        _mm_xor_si128(lhs.reg(idx), _mm_set1_epi64x(std::numeric_limits<int64_t>::lowest()))
                    );
            }
        }
        return ret;
    }
};

// a <= b => ~(b < a)
template <typename T, size_t W>
struct le<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static VecBool<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        return ~(sse::lt<T, W>::apply(rhs, lhs));
    }
};

template <typename T, size_t W>
struct gt<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static VecBool<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        static_check_supported_type<T>();

        VecBool<T, W> ret;
        constexpr bool is_signed = std::is_signed<T>::value;
        constexpr int nregs = VecBool<T, W>::n_regs();
        SIMD_IF_CONSTEXPR(sizeof(T) == 1) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = is_signed
                    ? _mm_cmpgt_epi8(lhs.reg(idx), rhs.reg(idx))
                    : _mm_cmpgt_epi8(
                        _mm_xor_si128(lhs.reg(idx), _mm_set1_epi8(std::numeric_limits<int8_t>::lowest())),
                        _mm_xor_si128(rhs.reg(idx), _mm_set1_epi8(std::numeric_limits<int8_t>::lowest()))
                    );
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 2) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = is_signed
                    ? _mm_cmpgt_epi16(lhs.reg(idx), rhs.reg(idx))
                    : _mm_cmpgt_epi16(
                        _mm_xor_si128(lhs.reg(idx), _mm_set1_epi16(std::numeric_limits<int16_t>::lowest())),
                        _mm_xor_si128(rhs.reg(idx), _mm_set1_epi16(std::numeric_limits<int16_t>::lowest()))
                    );
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 4) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = is_signed
                    ? _mm_cmpgt_epi32(lhs.reg(idx), rhs.reg(idx))
                    : _mm_cmpgt_epi32(
                        _mm_xor_si128(lhs.reg(idx), _mm_set1_epi32(std::numeric_limits<int32_t>::lowest())),
                        _mm_xor_si128(rhs.reg(idx), _mm_set1_epi32(std::numeric_limits<int32_t>::lowest()))
                    );
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 8) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = is_signed
                    ? _mm_cmpgt_epi64(lhs.reg(idx), rhs.reg(idx))
                    : _mm_cmpgt_epi64(
                        _mm_xor_si128(lhs.reg(idx), _mm_set1_epi64x(std::numeric_limits<int64_t>::lowest())),
                        _mm_xor_si128(rhs.reg(idx), _mm_set1_epi64x(std::numeric_limits<int64_t>::lowest()))
                    );
            }
        }
        return ret;
    }
};

// a >= b => ~(b < a)
template <typename T, size_t W>
struct ge<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static VecBool<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        return ~(sse::gt<T, W>::apply(rhs, lhs));
    }
};

} } } // namespace simd::kernel::sse
