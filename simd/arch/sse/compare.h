#pragma once

namespace simd { namespace kernel { namespace sse {
using namespace types;

namespace detail {
template <typename T>
struct cmp_eq_functor {
    template <typename U = T, REQUIRES(IS_INT_SIZE_1(U))>
    SIMD_INLINE
    sse_reg_i operator ()(const sse_reg_i& x, const sse_reg_i& y) noexcept {
        return _mm_cmpeq_epi8(x, y);
    }
    template <typename U = T, REQUIRES(IS_INT_SIZE_2(U))>
    SIMD_INLINE
    sse_reg_i operator ()(const sse_reg_i& x, const sse_reg_i& y) noexcept {
        return _mm_cmpeq_epi16(x, y);
    }
    template <typename U = T, REQUIRES(IS_INT_SIZE_4(U))>
    SIMD_INLINE
    sse_reg_i operator ()(const sse_reg_i& x, const sse_reg_i& y) noexcept {
        return _mm_cmpeq_epi32(x, y);
    }
    template <typename U = T, REQUIRES(IS_INT_SIZE_8(U))>
    SIMD_INLINE
    sse_reg_i operator ()(const sse_reg_i& x, const sse_reg_i& y) noexcept {
        return _mm_cmpeq_epi64(x, y);
    }

    SIMD_INLINE
    sse_reg_f operator ()(const sse_reg_f& x, const sse_reg_f& y) noexcept {
        return _mm_cmpeq_ps(x, y);
    }
    SIMD_INLINE
    sse_reg_d operator ()(const sse_reg_d& x, const sse_reg_d& y) noexcept {
        return _mm_cmpeq_pd(x, y);
    }

    SIMD_INLINE
    sse_reg_f operator ()(const sse_reg_f& x, const sse_reg_f& y, int) noexcept {
        return _mm_castsi128_ps(
                    _mm_cmpeq_epi32(
                        _mm_castps_si128(x),
                        _mm_castps_si128(y)
                    )
                );
    }
    SIMD_INLINE
    sse_reg_d operator ()(const sse_reg_d& x, const sse_reg_d& y, int) noexcept {
        return _mm_castsi128_pd(
                    _mm_cmpeq_epi64(
                        _mm_castpd_si128(x),
                        _mm_castpd_si128(y)
                    )
                );
    }
};

template <typename T>
struct cmp_ne_functor {
    SIMD_INLINE
    sse_reg_f operator ()(const sse_reg_f& x, const sse_reg_f& y) noexcept {
        return _mm_cmpneq_ps(x, y);
    }
    SIMD_INLINE
    sse_reg_d operator ()(const sse_reg_d& x, const sse_reg_d& y) noexcept {
        return _mm_cmpneq_pd(x, y);
    }

    SIMD_INLINE
    sse_reg_f operator ()(const sse_reg_f& x, const sse_reg_f& y, int) noexcept {
        return _mm_xor_ps(x, y);
    }
    SIMD_INLINE
    sse_reg_d operator ()(const sse_reg_d& x, const sse_reg_d& y, int) noexcept {
        return _mm_xor_pd(x, y);
    }
};

template <typename T>
struct cmp_lt_functor {
    SIMD_INLINE
    sse_reg_f operator ()(const sse_reg_f& x, const sse_reg_f& y) noexcept {
        return _mm_cmplt_ps(x, y);
    }
    SIMD_INLINE
    sse_reg_d operator ()(const sse_reg_d& x, const sse_reg_d& y) noexcept {
        return _mm_cmplt_pd(x, y);
    }
};

template <typename T>
struct cmp_le_functor {
    SIMD_INLINE
    sse_reg_f operator ()(const sse_reg_f& x, const sse_reg_f& y) noexcept {
        return _mm_cmple_ps(x, y);
    }
    SIMD_INLINE
    sse_reg_d operator ()(const sse_reg_d& x, const sse_reg_d& y) noexcept {
        return _mm_cmple_pd(x, y);
    }
};

template <typename T>
struct cmp_gt_functor {
    SIMD_INLINE
    sse_reg_f operator ()(const sse_reg_f& x, const sse_reg_f& y) noexcept {
        return _mm_cmpgt_ps(x, y);
    }
    SIMD_INLINE
    sse_reg_d operator ()(const sse_reg_d& x, const sse_reg_d& y) noexcept {
        return _mm_cmpgt_pd(x, y);
    }
};

template <typename T>
struct cmp_ge_functor {
    SIMD_INLINE
    sse_reg_f operator ()(const sse_reg_f& x, const sse_reg_f& y) noexcept {
        return _mm_cmpge_ps(x, y);
    }
    SIMD_INLINE
    sse_reg_d operator ()(const sse_reg_d& x, const sse_reg_d& y) noexcept {
        return _mm_cmpge_pd(x, y);
    }
};

}  // namespace detail

/// eq
template <typename T, size_t W>
struct eq<T, W>
    : ops::cmp_binary_op<T, W, detail::cmp_eq_functor<T>>
{};

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

template <typename T, size_t W>
struct ne<T, W, REQUIRE_FLOATING(T)>
    : ops::cmp_binary_op<T, W, detail::cmp_ne_functor<T>>
{};

/// lt
template <typename T, size_t W>
struct lt<T, W, REQUIRE_FLOATING(T)>
    : ops::cmp_binary_op<T, W, detail::cmp_lt_functor<T>>
{};

/// le
template <typename T, size_t W>
struct le<T, W, REQUIRE_FLOATING(T)>
    : ops::cmp_binary_op<T, W, detail::cmp_le_functor<T>>
{};

/// ge
template <typename T, size_t W>
struct ge<T, W, REQUIRE_FLOATING(T)>
    : ops::cmp_binary_op<T, W, detail::cmp_ge_functor<T>>
{};

/// gt
template <typename T, size_t W>
struct gt<T, W, REQUIRE_FLOATING(T)>
    : ops::cmp_binary_op<T, W, detail::cmp_gt_functor<T>>
{};

/// lt for integral
/// a < b
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

/// le for integral
/// a <= b => ~(b < a)
template <typename T, size_t W>
struct le<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static VecBool<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        auto ret = ~(sse::lt<T, W>::apply(rhs, lhs));
        return ret;
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

/// ge for integral
/// a >= b => ~(a < b)
template <typename T, size_t W>
struct ge<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static VecBool<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        auto ret = ~(sse::lt<T, W>::apply(lhs, rhs));
        return ret;
    }
};

} } } // namespace simd::kernel::sse
