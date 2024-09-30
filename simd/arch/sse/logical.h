#pragma once

namespace simd { namespace kernel { namespace sse {
using namespace types;

namespace detail {
struct and_functor {
    sse_reg_i operator ()(const sse_reg_i& x, const sse_reg_i& y) const noexcept {
        return _mm_and_si128(x, y);
    }
    sse_reg_f operator ()(const sse_reg_f& x, const sse_reg_f& y) const noexcept {
        return _mm_and_ps(x, y);
    }
    sse_reg_d operator ()(const sse_reg_d& x, const sse_reg_d& y) const noexcept {
        return _mm_and_pd(x, y);
    }
};
struct or_functor {
    sse_reg_i operator ()(const sse_reg_i& x, const sse_reg_i& y) const noexcept {
        return _mm_or_si128(x, y);
    }
    sse_reg_f operator ()(const sse_reg_f& x, const sse_reg_f& y) const noexcept {
        return _mm_or_ps(x, y);
    }
    sse_reg_d operator ()(const sse_reg_d& x, const sse_reg_d& y) const noexcept {
        return _mm_or_pd(x, y);
    }
};
struct xor_functor {
    sse_reg_i operator ()(const sse_reg_i& x, const sse_reg_i& y) const noexcept {
        return _mm_xor_si128(x, y);
    }
    sse_reg_f operator ()(const sse_reg_f& x, const sse_reg_f& y) const noexcept {
        return _mm_xor_ps(x, y);
    }
    sse_reg_d operator ()(const sse_reg_d& x, const sse_reg_d& y) const noexcept {
        return _mm_xor_pd(x, y);
    }
};

template <typename T, size_t W, typename F>
struct bitwise_op
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        Vec<T, W> ret;
        constexpr auto nregs = Vec<T, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = F()(lhs.reg(idx), rhs.reg(idx));
        }
        return ret;
    }
};
}  // namespace detail

template <typename T, size_t W>
struct bitwise_and<T, W> : detail::bitwise_op<T, W, detail::and_functor>
{
};

template <typename T, size_t W>
struct bitwise_or<T, W> : detail::bitwise_op<T, W, detail::or_functor>
{
};

template <typename T, size_t W>
struct bitwise_xor<T, W> : detail::bitwise_op<T, W, detail::xor_functor>
{
};

namespace detail {
inline static sse_reg_i bitwise_sra_int8(const sse_reg_i& x, int32_t y)
{
    sse_reg_i sign_mask = _mm_set1_epi16((0xFF00 >> y) & 0x00FF);
    sse_reg_i cmp_is_negative = _mm_cmpgt_epi8(_mm_setzero_si128(), x);
    sse_reg_i res = _mm_srai_epi16(x, y);
    return _mm_or_si128(_mm_and_si128(sign_mask, cmp_is_negative),
                        _mm_andnot_si128(sign_mask, res));
}
inline static sse_reg_i bitwise_srl_int8(const sse_reg_i& x, int32_t y)
{
    return _mm_and_si128(_mm_set1_epi8(0xFF >> y), _mm_srli_epi32(x, y));
}
inline static sse_reg_i bitwise_sra_int64(const sse_reg_i& x, int32_t y)
{
    return _mm_or_si128(
            _mm_srli_epi64(x, y),
            _mm_slli_epi64(
                _mm_srai_epi32(_mm_shuffle_epi32(x, _MM_SHUFFLE(3, 3, 1, 1)), 32),
                64 - y
            )
        );
}
inline static sse_reg_i bitwise_srl_int64(const sse_reg_i& x, int32_t y)
{
    return _mm_srli_epi64(x, y);
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
                ret.reg(idx) = _mm_and_si128(_mm_set1_epi8(0xFF << y),
                                             _mm_slli_epi32(x.reg(idx), y));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 2) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm_slli_epi16(x.reg(idx), y);
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 4) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm_slli_epi32(x.reg(idx), y);
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 8) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm_slli_epi64(x.reg(idx), y);
            }
        }
        return ret;
    }
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& x, const Vec<T, W>& y) noexcept
    {
        // TODO:
        return x;
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
        constexpr bool is_signed = std::is_signed<T>::value;
        constexpr auto nregs = Vec<T, W>::n_regs();
        SIMD_IF_CONSTEXPR(sizeof(T) == 1) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = is_signed
                    ? detail::bitwise_sra_int8(x.reg(idx), y)
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
        auto mask = _mm_set1_epi32(-1);
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_xor_si128(x.reg(idx), mask);
        }
        return ret;
    }
    SIMD_INLINE
    static VecBool<T, W> apply(const VecBool<T, W>& x) noexcept
    {
        static_check_supported_type<T>();

        VecBool<T, W> ret;
        constexpr auto nregs = VecBool<T, W>::n_regs();
        auto mask = _mm_set1_epi32(-1);
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_xor_si128(x.reg(idx), mask);
        }
        return ret;
    }
};

template <size_t W>
struct bitwise_not<float, W>
{
    SIMD_INLINE
    static Vec<float, W> apply(const Vec<float, W>& x) noexcept
    {
        Vec<float, W> ret;
        constexpr auto nregs = Vec<float, W>::n_regs();
        auto mask = _mm_castsi128_ps(_mm_set1_epi32(-1));
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_xor_ps(x.reg(idx), mask);
        }
        return ret;
    }
    SIMD_INLINE
    static VecBool<float, W> apply(const VecBool<float, W>& x) noexcept
    {
        VecBool<float, W> ret;
        constexpr auto nregs = VecBool<float, W>::n_regs();
        auto mask = _mm_castsi128_ps(_mm_set1_epi32(-1));
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_xor_ps(x.reg(idx), mask);
        }
        return ret;
    }
};

template <size_t W>
struct bitwise_not<double, W>
{
    SIMD_INLINE
    static Vec<double, W> apply(const Vec<double, W>& x) noexcept
    {
        Vec<double, W> ret;
        constexpr auto nregs = Vec<double, W>::n_regs();
        auto mask = _mm_castsi128_pd(_mm_set1_epi32(-1));
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_xor_pd(x.reg(idx), mask);
        }
        return ret;
    }
    SIMD_INLINE
    static VecBool<double, W> apply(const VecBool<double, W>& x) noexcept
    {
        VecBool<double, W> ret;
        constexpr auto nregs = VecBool<double, W>::n_regs();
        auto mask = _mm_castsi128_pd(_mm_set1_epi32(-1));
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_xor_pd(x.reg(idx), mask);
        }
        return ret;
    }
};

/// bitwise_andnot
template <typename T, size_t W>
struct bitwise_andnot<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const VecBool<T, W>& x, const Vec<T, W>& y) noexcept
    {
        static_check_supported_type<T>();

        Vec<T, W> ret;
        constexpr auto nregs = Vec<T, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_andnot_si128(x.reg(idx), y.reg(idx));
        }
        return ret;
    }
};

template <size_t W>
struct bitwise_andnot<float, W>
{
    SIMD_INLINE
    static Vec<float, W> apply(const VecBool<float, W>& x, const Vec<float, W>& y) noexcept
    {
        Vec<float, W> ret;
        constexpr auto nregs = Vec<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_andnot_ps(x.reg(idx), y.reg(idx));
        }
        return ret;
    }
};

template <size_t W>
struct bitwise_andnot<double, W>
{
    SIMD_INLINE
    static Vec<double, W> apply(const VecBool<double, W>& x, const Vec<double, W>& y) noexcept
    {
        Vec<double, W> ret;
        constexpr auto nregs = Vec<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_andnot_ps(x.reg(idx), y.reg(idx));
        }
        return ret;
    }
};
} } } // namespace simd::kernel::sse
