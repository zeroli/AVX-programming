#pragma once

namespace simd { namespace kernel { namespace avx512 {
using namespace types;

namespace detail {
struct avx2_bitwise_and {
    template <typename VO, typename VI>
    static VO apply(const VI& x, const VI& y) noexcept {
        return kernel::bitwise_and(x, y, AVX2{});
    }
};
struct avx2_bitwise_or {
    template <typename VO, typename VI>
    static VO apply(const VI& x, const VI& y) noexcept {
        return kernel::bitwise_or(x, y, AVX2{});
    }
};

struct avx2_bitwise_xor {
    template <typename VO, typename VI>
    static VO apply(const VI& x, const VI& y) noexcept {
        return kernel::bitwise_xor(x, y, AVX2{});
    }
};

template <typename T>
struct and_functor {
    /// epi8/epi16, not available on avx512 (BW)
    template <ENABLE_IF(sizeof(T) == 1 || sizeof(T) == 2)>
    avx512_reg_i operator ()(const avx512_reg_i& x, const avx512_reg_i& y) const noexcept {
        using avx2_vec_t = Vec<int32_t, 8>;
        return detail::forward_avx_op<detail::avx2_bitwise_and, avx2_vec_t>(x, y);
    }
    template <ENABLE_IF(sizeof(T) == 4)>
    avx512_reg_i operator ()(const avx512_reg_i& x, const avx512_reg_i& y) const noexcept {
        return _mm512_and_epi32(x, y);
    }
    template <ENABLE_IF(sizeof(T) == 8)>
    avx512_reg_i operator ()(const avx512_reg_i& x, const avx512_reg_i& y) const noexcept {
        return _mm512_and_epi64(x, y);
    }

    avx512_reg_f operator ()(const avx512_reg_f& x, const avx512_reg_f& y) const noexcept {
        return _mm512_and_ps(x, y);
    }
    avx512_reg_d operator ()(const avx512_reg_d& x, const avx512_reg_d& y) const noexcept {
        return _mm512_and_pd(x, y);
    }
};

template <typename T>
struct or_functor {
    /// epi8/epi16, not available on avx512 (BW)
    template <ENABLE_IF(sizeof(T) == 1 || sizeof(T) == 2)>
    avx512_reg_i operator ()(const avx512_reg_i& x, const avx512_reg_i& y) const noexcept {
        using avx2_vec_t = Vec<int32_t, 8>;
        return detail::forward_avx_op<detail::avx2_bitwise_or, avx2_vec_t>(x, y);
    }
    template <ENABLE_IF(sizeof(T) == 4)>
    avx512_reg_i operator ()(const avx512_reg_i& x, const avx512_reg_i& y) const noexcept {
        return _mm512_or_epi32(x, y);
    }
    template <ENABLE_IF(sizeof(T) == 8)>
    avx512_reg_i operator ()(const avx512_reg_i& x, const avx512_reg_i& y) const noexcept {
        return _mm512_or_epi64(x, y);
    }

    avx512_reg_f operator ()(const avx512_reg_f& x, const avx512_reg_f& y) const noexcept {
        return _mm512_or_ps(x, y);
    }
    avx512_reg_d operator ()(const avx512_reg_d& x, const avx512_reg_d& y) const noexcept {
        return _mm512_or_pd(x, y);
    }
};

template <typename T>
struct xor_functor {
    /// epi8/epi16, not available on avx512 (BW)
    template <ENABLE_IF(sizeof(T) == 1 || sizeof(T) == 2)>
    avx512_reg_i operator ()(const avx512_reg_i& x, const avx512_reg_i& y) const noexcept {
        using avx2_vec_t = Vec<int32_t, 8>;
        return detail::forward_avx_op<detail::avx2_bitwise_xor, avx2_vec_t>(x, y);
    }
    template <ENABLE_IF(sizeof(T) == 4)>
    avx512_reg_i operator ()(const avx512_reg_i& x, const avx512_reg_i& y) const noexcept {
        return _mm512_xor_epi32(x, y);
    }
    template <ENABLE_IF(sizeof(T) == 8)>
    avx512_reg_i operator ()(const avx512_reg_i& x, const avx512_reg_i& y) const noexcept {
        return _mm512_xor_epi64(x, y);
    }

    avx512_reg_f operator ()(const avx512_reg_f& x, const avx512_reg_f& y) const noexcept {
        return _mm512_xor_ps(x, y);
    }
    avx512_reg_d operator ()(const avx512_reg_d& x, const avx512_reg_d& y) const noexcept {
        return _mm512_xor_pd(x, y);
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
struct bitwise_and<T, W> : detail::bitwise_op<T, W, detail::and_functor<T>>
{
};

template <typename T, size_t W>
struct bitwise_or<T, W> : detail::bitwise_op<T, W, detail::or_functor<T>>
{
};

template <typename T, size_t W>
struct bitwise_xor<T, W> : detail::bitwise_op<T, W, detail::xor_functor<T>>
{
};
} } } // namespace simd::kernel::avx512
