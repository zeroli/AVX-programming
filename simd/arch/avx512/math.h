#pragma once

namespace simd { namespace kernel { namespace avx512 {
using namespace types;

namespace detail {
template <typename T>
struct abs_functor {
    template <ENABLE_IF(sizeof(T) == 1)>
    avx512_reg_i operator ()(const avx512_reg_i& x) const noexcept {
        return _mm512_abs_epi8(x);
    }
    template <ENABLE_IF(sizeof(T) == 2)>
    avx512_reg_i operator ()(const avx512_reg_i& x) const noexcept {
        return _mm512_abs_epi16(x);
    }
    template <ENABLE_IF(sizeof(T) == 4)>
    avx512_reg_i operator ()(const avx512_reg_i& x) const noexcept {
        return _mm512_abs_epi32(x);
    }
    template <ENABLE_IF(sizeof(T) == 8)>
    avx512_reg_i operator ()(const avx512_reg_i& x) const noexcept {
        return _mm512_abs_epi64(x);
    }

    avx512_reg_f operator ()(const avx512_reg_f& x) const noexcept {
        return _mm512_abs_ps(x);
    }
    avx512_reg_d operator ()(const avx512_reg_d& x) const noexcept {
        return _mm512_abs_pd(x);
    }
};

template <typename T, size_t W, typename F>
struct math_unary_op
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& x) noexcept
    {
        Vec<T, W> ret;
        constexpr auto nregs = Vec<T, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = F()(x.reg(idx));
        }
        return ret;
    }
};
}  // namespace detail

template <typename T, size_t W>
struct abs<T, W> : detail::math_unary_op<T, W, detail::abs_functor<T>>
{
};

} } } // namespace simd::kernel::avx512
