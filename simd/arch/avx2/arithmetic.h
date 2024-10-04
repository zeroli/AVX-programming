#pragma once

namespace simd { namespace kernel { namespace avx2 {
using namespace types;

namespace detail {
template <typename T>
struct add_functor {
    template <typename U = T, REQUIRES(IS_INT_SIZE_1(U))>
    SIMD_INLINE
    avx_reg_i operator ()(const avx_reg_i& x, const avx_reg_i& y) noexcept {
        return _mm256_add_epi8(x, y);
    }
    template <typename U = T, REQUIRES(IS_INT_SIZE_2(U))>
    SIMD_INLINE
    avx_reg_i operator ()(const avx_reg_i& x, const avx_reg_i& y) noexcept {
        return _mm256_add_epi16(x, y);
    }
    template <typename U = T, REQUIRES(IS_INT_SIZE_4(U))>
    SIMD_INLINE
    avx_reg_i operator ()(const avx_reg_i& x, const avx_reg_i& y) noexcept {
        return _mm256_add_epi32(x, y);
    }
    template <typename U = T, REQUIRES(IS_INT_SIZE_8(U))>
    SIMD_INLINE
    avx_reg_i operator ()(const avx_reg_i& x, const avx_reg_i& y) noexcept {
        return _mm256_add_epi64(x, y);
    }
};

template <typename T>
struct sub_functor {
    template <typename U = T, REQUIRES(IS_INT_SIZE_1(U))>
    SIMD_INLINE
    avx_reg_i operator ()(const avx_reg_i& x, const avx_reg_i& y) noexcept {
        return _mm256_sub_epi8(x, y);
    }
    template <typename U = T, REQUIRES(IS_INT_SIZE_2(U))>
    SIMD_INLINE
    avx_reg_i operator ()(const avx_reg_i& x, const avx_reg_i& y) noexcept {
        return _mm256_sub_epi16(x, y);
    }
    template <typename U = T, REQUIRES(IS_INT_SIZE_4(U))>
    SIMD_INLINE
    avx_reg_i operator ()(const avx_reg_i& x, const avx_reg_i& y) noexcept {
        return _mm256_sub_epi32(x, y);
    }
    template <typename U = T, REQUIRES(IS_INT_SIZE_8(U))>
    SIMD_INLINE
    avx_reg_i operator ()(const avx_reg_i& x, const avx_reg_i& y) noexcept {
        return _mm256_sub_epi64(x, y);
    }
};

template <typename T>
struct mul_functor {
    avx_reg_i operator ()(const avx_reg_i& x, const avx_reg_i& y) noexcept = delete;
};

template <typename T>
struct div_functor {
    avx_reg_i operator ()(const avx_reg_i& x, const avx_reg_i& y) noexcept = delete;
};

template <typename T>
struct mod_functor {
    avx_reg_i operator ()(const avx_reg_i& x, const avx_reg_i& y) noexcept {
        // TODO:
        return x;
    }
};
}  // namespace detail

/// add
template <typename T, size_t W>
struct add<T, W>
    : ops::arith_binary_op<T, W, detail::add_functor<T>>
{};

/// sub
template <typename T, size_t W>
struct sub<T, W>
    : ops::arith_binary_op<T, W, detail::sub_functor<T>>
{};

/// mul
template <typename T, size_t W>
struct mul<T, W>
    : ops::arith_binary_op<T, W, detail::mul_functor<T>>
{};

/// div
template <typename T, size_t W>
struct div<T, W>
    : ops::arith_binary_op<T, W, detail::div_functor<T>>
{};

/// mod for integral only (float/double, deleted)
template <typename T, size_t W>
struct mod<T, W>
    : ops::arith_binary_op<T, W, detail::mod_functor<T>>
{};

template <typename T, size_t W>
struct neg<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& x) noexcept
    {
        return avx2::sub<T, W>::apply(Vec<T, W>(0), x);
    }
};

} } } // namespace simd::kernel::avx2

#if 0
/// mul
template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
Vec<T, Arch> mul(const Vec<T, Arch>& self, const Vec<T, Arch>& other, requires_arch<AVX2>) noexcept
{
    if (sizeof(T) == 1) {
        auto mask_hi = _mm256_set1_epi32(0xFF00FF00);
        auto res_lo = _mm256_mullo_epi16(self, other);
        auto other_hi = _mm256_srli_epi16(other, 8);
        auto self_hi = _mm256_and_si256(self, mask_hi);
        auto res_hi = _mm256_mullo_epi16(self_hi, other_hi);
        auto res = _mm256_blendv_epi8(res_lo, res_hi, mask_hi);
        return res;
    } else if (sizeof(T) == 2) {
        return _mm256_mullo_epi16(self, other);
    } else if (sizeof(T) == 4) {
        return _mm256_mullo_epi32(self, other);
    } else {
        // TODO:
        assert(0);
    }
}

/// sadd
template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
Vec<T, Arch> sadd(const Vec<T, Arch>& self, const Vec<T, Arch>& other, requires_arch<AVX2>) noexcept
{
    if (std::is_signed<T>::value) {
        if (sizeof(T) == 1) {
            return _mm256_adds_epi8(self, other);
        } else if (sizeof(T) == 2) {
            return _mm256_adds_epi16(self, other);
        } else {
            // TODO
            return sadd(self, other, AVX{});
        }
    } else {
        if (sizeof(T) == 1) {
            return _m256_adds_epu8(self, other);
        } else if (sizeof(T) == 2) {
            return _mm256_adds_epu16(self, other);
        } else {
            // TODO
            return sadd(self, other, AVX{});
        }
    }
}

/// ssub
template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
Vec<T, Arch> ssub(const Vec<T, Arch>& self, const Vec<T, Arch>& other, requires_arch<AVX2>) noexcept
{
    if (std::is_signed<T>::value) {
        if (sizeof(T) == 1) {
            return _mm256_subs_epi8(self, other);
        } else if (sizeof(T) == 2) {
            return _mm256_subs_epi16(self, other);
        } else {
            // TODO
            return ssub(self, other, AVX{});
        }
    } else {
        if (sizeof(T) == 1) {
            return _m256_ssub_epu8(self, other);
        } else if (sizeof(T) == 2) {
            return _mm256_ssub_epu16(self, other);
        } else {
            // TODO
            return ssub(self, other, AVX{});
        }
    }
}
#endif
