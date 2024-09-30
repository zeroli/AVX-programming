#pragma once

#include "simd/types/avx2_register.h"
#include "simd/types/traits.h"

#include <limits>
#include <type_traits>
#include <cstddef>
#include <cstdint>

namespace simd {
namespace kernel {
namespace avx2 {
using namespace types;

/// add
template <typename T, size_t W>
struct add<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        static_check_supported_type<T>();

        Vec<T, W> ret;
        constexpr auto nregs = Vec<T, W>::n_regs();
        SIMD_IF_CONSTEXPR(sizeof(T) == 1) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm256_add_epi8(lhs.reg(idx), rhs.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 2) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm256_add_epi16(lhs.reg(idx), rhs.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 4) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm256_add_epi32(lhs.reg(idx), rhs.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 8) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm256_add_epi64(lhs.reg(idx), rhs.reg(idx));
            }
        }
        return ret;
    }
};
}  // namespace avx2
}  // namespace kernel
}  // namespace simd

#if 0
/// add
template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
Vec<T, Arch> add(const Vec<T, Arch>& self, const Vec<T, Arch>& other, requires_arch<AVX2>) noexcept
{
    if (sizeof(T) == 1) {
        return _mm256_add_epi8(self, other);
    } else if (sizeof(T) == 2) {
        return _mm256_add_epi16(self, other);
    } else if (sizeof(T) == 4) {
        return _mm256_add_epi32(self, other);
    } else if (sizeof(T) == 8) {
        return _mm256_add_epi64(self, other);
    } else {
        // TODO:
        assert(0);
    }
}

/// sub
template <typename Arch, typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
Vec<T, Arch> sub(const Vec<T, Arch>& self, const Vec<T, Arch>& other, requires_arch<AVX2>) noexcept
{
    if (sizeof(T) == 1) {
        return _mm256_sub_epi8(self, other);
    } else if (sizeof(T) == 2) {
        return _mm256_sub_epi16(self, other);
    } else if (sizeof(T) == 4) {
        return _mm256_sub_epi32(self, other);
    } else if (sizeof(T) == 8) {
        return _mm256_sub_epi64(self, other);
    } else {
        // TODO:
        assert(0);
    }
}

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
