#pragma once

#include "simd/types/sse_register.h"
#include "simd/types/traits.h"

#include <limits>
#include <type_traits>
#include <cstddef>
#include <cstdint>

namespace simd {
namespace kernel {
namespace sse {
using namespace types;

/// min
template <typename T, size_t W>
struct min<T, W, REQUIRE_INTEGRAL(T)>
{
    static Vec<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        static_check_supported_type<T, 4>();

        Vec<T, W> ret;
        constexpr int nregs = Vec<T, W>::n_regs();
        constexpr bool is_signed = std::is_signed<T>::value;
        SIMD_IF_CONSTEXPR(sizeof(T) == 1) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = is_signed
                                ? _mm_min_epi8(lhs.reg(idx), rhs.reg(idx))
                                : _mm_min_epu8(lhs.reg(idx), rhs.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 2) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = is_signed
                                ? _mm_min_epi16(lhs.reg(idx), rhs.reg(idx))
                                : _mm_min_epu16(lhs.reg(idx), rhs.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 4) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = is_signed
                                ? _mm_min_epi32(lhs.reg(idx), rhs.reg(idx))
                                : _mm_min_epu32(lhs.reg(idx), rhs.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 8) {
            assert(0 && "not implemented yet");
        }
        return ret;
    }
};

template <size_t W>
struct min<float, W>
{
    static Vec<float, W> apply(const Vec<float, W>& lhs, const Vec<float, W>& rhs) noexcept
    {
        Vec<float, W> ret;
        constexpr int nregs = Vec<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_min_ps(lhs.reg(idx), rhs.reg(idx));
        }
        return ret;
    }
};

template <size_t W>
struct min<double, W>
{
    static Vec<double, W> apply(const Vec<double, W>& lhs, const Vec<double, W>& rhs) noexcept
    {
        Vec<double, W> ret;
        constexpr int nregs = Vec<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_min_pd(lhs.reg(idx), rhs.reg(idx));
        }
        return ret;
    }
};

/// max
template <typename T, size_t W>
struct max<T, W, REQUIRE_INTEGRAL(T)>
{
    static Vec<T, W> apply(const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        static_check_supported_type<T, 4>();

        Vec<T, W> ret;
        constexpr int nregs = Vec<T, W>::n_regs();
        constexpr bool is_signed = std::is_signed<T>::value;
        SIMD_IF_CONSTEXPR(sizeof(T) == 1) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = is_signed
                                    ? _mm_max_epi8(lhs.reg(idx), rhs.reg(idx))
                                    : _mm_max_epu8(lhs.reg(idx), rhs.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 2) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = is_signed
                                    ? _mm_max_epi16(lhs.reg(idx), rhs.reg(idx))
                                    : _mm_max_epu16(lhs.reg(idx), rhs.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 4) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = is_signed
                                    ? _mm_max_epi32(lhs.reg(idx), rhs.reg(idx))
                                    : _mm_max_epu32(lhs.reg(idx), rhs.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 8) {
            assert(0 && "not implemented yet");
        }
        return ret;
    }
};

template <size_t W>
struct max<float, W>
{
    static Vec<float, W> apply(const Vec<float, W>& lhs, const Vec<float, W>& rhs) noexcept
    {
        Vec<float, W> ret;
        constexpr int nregs = Vec<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_max_ps(lhs.reg(idx), rhs.reg(idx));
        }
        return ret;
    }
};

template <size_t W>
struct max<double, W>
{
    static Vec<double, W> apply(const Vec<double, W>& lhs, const Vec<double, W>& rhs) noexcept
    {
        Vec<double, W> ret;
        constexpr int nregs = Vec<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_max_pd(lhs.reg(idx), rhs.reg(idx));
        }
        return ret;
    }
};

/// all_of
template <size_t W>
struct all_of<float, W>
{
    static bool apply(const VecBool<float, W>& x) noexcept
    {
        bool ret = true;
        constexpr auto nregs = VecBool<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret = ret && (_mm_movemask_ps(x.reg(idx)) == 0x0F);
        }
        return ret;
    }
};

template <size_t W>
struct all_of<double, W>
{
    static bool apply(const VecBool<double, W>& x) noexcept
    {
        bool ret = true;
        constexpr auto nregs = VecBool<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret = ret && (_mm_movemask_pd(x.reg(idx)) == 0x03);
        }
        return ret;
    }
};

template <typename T, size_t W>
struct all_of<T, W, REQUIRE_INTEGRAL(T)>
{
    static bool apply(const VecBool<T, W>& x) noexcept
    {
        static_check_supported_type<T>();

        bool ret = true;
        constexpr auto nregs = VecBool<T, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret = ret && (_mm_movemask_epi8(x.reg(idx)) == 0xFFFF);
        }
        return ret;
    }
};

/// any_of
template <size_t W>
struct any_of<float, W>
{
    static bool apply(const VecBool<float, W>& x) noexcept
    {
        bool ret = false;
        constexpr auto nregs = VecBool<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret = ret || (_mm_movemask_ps(x.reg(idx)) != 0);
        }
        return ret;
    }
};

template <size_t W>
struct any_of<double, W>
{
    static bool apply(const VecBool<double, W>& x) noexcept
    {
        bool ret = false;
        constexpr auto nregs = VecBool<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret = ret || (_mm_movemask_pd(x.reg(idx)) != 0);
        }
        return ret;
    }
};

template <typename T, size_t W>
struct any_of<T, W, REQUIRE_INTEGRAL(T)>
{
    static bool apply(const VecBool<T, W>& x) noexcept
    {
        static_check_supported_type<T>();

        bool ret = false;
        constexpr auto nregs = VecBool<T, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret = ret || (!_mm_testz_si128(x.reg(idx),x.reg(idx)));
        }
        return ret;
    }
};

/// select
template <typename T, size_t W>
struct select<T, W, REQUIRE_INTEGRAL(T)>
{
    static Vec<T, W> apply(const VecBool<T, W>& cond, const Vec<T, W>& lhs, const Vec<T, W>& rhs) noexcept
    {
        static_check_supported_type<T, 8>();

        Vec<T, W> ret;
        constexpr auto nregs = VecBool<T, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_or_si128(
                            _mm_and_si128(cond.reg(idx), lhs.reg(idx)),
                            _mm_andnot_si128(cond.reg(idx), rhs.reg(idx))
                        );
        }
        return ret;
    }
};

template <size_t W>
struct select<float, W>
{
    static Vec<float, W> apply(const VecBool<float, W>& cond, const Vec<float, W>& lhs, const Vec<float, W>& rhs) noexcept
    {
        Vec<float, W> ret;
        constexpr auto nregs = VecBool<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_blendv_ps(rhs.reg(idx), lhs.reg(idx), cond.reg(idx));
            #if 0  // naive implementation before sse4.2
            ret.reg(idx) = _mm_or_ps(
                            _mm_and_ps(cond.reg(idx), lhs.reg(idx)),
                            _mm_andnot_ps(cond.reg(idx), rhs.reg(idx))
                        );
            #endif
        }
        return ret;
    }
};

template <size_t W>
struct select<double, W>
{
    static Vec<double, W> apply(const VecBool<double, W>& cond, const Vec<double, W>& lhs, const Vec<double, W>& rhs) noexcept
    {
        Vec<double, W> ret;
        constexpr auto nregs = VecBool<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_blendv_pd(rhs.reg(idx), lhs.reg(idx), cond.reg(idx));
            #if 0  // naive implementation before sse4.2
            ret.reg(idx) = _mm_or_pd(
                            _mm_and_pd(cond.reg(idx), lhs.reg(idx)),
                            _mm_andnot_pd(cond.reg(idx), rhs.reg(idx))
                        );
            #endif
        }
        return ret;
    }
};

/// popcount
template <size_t W>
struct popcount<float, W>
{
    static int apply(const VecBool<float, W>& x) noexcept
    {
        int ret = 0;
        constexpr auto nregs = VecBool<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret += bits::count1(_mm_movemask_ps(x.reg(idx)));
        }
        return ret;
    }
};

template <size_t W>
struct popcount<double, W>
{
    static int apply(const VecBool<double, W>& x) noexcept
    {
        int ret = 0;
        constexpr auto nregs = VecBool<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret += bits::count1(_mm_movemask_pd(x.reg(idx)));
        }
        return ret;
    }
};

template <typename T, size_t W>
struct popcount<T, W, REQUIRE_INTEGRAL(T)>
{
    static int apply(const VecBool<T, W>& x) noexcept
    {
        static_check_supported_type<T>();

        int ret = 0;
        constexpr auto nregs = Vec<T, W>::n_regs();
        SIMD_IF_CONSTEXPR(sizeof(T) == 1) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret += bits::count1(_mm_movemask_epi8(x.reg(idx)));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 2) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret += bits::count1(_mm_movemask_epi8(x.reg(idx)));
            }
            ret >>= 1;
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 4) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret += bits::count1(_mm_movemask_ps(_mm_castsi128_ps(x.reg(idx))));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 8) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret += bits::count1(_mm_movemask_pd(_mm_castsi128_pd(x.reg(idx))));
            }
        }
        return ret;
    }
};

/// find_first_set
template <size_t W>
struct find_first_set<float, W>
{
    static int apply(const VecBool<float, W>& x) noexcept
    {
        int ret = 0;
        constexpr auto nregs = VecBool<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret += bits::count1(_mm_movemask_ps(x.reg(idx)));
        }
        return ret;
    }
};

template <size_t W>
struct find_first_set<double, W>
{
    static int apply(const VecBool<double, W>& x) noexcept
    {
        int ret = 0;
        constexpr auto nregs = VecBool<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret += bits::count1(_mm_movemask_pd(x.reg(idx)));
        }
        return ret;
    }
};

template <typename T, size_t W>
struct find_first_set<T, W, REQUIRE_INTEGRAL(T)>
{
    static int apply(const VecBool<T, W>& x) noexcept
    {
        static_check_supported_type<T>();

        int ret = 0;
        constexpr auto nregs = Vec<T, W>::n_regs();
        SIMD_IF_CONSTEXPR(sizeof(T) == 1) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret += bits::count1(_mm_movemask_epi8(x.reg(idx)));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 2) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret += bits::count1(_mm_movemask_epi8(x.reg(idx)));
            }
            ret >>= 1;
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 4) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret += bits::count1(_mm_movemask_ps(_mm_castsi128_ps(x.reg(idx))));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 8) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret += bits::count1(_mm_movemask_pd(_mm_castsi128_pd(x.reg(idx))));
            }
        }
        return ret;
    }
};

/// find_first_set
template <size_t W>
struct find_last_set<float, W>
{
    static int apply(const VecBool<float, W>& x) noexcept
    {
        int ret = 0;
        constexpr auto nregs = VecBool<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret += bits::count1(_mm_movemask_ps(x.reg(idx)));
        }
        return ret;
    }
};

template <size_t W>
struct find_last_set<double, W>
{
    static int apply(const VecBool<double, W>& x) noexcept
    {
        int ret = 0;
        constexpr auto nregs = VecBool<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret += bits::count1(_mm_movemask_pd(x.reg(idx)));
        }
        return ret;
    }
};

template <typename T, size_t W>
struct find_last_set<T, W, REQUIRE_INTEGRAL(T)>
{
    static int apply(const VecBool<T, W>& x) noexcept
    {
        static_check_supported_type<T>();

        int ret = 0;
        constexpr auto nregs = Vec<T, W>::n_regs();
        SIMD_IF_CONSTEXPR(sizeof(T) == 1) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret += bits::count1(_mm_movemask_epi8(x.reg(idx)));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 2) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret += bits::count1(_mm_movemask_epi8(x.reg(idx)));
            }
            ret >>= 1;
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 4) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret += bits::count1(_mm_movemask_ps(_mm_castsi128_ps(x.reg(idx))));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 8) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret += bits::count1(_mm_movemask_pd(_mm_castsi128_pd(x.reg(idx))));
            }
        }
        return ret;
    }
};
}  // namespace sse
}  // namespace kernel
}  // namespace simd
