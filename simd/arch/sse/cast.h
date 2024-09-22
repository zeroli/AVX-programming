#pragma once

#include "simd/types/sse_register.h"
#include "simd/types/vec.h"

namespace simd {
namespace kernel {
namespace sse {
using namespace types;

/// cast

namespace detail {
inline static __m128d _cvtepi64_pd(const __m128i& x)
{
    // from https://stackoverflow.com/questions/41144668/how-to-efficiently-perform-double-int64-conversions-with-sse-avx
    // adapted to sse2
    __m128i xH = _mm_srli_epi64(x, 32);
    xH = _mm_or_si128(xH, _mm_castpd_si128(_mm_set1_pd(19342813113834066795298816.))); //  2^84
    __m128i mask = _mm_setr_epi16(0xFFFF, 0xFFFF, 0x0000, 0x0000, 0xFFFF, 0xFFFF, 0x0000, 0x0000);
    __m128i xL = _mm_or_si128(_mm_and_si128(mask, x), _mm_andnot_si128(mask, _mm_castpd_si128(_mm_set1_pd(0x0010000000000000)))); //  2^52
    __m128d f = _mm_sub_pd(_mm_castsi128_pd(xH), _mm_set1_pd(19342813118337666422669312.)); //  2^84 + 2^52
    return _mm_add_pd(f, _mm_castsi128_pd(xL));
}

inline static __m128d _cvtepu64_pd(const __m128i& x)
{
    // from https://stackoverflow.com/questions/41144668/how-to-efficiently-perform-double-int64-conversions-with-sse-avx
    // adapted to sse2
    __m128i xH = _mm_srai_epi32(x, 16);
    xH = _mm_and_si128(xH, _mm_setr_epi16(0x0000, 0x0000, 0xFFFF, 0xFFFF, 0x0000, 0x0000, 0xFFFF, 0xFFFF));
    xH = _mm_add_epi64(xH, _mm_castpd_si128(_mm_set1_pd(442721857769029238784.))); //  3*2^67
    __m128i mask = _mm_setr_epi16(0xFFFF, 0xFFFF, 0xFFFF, 0x0000, 0xFFFF, 0xFFFF, 0xFFFF, 0x0000);
    __m128i xL = _mm_or_si128(_mm_and_si128(mask, x), _mm_andnot_si128(mask, _mm_castpd_si128(_mm_set1_pd(0x0010000000000000)))); //  2^52
    __m128d f = _mm_sub_pd(_mm_castsi128_pd(xH), _mm_set1_pd(442726361368656609280.)); //  3*2^67 + 2^52
    return _mm_add_pd(f, _mm_castsi128_pd(xL));
}
}  // namespace detail

template <typename T, size_t W>
struct cast<float, T, W, traits::enable_if_t<sizeof(T) == 4>>
{
    static Vec<float, W>
    apply(const Vec<T, W>& x) noexcept
    {
        Vec<float, W> ret;
        constexpr auto nregs = Vec<T, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_cvtepi32_ps(x.reg(idx));
        }
        return ret;
    }
};

template <size_t W>
struct cast<double, int64_t, W>
{
    static Vec<double, W>
    apply(const Vec<int64_t, W>& x) noexcept
    {
        Vec<double, W> ret;
        constexpr auto nregs = Vec<int64_t, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = detail::_cvtepi64_pd(x.reg(idx));
        }
        return ret;
    }
};
template <size_t W>
struct cast<double, uint64_t, W>
{
    static Vec<double, W>
    apply(const Vec<uint64_t, W>& x) noexcept
    {
        Vec<double, W> ret;
        constexpr auto nregs = Vec<uint64_t, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = detail::_cvtepu64_pd(x.reg(idx));
        }
        return ret;
    }
};

template <size_t W>
struct cast<int32_t, float, W>
{
    /// float => int32
    static Vec<int32_t, W>
    apply(const Vec<float, W>& x) noexcept
    {
        Vec<int32_t, W> ret;
        constexpr auto nregs = Vec<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_cvtps_epi32(x.reg(idx));
        }
        return ret;
    }
};
template <size_t W>
struct cast<int64_t, float, W>
{
    /// float => int64
    static Vec<int64_t, W>
    apply(const Vec<float, W>& x) noexcept
    {
        Vec<int64_t, W> ret;
        constexpr auto src_nregs = Vec<float, W>::n_regs();
        constexpr auto dst_nregs = Vec<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < src_nregs; idx++) {
            assert(0 && "not implemented yet");
        }
        return ret;
    }
};
template <size_t W>
struct cast<double, float, W>
{
    /// float => double
    static Vec<double, W>
    apply(const Vec<float, W>& x) noexcept
    {
        Vec<double, W> ret;
        constexpr auto src_nregs = Vec<float, W>::n_regs();
        constexpr auto dst_nregs = Vec<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < src_nregs; idx++) {
            assert(0 && "not implemented yet");
        }
        return ret;
    }
};

template <size_t W>
struct cast<int64_t, double, W>
{
    /// double => int64
    static Vec<int64_t, W>
    apply(const Vec<double, W>& x) noexcept
    {
        Vec<int64_t, W> ret;
        constexpr auto nregs = Vec<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            //ret.reg(idx) = detail::_cvtpd_epi64(x.reg(idx));
            assert(0 && "not implemented yet");
        }
        return ret;
    }
};
template <size_t W>
struct cast<float, double, W>
{
    /// double => float
    static Vec<float, W>
    apply(const Vec<double, W>& x) noexcept
    {
        Vec<float, W> ret;
        constexpr auto src_nregs = Vec<double, W>::n_regs();
        constexpr auto dst_nregs = Vec<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < src_nregs; idx++) {
            assert(0 && "not implemented yet");
        }
        return ret;
    }
};
}  // namespace sse
}  // namespace kernel
}  // namespace simd
