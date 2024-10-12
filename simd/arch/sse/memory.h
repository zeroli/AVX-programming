#pragma once

namespace simd { namespace kernel { namespace sse {

using namespace types;

/// broadcast
template <typename T, size_t W>
struct broadcast<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(T val) noexcept
    {
        static_check_supported_type<T, 8>();

        Vec<T, W> ret;
        constexpr auto nregs = Vec<T, W>::n_regs();
        SIMD_IF_CONSTEXPR(sizeof(T) == 1) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm_set1_epi8(val);
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 2) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm_set1_epi16(val);
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 4) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm_set1_epi32(val);
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 8) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm_set1_epi64x(val);
            }
        }
        return ret;
    }
};

template <size_t W>
struct broadcast<float, W>
{
    SIMD_INLINE
    static Vec<float, W> apply(float val) noexcept
    {
        Vec<float, W> ret;
        constexpr auto nregs = Vec<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_set1_ps(val);
        }
        return ret;
    }
};

template <size_t W>
struct broadcast<double, W>
{
    SIMD_INLINE
    static Vec<double, W> apply(double val) noexcept
    {
        Vec<double, W> ret;
        constexpr auto nregs = Vec<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_set1_pd(val);
        }
        return ret;
    }
};

/// setzero
template <typename T, size_t W>
struct setzero<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply() noexcept
    {
        static_check_supported_type<T, 8>();

        Vec<T, W> ret;
        constexpr auto nregs = Vec<T, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_setzero_si128();
        }
        return ret;
    }
};

template <size_t W>
struct setzero<float, W>
{
    SIMD_INLINE
    static Vec<float, W> apply() noexcept
    {
        Vec<float, W> ret;
        constexpr auto nregs = Vec<float, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_setzero_ps();
        }
        return ret;
    }
};

template <size_t W>
struct setzero<double, W>
{
    SIMD_INLINE
    static Vec<double, W> apply() noexcept
    {
        Vec<double, W> ret;
        constexpr auto nregs = Vec<double, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_setzero_pd();
        }
        return ret;
    }
};

/// set individual elements
template <typename T, size_t W>
struct set<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(T v0, T v1) noexcept
    {
        Vec<T, W> ret;
        ret.reg(0) = _mm_set_epi64x(v1, v0);
        return ret;
    }
    SIMD_INLINE
    static Vec<T, W> apply(T v0, T v1, T v2, T v3) noexcept
    {
        Vec<T, W> ret;
        SIMD_IF_CONSTEXPR(sizeof(T) == 4) {
            ret.reg(0) = _mm_set_epi32(v3, v2, v1, v0);
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 8) {
            ret.reg(0) = _mm_set_epi64x(v1, v0);
            ret.reg(1) = _mm_set_epi64x(v3, v2);
        } else {
            assert(0);
        }
        return ret;
    }
    SIMD_INLINE
    static Vec<T, W> apply(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7) noexcept
    {
        Vec<T, W> ret;
        SIMD_IF_CONSTEXPR(sizeof(T) == 2) {
            ret.reg(0) = _mm_set_epi16(v7, v6, v5, v4, v3, v2, v1, v0);
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 4) {
            ret.reg(0) = _mm_set_epi32(v3, v2, v1, v0);
            ret.reg(1) = _mm_set_epi32(v7, v6, v5, v4);
        } else {
            assert(0);
        }
        return ret;
    }
    SIMD_INLINE
    static Vec<T, W> apply(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7, T v8, T v9, T v10, T v11, T v12, T v13, T v14, T v15) noexcept
    {
        Vec<T, W> ret;
        SIMD_IF_CONSTEXPR(sizeof(T) == 1) {
            ret.reg(0) = _mm_set_epi8(v15, v14, v13, v12, v11, v10, v9, v8, v7, v6, v5, v4, v3, v2, v1, v0);
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 2) {
            ret.reg(0) = _mm_set_epi16(v7, v6, v5, v4, v3, v2, v1, v0);
            ret.reg(0) = _mm_set_epi16(v15, v14, v13, v12, v11, v10, v9, v8);
        } else {
            assert(0);
        }
        return ret;
    }
};

template <size_t W>
struct set<float, W>
{
    SIMD_INLINE
    static Vec<float, W> apply(float v0, float v1, float v2, float v3) noexcept
    {
        return _mm_set_ps(v3, v2, v1, v0);
    }
    SIMD_INLINE
    static Vec<float, W> apply(float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7) noexcept
    {
        Vec<float, W> ret;
        constexpr auto nregs = Vec<float, W>::n_regs();
        ret.reg(0) = _mm_set_ps(v3, v2, v1, v0);
        ret.reg(1) = _mm_set_ps(v7, v6, v5, v4);
        return ret;
    }
    SIMD_INLINE
    static Vec<float, W> apply(float v0, float v1, float v2,  float v3,  float v4,   float v5,   float v6,   float v7,
                                            float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15) noexcept
    {
        Vec<float, W> ret;
        constexpr auto nregs = Vec<float, W>::n_regs();
        ret.reg(0) = _mm_set_ps(v3, v2, v1, v0);
        ret.reg(1) = _mm_set_ps(v7, v6, v5, v4);
        ret.reg(2) = _mm_set_ps(v11, v10, v9, v8);
        ret.reg(3) = _mm_set_ps(v15, v14, v13, v12);
        return ret;
    }
};

template <size_t W>
struct set<double, W>
{
    SIMD_INLINE
    static Vec<double, W> apply(double v0, double v1) noexcept
    {
        return _mm_set_pd(v1, v0);
    }
    SIMD_INLINE
    static Vec<double, W> apply(double v0, double v1, double v2, double v3) noexcept
    {
        Vec<double, W> ret;
        constexpr auto nregs = Vec<double, W>::n_regs();
        ret.reg(0) = _mm_set_pd(v1, v0);
        ret.reg(1) = _mm_set_pd(v3, v2);
        return ret;
    }
    SIMD_INLINE
    static Vec<double, W> apply(double v0, double v1, double v2, double v3, float v4, float v5, float v6, float v7) noexcept
    {
        Vec<double, W> ret;
        constexpr auto nregs = Vec<double, W>::n_regs();
        ret.reg(0) = _mm_set_pd(v1, v0);
        ret.reg(1) = _mm_set_pd(v3, v2);
        ret.reg(2) = _mm_set_pd(v5, v4);
        ret.reg(3) = _mm_set_pd(v7, v6);
        return ret;
    }
};

/// load_aligned
template <typename T, size_t W>
struct load_aligned<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const T* mem) noexcept
    {
        static_check_supported_type<T, 8>();

        Vec<T, W> ret;
        constexpr auto nregs = Vec<T, W>::n_regs();
        constexpr auto reg_lanes = Vec<T, W>::reg_lanes();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_load_si128((const sse_reg_i*)(mem + idx * reg_lanes));
        }
        return ret;
    }
};

template <size_t W>
struct load_aligned<float, W>
{
    SIMD_INLINE
    static Vec<float, W> apply(const float* mem) noexcept
    {
        Vec<float, W> ret;
        constexpr auto nregs = Vec<float, W>::n_regs();
        constexpr auto reg_lanes = Vec<float, W>::reg_lanes();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_load_ps(mem + idx * reg_lanes);
        }
        return ret;
    }
};

template <size_t W>
struct load_aligned<double, W>
{
    SIMD_INLINE
    static Vec<double, W> apply(const double* mem) noexcept
    {
        Vec<double, W> ret;
        constexpr auto nregs = Vec<double, W>::n_regs();
        constexpr auto reg_lanes = Vec<double, W>::reg_lanes();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_load_pd(mem + idx * reg_lanes);
        }
        return ret;
    }
};

/// load_unaligned
template <typename T, size_t W>
struct load_unaligned<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const T* mem) noexcept
    {
        static_check_supported_type<T, 8>();

        Vec<T, W> ret;
        constexpr auto nregs = Vec<T, W>::n_regs();
        constexpr auto reg_lanes = Vec<T, W>::reg_lanes();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_loadu_si128((const sse_reg_i*)(mem + idx * reg_lanes));
        }
        return ret;
    }
};

template <size_t W>
struct load_unaligned<float, W>
{
    SIMD_INLINE
    static Vec<float, W> apply(const float* mem) noexcept
    {
        Vec<float, W> ret;
        constexpr auto nregs = Vec<float, W>::n_regs();
        constexpr auto reg_lanes = Vec<float, W>::reg_lanes();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_loadu_ps(mem + idx * reg_lanes);
        }
        return ret;
    }
};

template <size_t W>
struct load_unaligned<double, W>
{
    SIMD_INLINE
    static Vec<double, W> apply(const double* mem) noexcept
    {
        Vec<double, W> ret;
        constexpr auto nregs = Vec<double, W>::n_regs();
        constexpr auto reg_lanes = Vec<double, W>::reg_lanes();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_loadu_pd(mem + idx * reg_lanes);
        }
        return ret;
    }
};

/// store_aligned
template <typename T, size_t W>
struct store_aligned<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static void apply(T* mem, const Vec<T, W>& x) noexcept
    {
        static_check_supported_type<T, 8>();

        constexpr auto nregs = Vec<T, W>::n_regs();
        constexpr auto reg_lanes = Vec<T, W>::reg_lanes();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            _mm_store_si128((sse_reg_i*)(mem + idx * reg_lanes), x.reg(idx));
        }
    }
};

template <size_t W>
struct store_aligned<float, W>
{
    SIMD_INLINE
    static void apply(float* mem, const Vec<float, W>& x) noexcept
    {
        constexpr auto nregs = Vec<float, W>::n_regs();
        constexpr auto reg_lanes = Vec<float, W>::reg_lanes();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            _mm_store_ps(mem + idx * reg_lanes, x.reg(idx));
        }
    }
};

template <size_t W>
struct store_aligned<double, W>
{
    SIMD_INLINE
    static void apply(double* mem, const Vec<double, W>& x) noexcept
    {
        constexpr auto nregs = Vec<double, W>::n_regs();
        constexpr auto reg_lanes = Vec<double, W>::reg_lanes();

        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            _mm_store_pd(mem + idx * reg_lanes, x.reg(idx));
        }
    }
};

/// store_unaligned
template <typename T, size_t W>
struct store_unaligned<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static void apply(T* mem, const Vec<T, W>& x) noexcept
    {
        static_check_supported_type<T, 8>();

        constexpr auto nregs = Vec<T, W>::n_regs();
        constexpr auto reg_lanes = Vec<T, W>::reg_lanes();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            _mm_storeu_si128((sse_reg_i*)(mem + idx * reg_lanes), x.reg(idx));
        }
    }
};

template <size_t W>
struct store_unaligned<float, W>
{
    SIMD_INLINE
    static void apply(float* mem, const Vec<float, W>& x) noexcept
    {
        constexpr auto nregs = Vec<float, W>::n_regs();
        constexpr auto reg_lanes = Vec<float, W>::reg_lanes();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            _mm_storeu_ps(mem + idx * reg_lanes, x.reg(idx));
        }
    }
};

template <size_t W>
struct store_unaligned<double, W>
{
    SIMD_INLINE
    static void apply(double* mem, const Vec<double, W>& x) noexcept
    {
        constexpr auto nregs = Vec<double, W>::n_regs();
        constexpr auto reg_lanes = Vec<double, W>::reg_lanes();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            _mm_storeu_pd(mem + idx * reg_lanes, x.reg(idx));
        }
    }
};

namespace detail {
struct load_complex
{
    SIMD_INLINE
    sse_reg_f real(const sse_reg_f& lo, const sse_reg_f& hi) noexcept
    {
        return _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(2, 0, 2, 0));
    }
    SIMD_INLINE
    sse_reg_f imag(const sse_reg_f& lo, const sse_reg_f& hi) noexcept
    {
        return _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(3, 1, 3, 1));
    }
    SIMD_INLINE
    sse_reg_d real(const sse_reg_d& lo, const sse_reg_d& hi) noexcept
    {
        return _mm_shuffle_pd(lo, hi, _MM_SHUFFLE2(0, 0));
    }
    SIMD_INLINE
    sse_reg_d imag(const sse_reg_d& lo, const sse_reg_d& hi) noexcept
    {
        return _mm_shuffle_pd(lo, hi, _MM_SHUFFLE2(1, 1));
    }
};
}  // namespace detail

template <typename T, size_t W>
struct load_complex<T, W>
{
    using value_type = std::complex<T>;

    SIMD_INLINE
    static Vec<value_type, W> apply(const Vec<T, W>& vlo, const Vec<T, W>& vhi) noexcept
    {
        Vec<value_type, W> ret;
        constexpr auto nregs = Vec<T, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.real() = detail::load_complex().real(vlo.reg(idx), vhi.reg(idx));
            ret.imag() = detail::load_complex().imag(vlo.reg(idx), vhi.reg(idx));
        }
        return ret;
    }
};

namespace detail {
struct complex_packlo {
    sse_reg_f operator ()(const sse_reg_f& lo, const sse_reg_f& hi) noexcept
    {
        return _mm_unpacklo_ps(lo, hi);
    }
    sse_reg_d operator ()(const sse_reg_d& lo, const sse_reg_d& hi) noexcept
    {
        return _mm_unpacklo_pd(lo, hi);
    }
};

struct complex_packhi {
    sse_reg_f operator ()(const sse_reg_f& lo, const sse_reg_f& hi) noexcept
    {
        return _mm_unpackhi_ps(lo, hi);
    }
    sse_reg_d operator ()(const sse_reg_d& lo, const sse_reg_d& hi) noexcept
    {
        return _mm_unpackhi_pd(lo, hi);
    }
};
}  // namespace detail

template <typename T, size_t W>
struct complex_packlo<T, W>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& vlo, const Vec<T, W>& vhi) noexcept
    {
        Vec<T, W> ret;
        constexpr auto nregs = Vec<T, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = detail::complex_packlo()(vlo.reg(idx), vhi.reg(idx));
        }
        return ret;
    }
};

template <typename T, size_t W>
struct complex_packhi<T, W>
{
    SIMD_INLINE
    static Vec<T, W> apply(const Vec<T, W>& vlo, const Vec<T, W>& vhi) noexcept
    {
        Vec<T, W> ret;
        constexpr auto nregs = Vec<T, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = detail::complex_packhi()(vlo.reg(idx), vhi.reg(idx));
        }
        return ret;
    }
};

/// to_mask
namespace detail {
SIMD_INLINE
static int mask_lut(int mask)
{
    // clang-format off
    static const int mask_lut[256] = {
        0x0, 0x0, 0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x2, 0x0, 0x3, 0x0, 0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
        0x4, 0x0, 0x5, 0x0, 0x0, 0x0, 0x0, 0x0, 0x6, 0x0, 0x7, 0x0, 0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
        0x8, 0x0, 0x9, 0x0, 0x0, 0x0, 0x0, 0x0, 0xA, 0x0, 0xB, 0x0, 0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
        0xC, 0x0, 0xD, 0x0, 0x0, 0x0, 0x0, 0x0, 0xE, 0x0, 0xF, 0x0, 0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    };
    // clang-format on
    return mask_lut[mask & 0xAA];
}

SIMD_INLINE
static uint64_t movemask_epi16(const sse_reg_i& x)
{
    uint64_t mask8 = _mm_movemask_epi8(x);
    return mask_lut(mask8) | (mask_lut(mask8 >> 8) << 4);
}
SIMD_INLINE
static uint64_t movemask_epi32(const sse_reg_i& x)
{
    return _mm_movemask_ps(_mm_castsi128_ps(x));
}
SIMD_INLINE
static uint64_t movemask_epi64(const sse_reg_i& x)
{
    return _mm_movemask_pd(_mm_castsi128_pd(x));
}
}  // namespace detail

template <typename T, size_t W>
struct to_mask<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static uint64_t apply(const VecBool<T, W>& x) noexcept
    {
        static_check_supported_type<T, 8>();

        uint64_t ret = 0;
        constexpr int nregs = VecBool<T, W>::n_regs();
        SIMD_IF_CONSTEXPR(sizeof(T) == 1) {
            #pragma unroll
            for (int idx = nregs - 1; idx >= 0; idx--) {
                ret <<= 16;  /// 16 * elements for 16 bits
                ret |= _mm_movemask_epi8(x.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 2) {
            #pragma unroll
            for (int idx = nregs - 1; idx >= 0; idx--) {
                ret <= 8;  // 8 * elements for 8 bits
                ret |= detail::movemask_epi16(x.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 4) {
            #pragma unroll
            for (int idx = nregs - 1; idx >= 0; idx--) {
                ret <<= 4;  // 4 * elements for 4 bits
                ret |= detail::movemask_epi32(x.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 8) {
            #pragma unroll
            for (int idx = nregs - 1; idx >= 0; idx--) {
                ret <<= 2;  // 2 * elements for 2 bits
                ret |= detail::movemask_epi64(x.reg(idx));
            }
        }
        return ret;
    }
};

template <size_t W>
struct to_mask<float, W>
{
    SIMD_INLINE
    static uint64_t apply(const VecBool<float, W>& x) noexcept
    {
        uint64_t ret = 0;
        constexpr int nregs = VecBool<float, W>::n_regs();
        #pragma unroll
        for (int idx = nregs - 1; idx >= 0; idx--) {
            ret <<= 4;  // 4 * elements for 4 bits
            ret |= _mm_movemask_ps(x.reg(idx));
        }
        return ret;
    }
};

template <size_t W>
struct to_mask<double, W>
{
    SIMD_INLINE
    static uint64_t apply(const VecBool<double, W>& x) noexcept
    {
        uint64_t ret = 0;
        constexpr int nregs = VecBool<double, W>::n_regs();
        #pragma unroll
        for (int idx = nregs - 1; idx >= 0; idx--) {
            ret <<= 2;  // 2 * elements for 2 bits
            ret |= _mm_movemask_pd(x.reg(idx));
        }
        return ret;
    }
};

/// from_mask

template <size_t W>
struct from_mask<float, W>
{
    /// mask, lower 4 bits, each bit indicates one element (4 * sizeof(float) = 128)
    SIMD_INLINE
    static const sse_reg_i* mask_lut(uint64_t mask) noexcept
    {
        using A = typename VecBool<float, W>::arch_t;
        alignas(A::alignment()) static const uint32_t lut[][4] = {
            { 0x00000000, 0x00000000, 0x00000000, 0x00000000 },
            { 0xFFFFFFFF, 0x00000000, 0x00000000, 0x00000000 },
            { 0x00000000, 0xFFFFFFFF, 0x00000000, 0x00000000 },
            { 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0x00000000 },
            { 0x00000000, 0x00000000, 0xFFFFFFFF, 0x00000000 },
            { 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000 },
            { 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000 },
            { 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000 },
            { 0x00000000, 0x00000000, 0x00000000, 0xFFFFFFFF },
            { 0xFFFFFFFF, 0x00000000, 0x00000000, 0xFFFFFFFF },
            { 0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF },
            { 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF },
            { 0x00000000, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF },
            { 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF },
            { 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF },
            { 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF },
        };
        assert(!(mask & ~0xFul) && "inbound mask: [0, 0xF]");  // cannot beyond 16 (2^4)
        return (const sse_reg_i*)lut[mask];
    }
    SIMD_INLINE
    static VecBool<float, W> apply(uint64_t x) noexcept
    {
        VecBool<float, W> ret;
        constexpr auto nregs = VecBool<float, W>::n_regs();
        constexpr auto reg_lanes = VecBool<float, W>::reg_lanes();
        constexpr auto lanes_mask = (1ull << reg_lanes) - 1;
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_castsi128_ps(_mm_load_si128(mask_lut(x & lanes_mask)));
            x >>= reg_lanes;
        }
        return ret;
    }
};

template <size_t W>
struct from_mask<double, W>
{
    /// mask, lower 2 bits, each bit indicates one element (2 * sizeof(double) = 128)
    SIMD_INLINE
    static const sse_reg_i* mask_lut(uint64_t mask) noexcept
    {
        using A = typename VecBool<double, W>::arch_t;
        alignas(A::alignment()) static const uint64_t lut[][4] = {
            { 0x0000000000000000ul, 0x0000000000000000ul },
            { 0xFFFFFFFFFFFFFFFFul, 0x0000000000000000ul },
            { 0x0000000000000000ul, 0xFFFFFFFFFFFFFFFFul },
            { 0xFFFFFFFFFFFFFFFFul, 0xFFFFFFFFFFFFFFFFul },
        };
        assert(!(mask & ~0x3ul) && "inbound mask: [0, 3]");
        return (const sse_reg_i*)lut[mask];
    }
    SIMD_INLINE
    static VecBool<double, W> apply(uint64_t x) noexcept
    {
        VecBool<double, W> ret;
        constexpr auto nregs = VecBool<double, W>::n_regs();
        constexpr auto reg_lanes = VecBool<double, W>::reg_lanes();
        constexpr auto lanes_mask = (1ull << reg_lanes) - 1;
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_castsi128_pd(_mm_load_si128(mask_lut(x & lanes_mask)));
            x >>= reg_lanes;
        }
        return ret;
    }
};

template <typename T, size_t W>
struct from_mask<T, W, REQUIRE_INTEGRAL(T)>
{
    using A = typename VecBool<T, W>::arch_t;

    SIMD_INLINE
    static uint32_t mask_lut32(uint64_t mask) noexcept
    {
        /// 1 bit expands to 8 bits(1 byte), total 32bits
        alignas(A::alignment()) static const uint32_t lut[] = {
            0x00000000,
            0x000000FF,
            0x0000FF00,
            0x0000FFFF,
            0x00FF0000,
            0x00FF00FF,
            0x00FFFF00,
            0x00FFFFFF,
            0xFF000000,
            0xFF0000FF,
            0xFF00FF00,
            0xFF00FFFF,
            0xFFFF0000,
            0xFFFF00FF,
            0xFFFFFF00,
            0xFFFFFFFF,
        };
        assert(!(mask & ~0xF) && "inbound mask: [0, 0xF]");
        return lut[mask];
    }
    SIMD_INLINE
    static uint64_t mask_lut64(uint64_t mask) noexcept
    {
        /// 1 bit expands to 16 bits(2 bytes), total 64bits
        alignas(A::alignment()) static const uint64_t lut[] = {
            0x0000000000000000,
            0x000000000000FFFF,
            0x00000000FFFF0000,
            0x00000000FFFFFFFF,
            0x0000FFFF00000000,
            0x0000FFFF0000FFFF,
            0x0000FFFFFFFF0000,
            0x0000FFFFFFFFFFFF,
            0xFFFF000000000000,
            0xFFFF00000000FFFF,
            0xFFFF0000FFFF0000,
            0xFFFF0000FFFFFFFF,
            0xFFFFFFFF00000000,
            0xFFFFFFFF0000FFFF,
            0xFFFFFFFFFFFF0000,
            0xFFFFFFFFFFFFFFFF,
        };
        assert(!(mask & ~0xF) && "inbound mask: [0, 0xF]");
        return lut[mask];
    }

    SIMD_INLINE
    static VecBool<T, W> apply(uint64_t x) noexcept
    {
        static_check_supported_type<T, 8>();

        SIMD_IF_CONSTEXPR(sizeof(T) == 1) {
            VecBool<T, W> ret;
            constexpr auto nregs = VecBool<T, W>::n_regs();
            constexpr auto reg_lanes = VecBool<T, W>::reg_lanes();
            constexpr auto lanes_mask = (1ull << reg_lanes) - 1;
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                auto mask = x & lanes_mask;
                ret.reg(idx) = _mm_setr_epi32(  // each one gen 4 bytes(32bits)
                                    mask_lut32((mask >>  0) & 0xF),
                                    mask_lut32((mask >>  4) & 0xF),
                                    mask_lut32((mask >>  8) & 0xF),
                                    mask_lut32((mask >> 12) & 0xF)
                                );
                x >>= reg_lanes;
            }
            return ret;
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 2) {
            VecBool<T, W> ret;
            constexpr auto nregs = VecBool<T, W>::n_regs();
            constexpr auto reg_lanes = VecBool<T, W>::reg_lanes();
            constexpr auto lanes_mask = (1ull << reg_lanes) - 1;
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                auto mask = x & lanes_mask;
                ret.reg(idx) = _mm_set_epi64x(  // each one gen 8 bytes(64bits)
                                    mask_lut64((mask >> 4) & 0xF),
                                    mask_lut64((mask >> 0) & 0xF)
                                );
                x >>= reg_lanes;
            }
            return ret;
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 4) {
            VecBool<T, W> ret;
            constexpr auto nregs = VecBool<T, W>::n_regs();
            auto float_mask = sse::from_mask<float, W>::apply(x);
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm_castps_si128(float_mask.reg(idx));
            }
            return ret;
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 8) {
            VecBool<T, W> ret;
            constexpr auto nregs = VecBool<T, W>::n_regs();
            auto float_mask = sse::from_mask<double, W>::apply(x);
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm_castpd_si128(float_mask.reg(idx));
            }
            return ret;
        }
    }
};

/// gather
template <typename T, size_t W, typename U, typename V>
struct gather<T, W, U, V, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(const T* mem, const Vec<V, W>& index) noexcept
    {
        static_check_supported_type<T, 8>();

        Vec<T, W> ret;
        constexpr auto nregs = Vec<T, W>::n_regs();
        constexpr auto reg_lanes = Vec<T, W>::reg_lanes();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            // TODO
        }
        return ret;
    }
};

template <size_t W, typename U, typename V>
struct gather<float, W, U, V>
{
    SIMD_INLINE
    static Vec<float, W> apply(const U* mem, const Vec<V, W>& index) noexcept
    {
        Vec<float, W> ret;
        constexpr auto nregs = Vec<float, W>::n_regs();
        constexpr auto reg_lanes = Vec<float, W>::reg_lanes();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
        }
        return ret;
    }
};

template <size_t W, typename U, typename V>
struct gather<double, W, U, V>
{
    SIMD_INLINE
    static Vec<double, W> apply(const U* mem, const Vec<V, W>& index) noexcept
    {
        Vec<double, W> ret;
        constexpr auto nregs = Vec<double, W>::n_regs();
        constexpr auto reg_lanes = Vec<double, W>::reg_lanes();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
        }
        return ret;
    }
};

/// scatter
template <typename T, size_t W, typename U, typename V>
struct scatter<T, W, U, V, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static void apply(const Vec<T, W>& x, U* mem, const Vec<V, W>& index) noexcept
    {
        static_check_supported_type<T, 8>();

        constexpr auto nregs = Vec<T, W>::n_regs();
        constexpr auto reg_lanes = Vec<T, W>::reg_lanes();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
        }
    }
};

template <size_t W, typename U, typename V>
struct scatter<float, W, U, V>
{
    SIMD_INLINE
    static void apply(const Vec<float, W>& x, U* mem, const Vec<V, W>& index) noexcept
    {
        constexpr auto nregs = Vec<float, W>::n_regs();
        constexpr auto reg_lanes = Vec<float, W>::reg_lanes();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
        }
    }
};

template <size_t W, typename U, typename V>
struct scatter<double, W, U, V>
{
    SIMD_INLINE
    static void apply(const Vec<double, W>& x, U* mem, const Vec<V, W>& index) noexcept
    {
        constexpr auto nregs = Vec<double, W>::n_regs();
        constexpr auto reg_lanes = Vec<double, W>::reg_lanes();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
        }
    }
};

} } } // namespace simd::kernel::sse
