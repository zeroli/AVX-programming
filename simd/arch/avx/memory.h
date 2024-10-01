#pragma once

namespace simd { namespace kernel { namespace avx {
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
                ret.reg(idx) = _mm256_set1_epi8(val);
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 2) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm256_set1_epi16(val);
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 4) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm256_set1_epi32(val);
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 8) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm256_set1_epi64x(val);
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
            ret.reg(idx) = _mm256_set1_ps(val);
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
            ret.reg(idx) = _mm256_set1_pd(val);
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
            ret.reg(idx) = _mm256_setzero_si256();
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
            ret.reg(idx) = _mm256_setzero_ps();
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
            ret.reg(idx) = _mm256_setzero_pd();
        }
        return ret;
    }
};

/// set individual elements
template <typename T, size_t W>
struct set<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(T v0, T v1, T v2, T v3) noexcept
    {
        static_assert(W == 4);
        Vec<T, W> ret = _mm256_set_epi64x(v3, v2, v1, v0);
        return ret;
    }
    SIMD_INLINE
    static Vec<T, W> apply(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7) noexcept
    {
        static_assert(W == 8);
        Vec<T, W> ret;
        SIMD_IF_CONSTEXPR(sizeof(T) == 4) {  // 8 * 4 * 8 = 256 (1 reg)
            ret.reg(0) = _mm256_set_epi32(v7, v6, v5, v4, v3, v2, v1, v0);
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 8) {  // 8 * 8 * 8 = 512 (2 regs)
            ret.reg(0) = _mm256_set_epi64x(v3, v2, v1, v0);
            ret.reg(1) = _mm256_set_epi64x(v7, v6, v5, v4);
        } else {
            /// sizeof(T) = 1 invalid (64bits = 8 * 1 * 8)
            /// sizeof(T) = 2 goes to sse (128bits = 8 * 2 * 8)
            assert(0);
        }
        return ret;
    }
    SIMD_INLINE
    static Vec<T, W> apply(T v0, T v1, T v2,  T v3,  T v4,  T v5,  T v6,  T v7,
                           T v8, T v9, T v10, T v11, T v12, T v13, T v14, T v15) noexcept
    {
        static_assert(W == 16);
        Vec<T, W> ret;
        SIMD_IF_CONSTEXPR(sizeof(T) == 2) {  // 16 * 2 * 8 = 256 (1 reg)
            ret.reg(0) = _mm256_set_epi16(v15, v14, v13, v12, v11, v10, v9, v8,
                                          v7,  v6,  v5,  v4,  v3,  v2,  v1, v0);
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 4) {  // 16 * 4 * 8 = 512 (2 regs)
            ret.reg(0) = _mm256_set_epi32(v7,  v6,  v5,  v4,  v3,  v2,  v1, v0);
            ret.reg(1) = _mm256_set_epi32(v15, v14, v13, v12, v11, v10, v9, v8);
        } else {
            /// sizeof(T) = 1 goes to sse (128bits = 16 * 1 * 8)
            /// sizeof(T) = 8 invalid (1024bits)
            assert(0);
        }
        return ret;
    }
    static Vec<T, W> apply(T v0,  T v1,  T v2,  T v3,  T v4,  T v5,  T v6,  T v7,
                           T v8,  T v9,  T v10, T v11, T v12, T v13, T v14, T v15,
                           T v16, T v17, T v18, T v19, T v20, T v21, T v22, T v23,
                           T v24, T v25, T v26, T v27, T v28, T v29, T v30, T v31) noexcept
    {
        static_assert(W == 32);
        Vec<T, W> ret;
        SIMD_IF_CONSTEXPR(sizeof(T) == 1) {
            ret.reg(0) = _mm256_set_epi8(v31, v30, v29, v28, v27, v26, v25, v24,
                                         v23, v22, v21, v20, v19, v18, v17, v16,
                                         v15, v14, v13, v12, v11, v10, v9,  v8,
                                         v7,  v6,  v5,  v4,  v3,  v2,  v1,  v0);
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 2) {
            ret.reg(0) = _mm256_set_epi16(v15, v14, v13, v12, v11, v10, v9,  v8,
                                          v7,  v6,  v5,  v4,  v3,  v2,  v1,  v0);
            ret.reg(1) = _mm256_set_epi16(v31, v30, v29, v28, v27, v26, v25, v24,
                                          v23, v22, v21, v20, v19, v18, v17, v16);
        } else {
            assert(0);
        }
        return ret;
    }
    static Vec<T, W> apply(T v0,  T v1,  T v2,  T v3,  T v4,  T v5,  T v6,  T v7,
                           T v8,  T v9,  T v10, T v11, T v12, T v13, T v14, T v15,
                           T v16, T v17, T v18, T v19, T v20, T v21, T v22, T v23,
                           T v24, T v25, T v26, T v27, T v28, T v29, T v30, T v31,
                           T v32, T v33, T v34, T v35, T v36, T v37, T v38, T v39,
                           T v40, T v41, T v42, T v43, T v44, T v45, T v46, T v47,
                           T v48, T v49, T v50, T v51, T v52, T v53, T v54, T v55,
                           T v56, T v57, T v58, T v59, T v60, T v61, T v62, T v63) noexcept
    {
        static_assert(W == 64);
        static_assert(sizeof(T) == 1);
        Vec<T, W> ret;
        ret.reg(0) = _mm256_set_epi8(v31, v30, v29, v28, v27, v26, v25, v24,
                                     v23, v22, v21, v20, v19, v18, v17, v16,
                                     v15, v14, v13, v12, v11, v10, v9,  v8,
                                     v7,  v6,  v5,  v4,  v3,  v2,  v1,  v0);
        ret.reg(1) = _mm256_set_epi8(v63, v62, v61, v60, v59, v58, v57, v56,
                                     v55, v54, v53, v52, v51, v50, v49, v48,
                                     v47, v46, v45, v44, v43, v42, v41, v40,
                                     v39, v38, v37, v36, v35, v34, v33, v32);
        return ret;
    }
};

template <size_t W>
struct set<float, W>
{
    SIMD_INLINE
    static Vec<float, W> apply(float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7) noexcept
    {
        static_assert(W == 8);
        // 8 * 4 * 8 = 256 (1 reg)
        return _mm256_set_ps(v7, v6, v5, v4, v3, v2, v1, v0);
    }
    SIMD_INLINE
    static Vec<float, W> apply(float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7,
                               float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15) noexcept
    {
        static_assert(W == 16);
        // 16 * 4 * 8 = 512 (2 regs)
        Vec<float, W> ret;
        ret.reg(0) = _mm256_set_ps(v7,  v6,  v5,  v4,  v3,  v2,  v1, v0);
        ret.reg(1) = _mm256_set_ps(v15, v14, v13, v12, v11, v10, v9, v8);
        return ret;
    }
    /// support up to 512bits
};

template <size_t W>
struct set<double, W>
{
    SIMD_INLINE
    static Vec<double, W> apply(double v0, double v1, double v2, double v3) noexcept
    {
        static_assert(W == 4);
        // 4 * 8 * 8 = 256 (1 reg)
        return _mm256_set_pd(v3, v2, v1, v0);
    }
    SIMD_INLINE
    static Vec<double, W> apply(double v0, double v1, double v2, double v3, float v4, float v5, float v6, float v7) noexcept
    {
        static_assert(W == 8);
        // 8 * 8 * 8 = 512 (2 regs)
        Vec<double, W> ret;
        ret.reg(0) = _mm256_set_pd(v3, v2, v1, v0);
        ret.reg(1) = _mm256_set_pd(v7, v6, v5, v4);
        return ret;
    }
    /// support up to 512bits
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
            ret.reg(idx) = _mm256_load_si256((const avx_reg_i*)(mem + idx * reg_lanes));
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
            ret.reg(idx) = _mm256_load_ps(mem + idx * reg_lanes);
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
            ret.reg(idx) = _mm256_load_pd(mem + idx * reg_lanes);
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
            ret.reg(idx) = _mm256_loadu_si256((const avx_reg_i*)(mem + idx * reg_lanes));
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
            ret.reg(idx) = _mm256_loadu_ps(mem + idx * reg_lanes);
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
            ret.reg(idx) = _mm256_loadu_pd(mem + idx * reg_lanes);
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
            _mm256_store_si256((avx_reg_i*)(mem + idx * reg_lanes), x.reg(idx));
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
            _mm256_store_ps(mem + idx * reg_lanes, x.reg(idx));
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
            _mm256_store_pd(mem + idx * reg_lanes, x.reg(idx));
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
            _mm256_storeu_si256((avx_reg_i*)(mem + idx * reg_lanes), x.reg(idx));
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
            _mm256_storeu_ps(mem + idx * reg_lanes, x.reg(idx));
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
            _mm256_storeu_pd(mem + idx * reg_lanes, x.reg(idx));
        }
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
static uint64_t movemask_epi16(const avx_reg_i& x)
{
    uint64_t mask8 = _mm256_movemask_epi8(x);
    return mask_lut(mask8) |
          (mask_lut(mask8 >> 8 ) << 4 ) |
          (mask_lut(mask8 >> 16) << 8 ) |
          (mask_lut(mask8 >> 24) << 12);
}
SIMD_INLINE
static uint64_t movemask_epi32(const avx_reg_i& x)
{
    return _mm256_movemask_ps(_mm256_castsi256_ps(x));
}
SIMD_INLINE
static uint64_t movemask_epi64(const avx_reg_i& x)
{
    return _mm256_movemask_pd(_mm256_castsi256_pd(x));
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
        constexpr auto nregs = VecBool<T, W>::n_regs();
        SIMD_IF_CONSTEXPR(sizeof(T) == 1) {
            #pragma unroll
            for (int idx = (int)nregs - 1; idx >= 0; idx--) {
                ret <<= 32;  /// 32 * elements for 32 bits
                ret |= _mm256_movemask_epi8(x.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 2) {
            #pragma unroll
            for (int idx = (int)nregs - 1; idx >= 0; idx--) {
                ret <= 16;  // 16 * elements for 16 bits
                ret |= detail::movemask_epi16(x.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 4) {
            #pragma unroll
            for (int idx = (int)nregs - 1; idx >= 0; idx--) {
                ret <<= 8;  // 8 * elements for 8 bits
                ret |= detail::movemask_epi32(x.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 8) {
            #pragma unroll
            for (int idx = (int)nregs - 1; idx >= 0; idx--) {
                ret <<= 4;  // 4 * elements for 4 bits
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
        constexpr auto nregs = VecBool<float, W>::n_regs();
        #pragma unroll
        for (int idx = (int)nregs - 1; idx >= 0; idx--) {
            ret <<= 8;  // 8 * elements for 8 bits
            ret |= _mm256_movemask_ps(x.reg(idx));
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
        constexpr auto nregs = VecBool<double, W>::n_regs();
        #pragma unroll
        for (int idx = (int)nregs - 1; idx >= 0; idx--) {
            ret <<= 4;  // 4 * elements for 4 bits
            ret |= _mm_movemask_pd(x.reg(idx));
        }
        return ret;
    }
};

/// from_mask

template <size_t W>
struct from_mask<float, W>
{
    /// mask, lower 8 bits, each bit indicates one element (8 * sizeof(float) = 256)
    SIMD_INLINE
    static avx_reg_i mask_lut(uint64_t mask) noexcept
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
        assert(!(mask & ~0xFFul) && "inbound mask: [0, 0xFF]");  // cannot beyond 256 (2^8)
        return detail::merge_reg(*(const sse_reg_i*)lut[(mask >> 0) & 0xF],
                                 *(const sse_reg_i*)lut[(mask >> 4) & 0xF]);
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
            ret.reg(idx) = _mm256_castsi256_ps(mask_lut(x & lanes_mask));
            x >>= reg_lanes;
        }
        return ret;
    }
};

template <size_t W>
struct from_mask<double, W>
{
    /// mask, lower 4 bits, each bit indicates one element (4 * sizeof(double) = 256)
    SIMD_INLINE
    static avx_reg_d mask_lut(uint64_t mask) noexcept
    {
        using A = typename VecBool<double, W>::arch_t;
        alignas(A::alignment()) static const uint64_t lut[][4] = {
            { 0x0000000000000000ul, 0x0000000000000000ul },
            { 0xFFFFFFFFFFFFFFFFul, 0x0000000000000000ul },
            { 0x0000000000000000ul, 0xFFFFFFFFFFFFFFFFul },
            { 0xFFFFFFFFFFFFFFFFul, 0xFFFFFFFFFFFFFFFFul },
        };
        assert(!(mask & ~0xFul) && "inbound mask: [0, F]");
        return detail::merge_reg(*(const sse_reg_d*)lut[(mask >> 0) & 0x3],
                                 *(const sse_reg_d*)lut[(mask >> 2) & 0x3]);
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
            ret.reg(idx) = _mm256_castsi256_pd(mask_lut(x & lanes_mask));
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
    static uint64_t mask_lut8(uint64_t mask) noexcept
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
        assert(!(mask & ~0xFF) && "inbound mask: [0, 0xFF]");
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
                ret.reg(idx) = _mm256_setr_epi32(  // each one gen 4 bytes (32bits)
                                    mask_lut32((mask >>  0) & 0xF),
                                    mask_lut32((mask >>  4) & 0xF),
                                    mask_lut32((mask >>  8) & 0xF),
                                    mask_lut32((mask >> 12) & 0xF),
                                    mask_lut32((mask >> 16) & 0xF),
                                    mask_lut32((mask >> 20) & 0xF),
                                    mask_lut32((mask >> 24) & 0xF),
                                    mask_lut32((mask >> 28) & 0xF)
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
                ret.reg(idx) = _mm256_setr_epi64x(  // each one gen 8 bytes (64bits)
                                    mask_lut64((mask >> 0) & 0xF),
                                    mask_lut64((mask >> 4) & 0xF),
                                    mask_lut64((mask >> 8) & 0xF),
                                    mask_lut64((mask > 12) & 0xF)
                                );
                x >>= reg_lanes;
            }
            return ret;
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 4) {
            VecBool<T, W> ret;
            constexpr auto nregs = VecBool<T, W>::n_regs();
            auto float_mask = avx::from_mask<float, W>::apply(x);
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm256_castps_si256(float_mask.reg(idx));
            }
            return ret;
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 8) {
            VecBool<T, W> ret;
            constexpr auto nregs = VecBool<T, W>::n_regs();
            auto float_mask = avx::from_mask<double, W>::apply(x);
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm256_castpd_si256(float_mask.reg(idx));
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

} } } // namespace simd::kernel::avx
