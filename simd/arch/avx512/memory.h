#pragma once

namespace simd { namespace kernel { namespace avx512 {
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
                ret.reg(idx) = _mm512_set1_epi8(val);
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 2) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm512_set1_epi16(val);
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 4) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm512_set1_epi32(val);
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 8) {
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                ret.reg(idx) = _mm512_set1_epi64(val);
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
            ret.reg(idx) = _mm512_set1_ps(val);
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
            ret.reg(idx) = _mm512_set1_pd(val);
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
            ret.reg(idx) = _mm512_setzero_si512();
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
            ret.reg(idx) = _mm512_setzero_pd();
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
            ret.reg(idx) = _mm512_setzero_pd();
        }
        return ret;
    }
};

/// set individual elements
template <typename T, size_t W>
struct set<T, W, REQUIRE_INTEGRAL(T)>
{
    SIMD_INLINE
    static Vec<T, W> apply(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7) noexcept
    {
        static_assert(W == 8);
        Vec<T, W> ret = _mm512_set_epi64(v7, v6, v5, v4, v3, v2, v1, v0);
        return ret;
    }
    SIMD_INLINE
    static Vec<T, W> apply(T v0, T v1, T v2,  T v3,  T v4,  T v5,  T v6,  T v7,
                           T v8, T v9, T v10, T v11, T v12, T v13, T v14, T v15) noexcept
    {
        static_assert(W == 16);
        Vec<T, W> ret;
        SIMD_IF_CONSTEXPR(sizeof(T) == 4) {  // 16 * 4 * 8 = 512 (1 reg)
            ret.reg(0) = _mm512_set_epi32(v15, v14, v13, v12, v11, v10, v9, v8,
                                          v7,  v6,  v5,  v4,  v3,  v2,  v1, v0);
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 8) {  // 16 * 8 * 8 = 512 (2 regs)
            ret.reg(0) = _mm512_set_epi64(v7,  v6,  v5,  v4,  v3,  v2,  v1, v0);
            ret.reg(1) = _mm512_set_epi64(v15, v14, v13, v12, v11, v10, v9, v8);
        } else {
            /// sizeof(T) = 1 goes to sse (128bits = 16 * 1 * 8)
            /// sizeof(T) = 2 goes to avx2 (256bits = 16 * 2 * 8)
            assert(0);
        }
        return ret;
    }
    SIMD_INLINE
    static Vec<T, W> apply(T v0,  T v1,  T v2,  T v3,  T v4,  T v5,  T v6,  T v7,
                           T v8,  T v9,  T v10, T v11, T v12, T v13, T v14, T v15,
                           T v16, T v17, T v18, T v19, T v20, T v21, T v22, T v23,
                           T v24, T v25, T v26, T v27, T v28, T v29, T v30, T v31) noexcept
    {
        static_assert(W == 32);
        Vec<T, W> ret;
        SIMD_IF_CONSTEXPR(sizeof(T) == 2) {  // 32 * 2 * 8 = 512 (1 reg)
            ret.reg(0) = _mm512_set_epi16(v31, v30, v29, v28, v27, v26, v25, v24,
                                          v23, v22, v21, v20, v19, v18, v17, v16,
                                          v15, v14, v13, v12, v11, v10, v9,  v8,
                                          v7,  v6,  v5,  v4,  v3,  v2,  v1,  v0);
        } else {
            /// sizeof(T) = 1 goes to avx2 (256bits = 32 * 1 * 8)
            /// sizeof(T) = 4 invalid (1024bits)
            /// sizeof(T) = 8 invalid (1024bits)
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
        ret.reg(0) = _mm512_set_epi8(v63, v62, v61, v60, v59, v58, v57, v56,
                                     v55, v54, v53, v52, v51, v50, v49, v48,
                                     v47, v46, v45, v44, v43, v42, v41, v40,
                                     v39, v38, v37, v36, v35, v34, v33, v32,
                                     v31, v30, v29, v28, v27, v26, v25, v24,
                                     v23, v22, v21, v20, v19, v18, v17, v16,
                                     v15, v14, v13, v12, v11, v10, v9,  v8,
                                     v7,  v6,  v5,  v4,  v3,  v2,  v1,  v0);
        return ret;
    }
};

template <size_t W>
struct set<float, W>
{
    SIMD_INLINE
    static Vec<float, W> apply(float v0, float v1, float v2,  float v3,  float v4,  float v5,  float v6,  float v7,
                               float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15) noexcept
    {
        static_assert(W == 16);
        // 16 * 4 * 8 = 512 (1 reg)
        Vec<float, W> ret = _mm512_set_ps(v15, v14, v13, v12, v11, v10, v9, v8,
                                          v7,  v6,  v5,  v4,  v3,  v2,  v1, v0);
        return ret;
    }
    /// support up to 512bits
};

template <size_t W>
struct set<double, W>
{
    SIMD_INLINE
    static Vec<double, W> apply(double v0, double v1, double v2, double v3, float v4, float v5, float v6, float v7) noexcept
    {
        static_assert(W == 8);
        // 8 * 8 * 8 = 512 (1 reg)
        Vec<double, W> ret = _mm512_set_pd(v7, v6, v5, v4, v3, v2, v1, v0);
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
            ret.reg(idx) = _mm512_load_si512((const avx_reg_i*)(mem + idx * reg_lanes));
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
            ret.reg(idx) = _mm512_load_ps(mem + idx * reg_lanes);
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
            ret.reg(idx) = _mm512_load_pd(mem + idx * reg_lanes);
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
            ret.reg(idx) = _mm512_loadu_si512((const avx_reg_i*)(mem + idx * reg_lanes));
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
            ret.reg(idx) = _mm512_loadu_ps(mem + idx * reg_lanes);
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
            ret.reg(idx) = _mm512_loadu_pd(mem + idx * reg_lanes);
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
            _mm512_store_si512((avx_reg_i*)(mem + idx * reg_lanes), x.reg(idx));
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
            _mm512_store_ps(mem + idx * reg_lanes, x.reg(idx));
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
            _mm512_store_pd(mem + idx * reg_lanes, x.reg(idx));
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
            _mm512_storeu_si512((avx_reg_i*)(mem + idx * reg_lanes), x.reg(idx));
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
            _mm512_storeu_ps(mem + idx * reg_lanes, x.reg(idx));
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
            _mm512_storeu_pd(mem + idx * reg_lanes, x.reg(idx));
        }
    }
};

/// to_mask
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
                //ret <<= 64;  /// 64 * elements for 64 bits
                //ret |= _mm512_movemask_epi8(x.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 2) {
            #pragma unroll
            for (int idx = (int)nregs - 1; idx >= 0; idx--) {
                ret <= 32;  // 32 * elements for 32 bits
                //ret |= detail::movemask_epi16(x.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 4) {
            #pragma unroll
            for (int idx = (int)nregs - 1; idx >= 0; idx--) {
                ret <<= 16;  // 16 * elements for 16 bits
                //ret |= detail::movemask_epi32(x.reg(idx));
            }
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 8) {
            #pragma unroll
            for (int idx = (int)nregs - 1; idx >= 0; idx--) {
                ret <<= 8;  // 8 * elements for 8 bits
                //ret |= detail::movemask_epi64(x.reg(idx));
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
            ret <<= 16;  // 16 * elements for 16 bits
            //ret |= _mm512_movemask_ps(x.reg(idx));
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
            ret <<= 8;  // 8 * elements for 8 bits
            //ret |= _mm_movemask_pd(x.reg(idx));
        }
        return ret;
    }
};

/// from_mask
template <size_t W>
struct from_mask<float, W>
{
    SIMD_INLINE
    static VecBool<float, W> apply(uint64_t x) noexcept
    {
        VecBool<float, W> ret;
        constexpr auto nregs = VecBool<float, W>::n_regs();
        constexpr auto reg_lanes = VecBool<float, W>::reg_lanes();
        constexpr auto lanes_mask = (1ull << reg_lanes) - 1;
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            //ret.reg(idx) = _mm512_castsi512_ps(mask_lut(x & lanes_mask));
            x >>= reg_lanes;
        }
        return ret;
    }
};

template <size_t W>
struct from_mask<double, W>
{
    SIMD_INLINE
    static VecBool<double, W> apply(uint64_t x) noexcept
    {
        VecBool<double, W> ret;
        constexpr auto nregs = VecBool<double, W>::n_regs();
        constexpr auto reg_lanes = VecBool<double, W>::reg_lanes();
        constexpr auto lanes_mask = (1ull << reg_lanes) - 1;
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            //ret.reg(idx) = _mm512_castsi512_pd(mask_lut(x & lanes_mask));
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
                // TODO:
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
                // TODO:
                x >>= reg_lanes;
            }
            return ret;
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 4) {
            VecBool<T, W> ret;
            constexpr auto nregs = VecBool<T, W>::n_regs();
            auto float_mask = avx::from_mask<float, W>::apply(x);
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                // TODO:
            }
            return ret;
        } else SIMD_IF_CONSTEXPR(sizeof(T) == 8) {
            VecBool<T, W> ret;
            constexpr auto nregs = VecBool<T, W>::n_regs();
            auto float_mask = avx::from_mask<double, W>::apply(x);
            #pragma unroll
            for (auto idx = 0; idx < nregs; idx++) {
                // TODO:
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

} } } // namespace simd::kernel::avx512
