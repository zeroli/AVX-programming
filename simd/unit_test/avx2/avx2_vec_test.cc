#include <gtest/gtest.h>

#include "simd/simd.h"

#include "simd/unit_test/test_common.h"

using namespace simd;

STATIC_CHECK_ARCH_ENABLED(AVX2);

TEST(vec_avx2, test_types)
{
    // avx2 has same reg as avx
    ut::test_type<simd::vi8x16_t, 16, 1, 16, simd::SSE>();
    ut::test_type<simd::vi8x32_t, 32, 1, 32, simd::AVX2>();
    ut::test_type<simd::vi8x64_t, 64, 2, 32, simd::AVX2>();

    ut::test_type<simd::vu8x16_t, 16, 1, 16, simd::SSE>();
    ut::test_type<simd::vu8x32_t, 32, 1, 32, simd::AVX2>();
    ut::test_type<simd::vu8x64_t, 64, 2, 32, simd::AVX2>();

    ut::test_type<simd::vi16x8_t,  8,  1, 8,  simd::SSE>();
    ut::test_type<simd::vi16x16_t, 16, 1, 16, simd::AVX2>();
    ut::test_type<simd::vi16x32_t, 32, 2, 16, simd::AVX2>();

    ut::test_type<simd::vu16x8_t,  8,  1, 8,  simd::SSE>();
    ut::test_type<simd::vu16x16_t, 16, 1, 16, simd::AVX2>();
    ut::test_type<simd::vu16x32_t, 32, 2, 16, simd::AVX2>();

    ut::test_type<simd::vi32x4_t,  4,  1, 4, simd::SSE>();
    ut::test_type<simd::vi32x8_t,  8,  1, 8, simd::AVX2>();
    ut::test_type<simd::vi32x16_t, 16, 2, 8, simd::AVX2>();

    ut::test_type<simd::vu32x4_t,  4,  1, 4, simd::SSE>();
    ut::test_type<simd::vu32x8_t,  8,  1, 8, simd::AVX2>();
    ut::test_type<simd::vu32x16_t, 16, 2, 8, simd::AVX2>();

    ut::test_type<simd::vf32x4_t,  4,  1, 4, simd::SSE>();
    ut::test_type<simd::vf32x8_t,  8,  1, 8, simd::AVX2>();
    ut::test_type<simd::vf32x16_t, 16, 2, 8, simd::AVX2>();

    ut::test_type<simd::vf64x2_t, 2, 1, 2, simd::SSE>();
    ut::test_type<simd::vf64x4_t, 4, 1, 4, simd::AVX2>();
    ut::test_type<simd::vf64x8_t, 8, 2, 4, simd::AVX2>();
}
