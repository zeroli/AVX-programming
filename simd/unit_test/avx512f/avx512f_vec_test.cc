#include <gtest/gtest.h>

#include "simd/simd.h"

#include "simd/unit_test/test_common.h"

using namespace simd;

STATIC_CHECK_ARCH_ENABLED(AVX512F);

TEST(vec_avx512f, test_types)
{
    ut::test_type<simd::vi8x16_t, 16, 1, 16, simd::SSE>();
    ut::test_type<simd::vi8x32_t, 32, 1, 32, simd::AVX2>();
    ut::test_type<simd::vi8x64_t, 64, 1, 32, simd::AVX512F>();

    ut::test_type<simd::vu8x16_t, 16, 1, 16, simd::SSE>();
    ut::test_type<simd::vu8x32_t, 32, 1, 32, simd::AVX2>();
    ut::test_type<simd::vu8x64_t, 64, 1, 64, simd::AVX512F>();

    ut::test_type<simd::vi16x8_t,  8,  1, 8,  simd::SSE>();
    ut::test_type<simd::vi16x16_t, 16, 1, 16, simd::AVX2>();
    ut::test_type<simd::vi16x32_t, 32, 1, 32, simd::AVX512F>();

    ut::test_type<simd::vu16x8_t,  8,  1, 8,  simd::SSE>();
    ut::test_type<simd::vu16x16_t, 16, 1, 16, simd::AVX2>();
    ut::test_type<simd::vu16x32_t, 32, 1, 32, simd::AVX512F>();

    ut::test_type<simd::vi32x4_t,  4,  1, 4,  simd::SSE>();
    ut::test_type<simd::vi32x8_t,  8,  1, 8,  simd::AVX2>();
    ut::test_type<simd::vi32x16_t, 16, 1, 16,  simd::AVX512F>();

    ut::test_type<simd::vu32x4_t,  4,  1, 4,  simd::SSE>();
    ut::test_type<simd::vu32x8_t,  8,  1, 8,  simd::AVX2>();
    ut::test_type<simd::vu32x16_t, 16, 1, 16, simd::AVX512F>();

    ut::test_type<simd::vi64x2_t,  2,  1, 2, simd::SSE>();
    ut::test_type<simd::vi64x4_t,  4,  1, 4, simd::AVX2>();
    ut::test_type<simd::vi64x8_t,  8,  1, 8, simd::AVX512F>();

    ut::test_type<simd::vu64x2_t,  2,  1, 2, simd::SSE>();
    ut::test_type<simd::vu64x4_t,  4,  1, 4, simd::AVX2>();
    ut::test_type<simd::vu64x8_t,  8,  1, 8, simd::AVX512F>();

    ut::test_type<simd::vf32x4_t,  4,  1, 4,  simd::SSE>();
    ut::test_type<simd::vf32x8_t,  8,  1, 8,  simd::AVX2>();
    ut::test_type<simd::vf32x16_t, 16, 1, 16, simd::AVX512F>();

    ut::test_type<simd::vf64x2_t, 2, 1, 2, simd::SSE>();
    ut::test_type<simd::vf64x4_t, 4, 1, 4, simd::AVX2>();
    ut::test_type<simd::vf64x8_t, 8, 1, 8, simd::AVX512F>();
}
