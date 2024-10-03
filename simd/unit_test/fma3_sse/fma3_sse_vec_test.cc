#include <gtest/gtest.h>

#include "simd/simd.h"
#include "simd/unit_test/test_common.h"

using namespace simd;

STATIC_CHECK_ARCH_ENABLED(FMA3_SSE);

TEST(vec_fma3_sse, test_types)
{
    TEST_VEC_TYPE(simd::vi8x16_t, 16, 1, 16, simd::FMA3<SSE>);
    TEST_VEC_TYPE(simd::vi8x32_t, 32, 1, 32, simd::FMA3<AVX>);
    TEST_VEC_TYPE(simd::vi8x64_t, 64, 2, 32, simd::FMA3<AVX>);

    TEST_VEC_TYPE(simd::vu8x16_t, 16, 1, 16, simd::FMA3<SSE>);
    TEST_VEC_TYPE(simd::vu8x32_t, 32, 1, 32, simd::FMA3<AVX>);
    TEST_VEC_TYPE(simd::vu8x64_t, 64, 2, 32, simd::FMA3<AVX>);

    TEST_VEC_TYPE(simd::vi16x8_t,  8,  1, 8,  simd::FMA3<SSE>);
    TEST_VEC_TYPE(simd::vi16x16_t, 16, 1, 16, simd::FMA3<AVX>);
    TEST_VEC_TYPE(simd::vi16x32_t, 32, 2, 16, simd::FMA3<AVX>);

    TEST_VEC_TYPE(simd::vu16x8_t,  8,  1, 8,  simd::FMA3<SSE>);
    TEST_VEC_TYPE(simd::vu16x16_t, 16, 1, 16, simd::FMA3<AVX>);
    TEST_VEC_TYPE(simd::vu16x32_t, 32, 2, 16, simd::FMA3<AVX>);

    TEST_VEC_TYPE(simd::vi32x4_t,  4,  1, 4, simd::FMA3<SSE>);
    TEST_VEC_TYPE(simd::vi32x8_t,  8,  1, 8, simd::FMA3<AVX>);
    TEST_VEC_TYPE(simd::vi32x16_t, 16, 2, 8, simd::FMA3<AVX>);

    TEST_VEC_TYPE(simd::vu32x4_t,  4,  1, 4, simd::FMA3<SSE>);
    TEST_VEC_TYPE(simd::vu32x8_t,  8,  1, 8, simd::FMA3<AVX>);
    TEST_VEC_TYPE(simd::vu32x16_t, 16, 2, 8, simd::FMA3<AVX>);

    TEST_VEC_TYPE(simd::vi64x2_t, 2, 1, 2, simd::FMA3<SSE>);
    TEST_VEC_TYPE(simd::vi64x4_t, 4, 1, 4, simd::FMA3<AVX>);
    TEST_VEC_TYPE(simd::vi64x8_t, 8, 2, 4, simd::FMA3<AVX>);

    TEST_VEC_TYPE(simd::vu64x2_t, 2, 1, 2, simd::FMA3<SSE>);
    TEST_VEC_TYPE(simd::vu64x4_t, 4, 1, 4, simd::FMA3<AVX>);
    TEST_VEC_TYPE(simd::vu64x8_t, 8, 2, 4, simd::FMA3<AVX>);

    TEST_VEC_TYPE(simd::vf32x4_t,  4,  1, 4, simd::FMA3<SSE>);
    TEST_VEC_TYPE(simd::vf32x8_t,  8,  1, 8, simd::FMA3<AVX>);
    TEST_VEC_TYPE(simd::vf32x16_t, 16, 2, 8, simd::FMA3<AVX>);

    TEST_VEC_TYPE(simd::vf64x2_t, 2, 1, 2, simd::FMA3<SSE>);
    TEST_VEC_TYPE(simd::vf64x4_t, 4, 1, 4, simd::FMA3<AVX>);
    TEST_VEC_TYPE(simd::vf64x8_t, 8, 2, 4, simd::FMA3<AVX>);
}
