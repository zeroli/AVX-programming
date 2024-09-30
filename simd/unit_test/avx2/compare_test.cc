#include <gtest/gtest.h>

#include "simd/simd.h"

STATIC_CHECK_ARCH_ENABLED(AVX2);

TEST(vec_op_avx2, test_cmp_eq)
{
    {
        simd::vi32x8_t a(2), b(2);
        EXPECT_TRUE(simd::all_of(a == b));
    }
    {
        simd::vf32x8_t a(2.f), b(2.f);
        EXPECT_TRUE(simd::all_of(b == a));
    }
    {
        simd::vf64x4_t a(2.0), b(2.0);
        EXPECT_TRUE(simd::all_of(b == a));
    }
    {
        simd::vf64x4_t a(1.0), b(2.0);
        EXPECT_TRUE(simd::none_of(b == a));
    }
}

TEST(vec_op_avx2, test_cmp_ne)
{
    {
        simd::vi32x8_t a(2), b(1);
        EXPECT_TRUE(simd::all_of(a != b));
    }
    {
        simd::vf32x8_t a(1.f), b(2.f);
        EXPECT_TRUE(simd::all_of(b != a));
    }
    {
        simd::vf64x4_t a(1.0), b(2.0);
        EXPECT_TRUE(simd::all_of(b != a));
    }
}

TEST(vec_op_avx2, test_cmp_lt)
{
    {
        simd::vi32x8_t a(1), b(2);
        EXPECT_TRUE(simd::all_of(a < b));
    }
    {
        simd::vf32x8_t a(1.f), b(2.f);
        EXPECT_TRUE(simd::all_of(a < b));
    }
    {
        simd::vf64x4_t a(1.0), b(2.0);
        EXPECT_TRUE(simd::all_of(a < b));
    }
}

TEST(vec_op_avx2, test_cmp_le)
{
    {
        simd::vi32x8_t a(1), b(1);
        EXPECT_TRUE(simd::all_of(a <= b));
    }
    {
        simd::vf32x8_t a(1.f), b(2.f);
        EXPECT_TRUE(simd::all_of(a <= b));
    }
    {
        simd::vf64x4_t a(1.0), b(2.0);
        EXPECT_TRUE(simd::all_of(a <= b));
    }
}

TEST(vec_op_avx2, test_cmp_gt)
{
    {
        simd::vi32x8_t a(2), b(1);
        EXPECT_TRUE(simd::all_of(a > b));
    }
    {
        simd::vf32x8_t a(1.f), b(2.f);
        EXPECT_TRUE(simd::all_of(b > a));
    }
    {
        simd::vf64x4_t a(1.0), b(2.0);
        EXPECT_TRUE(simd::all_of(b > a));
    }
}

TEST(vec_op_avx2, test_cmp_ge)
{
    {
        simd::vi32x8_t a(2), b(1);
        EXPECT_TRUE(simd::all_of(a >= b));
    }
    {
        simd::vf32x8_t a(1.f), b(2.f);
        EXPECT_TRUE(simd::all_of(b >= a));
    }
    {
        simd::vf64x4_t a(1.0), b(2.0);
        EXPECT_TRUE(simd::all_of(b >= a));
    }
    {
        simd::vf64x4_t a(2.0), b(2.0);
        EXPECT_TRUE(simd::all_of(b >= a));
    }
}
