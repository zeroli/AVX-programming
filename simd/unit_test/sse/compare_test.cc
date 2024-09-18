#include <gtest/gtest.h>

#include "simd/simd.h"

TEST(vec_op_sse, test_cmp_lt)
{
    {
        simd::Vec<int32_t, 4> a(1), b(2);
        EXPECT_TRUE(simd::all(a < b));
    }
    {
        simd::Vec<float, 4> a(1.f), b(2.f);
        EXPECT_TRUE(simd::all(a < b));
    }
    {
        simd::Vec<double, 2> a(1.0), b(2.0);
        EXPECT_TRUE(simd::all(a < b));
    }
}

TEST(vec_op_sse, test_cmp_le)
{
    {
        simd::Vec<int32_t, 4> a(1), b(1);
        EXPECT_TRUE(simd::all(a <= b));
    }
    {
        simd::Vec<float, 4> a(1.f), b(2.f);
        EXPECT_TRUE(simd::all(a <= b));
    }
    {
        simd::Vec<double, 2> a(1.0), b(2.0);
        EXPECT_TRUE(simd::all(a <= b));
    }
}

TEST(vec_op_sse, test_cmp_gt)
{
    {
        simd::Vec<int32_t, 4> a(2), b(1);
        EXPECT_TRUE(simd::all(a > b));
    }
    {
        simd::Vec<float, 4> a(1.f), b(2.f);
        EXPECT_TRUE(simd::all(b > a));
    }
    {
        simd::Vec<double, 2> a(1.0), b(2.0);
        EXPECT_TRUE(simd::all(b > a));
    }
}

TEST(vec_op_sse, test_cmp_ge)
{
    {
        simd::Vec<int32_t, 4> a(2), b(1);
        EXPECT_TRUE(simd::all(a >= b));
    }
    {
        simd::Vec<float, 4> a(1.f), b(2.f);
        EXPECT_TRUE(simd::all(b >= a));
    }
    {
        simd::Vec<double, 2> a(1.0), b(2.0);
        EXPECT_TRUE(simd::all(b >= a));
    }
    {
        simd::Vec<double, 2> a(2.0), b(2.0);
        EXPECT_TRUE(simd::all(b >= a));
    }
}

TEST(vec_op_sse, test_cmp_ne)
{
    {
        simd::Vec<int32_t, 4> a(2), b(1);
        EXPECT_TRUE(simd::all(a != b));
    }
    {
        simd::Vec<float, 4> a(1.f), b(2.f);
        EXPECT_TRUE(simd::all(b != a));
    }
    {
        simd::Vec<double, 2> a(1.0), b(2.0);
        EXPECT_TRUE(simd::all(b != a));
    }
}
