#include <gtest/gtest.h>

#include "simd/simd.h"

using namespace simd;

TEST(vec_op_generic, test_algo_none_of)
{
    {
        simd::Vec<int32_t, 4> a(1), b(0), c(0, 1, 1, 0);
        EXPECT_TRUE(simd::none_of(a == 0));
        EXPECT_TRUE(simd::none_of(b == 1));
        EXPECT_FALSE(simd::none_of(c == 0));
    }
    {
        simd::Vec<float, 4> a(1), b(0), c(0, 1, 1, 0);
        EXPECT_TRUE(simd::none_of(a == 0));
        EXPECT_TRUE(simd::none_of(b == 1));
        EXPECT_FALSE(simd::none_of(c == 0));
    }
}

TEST(vec_op_generic, test_algo_some_of)
{
    {
        simd::Vec<int32_t, 4> a(1), b(0), c(0, 1, 1, 1), d(1, 1, 1, 0);
        EXPECT_FALSE(simd::some_of(a == 1));
        EXPECT_FALSE(simd::some_of(b == 0));
        EXPECT_TRUE(simd::some_of(c == 0));
        EXPECT_TRUE(simd::some_of(d == 1));
    }
    {
        simd::Vec<float, 4> a(1), b(0), c(0, 1, 1, 1), d(1, 1, 1, 0);
        EXPECT_FALSE(simd::some_of(a == 1));
        EXPECT_FALSE(simd::some_of(b == 0));
        EXPECT_TRUE(simd::some_of(c == 0));
        EXPECT_TRUE(simd::some_of(d == 1));
    }
}
