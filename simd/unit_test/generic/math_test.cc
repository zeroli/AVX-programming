#include <gtest/gtest.h>

#include "simd/simd.h"

using namespace simd;

TEST(vec_op_generic, test_math_sign)
{
    {
        simd::Vec<int32_t, 4> a(1, -1, 0, 2), p(+1, -1, 0, +1);
        auto c = simd::sign(a);
        EXPECT_TRUE(simd::all(p == c));
    }
    {
        simd::Vec<float, 4> a(1, -1, 0, 2), p(+1, -1, 0, +1);
        auto c = simd::sign(a);
        EXPECT_TRUE(simd::all(p == c));
    }
    {
        simd::Vec<double, 4> a(1, -1, 0, 2), p(+1, -1, 0, +1);
        auto c = simd::sign(a);
        EXPECT_TRUE(simd::all(p == c));
    }
}

TEST(vec_op_generic, test_math_bitofsign)
{
    {
        simd::Vec<int32_t, 4> a(1, -1, 0, 2);
        auto c = simd::bitofsign(a);
        EXPECT_TRUE(bits::at_msb(c[0]) == false);
        EXPECT_TRUE(bits::at_msb(c[1]) == true);
        EXPECT_TRUE(bits::at_msb(c[2]) == false);
        EXPECT_TRUE(bits::at_msb(c[3]) == false);
    }
    {
        simd::Vec<float, 4> a(1, -1, 0, 2);
        auto c = simd::bitofsign(a);
        EXPECT_TRUE(bits::at_msb(c[0]) == false);
        EXPECT_TRUE(bits::at_msb(c[1]) == true);
        EXPECT_TRUE(bits::at_msb(c[2]) == false);
        EXPECT_TRUE(bits::at_msb(c[3]) == false);
    }
    {
        simd::Vec<double, 4> a(1, -1, 0, 2);
        auto c = simd::bitofsign(a);
        EXPECT_TRUE(bits::at_msb(c[0]) == false);
        EXPECT_TRUE(bits::at_msb(c[1]) == true);
        EXPECT_TRUE(bits::at_msb(c[2]) == false);
        EXPECT_TRUE(bits::at_msb(c[3]) == false);
    }
}
