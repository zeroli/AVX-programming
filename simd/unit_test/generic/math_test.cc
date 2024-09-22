#include <gtest/gtest.h>

#include "simd/simd.h"

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
