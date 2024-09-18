#include <gtest/gtest.h>

#include "simd/simd.h"

TEST(vec_op_sse, test_vec_init_regs)
{
    {
        simd::Vec<float, 4> al(0, 1, 2, 3);
        simd::Vec<float, 4> ah(4, 5, 6, 7);
        float pa[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
        auto b = simd::Vec<float, 8>::load_unaligned(pa);
        auto a = simd::Vec<float, 8>(al, ah);
        EXPECT_TRUE(simd::all(a == b));
        for (int i = 0; i < 8; i ++) {
            EXPECT_FLOAT_EQ(pa[i], a[i]);
        }
    }
    {
        simd::Vec<double, 2> al(0, 1);
        simd::Vec<double, 2> ah(2, 3);
        double pa[] = { 0, 1, 2, 3 };
        auto b = simd::Vec<double, 4>::load_unaligned(pa);
        auto a = simd::Vec<double, 4>(al, ah);
        EXPECT_TRUE(simd::all(a == b));
        for (int i = 0; i < 4; i ++) {
            EXPECT_FLOAT_EQ(pa[i], a[i]);
        }
    }
}
