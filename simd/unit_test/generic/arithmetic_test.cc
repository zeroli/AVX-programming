#include <gtest/gtest.h>

#include "simd/simd.h"

using namespace simd;

TEST(vec_op_generic, test_arith_fmadd)
{
    {
        simd::vf32x4_t a(1.f, 2.f, 3.f, 4.f);
        simd::vf32x4_t b(1.f, 2.f, 3.f, 4.f);
        simd::vf32x4_t c(1.f, 2.f, 3.f, 4.f);
        simd::vf32x4_t p(1.f * 1.f + 1.f, 2.f * 2.f + 2.f, 3.f * 3.f + 3.f, 4.f * 4.f + 4.f);
        auto d = simd::fmadd(a, b, c);
        for (auto i = 0; i < d.size(); i++) {
            EXPECT_FLOAT_EQ(d[i], p[i]);
        }
    }
    {
        simd::vf64x2_t a(3.3, 4.4);
        simd::vf64x2_t b(5.3, 4.1);
        simd::vf64x2_t c(6.3, 9.8);
        simd::vf64x2_t p(3.3 * 5.3 + 6.3, 4.4 * 4.1 + 9.8);
        auto d = simd::fmadd(a, b, c);
        for (auto i = 0; i < d.size(); i++) {
            EXPECT_FLOAT_EQ(d[i], p[i]);
        }
    }
}
