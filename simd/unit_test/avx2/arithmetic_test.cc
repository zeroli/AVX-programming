#include <gtest/gtest.h>

#include "simd/simd.h"

using namespace simd;

STATIC_CHECK_ARCH_ENABLED(AVX2);

TEST(vec_op_avx2, test_arith_add)
{
    {
        simd::vi32x8_t a(1), b(2), p(3);
        auto c = a + b;
        EXPECT_TRUE(simd::all_of(p == c));
        auto d = a + 2;
        EXPECT_TRUE(simd::all_of(p == d));
        auto e = simd::add(a, b);
        EXPECT_TRUE(simd::all_of(p == e));
        auto f = simd::add(a, 2);
        EXPECT_TRUE(simd::all_of(p == f));
        auto g = simd::add(1, b);
        EXPECT_TRUE(simd::all_of(p == g));
    }
    {
        simd::vf32x8_t a(1.f), b(2.f), p(3.f);
        auto c = a + b;
        EXPECT_TRUE(simd::all_of(p == c));
        auto d = a + 2.f;
        EXPECT_TRUE(simd::all_of(p == d));
        auto e = simd::add(a, b);
        EXPECT_TRUE(simd::all_of(p == e));
        auto f = simd::add(a, 2.f);
        EXPECT_TRUE(simd::all_of(p == f));
        auto g = simd::add(1.f, b);
        EXPECT_TRUE(simd::all_of(p == g));
    }
    {
        simd::vf64x4_t a(1.0), b(2.0), p(3.0);
        auto c = a + b;
        EXPECT_TRUE(simd::all_of(p == c));
        auto d = a + 2.0;
        EXPECT_TRUE(simd::all_of(p == d));
        auto e = simd::add(a, b);
        EXPECT_TRUE(simd::all_of(p == e));
        auto f = simd::add(a, 2.0);
        EXPECT_TRUE(simd::all_of(p == f));
        auto g = simd::add(1.0, b);
        EXPECT_TRUE(simd::all_of(p == g));
    }
}
