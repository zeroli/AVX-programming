#include <gtest/gtest.h>

#include "simd/simd.h"

using namespace simd;

STATIC_CHECK_ARCH_ENABLED(AVX);

TEST(vec_op_avx, test_arith_add)
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
}
