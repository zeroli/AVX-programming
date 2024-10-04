#include <gtest/gtest.h>

#include "simd/simd.h"
#include "check_arch.h"

TEST(vec_op_sse, test_math_abs)
{
    {
        simd::Vec<int8_t, 16> a(-3), p(3);
        auto c = simd::abs(a);
        EXPECT_TRUE(simd::all_of(p == c));
    }
    {
        simd::Vec<int16_t, 8> a(-3), p(3);
        auto c = simd::abs(a);
        EXPECT_TRUE(simd::all_of(p == c));
    }
    {
        simd::Vec<int32_t, 4> a(-3), p(3);
        auto c = simd::abs(a);
        EXPECT_TRUE(simd::all_of(p == c));
    }
    // {
    //     simd::Vec<int64_t, 2> a(-3);
    //     auto c = simd::abs(a);
    // }
    {
        simd::Vec<float, 4> a(-4.f), p(4.f);
        auto c = simd::abs(a);
        EXPECT_TRUE(simd::all_of(p == c));
    }
    {
        simd::Vec<double, 2> a(-4.0), p(4.0);
        auto c = simd::abs(a);
        EXPECT_TRUE(simd::all_of(p == c));
    }
}

TEST(vec_op_sse, test_math_sqrt)
{
    #if 0 // compilation error, non supported op for integral types
    {
        simd::Vec<int32_t, 4> a(-3);
        auto c = simd::sqrt(a);
    }
    #endif
    {
        simd::Vec<float, 4> a(4.f), p(2.f);
        auto c = simd::sqrt(a);
        EXPECT_TRUE(simd::all_of(p == c));
    }
    {
        simd::Vec<double, 2> a(4.0), p(2.0);
        auto c = simd::sqrt(a);
        EXPECT_TRUE(simd::all_of(p == c));
    }
}

TEST(vec_op_sse, test_math_ceil)
{
    {
        simd::Vec<int32_t, 4> a(3), p(3);
        auto c = simd::ceil(a);
        EXPECT_TRUE(simd::all_of(p == c));
    }
    {
        simd::Vec<float, 4> a(3.8f, 3.1f, 0.3f, 2.9999f), p(4.f, 4.f, 1.f, 3.f);
        auto c = simd::ceil(a);
        EXPECT_TRUE(simd::all_of(p == c));
    }
    {
        simd::Vec<float, 4> a(-3.8f, -3.1f, -0.3f, -100.323f), p(-3.f, -3.f, 0.f, -100.f);
        auto c = simd::ceil(a);
        EXPECT_TRUE(simd::all_of(p == c)) << c;
    }
    {
        simd::Vec<double, 2> a(4.5), p(5.0);
        auto c = simd::ceil(a);
        EXPECT_TRUE(simd::all_of(p == c));
    }
}

TEST(vec_op_sse, test_math_floor)
{
    {
        simd::Vec<int32_t, 4> a(3), p(3);
        auto c = simd::floor(a);
        EXPECT_TRUE(simd::all_of(p == c));
    }
    {
        simd::Vec<float, 4> a(3.8f, 3.1f, 0.01f, 2.9999f), p(3.f, 3.f, 0.f, 2.f);
        auto c = simd::floor(a);
        EXPECT_TRUE(simd::all_of(p == c));
    }
    {
        simd::Vec<float, 4> a(-3.8f, 2.1f, 300.23f, -203.111f), p(-4.f, 2.f, 300.f, -204.f);
        auto c = simd::floor(a);
        EXPECT_TRUE(simd::all_of(p == c)) << c;
    }
    {
        simd::Vec<double, 2> a(4.5), p(4.0);
        auto c = simd::floor(a);
        EXPECT_TRUE(simd::all_of(p == c));
    }
}

TEST(vec_op_sse, test_math_log)
{
    {
        simd::vf32x4_t a(3.8f, 3.1f, 0.01f, 2.9999f);
        auto c = simd::log(a);
        for (auto i = 0; i < a.size(); i++) {
            EXPECT_FLOAT_EQ(std::log(a[i]), c[i]);
        }
    }
}
