#include <gtest/gtest.h>

#include "simd/simd.h"

TEST(vec_op_sse, test_algo_min)
{
    {
        simd::Vec<int8_t, 16> a(4), b(-2), p(-2);
        auto c = simd::min(a, b);
        EXPECT_TRUE(simd::all(p == c));
    }
    {
        simd::Vec<uint8_t, 16> a(4), b(2), p(2);
        auto c = simd::min(a, b);
        EXPECT_TRUE(simd::all(p == c));
    }
    {
        simd::Vec<int16_t, 8> a(4), b(-2), p(-2);
        auto c = simd::min(a, b);
        EXPECT_TRUE(simd::all(p == c));
    }
    {
        simd::Vec<uint16_t, 8> a(4), b(2), p(2);
        auto c = simd::min(a, b);
        EXPECT_TRUE(simd::all(p == c));
    }
    {
        simd::Vec<int32_t, 4> a(4), b(-2), p(-2);
        auto c = simd::min(a, b);
        EXPECT_TRUE(simd::all(p == c));
    }
    {
        simd::Vec<uint32_t, 4> a(4), b(2), p(2);
        auto c = simd::min(a, b);
        EXPECT_TRUE(simd::all(p == c));
    }
    // {
    //     simd::Vec<int64_t, 2> a(4), b(-2);
    //     auto c = simd::min(a, b);
    // }
    // {
    //     simd::Vec<uint64_t, 2> a(4), b(2);
    //     auto c = simd::min(a, b);
    // }
    {
        simd::Vec<float, 4> a(4.f), b(2.f), p(2.f);
        auto c = simd::min(a, b);
        EXPECT_TRUE(simd::all(p == c));
    }
    {
        simd::Vec<double, 2> a(8.0), b(2.0), p(2.0);
        auto c = simd::min(a, b);
        EXPECT_TRUE(simd::all(p == c));
    }
}

TEST(vec_op_sse, test_algo_max)
{
    {
        simd::Vec<int8_t, 16> a(4), b(-2), p(4);
        auto c = simd::max(a, b);
        EXPECT_TRUE(simd::all(p == c));
    }
    {
        simd::Vec<uint8_t, 16> a(4), b(2), p(4);
        auto c = simd::max(a, b);
        EXPECT_TRUE(simd::all(p == c));
    }
    {
        simd::Vec<int16_t, 8> a(4), b(-2), p(4);
        auto c = simd::max(a, b);
        EXPECT_TRUE(simd::all(p == c));
    }
    {
        simd::Vec<uint16_t, 8> a(4), b(2), p(4);
        auto c = simd::max(a, b);
        EXPECT_TRUE(simd::all(p == c));
    }
    {
        simd::Vec<int32_t, 4> a(4), b(-2), p(4);
        auto c = simd::max(a, b);
        EXPECT_TRUE(simd::all(p == c));
    }
    {
        simd::Vec<uint32_t, 4> a(4), b(2), p(4);
        auto c = simd::max(a, b);
        EXPECT_TRUE(simd::all(p == c));
    }
    // {
    //     simd::Vec<int64_t, 2> a(4), b(-2);
    //     auto c = simd::max(a, b);
    // }
    // {
    //     simd::Vec<uint64_t, 2> a(4), b(2);
    //     auto c = simd::max(a, b);
    // }
    {
        simd::Vec<float, 4> a(4.f), b(2.f), p(4.f);
        auto c = simd::max(a, b);
        EXPECT_TRUE(simd::all(p == c));
    }
    {
        simd::Vec<double, 2> a(8.0), b(2.0), p(8.0);
        auto c = simd::max(a, b);
        EXPECT_TRUE(simd::all(p == c));
    }
}

TEST(vec_op_sse, test_algo_any_all)
{
    {
        simd::Vec<int32_t, 4> a(1);
        EXPECT_TRUE(simd::any(a == 1));
        EXPECT_TRUE(simd::all(a == 1));
    }
    {
        simd::Vec<int32_t, 4> a(0);
        EXPECT_TRUE(simd::any(a == 0));
        EXPECT_FALSE(simd::all(a == 1));
    }
    {
        simd::Vec<float, 4> a(1.f);
        EXPECT_TRUE(simd::any(a == 1.f));
        EXPECT_TRUE(simd::all(a == 1.f));
    }
    {
        simd::Vec<float, 4> a(0.f);
        EXPECT_FALSE(simd::any(a == 1.f));
    }
    {
        simd::Vec<double, 2> a(1.0);
        EXPECT_TRUE(simd::any(a == 1.0));
    }
    {
        simd::Vec<double, 2> a(0.0);
        EXPECT_FALSE(simd::any(a == 1.0));
        EXPECT_TRUE(simd::any(a != 1.0));
    }
}

TEST(vec_op_sse, test_algo_select)
{
    {
        simd::VecBool<int32_t, 4> c(true, false, true, false);
        simd::Vec<int32_t, 4> a(1), b(2), p(1,2,1,2);
        auto r = simd::select(c, a, b);
        EXPECT_TRUE(simd::all(r == p));
    }
    {
        simd::Vec<int32_t, 4> a(1,2,3,-1), b(2), p(1,2,2,-1);
        auto r = simd::select(a <= b, a, b);
        EXPECT_TRUE(simd::all(r == p));
    }
    {
        simd::Vec<int32_t, 8> a(1,2,3,-1,0,1,2,3), b(2), p(1,2,2,-1,0,1,2,2);
        auto r = simd::select(a <= b, a, b);
        EXPECT_TRUE(simd::all(r == p));
    }
    {
        simd::Vec<float, 4> a(1,2,3,-1), b(2), p(1,2,2,-1);
        auto r = simd::select(a <= b, a, b);
        EXPECT_TRUE(simd::all(r == p));
    }
    {
        simd::Vec<double, 4> a(1,2,3,-1), b(2), p(1,2,2,-1);
        auto r = simd::select(a <= b, a, b);
        EXPECT_TRUE(simd::all(r == p));
    }
}
