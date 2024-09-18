#include <gtest/gtest.h>

#include "simd/simd.h"

TEST(vec_op_sse, test_algo_min)
{
    {
        simd::Vec<int8_t, 16> a(4), b(-2), p(-2);
        auto c = simd::min(a, b);
        std::cout << "min(" << a << "(a)" << ", " << b << "(b)) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
    }
    {
        simd::Vec<uint8_t, 16> a(4), b(2), p(2);
        auto c = simd::min(a, b);
        std::cout << "min(" << a << "(a)" << ", " << b << "(b)) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
    }
    {
        simd::Vec<int16_t, 8> a(4), b(-2), p(-2);
        auto c = simd::min(a, b);
        std::cout << "min(" << a << "(a)" << ", " << b << "(b)) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
    }
    {
        simd::Vec<uint16_t, 8> a(4), b(2), p(2);
        auto c = simd::min(a, b);
        std::cout << "min(" << a << "(a)" << ", " << b << "(b)) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
    }
    {
        simd::Vec<int32_t, 4> a(4), b(-2), p(-2);
        auto c = simd::min(a, b);
        std::cout << "min(" << a << "(a)" << ", " << b << "(b)) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
    }
    {
        simd::Vec<uint32_t, 4> a(4), b(2), p(2);
        auto c = simd::min(a, b);
        std::cout << "min(" << a << "(a)" << ", " << b << "(b)) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
    }
    // {
    //     simd::Vec<int64_t, 2> a(4), b(-2);
    //     auto c = simd::min(a, b);
    //     std::cout << "min(" << a << "(a)" << ", " << b << "(b)) = " << c << "(c)\n";
    // }
    // {
    //     simd::Vec<uint64_t, 2> a(4), b(2);
    //     auto c = simd::min(a, b);
    //     std::cout << "min(" << a << "(a)" << ", " << b << "(b)) = " << c << "(c)\n";
    // }
    {
        simd::Vec<float, 4> a(4.f), b(2.f), p(2.f);
        auto c = simd::min(a, b);
        std::cout << "min(" << a << "(a)" << ", " << b << "(b)) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
    }
    {
        simd::Vec<double, 2> a(8.0), b(2.0), p(2.0);
        auto c = simd::min(a, b);
        std::cout << "min(" << a << "(a)" << ", " << b << "(b)) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
    }
}

TEST(vec_op_sse, test_algo_max)
{
    {
        simd::Vec<int8_t, 16> a(4), b(-2), p(4);
        auto c = simd::max(a, b);
        std::cout << "max(" << a << "(a)" << ", " << b << "(b)) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
    }
    {
        simd::Vec<uint8_t, 16> a(4), b(2), p(4);
        auto c = simd::max(a, b);
        EXPECT_TRUE(simd::all(p == c));
        std::cout << "max(" << a << "(a)" << ", " << b << "(b)) = " << c << "(c)\n";
    }
    {
        simd::Vec<int16_t, 8> a(4), b(-2), p(4);
        auto c = simd::max(a, b);
        EXPECT_TRUE(simd::all(p == c));
        std::cout << "max(" << a << "(a)" << ", " << b << "(b)) = " << c << "(c)\n";
    }
    {
        simd::Vec<uint16_t, 8> a(4), b(2), p(4);
        auto c = simd::max(a, b);
        EXPECT_TRUE(simd::all(p == c));
        std::cout << "max(" << a << "(a)" << ", " << b << "(b)) = " << c << "(c)\n";
    }
    {
        simd::Vec<int32_t, 4> a(4), b(-2), p(4);
        auto c = simd::max(a, b);
        EXPECT_TRUE(simd::all(p == c));
        std::cout << "max(" << a << "(a)" << ", " << b << "(b)) = " << c << "(c)\n";
    }
    {
        simd::Vec<uint32_t, 4> a(4), b(2), p(4);
        auto c = simd::max(a, b);
        EXPECT_TRUE(simd::all(p == c));
        std::cout << "max(" << a << "(a)" << ", " << b << "(b)) = " << c << "(c)\n";
    }
    // {
    //     simd::Vec<int64_t, 2> a(4), b(-2);
    //     auto c = simd::max(a, b);
    //     std::cout << "max(" << a << "(a)" << ", " << b << "(b)) = " << c << "(c)\n";
    // }
    // {
    //     simd::Vec<uint64_t, 2> a(4), b(2);
    //     auto c = simd::max(a, b);
    //     std::cout << "max(" << a << "(a)" << ", " << b << "(b)) = " << c << "(c)\n";
    // }
    {
        simd::Vec<float, 4> a(4.f), b(2.f), p(4.f);
        auto c = simd::max(a, b);
        EXPECT_TRUE(simd::all(p == c));
        std::cout << "max(" << a << "(a)" << ", " << b << "(b)) = " << c << "(c)\n";
    }
    {
        simd::Vec<double, 2> a(8.0), b(2.0), p(8.0);
        auto c = simd::max(a, b);
        EXPECT_TRUE(simd::all(p == c));
        std::cout << "max(" << a << "(a)" << ", " << b << "(b)) = " << c << "(c)\n";
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
