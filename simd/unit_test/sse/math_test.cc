#include <gtest/gtest.h>

#include "simd/simd.h"

TEST(vec_op_sse, test_math_abs)
{
    {
        simd::Vec<int8_t, 16> a(-3), p(3);
        auto c = simd::abs(a);
        EXPECT_TRUE(simd::all(p == c));
        std::cout << "abs(" << a << "(a)" <<") = " << c << "(c)\n";
    }
    {
        simd::Vec<int16_t, 8> a(-3), p(3);
        auto c = simd::abs(a);
        EXPECT_TRUE(simd::all(p == c));
        std::cout << "abs(" << a << "(a)" <<") = " << c << "(c)\n";
    }
    {
        simd::Vec<int32_t, 4> a(-3), p(3);
        auto c = simd::abs(a);
        EXPECT_TRUE(simd::all(p == c));
        std::cout << "abs(" << a << "(a)" <<") = " << c << "(c)\n";
    }
    // {
    //     simd::Vec<int64_t, 2> a(-3);
    //     auto c = simd::abs(a);
    //     std::cout << "abs(" << a << "(a)" <<") = " << c << "(c)\n";
    // }
    {
        simd::Vec<float, 4> a(-4.f), p(4.f);
        auto c = simd::abs(a);
        EXPECT_TRUE(simd::all(p == c));
        std::cout << "abs(" << a << "(a)" <<") = " << c << "(c)\n";
    }
    {
        simd::Vec<double, 2> a(-4.0), p(4.0);
        auto c = simd::abs(a);
        EXPECT_TRUE(simd::all(p == c));
        std::cout << "abs(" << a << "(a)" <<") = " << c << "(c)\n";
    }
}

TEST(vec_op_sse, test_math_sqrt)
{
    #if 0 // compilation error, non supported op for integral types
    {
        simd::Vec<int32_t, 4> a(-3);
        auto c = simd::sqrt(a);
        std::cout << "sqrt(" << a << "(a)" <<") = " << c << "(c)\n";
    }
    #endif
    {
        simd::Vec<float, 4> a(4.f), p(2.f);
        auto c = simd::sqrt(a);
        EXPECT_TRUE(simd::all(p == c));
        std::cout << "sqrt(" << a << "(a)" <<") = " << c << "(c)\n";
    }
    {
        simd::Vec<double, 2> a(4.0), p(2.0);
        auto c = simd::sqrt(a);
        EXPECT_TRUE(simd::all(p == c));
        std::cout << "sqrt(" << a << "(a)" <<") = " << c << "(c)\n";
    }
}
