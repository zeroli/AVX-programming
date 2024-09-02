#include <gtest/gtest.h>

#include "simd/simd.h"

#include <sstream>

TEST(vec_binary_op_sse, test_add)
{
    {
        simd::Vec<int32_t, 4> a(1), b(2), p(3);
        auto c = a + b;
        std::cout << a << "(a)" << " + " << b << "(b) = " << c << "(c)\n";
        //EXPECT_EQ(p, c);
    }
    {
        simd::Vec<int32_t, 4> a(1);
        int b = 2;
        auto c = a + b;
        std::cout << a << "(a)" << " + " << b << "(b) = " << c << "(c)\n";
    }
    {
        simd::Vec<int32_t, 4> a(1);
        int b = 2;
        auto c = b + a;
        std::cout << b << "(b)" << " + " << a << "() = " << c << "(c)\n";
    }
    {
        simd::Vec<float, 4> a(1.f);
        float b = 2.f;
        auto c = a + b;
        std::cout << a << "(a)" << " + " << b << "(b) = " << c << "(c)\n";
    }
    {
        simd::Vec<double, 2> a(1.0);
        double b = 2.0;
        auto c = a + b;
        std::cout << a << "(a)" << " + " << b << "(b) = " << c << "(c)\n";
    }
    {
        simd::Vec<std::complex<float>, 2> a({1.f, 2.f}), b({1.f, 2.f});
        auto c = a + b;
    }
    {
        simd::Vec<std::complex<double>, 1> a({1.0, 2.0}), b({1.0, 2.0});
        auto c = a + b;
    }
}

TEST(vec_binary_op_sse, test_sub)
{
    {
        simd::Vec<int32_t, 4> a(1), b(2), p(3);
        auto c = a + b;
        std::cout << a << "(a)" << " - " << b << "(b) = " << c << "(c)\n";
        //EXPECT_EQ(p, c);
    }
    {
        simd::Vec<int32_t, 4> a(1);
        int b = 2;
        auto c = a - b;
        std::cout << a << "(a)" << " - " << b << "(b) = " << c << "(c)\n";
    }
    {
        simd::Vec<int32_t, 4> a(1);
        int b = 2;
        auto c = b - a;
        std::cout << b << "(b)" << " - " << a << "(a) = " << c << "(c)\n";
    }
}
