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
    {
        simd::Vec<float, 4> a(2.f), b(1.f);
        auto c = a - b;
        std::cout << a << "(a)" << " - " << b << "(b) = " << c << "(c)\n";
    }
    {
        simd::Vec<double, 2> a(2.0), b(1.0);
        auto c = a - b;
        std::cout << a << "(a)" << " - " << b << "(b) = " << c << "(c)\n";
    }
}

TEST(vec_binary_op_sse, test_mul)
{
    {
        simd::Vec<float, 4> a(2.f), b(1.f);
        auto c = a * b;
        std::cout << a << "(a)" << " * " << b << "(b) = " << c << "(c)\n";
    }
    {
        simd::Vec<double, 2> a(2.0), b(1.0);
        auto c = a * b;
        std::cout << a << "(a)" << " * " << b << "(b) = " << c << "(c)\n";
    }
}

TEST(vec_binary_op_sse, test_div)
{
    {
        simd::Vec<float, 4> a(4.f), b(2.f);
        auto c = a / b;
        std::cout << a << "(a)" << " / " << b << "(b) = " << c << "(c)\n";
    }
    {
        simd::Vec<double, 2> a(8.0), b(2.0);
        auto c = a / b;
        std::cout << a << "(a)" << " / " << b << "(b) = " << c << "(c)\n";
    }
}

TEST(vec_binary_op_sse, test_min)
{
    {
        simd::Vec<int8_t, 16> a(4), b(-2);
        auto c = simd::min(a, b);
        std::cout << "min(" << a << "(a)" << ", " << b << "(b)) = " << c << "(c)\n";
    }
    {
        simd::Vec<uint8_t, 16> a(4), b(2);
        auto c = simd::min(a, b);
        std::cout << "min(" << a << "(a)" << ", " << b << "(b)) = " << c << "(c)\n";
    }
    {
        simd::Vec<int16_t, 8> a(4), b(-2);
        auto c = simd::min(a, b);
        std::cout << "min(" << a << "(a)" << ", " << b << "(b)) = " << c << "(c)\n";
    }
    {
        simd::Vec<uint16_t, 8> a(4), b(2);
        auto c = simd::min(a, b);
        std::cout << "min(" << a << "(a)" << ", " << b << "(b)) = " << c << "(c)\n";
    }
    {
        simd::Vec<int32_t, 4> a(4), b(-2);
        auto c = simd::min(a, b);
        std::cout << "min(" << a << "(a)" << ", " << b << "(b)) = " << c << "(c)\n";
    }
    {
        simd::Vec<uint32_t, 4> a(4), b(2);
        auto c = simd::min(a, b);
        std::cout << "min(" << a << "(a)" << ", " << b << "(b)) = " << c << "(c)\n";
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
        simd::Vec<float, 4> a(4.f), b(2.f);
        auto c = simd::min(a, b);
        std::cout << "min(" << a << "(a)" << ", " << b << "(b)) = " << c << "(c)\n";
    }
    {
        simd::Vec<double, 2> a(8.0), b(2.0);
        auto c = simd::min(a, b);
        std::cout << "min(" << a << "(a)" << ", " << b << "(b)) = " << c << "(c)\n";
    }
}

TEST(vec_binary_op_sse, test_max)
{
    {
        simd::Vec<int8_t, 16> a(4), b(-2);
        auto c = simd::max(a, b);
        std::cout << "max(" << a << "(a)" << ", " << b << "(b)) = " << c << "(c)\n";
    }
    {
        simd::Vec<uint8_t, 16> a(4), b(2);
        auto c = simd::max(a, b);
        std::cout << "max(" << a << "(a)" << ", " << b << "(b)) = " << c << "(c)\n";
    }
    {
        simd::Vec<int16_t, 8> a(4), b(-2);
        auto c = simd::max(a, b);
        std::cout << "max(" << a << "(a)" << ", " << b << "(b)) = " << c << "(c)\n";
    }
    {
        simd::Vec<uint16_t, 8> a(4), b(2);
        auto c = simd::max(a, b);
        std::cout << "max(" << a << "(a)" << ", " << b << "(b)) = " << c << "(c)\n";
    }
    {
        simd::Vec<int32_t, 4> a(4), b(-2);
        auto c = simd::max(a, b);
        std::cout << "max(" << a << "(a)" << ", " << b << "(b)) = " << c << "(c)\n";
    }
    {
        simd::Vec<uint32_t, 4> a(4), b(2);
        auto c = simd::max(a, b);
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
        simd::Vec<float, 4> a(4.f), b(2.f);
        auto c = simd::max(a, b);
        std::cout << "max(" << a << "(a)" << ", " << b << "(b)) = " << c << "(c)\n";
    }
    {
        simd::Vec<double, 2> a(8.0), b(2.0);
        auto c = simd::max(a, b);
        std::cout << "max(" << a << "(a)" << ", " << b << "(b)) = " << c << "(c)\n";
    }
}

TEST(vec_binary_op_sse, test_abs)
{
    {
        simd::Vec<int8_t, 16> a(-3);
        auto c = simd::abs(a);
        std::cout << "abs(" << a << "(a)" <<") = " << c << "(c)\n";
    }
    {
        simd::Vec<int16_t, 8> a(-3);
        auto c = simd::abs(a);
        std::cout << "abs(" << a << "(a)" <<") = " << c << "(c)\n";
    }
    {
        simd::Vec<int32_t, 4> a(-3);
        auto c = simd::abs(a);
        std::cout << "abs(" << a << "(a)" <<") = " << c << "(c)\n";
    }
    // {
    //     simd::Vec<int64_t, 2> a(-3);
    //     auto c = simd::abs(a);
    //     std::cout << "abs(" << a << "(a)" <<") = " << c << "(c)\n";
    // }
    {
        simd::Vec<float, 4> a(-4.f);
        auto c = simd::abs(a);
        std::cout << "abs(" << a << "(a)" <<") = " << c << "(c)\n";
    }
    {
        simd::Vec<double, 2> a(-4.0);
        auto c = simd::abs(a);
        std::cout << "abs(" << a << "(a)" <<") = " << c << "(c)\n";
    }
}
