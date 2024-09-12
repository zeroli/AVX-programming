#include <gtest/gtest.h>

#include "simd/simd.h"

#include <sstream>

TEST(vec_op_sse, test_arith_add)
{
    {
        simd::Vec<int32_t, 4> a(1), b(2), p(3);
        auto c = a + b;
        std::cout << a << "(a)" << " + " << b << "(b) = " << c << "(c)\n";
        //EXPECT_EQ(p, c);
        auto d = simd::add(a, b);
    }
    {
        simd::Vec<int32_t, 4> a(1);
        int b = 2;
        auto c = a + b;
        std::cout << a << "(a)" << " + " << b << "(b) = " << c << "(c)\n";
        auto d = simd::add(a, b);
    }
    {
        simd::Vec<int32_t, 4> a(1);
        int b = 2;
        auto c = b + a;
        std::cout << b << "(b)" << " + " << a << "() = " << c << "(c)\n";
        auto d = simd::add(b, a);
    }
    {
        simd::Vec<float, 4> a(1.f), b(2.f);
        auto c = a + b;
        auto d = a + 2.f;
        std::cout << a << "(a)" << " + " << b << "(b) = " << c << "(c)\n";
        auto e = simd::add(a, b);
        auto f = simd::add(a, 2.f);
        auto g = simd::add(1.f, b);
    }
    {
        simd::Vec<double, 2> a(1.0), b(2.0);
        auto c = a + b;
        auto d = a + 2.0;
        std::cout << a << "(a)" << " + " << b << "(b) = " << c << "(c)\n";
        auto e = simd::add(a, b);
        auto f = simd::add(a, 2.0);
        auto g = simd::add(1.0, b);
    }
    {
        simd::Vec<std::complex<float>, 2> a({1.f, 2.f}), b({1.f, 2.f});
        auto c = a + b;
        auto d = simd::add(a, b);
    }
    {
        simd::Vec<std::complex<double>, 1> a({1.0, 2.0}), b({1.0, 2.0});
        auto c = a + b;
        auto d = simd::add(a, b);
    }
}

TEST(vec_op_sse, test_arith_add_inplace)
{
    {
        simd::Vec<int32_t, 4> a(1), b(2), p(3);
        a += b;
        std::cout << a << "(a)\n";
        //EXPECT_EQ(p, c);
    }
    {
        simd::Vec<float, 4> a(1.f), b(2.f);
        a += b;
        std::cout << a << "(a)\n";
    }
    {
        simd::Vec<double, 2> a(1.0), b(2.0);
        a += b;
        std::cout << a << "(a)\n";
    }
}

TEST(vec_op_sse, test_arith_sub)
{
    {
        simd::Vec<int32_t, 4> a(1), b(2);
        auto c = a - b;
        std::cout << a << "(a)" << " - " << b << "(b) = " << c << "(c)\n";
        //EXPECT_EQ(p, c);
        auto d = simd::sub(a, b);
    }
    {
        simd::Vec<int32_t, 4> a(1);
        int b = 2;
        auto c = a - b;
        std::cout << a << "(a)" << " - " << b << "(b) = " << c << "(c)\n";
        auto d = simd::sub(a, b);
    }
    {
        simd::Vec<int32_t, 4> a(1);
        int b = 2;
        auto c = b - a;
        std::cout << b << "(b)" << " - " << a << "(a) = " << c << "(c)\n";
        auto d = simd::sub(b, a);
    }
    {
        simd::Vec<float, 4> a(2.f), b(1.f);
        auto c = a - b;
        std::cout << a << "(a)" << " - " << b << "(b) = " << c << "(c)\n";
        auto d = simd::sub(a, b);
        auto e = simd::sub(a, 1.f);
        auto f = simd::sub(2.f, b);
    }
    {
        simd::Vec<double, 2> a(2.0), b(1.0);
        auto c = a - b;
        std::cout << a << "(a)" << " - " << b << "(b) = " << c << "(c)\n";
        auto d = simd::sub(a, b);
        auto e = simd::sub(a, 1.0);
        auto f = simd::sub(2.0, b);
    }
    {
        simd::Vec<std::complex<float>, 2> a({1.f, 2.f}), b({1.f, 2.f});
        auto c = a - b;
        auto d = simd::sub(a, b);
    }
    {
        simd::Vec<std::complex<double>, 1> a({1.0, 2.0}), b({1.0, 2.0});
        auto c = a - b;
        auto d = simd::sub(a, b);
    }
}

TEST(vec_op_sse, test_arith_sub_inplace)
{
    {
        simd::Vec<int32_t, 4> a(1), b(2), p(3);
        a -= b;
        std::cout << a << "(a)\n";
        //EXPECT_EQ(p, c);
    }
    {
        simd::Vec<float, 4> a(1.f), b(2.f);
        a -= b;
        std::cout << a << "(a)\n";
    }
    {
        simd::Vec<double, 2> a(1.0), b(2.0);
        a -= b;
        std::cout << a << "(a)\n";
    }
}

TEST(vec_op_sse, test_arith_mul)
{
    {
        simd::Vec<float, 4> a(2.f), b(1.f);
        auto c = a * b;
        std::cout << a << "(a)" << " * " << b << "(b) = " << c << "(c)\n";
        auto d = simd::mul(a, b);
        auto e = simd::mul(a, 1.f);
        auto f = simd::mul(2.f, b);
    }
    {
        simd::Vec<double, 2> a(2.0), b(1.0);
        auto c = a * b;
        std::cout << a << "(a)" << " * " << b << "(b) = " << c << "(c)\n";
        auto d = simd::mul(a, b);
        auto e = simd::mul(a, 1.0);
        auto f = simd::mul(2.0, b);
    }
}

TEST(vec_op_sse, test_mul_inplace)
{
    {
        simd::Vec<float, 4> a(1.f), b(2.f);
        a *= b;
        std::cout << a << "(a)\n";
    }
    {
        simd::Vec<double, 2> a(1.0), b(2.0);
        a *= b;
        std::cout << a << "(a)\n";
    }
}

TEST(vec_op_sse, test_arith_div)
{
    {
        simd::Vec<float, 4> a(4.f), b(2.f);
        auto c = a / b;
        std::cout << a << "(a)" << " / " << b << "(b) = " << c << "(c)\n";
        auto d = simd::div(a, b);
        auto e = simd::div(a, 2.f);
        auto f = simd::div(4.f, b);
    }
    {
        simd::Vec<double, 2> a(8.0), b(2.0);
        auto c = a / b;
        std::cout << a << "(a)" << " / " << b << "(b) = " << c << "(c)\n";
        auto d = simd::div(a, b);
        auto e = simd::div(a, 2.0);
        auto f = simd::div(8.0, b);
    }
}

TEST(vec_op_sse, test_div_inplace)
{
    {
        simd::Vec<float, 4> a(1.f), b(2.f);
        a /= b;
        std::cout << a << "(a)\n";
    }
    {
        simd::Vec<double, 2> a(1.0), b(2.0);
        a /= b;
        std::cout << a << "(a)\n";
    }
}

TEST(vec_op_sse, test_min)
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

TEST(vec_op_sse, test_max)
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

TEST(vec_op_sse, test_math_abs)
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
        simd::Vec<float, 4> a(4.f);
        auto c = simd::sqrt(a);
        std::cout << "sqrt(" << a << "(a)" <<") = " << c << "(c)\n";
    }
    {
        simd::Vec<double, 2> a(4.0);
        auto c = simd::sqrt(a);
        std::cout << "sqrt(" << a << "(a)" <<") = " << c << "(c)\n";
    }
}

TEST(vec_op_sse, test_bitwise_and)
{
    {
        simd::Vec<int32_t, 4> a(1), b(3);
        auto c = a & b;
        std::cout << a << "(a)" <<" & " << b << "(b) = " << c << "(c)\n";
        auto d = simd::bitwise_and(a, b);
        a &= b;
    }
    {
        simd::Vec<float, 4> a(1), b(3);
        auto c = a & b;
        std::cout << a << "(a)" <<" & " << b << "(b) = " << c << "(c)\n";
        auto d = simd::bitwise_and(a, b);
        a &= b;
    }
    {
        simd::Vec<double, 2> a(1), b(3);
        auto c = a & b;
        std::cout << a << "(a)" <<" & " << b << "(b) = " << c << "(c)\n";
        auto d = simd::bitwise_and(a, b);
        a &= b;
    }
}

TEST(vec_op_sse, test_bitwise_or)
{
    {
        simd::Vec<int32_t, 4> a(1), b(3);
        auto c = a | b;
        std::cout << a << "(a)" <<" & " << b << "(b) = " << c << "(c)\n";
        auto d = simd::bitwise_or(a, b);
        a |= b;
    }
    {
        simd::Vec<float, 4> a(1), b(3);
        auto c = a | b;
        std::cout << a << "(a)" <<" & " << b << "(b) = " << c << "(c)\n";
        auto d = simd::bitwise_or(a, b);
        a |= b;
    }
    {
        simd::Vec<double, 2> a(1), b(3);
        auto c = a | b;
        std::cout << a << "(a)" <<" & " << b << "(b) = " << c << "(c)\n";
        auto d = simd::bitwise_or(a, b);
        a |= b;
    }
}

TEST(vec_op_sse, test_bitwise_xor)
{
    {
        simd::Vec<int32_t, 4> a(1), b(3);
        auto c = a ^ b;
        std::cout << a << "(a)" <<" & " << b << "(b) = " << c << "(c)\n";
        auto d = simd::bitwise_xor(a, b);
        a ^= b;
    }
    {
        simd::Vec<float, 4> a(1), b(3);
        auto c = a ^ b;
        std::cout << a << "(a)" <<" & " << b << "(b) = " << c << "(c)\n";
        auto d = simd::bitwise_xor(a, b);
        a ^= b;
    }
    {
        simd::Vec<double, 2> a(1), b(3);
        auto c = a ^ b;
        std::cout << a << "(a)" <<" & " << b << "(b) = " << c << "(c)\n";
        auto d = simd::bitwise_xor(a, b);
        a ^= b;
    }
}
