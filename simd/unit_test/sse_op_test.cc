#include <gtest/gtest.h>

#include "simd/simd.h"

#include <sstream>

TEST(vec_op_sse, test_arith_add)
{
    {
        simd::Vec<int32_t, 4> a(1), b(2), p(3);
        auto c = a + b;
        std::cout << a << "(a)" << " + " << b << "(b) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
        auto d = simd::add(a, b);
        EXPECT_TRUE(simd::all(p == d));
    }
    {
        simd::Vec<int32_t, 4> a(1), p(3);
        int b = 2;
        auto c = a + b;
        std::cout << a << "(a)" << " + " << b << "(b) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
        auto d = simd::add(a, b);
        EXPECT_TRUE(simd::all(p == d));
    }
    {
        simd::Vec<int32_t, 4> a(1), p(3);
        int b = 2;
        auto c = b + a;
        std::cout << b << "(b)" << " + " << a << "() = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
        auto d = simd::add(b, a);
        EXPECT_TRUE(simd::all(p == d));
    }
    {
        simd::Vec<float, 4> a(1.f), b(2.f), p(3.f);
        auto c = a + b;
        EXPECT_TRUE(simd::all(p == c));
        auto d = a + 2.f;
        EXPECT_TRUE(simd::all(p == d));
        std::cout << a << "(a)" << " + " << b << "(b) = " << c << "(c)\n";
        auto e = simd::add(a, b);
        EXPECT_TRUE(simd::all(p == e));
        auto f = simd::add(a, 2.f);
        EXPECT_TRUE(simd::all(p == f));
        auto g = simd::add(1.f, b);
        EXPECT_TRUE(simd::all(p == g));
    }
    {
        simd::Vec<double, 2> a(1.0), b(2.0), p(3.);
        auto c = a + b;
        EXPECT_TRUE(simd::all(p == c));
        auto d = a + 2.0;
        EXPECT_TRUE(simd::all(p == d));
        std::cout << a << "(a)" << " + " << b << "(b) = " << c << "(c)\n";
        auto e = simd::add(a, b);
        EXPECT_TRUE(simd::all(p == e));
        auto f = simd::add(a, 2.0);
        EXPECT_TRUE(simd::all(p == f));
        auto g = simd::add(1.0, b);
        EXPECT_TRUE(simd::all(p == g));
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
        EXPECT_TRUE(simd::all(p == a));
    }
    {
        simd::Vec<float, 4> a(1.f), b(2.f), p(3.f);
        a += b;
        std::cout << a << "(a)\n";
        EXPECT_TRUE(simd::all(p == a));
    }
    {
        simd::Vec<double, 2> a(1.0), b(2.0), p(3.0);
        a += b;
        std::cout << a << "(a)\n";
        EXPECT_TRUE(simd::all(p == a)) << a << ", " << p;
    }
}

TEST(vec_op_sse, test_arith_sub)
{
    {
        simd::Vec<int32_t, 4> a(1), b(2), p(-1);
        auto c = a - b;
        std::cout << a << "(a)" << " - " << b << "(b) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
        auto d = simd::sub(a, b);
        EXPECT_TRUE(simd::all(p == d));
    }
    {
        simd::Vec<int32_t, 4> a(1), p(-1);
        int b = 2;
        auto c = a - b;
        std::cout << a << "(a)" << " - " << b << "(b) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
        auto d = simd::sub(a, b);
        EXPECT_TRUE(simd::all(p == d));
    }
    {
        simd::Vec<int32_t, 4> a(1), p(1);
        int b = 2;
        auto c = b - a;
        std::cout << b << "(b)" << " - " << a << "(a) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
        auto d = simd::sub(b, a);
        EXPECT_TRUE(simd::all(p == d));
    }
    {
        simd::Vec<float, 4> a(2.f), b(1.f), p(1.f);
        auto c = a - b;
        std::cout << a << "(a)" << " - " << b << "(b) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
        auto d = simd::sub(a, b);
        EXPECT_TRUE(simd::all(p == d));
        auto e = simd::sub(a, 1.f);
        EXPECT_TRUE(simd::all(p == e));
        auto f = simd::sub(2.f, b);
        EXPECT_TRUE(simd::all(p == f));
    }
    {
        simd::Vec<double, 2> a(2.0), b(1.0), p(1.0);
        auto c = a - b;
        std::cout << a << "(a)" << " - " << b << "(b) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
        auto d = simd::sub(a, b);
        EXPECT_TRUE(simd::all(p == d));
        auto e = simd::sub(a, 1.0);
        EXPECT_TRUE(simd::all(p == e));
        auto f = simd::sub(2.0, b);
        EXPECT_TRUE(simd::all(p == f));
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
        simd::Vec<int32_t, 4> a(1), b(2), p(-1);
        a -= b;
        std::cout << a << "(a)\n";
        EXPECT_TRUE(simd::all(p == a));
    }
    {
        simd::Vec<float, 4> a(1.f), b(2.f), p(-1);
        a -= b;
        std::cout << a << "(a)\n";
        EXPECT_TRUE(simd::all(p == a));
    }
    {
        simd::Vec<double, 2> a(1.0), b(2.0), p(-1);
        a -= b;
        std::cout << a << "(a)\n";
        EXPECT_TRUE(simd::all(p == a));
    }
}

TEST(vec_op_sse, test_arith_mul)
{
    {
        simd::Vec<float, 4> a(2.f), b(1.f), p(2.f);
        auto c = a * b;
        std::cout << a << "(a)" << " * " << b << "(b) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
        auto d = simd::mul(a, b);
        EXPECT_TRUE(simd::all(p == d));
        auto e = simd::mul(a, 1.f);
        EXPECT_TRUE(simd::all(p == e));
        auto f = simd::mul(2.f, b);
        EXPECT_TRUE(simd::all(p == f));
    }
    {
        simd::Vec<double, 2> a(2.0), b(1.0), p(2.0);
        auto c = a * b;
        std::cout << a << "(a)" << " * " << b << "(b) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
        auto d = simd::mul(a, b);
        EXPECT_TRUE(simd::all(p == d));
        auto e = simd::mul(a, 1.0);
        EXPECT_TRUE(simd::all(p == e));
        auto f = simd::mul(2.0, b);
        EXPECT_TRUE(simd::all(p == f));
    }
}

TEST(vec_op_sse, test_mul_inplace)
{
    {
        simd::Vec<float, 4> a(1.f), b(2.f), p(2.f);
        a *= b;
        std::cout << a << "(a)\n";
        EXPECT_TRUE(simd::all(p == a));
    }
    {
        simd::Vec<double, 2> a(1.0), b(2.0), p(2.0);
        a *= b;
        std::cout << a << "(a)\n";
        EXPECT_TRUE(simd::all(p == a));
    }
}

TEST(vec_op_sse, test_arith_div)
{
    {
        simd::Vec<float, 4> a(4.f), b(2.f), p(2.f);
        auto c = a / b;
        std::cout << a << "(a)" << " / " << b << "(b) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
        auto d = simd::div(a, b);
        EXPECT_TRUE(simd::all(p == d));
        auto e = simd::div(a, 2.f);
        EXPECT_TRUE(simd::all(p == e));
        auto f = simd::div(4.f, b);
        EXPECT_TRUE(simd::all(p == f));
    }
    {
        simd::Vec<double, 2> a(8.0), b(2.0), p(4.0);
        auto c = a / b;
        std::cout << a << "(a)" << " / " << b << "(b) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
        auto d = simd::div(a, b);
        EXPECT_TRUE(simd::all(p == d));
        auto e = simd::div(a, 2.0);
        EXPECT_TRUE(simd::all(p == e));
        auto f = simd::div(8.0, b);
        EXPECT_TRUE(simd::all(p == f));
    }
}

TEST(vec_op_sse, test_div_inplace)
{
    {
        simd::Vec<float, 4> a(1.f), b(2.f), p(0.5f);
        a /= b;
        std::cout << a << "(a)\n";
        EXPECT_TRUE(simd::all(p == a));
    }
    {
        simd::Vec<double, 2> a(1.0), b(2.0), p(0.5);
        a /= b;
        std::cout << a << "(a)\n";
        EXPECT_TRUE(simd::all(p == a));
    }
}

TEST(vec_op_sse, test_min)
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

TEST(vec_op_sse, test_max)
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

#define OP_FLOATS(op, x, y) \
({ \
    float x_ = x, y_ = y; \
    int32_t z = (*reinterpret_cast<int32_t*>(&x_) op \
                        *reinterpret_cast<int32_t*>(&y_)); \
    *(reinterpret_cast<float*>(&z)); \
}) \
///

#define OP_DOUBLES(op, x, y) \
({ \
    double x_ = x, y_ = y; \
    int64_t z = (*reinterpret_cast<int64_t*>(&x_) op \
                        *reinterpret_cast<int64_t*>(&y_)); \
    *(reinterpret_cast<double*>(&z)); \
}) \
///

TEST(vec_op_sse, test_bitwise_and)
{
    {
        simd::Vec<int32_t, 4> a(1), b(3), p(1 & 3);
        auto c = a & b;
        std::cout << a << "(a)" <<" & " << b << "(b) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
        auto d = simd::bitwise_and(a, b);
        EXPECT_TRUE(simd::all(p == d));
        a &= b;
        EXPECT_TRUE(simd::all(p == a));
    }
    {
        simd::Vec<float, 4> a(1), b(3), p(OP_FLOATS(&, 1, 3));
        auto c = a & b;
        std::cout << a << "(a)" <<" & " << b << "(b) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
        auto d = simd::bitwise_and(a, b);
        EXPECT_TRUE(simd::all(p == d));
        a &= b;
        EXPECT_TRUE(simd::all(p == a));
    }
    {
        simd::Vec<double, 2> a(1), b(3), p(OP_DOUBLES(&, 1, 3));
        auto c = a & b;
        std::cout << a << "(a)" <<" & " << b << "(b) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
        auto d = simd::bitwise_and(a, b);
        EXPECT_TRUE(simd::all(p == d));
        a &= b;
        EXPECT_TRUE(simd::all(p == a));
    }
}

TEST(vec_op_sse, test_bitwise_or)
{
    {
        simd::Vec<int32_t, 4> a(1), b(3), p(1 | 3);
        auto c = a | b;
        std::cout << a << "(a)" <<" | " << b << "(b) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
        auto d = simd::bitwise_or(a, b);
        EXPECT_TRUE(simd::all(p == d));
        a |= b;
        EXPECT_TRUE(simd::all(p == a));
    }
    {
        simd::Vec<float, 4> a(1), b(2), p(OP_FLOATS(|, 1, 2));
        auto c = a | b;
        std::cout << a << "(a)" <<" | " << b << "(b) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
        auto d = simd::bitwise_or(a, b);
        EXPECT_TRUE(simd::all(p == d));
        a |= b;
        EXPECT_TRUE(simd::all(p == a));
    }
    {
        simd::Vec<double, 2> a(1), b(2), p(OP_DOUBLES(|, 1, 2));
        auto c = a | b;
        std::cout << a << "(a)" <<" | " << b << "(b) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
        auto d = simd::bitwise_or(a, b);
        EXPECT_TRUE(simd::all(p == d));
        a |= b;
        EXPECT_TRUE(simd::all(p == a));
    }
}

TEST(vec_op_sse, test_bitwise_xor)
{
    {
        simd::Vec<int32_t, 4> a(1), b(3), p(1 ^ 3);
        auto c = a ^ b;
        std::cout << a << "(a)" <<" ^ " << b << "(b) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
        auto d = simd::bitwise_xor(a, b);
        EXPECT_TRUE(simd::all(p == d));
        a ^= b;
        EXPECT_TRUE(simd::all(p == a));
    }
    {
        simd::Vec<float, 4> a(1), b(2), p(OP_FLOATS(^, 1, 2));
        auto c = a ^ b;
        std::cout << a << "(a)" <<" ^ " << b << "(b) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
        auto d = simd::bitwise_xor(a, b);
        EXPECT_TRUE(simd::all(p == d));
        a ^= b;
        EXPECT_TRUE(simd::all(p == a));
    }
    {
        simd::Vec<double, 2> a(1), b(1), p(OP_DOUBLES(^, 1, 1));
        auto c = a ^ b;
        std::cout << a << "(a)" <<" ^ " << b << "(b) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
        auto d = simd::bitwise_xor(a, b);
        EXPECT_TRUE(simd::all(p == d));
        a ^= b;
        EXPECT_TRUE(simd::all(p == a));
    }
}

TEST(vec_op_sse, test_bitwise_not)
{
    {
        simd::Vec<int32_t, 4> a(1), p(~1);
        auto c = ~a;
        std::cout << "~a: " << a << "(a) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
        auto d = simd::bitwise_not(a);
        EXPECT_TRUE(simd::all(p == d));
    }
    auto not_float = [](float x) {
        int32_t z = ~(*reinterpret_cast<int32_t*>(&x));
        return *(reinterpret_cast<float*>(&z));
    };
    {
        simd::Vec<float, 4> a(1.f), p(not_float(1.f));
        auto c = ~a;
        std::cout << "~a: " << a << "(a) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
        auto d = simd::bitwise_not(a);
        EXPECT_TRUE(simd::all(p == d));
    }
    auto not_double = [](double x) {
        int64_t z = ~(*reinterpret_cast<int64_t*>(&x));
        return *(reinterpret_cast<double*>(&z));
    };
    {
        simd::Vec<double, 2> a(1.0), p(not_double(1.0));
        auto c = ~a;
        std::cout << "~a: " << a << "(a) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
        auto d = simd::bitwise_not(a);
        EXPECT_TRUE(simd::all(p == d));
    }
}

TEST(vec_op_sse, test_inc_by1)
{
    {
        simd::Vec<int32_t, 4> a(1), p(2);
        auto c = ++a;
        EXPECT_TRUE(simd::all(p == c));
        EXPECT_TRUE(simd::all(p == a));
        std::cout << "++a: " << a << "(a) = " << c << "(c)\n";
        auto d = a++;
        EXPECT_TRUE(simd::all(p == d));
        p++;
        EXPECT_TRUE(simd::all(p == a));
    }
    {
        simd::Vec<float, 4> a(1.f), p(2.f);
        auto c = ++a;
        std::cout << "++a: " << a << "(a) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
        EXPECT_TRUE(simd::all(p == a));
        auto d = a++;
        EXPECT_TRUE(simd::all(p == d));
        p++;
        EXPECT_TRUE(simd::all(p == a));
    }
    {
        simd::Vec<double, 2> a(1.0), p(2.0);
        auto c = ++a;
        std::cout << "++a: " << a << "(a) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
        EXPECT_TRUE(simd::all(p == a));
        auto d = a++;
        EXPECT_TRUE(simd::all(p == d));
        p++;
        EXPECT_TRUE(simd::all(p == a));
    }
}

TEST(vec_op_sse, test_dec_by1)
{
    {
        simd::Vec<int32_t, 4> a(1), p(0);
        auto c = --a;
        std::cout << "--a: " << a << "(a) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
        EXPECT_TRUE(simd::all(p == a));
        auto d = a--;
        EXPECT_TRUE(simd::all(p == d));
        p--;
        EXPECT_TRUE(simd::all(p == a));
    }
    {
        simd::Vec<float, 4> a(1.f), p(0.f);
        auto c = --a;
        std::cout << "--a: " << a << "(a) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
        EXPECT_TRUE(simd::all(p == a));
        auto d = a--;
        EXPECT_TRUE(simd::all(p == d));
        p--;
        EXPECT_TRUE(simd::all(p == a));
    }
    {
        simd::Vec<double, 2> a(1.0), p(0.0);
        auto c = --a;
        std::cout << "--a: " << a << "(a) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
        EXPECT_TRUE(simd::all(p == a));
        auto d = a--;
        EXPECT_TRUE(simd::all(p == d));
        p--;
        EXPECT_TRUE(simd::all(p == a));
    }
}

TEST(vec_op_sse, test_neg)
{
    {
        simd::Vec<int32_t, 4> a(1), p(-1);
        auto c = -a;
        std::cout << "-a: " << a << "(a) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
    }
    {
        simd::Vec<float, 4> a(1.f), p(-1.f);
        auto c = -a;
        std::cout << "-a: " << a << "(a) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
    }
    {
        simd::Vec<double, 2> a(1.0), p(-1.0);
        auto c = -a;
        std::cout << "-a: " << a << "(a) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
    }
}

TEST(vec_op_sse, test_any_all)
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

TEST(vec_op_sse, test_cmp_lt)
{
    {
        simd::Vec<int32_t, 4> a(1), b(2);
        EXPECT_TRUE(simd::all(a < b));
    }
    {
        simd::Vec<float, 4> a(1.f), b(2.f);
        EXPECT_TRUE(simd::all(a < b));
    }
    {
        simd::Vec<double, 2> a(1.0), b(2.0);
        EXPECT_TRUE(simd::all(a < b));
    }
}

TEST(vec_op_sse, test_cmp_le)
{
    {
        simd::Vec<int32_t, 4> a(1), b(1);
        EXPECT_TRUE(simd::all(a <= b));
    }
    {
        simd::Vec<float, 4> a(1.f), b(2.f);
        EXPECT_TRUE(simd::all(a <= b));
    }
    {
        simd::Vec<double, 2> a(1.0), b(2.0);
        EXPECT_TRUE(simd::all(a <= b));
    }
}

TEST(vec_op_sse, test_cmp_gt)
{
    {
        simd::Vec<int32_t, 4> a(2), b(1);
        EXPECT_TRUE(simd::all(a > b));
    }
    {
        simd::Vec<float, 4> a(1.f), b(2.f);
        EXPECT_TRUE(simd::all(b > a));
    }
    {
        simd::Vec<double, 2> a(1.0), b(2.0);
        EXPECT_TRUE(simd::all(b > a));
    }
}

TEST(vec_op_sse, test_cmp_ge)
{
    {
        simd::Vec<int32_t, 4> a(2), b(1);
        EXPECT_TRUE(simd::all(a >= b));
    }
    {
        simd::Vec<float, 4> a(1.f), b(2.f);
        EXPECT_TRUE(simd::all(b >= a));
    }
    {
        simd::Vec<double, 2> a(1.0), b(2.0);
        EXPECT_TRUE(simd::all(b >= a));
    }
    {
        simd::Vec<double, 2> a(2.0), b(2.0);
        EXPECT_TRUE(simd::all(b >= a));
    }
}

TEST(vec_op_sse, test_cmp_ne)
{
    {
        simd::Vec<int32_t, 4> a(2), b(1);
        EXPECT_TRUE(simd::all(a != b));
    }
    {
        simd::Vec<float, 4> a(1.f), b(2.f);
        EXPECT_TRUE(simd::all(b != a));
    }
    {
        simd::Vec<double, 2> a(1.0), b(2.0);
        EXPECT_TRUE(simd::all(b != a));
    }
}
