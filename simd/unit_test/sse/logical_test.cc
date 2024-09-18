#include <gtest/gtest.h>

#include "simd/simd.h"

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

TEST(vec_op_sse, test_logical_bitwise_and)
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

TEST(vec_op_sse, test_logical_bitwise_or)
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

TEST(vec_op_sse, test_logical_bitwise_xor)
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

TEST(vec_op_sse, test_logical_bitwise_not)
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

TEST(vec_op_sse, test_logical_bitwise_lshift)
{
    {
        simd::Vec<int8_t, 16> a(1), p(2);
        auto b = a << 1;
        EXPECT_TRUE(simd::all(p == b));
    }
    {
        simd::Vec<int16_t, 8> a(1), p(2);
        auto b = a << 1;
        EXPECT_TRUE(simd::all(p == b));
    }
    {
        simd::Vec<int32_t, 4> a(1), p(2);
        auto b = a << 1;
        EXPECT_TRUE(simd::all(p == b));
    }
    {
        simd::Vec<int64_t, 2> a(1), p(2);
        auto b = a << 1;
        EXPECT_TRUE(simd::all(p == b));
    }
}

TEST(vec_op_sse, test_logical_bitwise_rshift)
{
    {
        simd::Vec<uint8_t, 16> a(2), p(+2 >> 1);
        auto b = a >> 1;
        EXPECT_TRUE(simd::all(p == b));
    }
    {
        simd::Vec<int8_t, 16> a(-2), p(-2 >> 1);
        auto b = a >> 1;
        EXPECT_TRUE(simd::all(p == b));
    }
    {
        simd::Vec<uint16_t, 8> a(2), p(+2 >> 1);
        auto b = a >> 1;
        EXPECT_TRUE(simd::all(p == b)) << p << "\n" << b;
    }
    {
        simd::Vec<int16_t, 8> a(-2), p(-2 >> 1);
        auto b = a >> 1;
        EXPECT_TRUE(simd::all(p == b)) << p << "\n" << b;
    }
    {
        simd::Vec<uint32_t, 4> a(2), p(+2 >> 1);
        auto b = a >> 1;
        EXPECT_TRUE(simd::all(p == b));
    }
    {
        simd::Vec<int32_t, 4> a(-2), p(-2 >> 1);
        auto b = a >> 1;
        EXPECT_TRUE(simd::all(p == b));
    }
    {
        simd::Vec<uint64_t, 2> a(2), p(+2 >> 1);
        auto b = a >> 1;
        EXPECT_TRUE(simd::all(p == b));
    }
    {
        simd::Vec<int64_t, 2> a(-2), p(-2 >> 1);
        auto b = a >> 1;
        EXPECT_TRUE(simd::all(p == b)) << p << "\n" << b;
    }
}
