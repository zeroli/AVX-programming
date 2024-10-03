#include <gtest/gtest.h>

#include "simd/simd.h"
#include "check_arch.h"

using namespace simd;

#define OP_FLOATS(op, x, y) \
({ \
    float x_ = x, y_ = y; \
    bits::cast<float>(bits::cast<int>(x_) op bits::cast<int>(y_)); \
}) \
///

#define OP_DOUBLES(op, x, y) \
({ \
    double x_ = x, y_ = y; \
    bits::cast<double>(bits::cast<int64_t>(x_) op bits::cast<int64_t>(y_)); \
}) \
///

TEST(vec_op_sse, test_logical_bitwise_and)
{
    {
        simd::Vec<int32_t, 4> a(1), b(3), p(1 & 3);
        auto c = a & b;
        EXPECT_TRUE(simd::all_of(p == c));
        auto d = simd::bitwise_and(a, b);
        EXPECT_TRUE(simd::all_of(p == d));
        a &= b;
        EXPECT_TRUE(simd::all_of(p == a));
    }
    {
        simd::Vec<float, 4> a(1), b(3), p(OP_FLOATS(&, 1, 3));
        auto c = a & b;
        EXPECT_TRUE(simd::all_of(p == c));
        auto d = simd::bitwise_and(a, b);
        EXPECT_TRUE(simd::all_of(p == d));
        a &= b;
        EXPECT_TRUE(simd::all_of(p == a));
    }
    {
        simd::Vec<double, 2> a(1), b(3), p(OP_DOUBLES(&, 1, 3));
        auto c = a & b;
        EXPECT_TRUE(simd::all_of(p == c));
        auto d = simd::bitwise_and(a, b);
        EXPECT_TRUE(simd::all_of(p == d));
        a &= b;
        EXPECT_TRUE(simd::all_of(p == a));
    }
}

TEST(vec_op_sse, test_logical_bitwise_or)
{
    {
        simd::Vec<int32_t, 4> a(1), b(3), p(1 | 3);
        auto c = a | b;
        EXPECT_TRUE(simd::all_of(p == c));
        auto d = simd::bitwise_or(a, b);
        EXPECT_TRUE(simd::all_of(p == d));
        a |= b;
        EXPECT_TRUE(simd::all_of(p == a));
    }
    {
        simd::Vec<float, 4> a(1), b(2), p(OP_FLOATS(|, 1, 2));
        auto c = a | b;
        EXPECT_TRUE(simd::all_of(p == c));
        auto d = simd::bitwise_or(a, b);
        EXPECT_TRUE(simd::all_of(p == d));
        a |= b;
        EXPECT_TRUE(simd::all_of(p == a));
    }
    {
        simd::Vec<double, 2> a(1), b(2), p(OP_DOUBLES(|, 1, 2));
        auto c = a | b;
        EXPECT_TRUE(simd::all_of(p == c));
        auto d = simd::bitwise_or(a, b);
        EXPECT_TRUE(simd::all_of(p == d));
        a |= b;
        EXPECT_TRUE(simd::all_of(p == a));
    }
}

TEST(vec_op_sse, test_logical_bitwise_xor)
{
    {
        simd::Vec<int32_t, 4> a(1), b(3), p(1 ^ 3);
        auto c = a ^ b;
        EXPECT_TRUE(simd::all_of(p == c));
        auto d = simd::bitwise_xor(a, b);
        EXPECT_TRUE(simd::all_of(p == d));
        a ^= b;
        EXPECT_TRUE(simd::all_of(p == a));
    }
    {
        simd::Vec<float, 4> a(1), b(2), p(OP_FLOATS(^, 1, 2));
        auto c = a ^ b;
        EXPECT_TRUE(simd::all_of(p == c));
        auto d = simd::bitwise_xor(a, b);
        EXPECT_TRUE(simd::all_of(p == d));
        a ^= b;
        EXPECT_TRUE(simd::all_of(p == a));
    }
    {
        simd::Vec<double, 2> a(1), b(1), p(OP_DOUBLES(^, 1, 1));
        auto c = a ^ b;
        EXPECT_TRUE(simd::all_of(p == c));
        auto d = simd::bitwise_xor(a, b);
        EXPECT_TRUE(simd::all_of(p == d));
        a ^= b;
        EXPECT_TRUE(simd::all_of(p == a));
    }
}

TEST(vec_op_sse, test_logical_bitwise_not)
{
    {
        simd::Vec<int32_t, 4> a(1), p(~1);
        auto c = ~a;
        EXPECT_TRUE(simd::all_of(p == c));
        auto d = simd::bitwise_not(a);
        EXPECT_TRUE(simd::all_of(p == d));
    }
    auto not_float = [](float x) {
        return bits::bitwise_not(x);
    };
    {
        simd::Vec<float, 4> a(1.f), p(not_float(1.f));
        auto c = ~a;
        EXPECT_TRUE(simd::all_of(p == c));
        auto d = simd::bitwise_not(a);
        EXPECT_TRUE(simd::all_of(p == d));
    }
    auto not_double = [](double x) {
        return bits::bitwise_not(x);
    };
    {
        simd::Vec<double, 2> a(1.0), p(not_double(1.0));
        auto c = ~a;
        EXPECT_TRUE(simd::all_of(p == c));
        auto d = simd::bitwise_not(a);
        EXPECT_TRUE(simd::all_of(p == d));
    }
}

TEST(vec_op_sse, test_logical_bitwise_lshift)
{
    {
        simd::Vec<int8_t, 16> a(1), p(2);
        auto b = a << 1;
        EXPECT_TRUE(simd::all_of(p == b));
    }
    {
        simd::Vec<int16_t, 8> a(1), p(2);
        auto b = a << 1;
        EXPECT_TRUE(simd::all_of(p == b));
    }
    {
        simd::Vec<int32_t, 4> a(1), p(2);
        auto b = a << 1;
        EXPECT_TRUE(simd::all_of(p == b));
    }
    {
        simd::Vec<int64_t, 2> a(1), p(2);
        auto b = a << 1;
        EXPECT_TRUE(simd::all_of(p == b));
    }
}

TEST(vec_op_sse, test_logical_bitwise_rshift)
{
    {
        simd::Vec<uint8_t, 16> a(2), p(+2 >> 1);
        auto b = a >> 1;
        EXPECT_TRUE(simd::all_of(p == b));
    }
    {
        simd::Vec<int8_t, 16> a(-2), p(-2 >> 1);
        auto b = a >> 1;
        EXPECT_TRUE(simd::all_of(p == b));
    }
    {
        simd::Vec<uint16_t, 8> a(2), p(+2 >> 1);
        auto b = a >> 1;
        EXPECT_TRUE(simd::all_of(p == b)) << p << "\n" << b;
    }
    {
        simd::Vec<int16_t, 8> a(-2), p(-2 >> 1);
        auto b = a >> 1;
        EXPECT_TRUE(simd::all_of(p == b)) << p << "\n" << b;
    }
    {
        simd::Vec<uint32_t, 4> a(2), p(+2 >> 1);
        auto b = a >> 1;
        EXPECT_TRUE(simd::all_of(p == b));
    }
    {
        simd::Vec<int32_t, 4> a(-2), p(-2 >> 1);
        auto b = a >> 1;
        EXPECT_TRUE(simd::all_of(p == b));
    }
    {
        simd::Vec<uint64_t, 2> a(2), p(+2 >> 1);
        auto b = a >> 1;
        EXPECT_TRUE(simd::all_of(p == b));
    }
    {
        simd::Vec<int64_t, 2> a(-2), p(-2 >> 1);
        auto b = a >> 1;
        EXPECT_TRUE(simd::all_of(p == b)) << p << "\n" << b;
    }
}
