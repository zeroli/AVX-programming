#include <gtest/gtest.h>

#include "simd/simd.h"

STATIC_CHECK_ARCH_ENABLED(AVX);

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

TEST(vec_op_avx, test_logical_bitwise_and)
{
    {
        simd::vi32x8_t a(1), b(3), p(1 & 3);
        auto c = a & b;
        EXPECT_TRUE(simd::all_of(p == c));
        auto d = simd::bitwise_and(a, b);
        EXPECT_TRUE(simd::all_of(p == d));
        a &= b;
        EXPECT_TRUE(simd::all_of(p == a));
    }
    {
        simd::vf32x8_t a(1), b(3), p(OP_FLOATS(&, 1, 3));
        auto c = a & b;
        EXPECT_TRUE(simd::all_of(p == c));
        auto d = simd::bitwise_and(a, b);
        EXPECT_TRUE(simd::all_of(p == d));
        a &= b;
        EXPECT_TRUE(simd::all_of(p == a));
    }
    {
        simd::vf64x4_t a(1), b(3), p(OP_DOUBLES(&, 1, 3));
        auto c = a & b;
        EXPECT_TRUE(simd::all_of(p == c));
        auto d = simd::bitwise_and(a, b);
        EXPECT_TRUE(simd::all_of(p == d));
        a &= b;
        EXPECT_TRUE(simd::all_of(p == a));
    }
}

TEST(vec_op_avx, test_logical_bitwise_or)
{
    {
        simd::vi32x8_t a(1), b(3), p(1 | 3);
        auto c = a | b;
        EXPECT_TRUE(simd::all_of(p == c));
        auto d = simd::bitwise_or(a, b);
        EXPECT_TRUE(simd::all_of(p == d));
        a |= b;
        EXPECT_TRUE(simd::all_of(p == a));
    }
    {
        simd::vf32x8_t a(1), b(2), p(OP_FLOATS(|, 1, 2));
        auto c = a | b;
        EXPECT_TRUE(simd::all_of(p == c));
        auto d = simd::bitwise_or(a, b);
        EXPECT_TRUE(simd::all_of(p == d));
        a |= b;
        EXPECT_TRUE(simd::all_of(p == a));
    }
    {
        simd::vf64x4_t a(1), b(2), p(OP_DOUBLES(|, 1, 2));
        auto c = a | b;
        EXPECT_TRUE(simd::all_of(p == c));
        auto d = simd::bitwise_or(a, b);
        EXPECT_TRUE(simd::all_of(p == d));
        a |= b;
        EXPECT_TRUE(simd::all_of(p == a));
    }
}

TEST(vec_op_avx, test_logical_bitwise_xor)
{
    {
        simd::vi32x8_t a(1), b(3), p(1 ^ 3);
        auto c = a ^ b;
        EXPECT_TRUE(simd::all_of(p == c));
        auto d = simd::bitwise_xor(a, b);
        EXPECT_TRUE(simd::all_of(p == d));
        a ^= b;
        EXPECT_TRUE(simd::all_of(p == a));
    }
    {
        simd::vf32x8_t a(1), b(2), p(OP_FLOATS(^, 1, 2));
        auto c = a ^ b;
        EXPECT_TRUE(simd::all_of(p == c));
        auto d = simd::bitwise_xor(a, b);
        EXPECT_TRUE(simd::all_of(p == d));
        a ^= b;
        EXPECT_TRUE(simd::all_of(p == a));
    }
    {
        simd::vf64x4_t a(1), b(1), p(OP_DOUBLES(^, 1, 1));
        auto c = a ^ b;
        EXPECT_TRUE(simd::all_of(p == c));
        auto d = simd::bitwise_xor(a, b);
        EXPECT_TRUE(simd::all_of(p == d));
        a ^= b;
        EXPECT_TRUE(simd::all_of(p == a));
    }
}

TEST(vec_op_avx, test_logical_bitwise_not)
{
    {
        simd::vi32x8_t a(1), p(~1);
        auto c = ~a;
        EXPECT_TRUE(simd::all_of(p == c));
        auto d = simd::bitwise_not(a);
        EXPECT_TRUE(simd::all_of(p == d));
    }
    auto not_float = [](float x) {
        return bits::bitwise_not(x);
    };
    {
        simd::vf32x8_t a(1.f), p(not_float(1.f));
        auto c = ~a;
        EXPECT_TRUE(simd::all_of(p == c));
        auto d = simd::bitwise_not(a);
        EXPECT_TRUE(simd::all_of(p == d));
    }
    auto not_double = [](double x) {
        return bits::bitwise_not(x);
    };
    {
        simd::vf64x4_t a(1.0), p(not_double(1.0));
        auto c = ~a;
        EXPECT_TRUE(simd::all_of(p == c));
        auto d = simd::bitwise_not(a);
        EXPECT_TRUE(simd::all_of(p == d));
    }
}

TEST(vec_op_avx, test_logical_bitwise_lshift)
{
    {
        simd::vi8x32_t a(1), p(2);
        auto b = a << 1;
        EXPECT_TRUE(simd::all_of(p == b));
    }
    {
        simd::vi16x16_t a(1), p(2);
        auto b = a << 1;
        EXPECT_TRUE(simd::all_of(p == b));
    }
    {
        simd::vi32x8_t a(1), p(2);
        auto b = a << 1;
        EXPECT_TRUE(simd::all_of(p == b));
    }
    {
        simd::vi64x4_t a(1), p(2);
        auto b = a << 1;
        EXPECT_TRUE(simd::all_of(p == b));
    }
}

TEST(vec_op_avx, test_logical_bitwise_rshift)
{
    {
        simd::vu8x32_t a(2), p(+2 >> 1);
        auto b = a >> 1;
        EXPECT_TRUE(simd::all_of(p == b));
    }
    {
        simd::vi8x32_t a(-2), p(-2 >> 1);
        auto b = a >> 1;
        EXPECT_TRUE(simd::all_of(p == b));
    }
    {
        simd::vu16x16_t a(2), p(+2 >> 1);
        auto b = a >> 1;
        EXPECT_TRUE(simd::all_of(p == b)) << p << "\n" << b;
    }
    {
        simd::vi16x16_t a(-2), p(-2 >> 1);
        auto b = a >> 1;
        EXPECT_TRUE(simd::all_of(p == b)) << p << "\n" << b;
    }
    {
        simd::vu32x8_t a(2), p(+2 >> 1);
        auto b = a >> 1;
        EXPECT_TRUE(simd::all_of(p == b));
    }
    {
        simd::vi32x8_t a(-2), p(-2 >> 1);
        auto b = a >> 1;
        EXPECT_TRUE(simd::all_of(p == b));
    }
    {
        simd::vu64x4_t a(2), p(+2 >> 1);
        auto b = a >> 1;
        EXPECT_TRUE(simd::all_of(p == b));
    }
    {
        simd::vi64x4_t a(-2), p(-2 >> 1);
        auto b = a >> 1;
        EXPECT_TRUE(simd::all_of(p == b)) << p << "\n" << b;
    }
}
