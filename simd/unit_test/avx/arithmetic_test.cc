#include <gtest/gtest.h>

#include "simd/simd.h"

using namespace simd;

STATIC_CHECK_ARCH_ENABLED(AVX);

TEST(vec_op_avx, test_arith_add)
{
    {
        simd::vi32x8_t a(1), b(2), p(3);
        auto c = a + b;
        EXPECT_TRUE(simd::all_of(p == c));
        auto d = a + 2;
        EXPECT_TRUE(simd::all_of(p == d));
        auto e = simd::add(a, b);
        EXPECT_TRUE(simd::all_of(p == e));
        auto f = simd::add(a, 2);
        EXPECT_TRUE(simd::all_of(p == f));
        auto g = simd::add(1, b);
        EXPECT_TRUE(simd::all_of(p == g));
    }
    {
        simd::vf32x8_t a(1.f), b(2.f), p(3.f);
        auto c = a + b;
        EXPECT_TRUE(simd::all_of(p == c));
        auto d = a + 2.f;
        EXPECT_TRUE(simd::all_of(p == d));
        auto e = simd::add(a, b);
        EXPECT_TRUE(simd::all_of(p == e));
        auto f = simd::add(a, 2.f);
        EXPECT_TRUE(simd::all_of(p == f));
        auto g = simd::add(1.f, b);
        EXPECT_TRUE(simd::all_of(p == g));
    }
    {
        simd::vf64x4_t a(1.0), b(2.0), p(3.0);
        auto c = a + b;
        EXPECT_TRUE(simd::all_of(p == c));
        auto d = a + 2.0;
        EXPECT_TRUE(simd::all_of(p == d));
        auto e = simd::add(a, b);
        EXPECT_TRUE(simd::all_of(p == e));
        auto f = simd::add(a, 2.0);
        EXPECT_TRUE(simd::all_of(p == f));
        auto g = simd::add(1.0, b);
        EXPECT_TRUE(simd::all_of(p == g));
    }
}

TEST(vec_op_sse, test_arith_add_inplace)
{
    {
        simd::vi32x8_t a(1), b(2), p(3);
        a += b;
        EXPECT_TRUE(simd::all_of(p == a));
    }
    {
        simd::vf32x8_t a(1.f), b(2.f), p(3.f);
        a += b;
        EXPECT_TRUE(simd::all_of(p == a));
    }
    {
        simd::vf64x4_t a(1.0), b(2.0), p(3.0);
        a += b;
        EXPECT_TRUE(simd::all_of(p == a)) << a << ", " << p;
    }
}

TEST(vec_op_sse, test_arith_sub)
{
    {
        simd::vi32x8_t a(1), b(2), p(-1);
        auto c = a - b;
        EXPECT_TRUE(simd::all_of(p == c));
        auto d = simd::sub(a, b);
        EXPECT_TRUE(simd::all_of(p == d));
    }
    {
        simd::vi32x8_t a(1), p(-1);
        int b = 2;
        auto c = a - b;
        EXPECT_TRUE(simd::all_of(p == c));
        auto d = simd::sub(a, b);
        EXPECT_TRUE(simd::all_of(p == d));
    }
    {
        simd::vi32x8_t a(1), p(1);
        int b = 2;
        auto c = b - a;
        EXPECT_TRUE(simd::all_of(p == c));
        auto d = simd::sub(b, a);
        EXPECT_TRUE(simd::all_of(p == d));
    }
    {
        simd::vf32x8_t a(2.f), b(1.f), p(1.f);
        auto c = a - b;
        EXPECT_TRUE(simd::all_of(p == c));
        auto d = simd::sub(a, b);
        EXPECT_TRUE(simd::all_of(p == d));
        auto e = simd::sub(a, 1.f);
        EXPECT_TRUE(simd::all_of(p == e));
        auto f = simd::sub(2.f, b);
        EXPECT_TRUE(simd::all_of(p == f));
    }
    {
        simd::vf64x4_t a(2.0), b(1.0), p(1.0);
        auto c = a - b;
        EXPECT_TRUE(simd::all_of(p == c));
        auto d = simd::sub(a, b);
        EXPECT_TRUE(simd::all_of(p == d));
        auto e = simd::sub(a, 1.0);
        EXPECT_TRUE(simd::all_of(p == e));
        auto f = simd::sub(2.0, b);
        EXPECT_TRUE(simd::all_of(p == f));
    }
}

TEST(vec_op_sse, test_arith_sub_inplace)
{
    {
        simd::vi32x8_t a(1), b(2), p(-1);
        a -= b;
        EXPECT_TRUE(simd::all_of(p == a));
    }
    {
        simd::vf32x8_t a(1.f), b(2.f), p(-1);
        a -= b;
        EXPECT_TRUE(simd::all_of(p == a));
    }
    {
        simd::vf64x4_t a(1.0), b(2.0), p(-1);
        a -= b;
        EXPECT_TRUE(simd::all_of(p == a));
    }
}

TEST(vec_op_sse, test_arith_mul)
{
    {
        simd::vf32x8_t a(2.f), b(1.f), p(2.f);
        auto c = a * b;
        EXPECT_TRUE(simd::all_of(p == c));
        auto d = simd::mul(a, b);
        EXPECT_TRUE(simd::all_of(p == d));
        auto e = simd::mul(a, 1.f);
        EXPECT_TRUE(simd::all_of(p == e));
        auto f = simd::mul(2.f, b);
        EXPECT_TRUE(simd::all_of(p == f));
    }
    {
        simd::vf64x4_t a(2.0), b(1.0), p(2.0);
        auto c = a * b;
        EXPECT_TRUE(simd::all_of(p == c));
        auto d = simd::mul(a, b);
        EXPECT_TRUE(simd::all_of(p == d));
        auto e = simd::mul(a, 1.0);
        EXPECT_TRUE(simd::all_of(p == e));
        auto f = simd::mul(2.0, b);
        EXPECT_TRUE(simd::all_of(p == f));
    }
}

TEST(vec_op_sse, test_arithmul_inplace)
{
    {
        simd::vf32x8_t a(1.f), b(2.f), p(2.f);
        a *= b;
        EXPECT_TRUE(simd::all_of(p == a));
    }
    {
        simd::vf64x4_t a(1.0), b(2.0), p(2.0);
        a *= b;
        EXPECT_TRUE(simd::all_of(p == a));
    }
}

TEST(vec_op_sse, test_arith_div)
{
    {
        simd::vf32x8_t a(4.f), b(2.f), p(2.f);
        auto c = a / b;
        EXPECT_TRUE(simd::all_of(p == c));
        auto d = simd::div(a, b);
        EXPECT_TRUE(simd::all_of(p == d));
        auto e = simd::div(a, 2.f);
        EXPECT_TRUE(simd::all_of(p == e));
        auto f = simd::div(4.f, b);
        EXPECT_TRUE(simd::all_of(p == f));
    }
    {
        simd::vf64x4_t a(8.0), b(2.0), p(4.0);
        auto c = a / b;
        EXPECT_TRUE(simd::all_of(p == c));
        auto d = simd::div(a, b);
        EXPECT_TRUE(simd::all_of(p == d));
        auto e = simd::div(a, 2.0);
        EXPECT_TRUE(simd::all_of(p == e));
        auto f = simd::div(8.0, b);
        EXPECT_TRUE(simd::all_of(p == f));
    }
}

TEST(vec_op_sse, test_arith_div_inplace)
{
    {
        simd::vf32x8_t a(1.f), b(2.f), p(0.5f);
        a /= b;
        EXPECT_TRUE(simd::all_of(p == a));
    }
    {
        simd::vf64x4_t a(1.0), b(2.0), p(0.5);
        a /= b;
        EXPECT_TRUE(simd::all_of(p == a));
    }
}

TEST(vec_op_sse, test_arith_inc_by1)
{
    {
        simd::vi32x8_t a(1), p(2);
        auto c = ++a;
        EXPECT_TRUE(simd::all_of(p == c));
        EXPECT_TRUE(simd::all_of(p == a));
        auto d = a++;
        EXPECT_TRUE(simd::all_of(p == d));
        p++;
        EXPECT_TRUE(simd::all_of(p == a));
    }
    {
        simd::vf32x8_t a(1.f), p(2.f);
        auto c = ++a;
        EXPECT_TRUE(simd::all_of(p == c));
        EXPECT_TRUE(simd::all_of(p == a));
        auto d = a++;
        EXPECT_TRUE(simd::all_of(p == d));
        p++;
        EXPECT_TRUE(simd::all_of(p == a));
    }
    {
        simd::vf64x4_t a(1.0), p(2.0);
        auto c = ++a;
        EXPECT_TRUE(simd::all_of(p == c));
        EXPECT_TRUE(simd::all_of(p == a));
        auto d = a++;
        EXPECT_TRUE(simd::all_of(p == d));
        p++;
        EXPECT_TRUE(simd::all_of(p == a));
    }
}

TEST(vec_op_sse, test_arith_dec_by1)
{
    {
        simd::vi32x8_t a(1), p(0);
        auto c = --a;
        EXPECT_TRUE(simd::all_of(p == c));
        EXPECT_TRUE(simd::all_of(p == a));
        auto d = a--;
        EXPECT_TRUE(simd::all_of(p == d));
        p--;
        EXPECT_TRUE(simd::all_of(p == a));
    }
    {
        simd::vf32x8_t a(1.f), p(0.f);
        auto c = --a;
        EXPECT_TRUE(simd::all_of(p == c));
        EXPECT_TRUE(simd::all_of(p == a));
        auto d = a--;
        EXPECT_TRUE(simd::all_of(p == d));
        p--;
        EXPECT_TRUE(simd::all_of(p == a));
    }
    {
        simd::vf64x4_t a(1.0), p(0.0);
        auto c = --a;
        EXPECT_TRUE(simd::all_of(p == c));
        EXPECT_TRUE(simd::all_of(p == a));
        auto d = a--;
        EXPECT_TRUE(simd::all_of(p == d));
        p--;
        EXPECT_TRUE(simd::all_of(p == a));
    }
}

TEST(vec_op_sse, test_arith_neg)
{
    {
        simd::vi32x8_t a(1), p(-1);
        auto c = -a;
        EXPECT_TRUE(simd::all_of(p == c));
    }
    {
        simd::vf32x8_t a(1.f), p(-1.f);
        auto c = -a;
        EXPECT_TRUE(simd::all_of(p == c));
    }
    {
        simd::vf64x4_t a(1.0), p(-1.0);
        auto c = -a;
        EXPECT_TRUE(simd::all_of(p == c));
    }
}
