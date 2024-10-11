#include <gtest/gtest.h>

#include "simd/simd.h"
#include "check_arch.h"

using namespace simd;

TEST(vec_complex_avx, test_arith_add)
{
    {
        cf32_t x(1.f, 2.f), y(2.f, 3.f);
        simd::Vec<cf32_t, 8> a(x), b(y), p(x + y);
        auto c = a + b;
        for (auto i = 0; i < c.size(); i++) {
            EXPECT_FLOAT_EQ(p.real()[i], c.real()[i]);
            EXPECT_FLOAT_EQ(p.imag()[i], c.imag()[i]);
        }
    }
    {
        cf64_t x(1.5, 2.3), y(2.1, 3.0);
        simd::Vec<cf64_t, 4> a(x), b(y), p(x + y);
        auto c = a + b;
        for (auto i = 0; i < c.size(); i++) {
            EXPECT_FLOAT_EQ(p.real()[i], c.real()[i]);
            EXPECT_FLOAT_EQ(p.imag()[i], c.imag()[i]);
        }
    }
    {
        cf64_t x(1.5, 2.3), y(2.1, 3.0);
        simd::Vec<cf64_t, 8> a(x), b(y), p(x + y);
        auto c = a + b;
        for (auto i = 0; i < c.size(); i++) {
            EXPECT_FLOAT_EQ(p.real()[i], c.real()[i]);
            EXPECT_FLOAT_EQ(p.imag()[i], c.imag()[i]);
        }
    }
}

TEST(vec_complex_avx, test_arith_sub)
{
    {
        cf32_t x(1.f, 2.f), y(2.f, 3.f);
        simd::Vec<cf32_t, 8> a(x), b(y), p(x - y);
        auto c = a - b;
        for (auto i = 0; i < c.size(); i++) {
            EXPECT_FLOAT_EQ(p.real()[i], c.real()[i]);
            EXPECT_FLOAT_EQ(p.imag()[i], c.imag()[i]);
        }
    }
    {
        cf64_t x(1.5, 2.3), y(2.1, 3.0);
        simd::Vec<cf64_t, 4> a(x), b(y), p(x - y);
        auto c = a - b;
        for (auto i = 0; i < c.size(); i++) {
            EXPECT_FLOAT_EQ(p.real()[i], c.real()[i]);
            EXPECT_FLOAT_EQ(p.imag()[i], c.imag()[i]);
        }
    }
    {
        cf64_t x(1.5, 2.3), y(2.1, 3.0);
        simd::Vec<cf64_t, 8> a(x), b(y), p(x - y);
        auto c = a - b;
        for (auto i = 0; i < c.size(); i++) {
            EXPECT_FLOAT_EQ(p.real()[i], c.real()[i]);
            EXPECT_FLOAT_EQ(p.imag()[i], c.imag()[i]);
        }
    }
}

TEST(vec_complex_avx, test_arith_mul)
{
    {
        cf32_t x(1.f, 2.f), y(2.f, 3.f);
        simd::Vec<cf32_t, 8> a(x), b(y), p(x * y);
        auto c = a * b;
        for (auto i = 0; i < c.size(); i++) {
            EXPECT_FLOAT_EQ(p.real()[i], c.real()[i]);
            EXPECT_FLOAT_EQ(p.imag()[i], c.imag()[i]);
        }
    }
    {
        cf64_t x(1.5, 2.3), y(2.1, 3.0);
        simd::Vec<cf64_t, 4> a(x), b(y), p(x * y);
        auto c = a * b;
        for (auto i = 0; i < c.size(); i++) {
            EXPECT_FLOAT_EQ(p.real()[i], c.real()[i]);
            EXPECT_FLOAT_EQ(p.imag()[i], c.imag()[i]);
        }
    }
    {
        cf64_t x(1.5, 2.3), y(2.1, 3.0);
        simd::Vec<cf64_t, 8> a(x), b(y), p(x * y);
        auto c = a * b;
        for (auto i = 0; i < c.size(); i++) {
            EXPECT_FLOAT_EQ(p.real()[i], c.real()[i]);
            EXPECT_FLOAT_EQ(p.imag()[i], c.imag()[i]);
        }
    }
}

TEST(vec_complex_avx, test_arith_div)
{
    {
        cf32_t x(1.f, 2.f), y(2.f, 3.f);
        simd::Vec<cf32_t, 4> a(x), b(y), p(x / y);
        auto c = a / b;
        for (auto i = 0; i < c.size(); i++) {
            EXPECT_FLOAT_EQ(p.real()[i], c.real()[i]);
            EXPECT_FLOAT_EQ(p.imag()[i], c.imag()[i]);
        }
    }
    {
        cf64_t x(1.5, 2.3), y(2.1, 3.0);
        simd::Vec<cf64_t, 2> a(x), b(y), p(x / y);
        auto c = a / b;
        for (auto i = 0; i < c.size(); i++) {
            EXPECT_FLOAT_EQ(p.real()[i], c.real()[i]);
            EXPECT_FLOAT_EQ(p.imag()[i], c.imag()[i]);
        }
    }
    {
        cf64_t x(1.5, 2.3), y(2.1, 3.0);
        simd::Vec<cf64_t, 4> a(x), b(y), p(x / y);
        auto c = a / b;
        for (auto i = 0; i < c.size(); i++) {
            EXPECT_FLOAT_EQ(p.real()[i], c.real()[i]);
            EXPECT_FLOAT_EQ(p.imag()[i], c.imag()[i]);
        }
    }
}
