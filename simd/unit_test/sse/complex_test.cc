#include <gtest/gtest.h>

#include "simd/simd.h"
#include "check_arch.h"

using namespace simd;

TEST(vec_complex_sse, test_arith_add)
{
    {
        cf32_t x(1.f, 2.f), y(2.f, 3.f);
        simd::Vec<cf32_t, 4> a(x), b(y), p(x + y);
        auto c = a + b;
        for (auto i = 0; i < c.size(); i++) {
            EXPECT_FLOAT_EQ(p.real()[i], c.real()[i]);
            EXPECT_FLOAT_EQ(p.imag()[i], c.imag()[i]);
        }
    }
    {
        cf64_t x(1.5, 2.3), y(2.1, 3.0);
        simd::Vec<cf64_t, 2> a(x), b(y), p(x + y);
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
}

TEST(vec_complex_sse, test_arith_sub)
{
    {
        cf32_t x(1.f, 2.f), y(2.f, 3.f);
        simd::Vec<cf32_t, 4> a(x), b(y), p(x - y);
        auto c = a - b;
        for (auto i = 0; i < c.size(); i++) {
            EXPECT_FLOAT_EQ(p.real()[i], c.real()[i]);
            EXPECT_FLOAT_EQ(p.imag()[i], c.imag()[i]);
        }
    }
    {
        cf64_t x(1.5, 2.3), y(2.1, 3.0);
        simd::Vec<cf64_t, 2> a(x), b(y), p(x - y);
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
}

TEST(vec_complex_sse, test_arith_mul)
{
    {
        cf32_t x(1.f, 2.f), y(2.f, 3.f);
        simd::Vec<cf32_t, 4> a(x), b(y), p(x * y);
        auto c = a * b;
        for (auto i = 0; i < c.size(); i++) {
            EXPECT_FLOAT_EQ(p.real()[i], c.real()[i]);
            EXPECT_FLOAT_EQ(p.imag()[i], c.imag()[i]);
        }
    }
    {
        cf64_t x(1.5, 2.3), y(2.1, 3.0);
        simd::Vec<cf64_t, 2> a(x), b(y), p(x * y);
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
}

TEST(vec_complex_sse, test_arith_div)
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

TEST(vec_complex_sse, test_memory_load_aligned)
{
    {
        alignas(16) float mem[] = { 1.f, 2.f, 3.f, 4.f };
        auto a = simd::Vec<cf32_t, 4>::load_aligned(mem);
        for (auto i = 0; i < a.size(); i++) {
            EXPECT_FLOAT_EQ(mem[i], a.real()[i]);
            EXPECT_FLOAT_EQ(0.f, a.imag()[i]);
        }
    }
    {
        alignas(16) float rmem[] = { 1.f, 2.f, 3.f, 4.f };
        alignas(16) float imem[] = { 5.f, 6.f, 7.f, 8.f };
        auto a = simd::Vec<cf32_t, 4>::load_aligned(rmem, imem);
        for (auto i = 0; i < a.size(); i++) {
            EXPECT_FLOAT_EQ(rmem[i], a.real()[i]);
            EXPECT_FLOAT_EQ(imem[i], a.imag()[i]);
        }
    }
    {
        alignas(16) float mem[] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f };
        auto a = simd::Vec<cf32_t, 4>::load(mem, simd::aligned_mode{});
        for (auto i = 0; i < a.size(); i++) {
            EXPECT_FLOAT_EQ(mem[2*i],   a.real()[i]);
            EXPECT_FLOAT_EQ(mem[2*i+1], a.imag()[i]);
        }
    }
}
