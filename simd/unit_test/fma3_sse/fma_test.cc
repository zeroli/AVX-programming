#include <gtest/gtest.h>

#include "simd/simd.h"

using namespace simd;

STATIC_CHECK_ARCH_ENABLED(FMA3_SSE);

TEST(vec_op_fma3_sse, test_arith_fmadd)
{
    {
        simd::vf32x4_t a(1.f, 2.f, 3.f, 4.f);
        simd::vf32x4_t b(1.f, 2.f, 3.f, 4.f);
        simd::vf32x4_t c(1.f, 2.f, 3.f, 4.f);
        simd::vf32x4_t p(1.f * 1.f + 1.f, 2.f * 2.f + 2.f, 3.f * 3.f + 3.f, 4.f * 4.f + 4.f);
        auto d = simd::fmadd(a, b, c);
        for (auto i = 0; i < d.size(); i++) {
            EXPECT_FLOAT_EQ(d[i], p[i]);
        }
    }
    {
        simd::vf64x2_t a(3.3, 4.4);
        simd::vf64x2_t b(5.3, 4.1);
        simd::vf64x2_t c(6.3, 9.8);
        simd::vf64x2_t p(3.3 * 5.3 + 6.3, 4.4 * 4.1 + 9.8);
        auto d = simd::fmadd(a, b, c);
        for (auto i = 0; i < d.size(); i++) {
            EXPECT_FLOAT_EQ(d[i], p[i]);
        }
    }
}

TEST(vec_op_fma3_sse, test_arith_fmsub)
{
    {
        simd::vf32x4_t a(1.f, 2.f, 3.f, 4.f);
        simd::vf32x4_t b(1.f, 2.f, 3.f, 4.f);
        simd::vf32x4_t c(1.f, 2.f, 3.f, 4.f);
        simd::vf32x4_t p(
            1.f * 1.f - 1.f,
            2.f * 2.f - 2.f,
            3.f * 3.f - 3.f,
            4.f * 4.f - 4.f
        );
        auto d = simd::fmsub(a, b, c);
        for (auto i = 0; i < d.size(); i++) {
            EXPECT_FLOAT_EQ(d[i], p[i]);
        }
    }
    {
        simd::vf64x2_t a(3.3, 4.4);
        simd::vf64x2_t b(5.3, 4.1);
        simd::vf64x2_t c(6.3, 9.8);
        simd::vf64x2_t p(
            3.3 * 5.3 - 6.3,
            4.4 * 4.1 - 9.8
        );
        auto d = simd::fmsub(a, b, c);
        for (auto i = 0; i < d.size(); i++) {
            EXPECT_FLOAT_EQ(d[i], p[i]);
        }
    }
}

TEST(vec_op_fma3_sse, test_arith_fnmadd)
{
    {
        simd::vf32x4_t a(1.f, 2.f, 3.f, 4.f);
        simd::vf32x4_t b(1.f, 2.f, 3.f, 4.f);
        simd::vf32x4_t c(1.f, 2.f, 3.f, 4.f);
        simd::vf32x4_t p(
            -(1.f * 1.f) + 1.f,
            -(2.f * 2.f) + 2.f,
            -(3.f * 3.f) + 3.f,
            -(4.f * 4.f) + 4.f
        );
        auto d = simd::fnmadd(a, b, c);
        for (auto i = 0; i < d.size(); i++) {
            EXPECT_FLOAT_EQ(d[i], p[i]);
        }
    }
    {
        simd::vf64x2_t a(3.3, 4.4);
        simd::vf64x2_t b(5.3, 4.1);
        simd::vf64x2_t c(6.3, 9.8);
        simd::vf64x2_t p(
            -(3.3 * 5.3) + 6.3,
            -(4.4 * 4.1) + 9.8
        );
        auto d = simd::fnmadd(a, b, c);
        for (auto i = 0; i < d.size(); i++) {
            EXPECT_FLOAT_EQ(d[i], p[i]);
        }
    }
}

TEST(vec_op_fma3_sse, test_arith_fnmsub)
{
    {
        simd::vf32x4_t a(1.f, 2.f, 3.f, 4.f);
        simd::vf32x4_t b(1.f, 2.f, 3.f, 4.f);
        simd::vf32x4_t c(1.f, 2.f, 3.f, 4.f);
        simd::vf32x4_t p(
            -(1.f * 1.f) - 1.f,
            -(2.f * 2.f) - 2.f,
            -(3.f * 3.f) - 3.f,
            -(4.f * 4.f) - 4.f
        );
        auto d = simd::fnmsub(a, b, c);
        for (auto i = 0; i < d.size(); i++) {
            EXPECT_FLOAT_EQ(d[i], p[i]);
        }
    }
    {
        simd::vf64x2_t a(3.3, 4.4);
        simd::vf64x2_t b(5.3, 4.1);
        simd::vf64x2_t c(6.3, 9.8);
        simd::vf64x2_t p(
            -(3.3 * 5.3) - 6.3,
            -(4.4 * 4.1) - 9.8
        );
        auto d = simd::fnmsub(a, b, c);
        for (auto i = 0; i < d.size(); i++) {
            EXPECT_FLOAT_EQ(d[i], p[i]);
        }
    }
}

TEST(vec_op_fma3_sse, test_arith_fmaddsub)
{
    {
        simd::vf32x4_t a(1.f, 2.f, 3.f, 4.f);
        simd::vf32x4_t b(1.f, 2.f, 3.f, 4.f);
        simd::vf32x4_t c(1.f, 2.f, 3.f, 4.f);
        simd::vf32x4_t p(
                1.f * 1.f - 1.f,
                2.f * 2.f + 2.f,
                3.f * 3.f - 3.f,
                4.f * 4.f + 4.f);
        auto d = simd::fmaddsub(a, b, c);
        for (auto i = 0; i < d.size(); i++) {
            EXPECT_FLOAT_EQ(d[i], p[i]);
        }
    }
    {
        simd::vf32x8_t a(1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f);
        simd::vf32x8_t b(1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f);
        simd::vf32x8_t c(1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f);
        simd::vf32x8_t p(
                1.f * 1.f - 1.f,
                2.f * 2.f + 2.f,
                3.f * 3.f - 3.f,
                4.f * 4.f + 4.f,
                1.f * 1.f - 1.f,
                2.f * 2.f + 2.f,
                3.f * 3.f - 3.f,
                4.f * 4.f + 4.f
        );
        auto d = simd::fmaddsub(a, b, c);
        for (auto i = 0; i < d.size(); i++) {
            EXPECT_FLOAT_EQ(d[i], p[i]);
        }
    }

    {
        simd::vf64x2_t a(3.3, 4.4);
        simd::vf64x2_t b(5.3, 4.1);
        simd::vf64x2_t c(6.3, 9.8);
        simd::vf64x2_t p(
                3.3 * 5.3 - 6.3,
                4.4 * 4.1 + 9.8
        );
        auto d = simd::fmaddsub(a, b, c);
        for (auto i = 0; i < d.size(); i++) {
            EXPECT_FLOAT_EQ(d[i], p[i]);
        }
    }
    {
        simd::vf64x4_t a(3.3, 4.4, 3.3, 4.4);
        simd::vf64x4_t b(5.3, 4.1, 5.3, 4.1);
        simd::vf64x4_t c(6.3, 9.8, 6.3, 9.8);
        simd::vf64x4_t p(
                3.3 * 5.3 - 6.3,
                4.4 * 4.1 + 9.8,
                3.3 * 5.3 - 6.3,
                4.4 * 4.1 + 9.8
        );
        auto d = simd::fmaddsub(a, b, c);
        for (auto i = 0; i < d.size(); i++) {
            EXPECT_FLOAT_EQ(d[i], p[i]);
        }
    }
}

TEST(vec_op_fma3_sse, test_arith_fmsubadd)
{
    {
        simd::vf32x4_t a(1.f, 2.f, 3.f, 4.f);
        simd::vf32x4_t b(1.f, 2.f, 3.f, 4.f);
        simd::vf32x4_t c(1.f, 2.f, 3.f, 4.f);
        simd::vf32x4_t p(
                1.f * 1.f + 1.f,
                2.f * 2.f - 2.f,
                3.f * 3.f + 3.f,
                4.f * 4.f - 4.f
        );
        auto d = simd::fmsubadd(a, b, c);
        for (auto i = 0; i < d.size(); i++) {
            EXPECT_FLOAT_EQ(d[i], p[i]);
        }
    }
    {
        simd::vf32x8_t a(1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f);
        simd::vf32x8_t b(1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f);
        simd::vf32x8_t c(1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f);
        simd::vf32x8_t p(
                1.f * 1.f + 1.f,
                2.f * 2.f - 2.f,
                3.f * 3.f + 3.f,
                4.f * 4.f - 4.f,
                1.f * 1.f + 1.f,
                2.f * 2.f - 2.f,
                3.f * 3.f + 3.f,
                4.f * 4.f - 4.f
        );
        auto d = simd::fmsubadd(a, b, c);
        for (auto i = 0; i < d.size(); i++) {
            EXPECT_FLOAT_EQ(d[i], p[i]);
        }
    }

    {
        simd::vf64x2_t a(3.3, 4.4);
        simd::vf64x2_t b(5.3, 4.1);
        simd::vf64x2_t c(6.3, 9.8);
        simd::vf64x2_t p(
                3.3 * 5.3 + 6.3,
                4.4 * 4.1 - 9.8
        );
        auto d = simd::fmsubadd(a, b, c);
        for (auto i = 0; i < d.size(); i++) {
            EXPECT_FLOAT_EQ(d[i], p[i]);
        }
    }
    {
        simd::vf64x4_t a(3.3, 4.4, 3.3, 4.4);
        simd::vf64x4_t b(5.3, 4.1, 5.3, 4.1);
        simd::vf64x4_t c(6.3, 9.8, 6.3, 9.8);
        simd::vf64x4_t p(
                3.3 * 5.3 + 6.3,
                4.4 * 4.1 - 9.8,
                3.3 * 5.3 + 6.3,
                4.4 * 4.1 - 9.8
        );
        auto d = simd::fmsubadd(a, b, c);
        for (auto i = 0; i < d.size(); i++) {
            EXPECT_FLOAT_EQ(d[i], p[i]);
        }
    }
}