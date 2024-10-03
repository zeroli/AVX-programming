#include <gtest/gtest.h>

#include "simd/simd.h"
#include "check_arch.h"

using namespace simd;

TEST(vec_op_fma3_avx, test_arith_fmadd)
{
    {
        simd::vf32x8_t a(1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.0f, 8.0f);
        simd::vf32x8_t b(1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.0f, 8.0f);
        simd::vf32x8_t c(1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.0f, 8.0f);
        simd::vf32x8_t p(
            1.f * 1.f + 1.f,
            2.f * 2.f + 2.f,
            3.f * 3.f + 3.f,
            4.f * 4.f + 4.f,
            5.f * 5.f + 5.f,
            6.f * 6.f + 6.f,
            7.f * 7.f + 7.f,
            8.f * 8.f + 8.f
        );
        auto d = simd::fmadd(a, b, c);
        for (auto i = 0; i < d.size(); i++) {
            EXPECT_FLOAT_EQ(d[i], p[i]);
        }
    }
    {
        simd::vf64x4_t a(3.3, 4.4, 5.6, 7.8);
        simd::vf64x4_t b(5.3, 4.1, 5.2, 8.9);
        simd::vf64x4_t c(6.3, 9.8, 2.3, 9.4);
        simd::vf64x4_t p(
            3.3 * 5.3 + 6.3,
            4.4 * 4.1 + 9.8,
            5.6 * 5.2 + 2.3,
            7.8 * 8.9 + 9.4
        );
        auto d = simd::fmadd(a, b, c);
        for (auto i = 0; i < d.size(); i++) {
            EXPECT_FLOAT_EQ(d[i], p[i]);
        }
    }
}

TEST(vec_op_fma3_avx, test_arith_fmsub)
{
    {
        simd::vf32x8_t a(1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.0f, 8.0f);
        simd::vf32x8_t b(1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.0f, 8.0f);
        simd::vf32x8_t c(1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.0f, 8.0f);
        simd::vf32x8_t p(
            1.f * 1.f - 1.f,
            2.f * 2.f - 2.f,
            3.f * 3.f - 3.f,
            4.f * 4.f - 4.f,
            5.f * 5.f - 5.f,
            6.f * 6.f - 6.f,
            7.f * 7.f - 7.f,
            8.f * 8.f - 8.f
        );
        auto d = simd::fmsub(a, b, c);
        for (auto i = 0; i < d.size(); i++) {
            EXPECT_FLOAT_EQ(d[i], p[i]);
        }
    }
    {
        simd::vf64x4_t a(3.3, 4.4, 5.6, 7.8);
        simd::vf64x4_t b(5.3, 4.1, 5.2, 8.9);
        simd::vf64x4_t c(6.3, 9.8, 2.3, 9.4);
        simd::vf64x4_t p(
            3.3 * 5.3 - 6.3,
            4.4 * 4.1 - 9.8,
            5.6 * 5.2 - 2.3,
            7.8 * 8.9 - 9.4
        );
        auto d = simd::fmsub(a, b, c);
        for (auto i = 0; i < d.size(); i++) {
            EXPECT_FLOAT_EQ(d[i], p[i]);
        }
    }
}

TEST(vec_op_fma3_avx, test_arith_fnmadd)
{
    {
        simd::vf32x8_t a(1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.0f, 8.0f);
        simd::vf32x8_t b(1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.0f, 8.0f);
        simd::vf32x8_t c(1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.0f, 8.0f);
        simd::vf32x8_t p(
            -(1.f * 1.f) + 1.f,
            -(2.f * 2.f) + 2.f,
            -(3.f * 3.f) + 3.f,
            -(4.f * 4.f) + 4.f,
            -(5.f * 5.f) + 5.f,
            -(6.f * 6.f) + 6.f,
            -(7.f * 7.f) + 7.f,
            -(8.f * 8.f) + 8.f
        );
        auto d = simd::fnmadd(a, b, c);
        for (auto i = 0; i < d.size(); i++) {
            EXPECT_FLOAT_EQ(d[i], p[i]);
        }
    }
    {
        simd::vf64x4_t a(3.3, 4.4, 5.6, 7.8);
        simd::vf64x4_t b(5.3, 4.1, 5.2, 8.9);
        simd::vf64x4_t c(6.3, 9.8, 2.3, 9.4);
        simd::vf64x4_t p(
            -(3.3 * 5.3) + 6.3,
            -(4.4 * 4.1) + 9.8,
            -(5.6 * 5.2) + 2.3,
            -(7.8 * 8.9) + 9.4
        );
        auto d = simd::fnmadd(a, b, c);
        for (auto i = 0; i < d.size(); i++) {
            EXPECT_FLOAT_EQ(d[i], p[i]);
        }
    }
}

TEST(vec_op_fma3_avx, test_arith_fnmsub)
{
    {
        simd::vf32x8_t a(1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.0f, 8.0f);
        simd::vf32x8_t b(1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.0f, 8.0f);
        simd::vf32x8_t c(1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.0f, 8.0f);
        simd::vf32x8_t p(
            -(1.f * 1.f) - 1.f,
            -(2.f * 2.f) - 2.f,
            -(3.f * 3.f) - 3.f,
            -(4.f * 4.f) - 4.f,
            -(5.f * 5.f) - 5.f,
            -(6.f * 6.f) - 6.f,
            -(7.f * 7.f) - 7.f,
            -(8.f * 8.f) - 8.f
        );
        auto d = simd::fnmsub(a, b, c);
        for (auto i = 0; i < d.size(); i++) {
            EXPECT_FLOAT_EQ(d[i], p[i]);
        }
    }
    {
        simd::vf64x4_t a(3.3, 4.4, 5.6, 7.8);
        simd::vf64x4_t b(5.3, 4.1, 5.2, 8.9);
        simd::vf64x4_t c(6.3, 9.8, 2.3, 9.4);
        simd::vf64x4_t p(
            -(3.3 * 5.3) - 6.3,
            -(4.4 * 4.1) - 9.8,
            -(5.6 * 5.2) - 2.3,
            -(7.8 * 8.9) - 9.4
        );
        auto d = simd::fnmsub(a, b, c);
        for (auto i = 0; i < d.size(); i++) {
            EXPECT_FLOAT_EQ(d[i], p[i]);
        }
    }
}

TEST(vec_op_fma3_avx, test_arith_fmaddsub)
{
    {
        simd::vf32x8_t a(1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.0f, 8.0f);
        simd::vf32x8_t b(1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.0f, 8.0f);
        simd::vf32x8_t c(1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.0f, 8.0f);
        simd::vf32x8_t p(
            1.f * 1.f - 1.f,
            2.f * 2.f + 2.f,
            3.f * 3.f - 3.f,
            4.f * 4.f + 4.f,
            5.f * 5.f - 5.f,
            6.f * 6.f + 6.f,
            7.f * 7.f - 7.f,
            8.f * 8.f + 8.f
        );
        auto d = simd::fmaddsub(a, b, c);
        for (auto i = 0; i < d.size(); i++) {
            EXPECT_FLOAT_EQ(d[i], p[i]);
        }
    }
    {
        simd::vf64x4_t a(3.3, 4.4, 5.6, 7.8);
        simd::vf64x4_t b(5.3, 4.1, 5.2, 8.9);
        simd::vf64x4_t c(6.3, 9.8, 2.3, 9.4);
        simd::vf64x4_t p(
            3.3 * 5.3 - 6.3,
            4.4 * 4.1 + 9.8,
            5.6 * 5.2 - 2.3,
            7.8 * 8.9 + 9.4
        );
        auto d = simd::fmaddsub(a, b, c);
        for (auto i = 0; i < d.size(); i++) {
            EXPECT_FLOAT_EQ(d[i], p[i]);
        }
    }
}

TEST(vec_op_fma3_avx, test_arith_fmsubadd)
{
    {
        simd::vf32x8_t a(1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.0f, 8.0f);
        simd::vf32x8_t b(1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.0f, 8.0f);
        simd::vf32x8_t c(1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.0f, 8.0f);
        simd::vf32x8_t p(
            1.f * 1.f + 1.f,
            2.f * 2.f - 2.f,
            3.f * 3.f + 3.f,
            4.f * 4.f - 4.f,
            5.f * 5.f + 5.f,
            6.f * 6.f - 6.f,
            7.f * 7.f + 7.f,
            8.f * 8.f - 8.f
        );
        auto d = simd::fmsubadd(a, b, c);
        for (auto i = 0; i < d.size(); i++) {
            EXPECT_FLOAT_EQ(d[i], p[i]);
        }
    }

    {
        simd::vf64x4_t a(3.3, 4.4, 5.6, 7.8);
        simd::vf64x4_t b(5.3, 4.1, 5.2, 8.9);
        simd::vf64x4_t c(6.3, 9.8, 2.3, 9.4);
        simd::vf64x4_t p(
            3.3 * 5.3 + 6.3,
            4.4 * 4.1 - 9.8,
            5.6 * 5.2 + 2.3,
            7.8 * 8.9 - 9.4
        );
        auto d = simd::fmsubadd(a, b, c);
        for (auto i = 0; i < d.size(); i++) {
            EXPECT_FLOAT_EQ(d[i], p[i]);
        }
    }
}
