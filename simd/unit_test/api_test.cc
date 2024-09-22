#include <gtest/gtest.h>

#include "simd/simd.h"

using namespace simd;

TEST(api, test_setzero)
{
    {
        auto a = simd::setzero<int32_t, 4>();
        for (int i = 0; i < 4; i++) {
            EXPECT_EQ(0, a[i]);
        }
    }
    {
        auto a = simd::setzero<float, 4>();
        for (int i = 0; i < 4; i++) {
            EXPECT_FLOAT_EQ(0, a[i]);
        }
    }
    {
        auto a = simd::setzero<double, 4>();
        for (int i = 0; i < 4; i++) {
            EXPECT_FLOAT_EQ(0, a[i]);
        }
    }
}

TEST(api, test_load)
{
    {
        alignas(16) int32_t p[] = {1, 2, 3, 4};
        auto a = simd::load<4>(p);
        for (int i = 0; i < 4; i++) {
            EXPECT_EQ(p[i], a[i]);
        }
        a = simd::loadu<4>(p);
        for (int i = 0; i < 4; i++) {
            EXPECT_EQ(p[i], a[i]);
        }
    }
    {
        alignas(16) int32_t p[] = {1, 2, 3, 4};
        auto a = simd::load_aligned<int32_t, 4>(p);
        for (int i = 0; i < 4; i++) {
            EXPECT_EQ(p[i], a[i]);
        }
        a = simd::load_unaligned<int32_t, 4>(p);
        for (int i = 0; i < 4; i++) {
            EXPECT_EQ(p[i], a[i]);
        }
    }
    {
        alignas(16) int32_t p[] = {1, 2, 3, 4};
        auto a = simd::load<4>(p);
        for (int i = 0; i < 4; i++) {
            EXPECT_EQ(p[i], a[i]);
        }
    }
    {
        alignas(16) float p[] = {1, 2, 3, 4};
        auto a = simd::load<4>(p);
        for (int i = 0; i < 4; i++) {
            EXPECT_FLOAT_EQ(p[i], a[i]);
        }
    }
    {
        alignas(32) double p[] = {1, 2, 3, 4};
        auto a = simd::load<4>(p);
        for (int i = 0; i < 4; i++) {
            EXPECT_FLOAT_EQ(p[i], a[i]);
        }
    }
}
