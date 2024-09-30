#include <gtest/gtest.h>

#include "simd/simd.h"

using namespace simd;

TEST(vec_avx, test_types)
{
    {
        EXPECT_EQ(8, simd::vi32x8_t::size());
        EXPECT_EQ(1, simd::vi32x8_t::n_regs());
        EXPECT_EQ(8, simd::vi32x8_t::reg_lanes());
        EXPECT_TRUE(simd::vi32x8_t::size() ==
            simd::vi32x8_t::n_regs() * simd::vi32x8_t::reg_lanes());
    }
    {
        EXPECT_EQ(16, simd::vi32x16_t::size());
        EXPECT_EQ(2,  simd::vi32x16_t::n_regs());
        EXPECT_EQ(8,  simd::vi32x16_t::reg_lanes());
        EXPECT_TRUE(simd::vi32x16_t::size() ==
            simd::vi32x16_t::n_regs() * simd::vi32x16_t::reg_lanes());
    }
    {
        EXPECT_EQ(8, simd::vf32x8_t::size());
        EXPECT_EQ(1, simd::vf32x8_t::n_regs());
        EXPECT_EQ(8, simd::vf32x8_t::reg_lanes());
        EXPECT_TRUE(simd::vf32x8_t::size() ==
            simd::vf32x8_t::n_regs() * simd::vf32x8_t::reg_lanes());
    }
    {
        EXPECT_EQ(16, simd::vf32x16_t::size());
        EXPECT_EQ(2,  simd::vf32x16_t::n_regs());
        EXPECT_EQ(8,  simd::vf32x16_t::reg_lanes());
        EXPECT_TRUE(simd::vf32x16_t::size() ==
            simd::vf32x16_t::n_regs() * simd::vf32x16_t::reg_lanes());
    }
    {
        EXPECT_EQ(4, simd::vf64x4_t::size());
        EXPECT_EQ(1, simd::vf64x4_t::n_regs());
        EXPECT_EQ(4, simd::vf64x4_t::reg_lanes());
        EXPECT_TRUE(simd::vf64x4_t::size() ==
            simd::vf64x4_t::n_regs() * simd::vf64x4_t::reg_lanes());
    }
    {
        EXPECT_EQ(8, simd::vf64x8_t::size());
        EXPECT_EQ(2, simd::vf64x8_t::n_regs());
        EXPECT_EQ(4, simd::vf64x8_t::reg_lanes());
        EXPECT_TRUE(simd::vf64x8_t::size() ==
            simd::vf64x8_t::n_regs() * simd::vf64x8_t::reg_lanes());
    }
}
