#include <gtest/gtest.h>

#include "simd/simd.h"

using namespace simd;

TEST(api, test_generic_real_imag)
{
    {
        simd::Vec<float, 4> x(1.f);
        EXPECT_TRUE(simd::all_of(x == simd::real(x)));
    }
    {
        simd::Vec<cf32_t, 4> x(cf32_t(1.f, 2.f));
        EXPECT_TRUE(simd::all_of(1.f == simd::real(x)));
    }
    {
        simd::Vec<float, 4> x(1.f);
        EXPECT_TRUE(simd::all_of(0.f == simd::imag(x)));
    }
    {
        simd::Vec<cf32_t, 4> x(cf32_t(1.f, 2.f));
        EXPECT_TRUE(simd::all_of(2.f == simd::imag(x)));
    }
}
