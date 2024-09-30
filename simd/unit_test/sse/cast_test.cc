#include <gtest/gtest.h>

#include "simd/simd.h"

STATIC_CHECK_ARCH_ENABLED(SSE);

TEST(vec_sse, test_vec_cast)
{
    {
        simd::Vec<int32_t, 4> a(1,2,3,4);
        simd::Vec<float, 4> p(1,2,3,4);
        auto b = simd::cast<float>(a);
        EXPECT_TRUE(simd::all_of(p == b));
    }
    {
        simd::Vec<int32_t, 4> p(1,2,3,4);
        simd::Vec<float, 4> a(1,2,3,4);
        auto b = simd::cast<int32_t>(a);
        EXPECT_TRUE(simd::all_of(p == b));
    }
    {
        simd::Vec<int32_t, 8> p(-1,2,3,4,5,-6,7,8);
        simd::Vec<float, 8> a(-1,2,3,4,5,-6,7,8);
        auto b = simd::cast<int32_t>(a);
        EXPECT_TRUE(simd::all_of(p == b));
    }
    {
        simd::Vec<int64_t, 4> a(1,2,3,4);
        simd::Vec<double, 4> p(1,2,3,4);
        auto b = simd::cast<double>(a);
        EXPECT_TRUE(simd::all_of(p == b));
    }
}
