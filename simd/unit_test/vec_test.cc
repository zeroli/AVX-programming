#include <gtest/gtest.h>

#include "simd/simd.h"

#include <sstream>

TEST(vec, test_1)
{
    {
        simd::Vec<int32_t, 4> a, b;
    }
    {
        simd::Vec<int32_t, 4> a(1), b(2);
    }
    {
        simd::Vec<int32_t, 4> a(1), b(2);
        std::ostringstream os;
        os << a;
        EXPECT_EQ("vi32x4[1, 1, 1, 1]", os.str());
        os.str("");
        os << b;
        EXPECT_EQ("vi32x4[2, 2, 2, 2]", os.str());
    }
}
