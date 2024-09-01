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

TEST(vec, test_add)
{
    {
        simd::Vec<int32_t, 4> a(1), b(2), p(3);
        auto c = a + b;
        std::cout << a << "(a)" << " + " << b << "(b) = " << c << "(c)\n";
        //EXPECT_EQ(p, c);
    }
    {
        simd::Vec<int32_t, 4> a(1);
        int b = 2;
        auto c = a + b;
        std::cout << a << "(a)" << " + " << b << "(b) = " << c << "(c)\n";
    }
    {
        simd::Vec<int32_t, 4> a(1);
        int b = 2;
        auto c = b + a;
        std::cout << b << "(b)" << " + " << a << "() = " << c << "(c)\n";
    }
}

TEST(vec, test_sub)
{
    {
        simd::Vec<int32_t, 4> a(1), b(2), p(3);
        auto c = a + b;
        std::cout << a << "(a)" << " - " << b << "(b) = " << c << "(c)\n";
        //EXPECT_EQ(p, c);
    }
    {
        simd::Vec<int32_t, 4> a(1);
        int b = 2;
        auto c = a - b;
        std::cout << a << "(a)" << " - " << b << "(b) = " << c << "(c)\n";
    }
    {
        simd::Vec<int32_t, 4> a(1);
        int b = 2;
        auto c = b - a;
        std::cout << b << "(b)" << " - " << a << "(a) = " << c << "(c)\n";
    }
}
