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

template <typename T, size_t W>
void check_vec_aligned()
{
    simd::Vec<T, W> a;
    EXPECT_TRUE(simd::is_aligned(&a, sizeof(T) * W));
    EXPECT_TRUE(simd::is_aligned(&a, sizeof(a)));
}

TEST(vec, test_alignment)
{
    /// 128bits
    check_vec_aligned<int8_t, 16>();
    check_vec_aligned<uint8_t, 16>();
    check_vec_aligned<int16_t, 8>();
    check_vec_aligned<uint16_t, 8>();
    check_vec_aligned<int32_t, 4>();
    check_vec_aligned<uint32_t, 4>();
    check_vec_aligned<int64_t, 2>();
    check_vec_aligned<uint64_t, 2>();
    check_vec_aligned<float, 4>();
    check_vec_aligned<double, 2>();

    /// 256bits
    // check_vec_aligned<int8_t, 32>();
    // check_vec_aligned<uint8_t, 32>();
    // check_vec_aligned<int16_t, 16>();
    // check_vec_aligned<uint16_t, 16>();
    // check_vec_aligned<int32_t, 8>();
    // check_vec_aligned<uint32_t, 8>();
    // check_vec_aligned<int64_t, 4>();
    // check_vec_aligned<uint64_t, 4>();
    // check_vec_aligned<float, 8>();
    // check_vec_aligned<double, 4>();

    /// 512bits
    // check_vec_aligned<int8_t, 64>();
    // check_vec_aligned<uint8_t, 64>();
    // check_vec_aligned<int16_t, 32>();
    // check_vec_aligned<uint16_t, 32>();
    // check_vec_aligned<int32_t, 16>();
    // check_vec_aligned<uint32_t, 16>();
    // check_vec_aligned<int64_t, 8>();
    // check_vec_aligned<uint64_t, 8>();
    // check_vec_aligned<float, 16>();
    // check_vec_aligned<double, 8>();
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
