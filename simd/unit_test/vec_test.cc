#include <gtest/gtest.h>

#include "simd/simd.h"

#include <sstream>

TEST(vec, test_integral)
{
    {
        simd::Vec<int32_t, 4> a, b;
    }
    {
        simd::Vec<int32_t, 4> a(1), b(2);
        for (int i = 0; i < 4; i++) {
            EXPECT_EQ(1, a[i]);
            EXPECT_EQ(1, a.at(i));
        }
        for (int i = 0; i < 4; i++) {
            EXPECT_EQ(2, b[i]);
            EXPECT_EQ(2, b.at(i));
        }
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
    check_vec_aligned<int8_t, 32>();
    check_vec_aligned<uint8_t, 32>();
    check_vec_aligned<int16_t, 16>();
    check_vec_aligned<uint16_t, 16>();
    check_vec_aligned<int32_t, 8>();
    check_vec_aligned<uint32_t, 8>();
    check_vec_aligned<int64_t, 4>();
    check_vec_aligned<uint64_t, 4>();
    check_vec_aligned<float, 8>();
    check_vec_aligned<double, 4>();

    /// 512bits
    check_vec_aligned<int8_t, 64>();
    check_vec_aligned<uint8_t, 64>();
    check_vec_aligned<int16_t, 32>();
    check_vec_aligned<uint16_t, 32>();
    check_vec_aligned<int32_t, 16>();
    check_vec_aligned<uint32_t, 16>();
    check_vec_aligned<int64_t, 8>();
    check_vec_aligned<uint64_t, 8>();
    check_vec_aligned<float, 16>();
    check_vec_aligned<double, 8>();
}

TEST(vec, test_pretty_print)
{
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

TEST(vecbool, test_pretty_print)
{
    {
        simd::VecBool<int32_t, 4> a(true, false, false, true);
        std::ostringstream os;
        os << a;
        EXPECT_EQ("vi32bx4[T, F, F, T]", os.str());
    }
}
