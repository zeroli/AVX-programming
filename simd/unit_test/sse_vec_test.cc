#include <gtest/gtest.h>

#include "simd/simd.h"

using namespace simd;

TEST(vec_op_sse, test_vec_init_regs)
{
    {
        simd::Vec<float, 4> al(0, 1, 2, 3);
        simd::Vec<float, 4> ah(4, 5, 6, 7);
        float pa[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
        auto b = simd::Vec<float, 8>::load_unaligned(pa);
        auto a = simd::Vec<float, 8>(al, ah);
        EXPECT_TRUE(simd::all(a == b));
        for (int i = 0; i < 8; i ++) {
            EXPECT_FLOAT_EQ(pa[i], a[i]);
        }
    }
    {
        simd::Vec<double, 2> al(0, 1);
        simd::Vec<double, 2> ah(2, 3);
        double pa[] = { 0, 1, 2, 3 };
        auto b = simd::Vec<double, 4>::load_unaligned(pa);
        auto a = simd::Vec<double, 4>(al, ah);
        EXPECT_TRUE(simd::all(a == b));
        for (int i = 0; i < 4; i ++) {
            EXPECT_FLOAT_EQ(pa[i], a[i]);
        }
    }
}

TEST(vecbool_sse, test_vecbool_ctor)
{
    #define TEST_INT(TYPE, W) \
    { \
        simd::VecBool<TYPE, W> a(false), b(true); \
        for (int i = 0; i < W; i ++) { \
            EXPECT_EQ(0, a[i]); \
        } \
        for (int i = 0; i < W; i ++) { \
            EXPECT_EQ(-1, b[i]); \
        } \
    } \
    ///
    TEST_INT(int8_t, 16);
    TEST_INT(int16_t, 8);
    TEST_INT(int32_t, 4);
    TEST_INT(int64_t, 2);

    TEST_INT(int8_t,  32);
    TEST_INT(int16_t, 16);
    TEST_INT(int32_t,  8);
    TEST_INT(int64_t,  4);

    #undef TEST_INT

    {
        simd::VecBool<int32_t, 4> a(false, true, false, true);
        EXPECT_EQ( 0, a[0]);
        EXPECT_EQ(-1, a[1]);
        EXPECT_EQ( 0, a[2]);
        EXPECT_EQ(-1, a[3]);
    }
    {
        simd::VecBool<int32_t, 8> a(false, true, false, true, false, true, false, true);
        EXPECT_EQ( 0, a[0]);
        EXPECT_EQ(-1, a[1]);
        EXPECT_EQ( 0, a[2]);
        EXPECT_EQ(-1, a[3]);
        EXPECT_EQ( 0, a[4]);
        EXPECT_EQ(-1, a[5]);
        EXPECT_EQ( 0, a[6]);
        EXPECT_EQ(-1, a[7]);
    }
}

TEST(vecbool_sse, test_vec_ctor)
{
    #define TEST_INT_SINGLE_VAL(T, W, val) \
    { \
        simd::Vec<T, W> a(val); \
        for (int i = 0; i < a.size(); i++) { \
            EXPECT_EQ(val, a[i]); \
        } \
    } \
    ///
    TEST_INT_SINGLE_VAL(int8_t,  16, -100);
    TEST_INT_SINGLE_VAL(uint8_t, 16, +100);
    TEST_INT_SINGLE_VAL(int16_t,  8, -100);
    TEST_INT_SINGLE_VAL(uint16_t, 8, +100);
    TEST_INT_SINGLE_VAL(int32_t,  4, -100);
    TEST_INT_SINGLE_VAL(uint32_t, 4, +100);
    TEST_INT_SINGLE_VAL(int64_t,  2, -100);
    TEST_INT_SINGLE_VAL(uint64_t, 2, +100);

    TEST_INT_SINGLE_VAL(int8_t,  32, -100);
    TEST_INT_SINGLE_VAL(uint8_t, 32, +100);
    TEST_INT_SINGLE_VAL(int16_t, 16, -100);
    TEST_INT_SINGLE_VAL(uint16_t,16, +100);
    TEST_INT_SINGLE_VAL(int32_t,  8, -100);
    TEST_INT_SINGLE_VAL(uint32_t, 8, +100);
    TEST_INT_SINGLE_VAL(int64_t,  4, -100);
    TEST_INT_SINGLE_VAL(uint64_t, 4, +100);

    #undef TEST_INT_SINGLE_VAL

    #define TEST_FLOAT_SINGLE_VAL(T, W, val) \
    { \
        simd::Vec<T, W> a(val); \
        for (int i = 0; i < a.size(); i++) { \
            EXPECT_FLOAT_EQ(val, a[i]); \
        } \
    } \
    ///
    TEST_FLOAT_SINGLE_VAL(float, 4,  2.3f);
    TEST_FLOAT_SINGLE_VAL(float, 8,  2.3f);
    TEST_FLOAT_SINGLE_VAL(float, 16, 2.3f);

    TEST_FLOAT_SINGLE_VAL(double, 2, 2.3);
    TEST_FLOAT_SINGLE_VAL(double, 4, 2.3);
    TEST_FLOAT_SINGLE_VAL(double, 8, 2.3);

    #undef TEST_FLOAT_SINGLE_VAL

    {
        simd::Vec<int32_t, 4> a(1, 2, 3, 4);
        for (int i = 0; i < a.size(); i++) {
            EXPECT_EQ(i+1, a[i]);
        }
    }
    {
        simd::Vec<float, 4> a(1, 2, 3, 4);
        for (int i = 0; i < a.size(); i++) {
            EXPECT_FLOAT_EQ(i+1, a[i]);
        }
    }
    {
        simd::Vec<double, 4> a(1, 2, 3, 4);
        for (int i = 0; i < a.size(); i++) {
            EXPECT_FLOAT_EQ(i+1, a[i]);
        }
    }
    {
        simd::Vec<int32_t, 4> al(1, 2, 3, 4);
        simd::Vec<int32_t, 4> ah(5, 6, 7, 8);
        simd::Vec<int32_t, 8> a(al, ah);
        for (int i = 0; i < a.size(); i++) {
            EXPECT_EQ(i+1, a[i]);
        }
    }
    {
        simd::Vec<float, 4> al(1, 2, 3, 4);
        simd::Vec<float, 4> ah(5, 6, 7, 8);
        simd::Vec<float, 8> a(al, ah);
        for (int i = 0; i < a.size(); i++) {
            EXPECT_FLOAT_EQ(i+1, a[i]);
        }
    }
    {
        simd::VecBool<int32_t, 4> b(false, true, false, true);
        simd::Vec<int32_t, 4> a(b);
        EXPECT_EQ( 0, a[0]);
        EXPECT_EQ(-1, a[1]);
        EXPECT_EQ( 0, a[2]);
        EXPECT_EQ(-1, a[3]);
    }
    {
        simd::VecBool<float, 4> b(false, true, false, true);
        simd::Vec<float, 4> a(b);
        EXPECT_EQ( 0, bits::cast<int32_t>(a[0]));
        EXPECT_EQ(-1, bits::cast<int32_t>(a[1]));
        EXPECT_EQ( 0, bits::cast<int32_t>(a[2]));
        EXPECT_EQ(-1, bits::cast<int32_t>(a[3]));
    }
    {
        simd::VecBool<double, 4> b(false, true, false, true);
        simd::Vec<double, 4> a(b);
        EXPECT_EQ( 0, bits::cast<int64_t>(a[0]));
        EXPECT_EQ(-1, bits::cast<int64_t>(a[1]));
        EXPECT_EQ( 0, bits::cast<int64_t>(a[2]));
        EXPECT_EQ(-1, bits::cast<int64_t>(a[3]));
    }
}
