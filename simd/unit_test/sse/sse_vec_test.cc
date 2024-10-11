#include <gtest/gtest.h>

#include "simd/simd.h"

#include "simd/unit_test/test_common.h"
#include "check_arch.h"

using namespace simd;

TEST(vec_sse, test_type_arch)
{
    EXPECT_EQ(simd::util::arch_name<simd::vi32x4_t>(), simd::vi32x4_t::arch_name());
    simd::vi32x4_t v;
    EXPECT_EQ(simd::util::arch_name(&v), simd::vi32x4_t::arch_name());
}

TEST(vec_sse, test_types)
{
    TEST_VEC_TYPE(simd::vi8x16_t, 16, 1, 16, simd::SSE);
    TEST_VEC_TYPE(simd::vi8x32_t, 32, 2, 16, simd::SSE);
    TEST_VEC_TYPE(simd::vi8x64_t, 64, 4, 16, simd::SSE);

    TEST_VEC_TYPE(simd::vu8x16_t, 16, 1, 16, simd::SSE);
    TEST_VEC_TYPE(simd::vu8x32_t, 32, 2, 16, simd::SSE);
    TEST_VEC_TYPE(simd::vu8x64_t, 64, 4, 16, simd::SSE);

    TEST_VEC_TYPE(simd::vi16x8_t,  8,  1, 8, simd::SSE);
    TEST_VEC_TYPE(simd::vi16x16_t, 16, 2, 8, simd::SSE);
    TEST_VEC_TYPE(simd::vi16x32_t, 32, 4, 8, simd::SSE);

    TEST_VEC_TYPE(simd::vu16x8_t,  8,  1, 8, simd::SSE);
    TEST_VEC_TYPE(simd::vu16x16_t, 16, 2, 8, simd::SSE);
    TEST_VEC_TYPE(simd::vu16x32_t, 32, 4, 8, simd::SSE);

    TEST_VEC_TYPE(simd::vi32x4_t,  4,  1, 4, simd::SSE);
    TEST_VEC_TYPE(simd::vi32x8_t,  8,  2, 4, simd::SSE);
    TEST_VEC_TYPE(simd::vi32x16_t, 16, 4, 4, simd::SSE);

    TEST_VEC_TYPE(simd::vu32x4_t,  4,  1, 4, simd::SSE);
    TEST_VEC_TYPE(simd::vu32x8_t,  8,  2, 4, simd::SSE);
    TEST_VEC_TYPE(simd::vu32x16_t, 16, 4, 4, simd::SSE);

    TEST_VEC_TYPE(simd::vi64x2_t, 2, 1, 2, simd::SSE);
    TEST_VEC_TYPE(simd::vi64x4_t, 4, 2, 2, simd::SSE);
    TEST_VEC_TYPE(simd::vi64x8_t, 8, 4, 2, simd::SSE);

    TEST_VEC_TYPE(simd::vu64x2_t, 2, 1, 2, simd::SSE);
    TEST_VEC_TYPE(simd::vu64x4_t, 4, 2, 2, simd::SSE);
    TEST_VEC_TYPE(simd::vu64x8_t, 8, 4, 2, simd::SSE);

    TEST_VEC_TYPE(simd::vf32x4_t,  4,  1, 4, simd::SSE);
    TEST_VEC_TYPE(simd::vf32x8_t,  8,  2, 4, simd::SSE);
    TEST_VEC_TYPE(simd::vf32x16_t, 16, 4, 4, simd::SSE);

    TEST_VEC_TYPE(simd::vf64x2_t, 2, 1, 2, simd::SSE);
    TEST_VEC_TYPE(simd::vf64x4_t, 4, 2, 2, simd::SSE);
    TEST_VEC_TYPE(simd::vf64x8_t, 8, 4, 2, simd::SSE);

    TEST_VEC_TYPE(simd::vcf32x2_t, 2, 1, 2, simd::SSE);
    TEST_VEC_TYPE(simd::vcf32x4_t, 4, 2, 2, simd::SSE);
    TEST_VEC_TYPE(simd::vcf32x8_t, 8, 4, 2, simd::SSE);

    TEST_VEC_TYPE(simd::vcf64x1_t, 1, 1, 1, simd::SSE);
    TEST_VEC_TYPE(simd::vcf64x2_t, 2, 2, 1, simd::SSE);
    TEST_VEC_TYPE(simd::vcf64x4_t, 4, 4, 1, simd::SSE);
}

TEST(vec_sse, test_vec_ctor_generator)
{
    {
        using vec_t = simd::vi32x4_t;
        vec_t a([](int i) { return i + 1; });
        for (int i = 0; i < vec_t::size(); i++) {
            EXPECT_EQ(i + 1, a[i]);
        }
    }
}

TEST(vec_sse, test_vec_init_regs)
{
    {
        simd::Vec<float, 4> al(0, 1, 2, 3);
        simd::Vec<float, 4> ah(4, 5, 6, 7);
        float pa[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
        auto b = simd::Vec<float, 8>::load_unaligned(pa);
        auto a = simd::Vec<float, 8>(al, ah);
        EXPECT_TRUE(simd::all_of(a == b));
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
        EXPECT_TRUE(simd::all_of(a == b));
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

TEST(vec_sse, test_vec_ctor_single_val)
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
}

TEST(vec_sse, test_vec_ctor_multiple_vals)
{
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
}

TEST(vec_sse, test_vec_ctor_multiple_vecs)
{
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
}

TEST(vec_sse, test_vec_ctor_from_vecbool)
{
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

TEST(vec_sse, test_vec_copy_ctor)
{
    {
        simd::Vec<int32_t, 4> a = 10;
        simd::Vec<int32_t, 4> b(10);
        for (int i = 0; i < 4; i++) {
            EXPECT_EQ(a[i], b[i]);
        }
        b = 300;
        for (int i = 0; i < 4; i++) {
            EXPECT_EQ(b[i], 300);
        }
    }
}

TEST(vecbool_sse, test_vecbool_load)
{
    {
        const bool bp[] = { false, true, true, false };
        auto b = simd::VecBool<int32_t, 4>::load_aligned(bp);
        for (int i = 0; i < 4; i++) {
            EXPECT_EQ(bp[i] ? bits::ones<int32_t>() : bits::zeros<int32_t>(), b[i]);
        }
    }
    {
        const bool bp[] = { false, true, true, false };
        auto b = simd::VecBool<float, 4>::load_aligned(bp);
        for (int i = 0; i < 4; i++) {
            EXPECT_EQ(bp[i] ? bits::ones<int32_t>() : bits::zeros<int32_t>(), bits::cast<int32_t>(b[i]));
        }
    }
}

TEST(vecbool_sse, test_vecbool_store)
{
    {
        bool bp[] = { true };
        simd::VecBool<int32_t, 4> b(false, true, true, false);
        b.store_aligned(bp);
        EXPECT_EQ(bits::zeros<int32_t>(), bits::cast<int32_t>(b[0]));
        EXPECT_EQ(bits::ones<int32_t>(),  bits::cast<int32_t>(b[1]));
        EXPECT_EQ(bits::ones<int32_t>(),  bits::cast<int32_t>(b[2]));
        EXPECT_EQ(bits::zeros<int32_t>(), bits::cast<int32_t>(b[3]));
    }
    {
        bool bp[] = { true };
        simd::VecBool<float, 4> b(false, true, true, false);
        b.store_aligned(bp);
        EXPECT_EQ(bits::zeros<int32_t>(), bits::cast<int32_t>(b[0]));
        EXPECT_EQ(bits::ones<int32_t>(),  bits::cast<int32_t>(b[1]));
        EXPECT_EQ(bits::ones<int32_t>(),  bits::cast<int32_t>(b[2]));
        EXPECT_EQ(bits::zeros<int32_t>(), bits::cast<int32_t>(b[3]));
    }
}

TEST(vecbool_sse, test_vecbool_to_mask)
{
    {
        simd::VecBool<int8_t, 16> b(true);
        EXPECT_EQ(0xFFFF, b.to_mask());
    }
    {
        simd::VecBool<int16_t, 8> b(true);
        EXPECT_EQ(0xFF, b.to_mask());
    }
    {
        simd::VecBool<int32_t, 4> b(false, true, true, true);
        EXPECT_EQ(0b1110, b.to_mask());
    }
    {
        simd::VecBool<float, 4> b(false, true, true, true);
        EXPECT_EQ(0b1110, b.to_mask());
    }
    {
        simd::VecBool<double, 4> b(false, true, true, true);
        EXPECT_EQ(0b1110, b.to_mask());
    }
}

TEST(vecbool_sse, test_vecbool_from_mask)
{
    {
        auto b = simd::VecBool<int8_t, 16>::from_mask(0b0101110110011010);
        EXPECT_EQ(0b0101110110011010, b.to_mask());
    }
    {
        auto b = simd::VecBool<int16_t, 8>::from_mask(0b10011010);
        EXPECT_EQ(0b10011010, b.to_mask());
    }
    {
        auto b = simd::VecBool<int32_t, 4>::from_mask(0b1010);
        EXPECT_EQ(0b1010, b.to_mask());
    }
    {
        auto b = simd::VecBool<int64_t, 2>::from_mask(0b10);
        EXPECT_EQ(0b10, b.to_mask());
    }
    {
        auto b = simd::VecBool<float, 4>::from_mask(0b1010);
        EXPECT_EQ(0b1010, b.to_mask());
    }
    {
        auto b = simd::VecBool<double, 4>::from_mask(0b1010);
        EXPECT_EQ(0b1010, b.to_mask());
    }
}
