#include <gtest/gtest.h>

#include "simd/simd.h"

STATIC_CHECK_ARCH_ENABLED(SSE);

TEST(vec_op_sse, test_memory_clear)
{
    {
        simd::Vec<int32_t, 4> a(1);
        a.clear();
        for (int i = 0; i < 4; i++) {
            EXPECT_EQ(0, a[i]);
        }
    }
    {
        simd::Vec<float, 4> a(1);
        a.clear();
        for (int i = 0; i < 4; i++) {
            EXPECT_FLOAT_EQ(0, a[i]);
        }
    }
    {
        simd::Vec<double, 4> a(1);
        a.clear();
        for (int i = 0; i < 4; i++) {
            EXPECT_FLOAT_EQ(0, a[i]);
        }
    }
}

TEST(vec_op_sse, test_memory_load_aligned)
{
    {
        alignas(16) int32_t pa[] = { 1, 1, 1, 1 };
        auto a = simd::Vec<int32_t, 4>::load_aligned(pa);
        simd::Vec<int32_t, 4> b(1);
        EXPECT_TRUE(simd::all_of(a == b));
    }
    {
        alignas(16) float pa[] = { 1.f, 1.f, 1.f, 1.f };
        auto a = simd::Vec<float, 4>::load_aligned(pa);
        simd::Vec<float, 4> b(1.f);
        EXPECT_TRUE(simd::all_of(b == a));
    }
    {
        alignas(16) double pa[] = { 1.f, 1.f };
        auto a = simd::Vec<double, 2>::load_aligned(pa);
        simd::Vec<double, 2> b(1.0);
        EXPECT_TRUE(simd::all_of(b == a));
    }
}

TEST(vec_op_sse, test_memory_load_unaligned)
{
    {
        int32_t pa[] = { 1, 1, 1, 1 };
        auto a = simd::Vec<int32_t, 4>::load_unaligned(pa);
        simd::Vec<int32_t, 4> b(1);
        EXPECT_TRUE(simd::all_of(a == b));
    }
    {
        float pa[] = { 1.f, 1.f, 1.f, 1.f };
        auto a = simd::Vec<float, 4>::load_unaligned(pa);
        simd::Vec<float, 4> b(1.f);
        EXPECT_TRUE(simd::all_of(b == a));
    }
    {
        double pa[] = { 1.f, 1.f };
        auto a = simd::Vec<double, 2>::load_unaligned(pa);
        simd::Vec<double, 2> b(1.0);
        EXPECT_TRUE(simd::all_of(b == a));
    }
}

TEST(vec_op_sse, test_memory_store_aligned)
{
    {
        alignas(16) int32_t pa[] = { 1, 1, 1, 1 };
        simd::Vec<int32_t, 4> a(2);
        a.store_aligned(pa);
        auto b = simd::Vec<int32_t, 4>::load_aligned(pa);
        EXPECT_TRUE(simd::all_of(a == b));
    }
    {
        alignas(16) float pa[] = { 1.f, 1.f, 1.f, 1.f };
        simd::Vec<float, 4> a(2.f);
        a.store_aligned(pa);
        auto b = simd::Vec<float, 4>::load_aligned(pa);
        EXPECT_TRUE(simd::all_of(b == a));
    }
    {
        alignas(16) double pa[] = { 1.f, 1.f };
        simd::Vec<double, 2> a(2.0);
        a.store_aligned(pa);
        auto b = simd::Vec<double, 2>::load_aligned(pa);
        EXPECT_TRUE(simd::all_of(b == a));
    }
}

TEST(vec_op_sse, test_memory_store_unaligned)
{
    {
        int32_t pa[] = { 1, 1, 1, 1 };
        simd::Vec<int32_t, 4> a(2);
        a.store_unaligned(pa);
        auto b = simd::Vec<int32_t, 4>::load_aligned(pa);
        EXPECT_TRUE(simd::all_of(a == b));
    }
    {
        float pa[] = { 1.f, 1.f, 1.f, 1.f };
        simd::Vec<float, 4> a(2.f);
        a.store_unaligned(pa);
        auto b = simd::Vec<float, 4>::load_unaligned(pa);
        EXPECT_TRUE(simd::all_of(b == a));
    }
    {
        double pa[] = { 1.0, 1.0 };
        simd::Vec<double, 2> a(2.0);
        a.store_unaligned(pa);
        auto b = simd::Vec<double, 2>::load_unaligned(pa);
        EXPECT_TRUE(simd::all_of(b == a));
    }
}

TEST(vec_op_sse, test_memory_set)
{
    {
        simd::Vec<int64_t, 2> a(0, 1);
        int64_t pa[] = { 0, 1 };
        auto b = simd::Vec<int64_t, 2>::load_unaligned(pa);
        EXPECT_TRUE(simd::all_of(a == b));
        for (int i = 0; i < 2; i ++) {
            EXPECT_EQ(pa[i], a[i]);
        }
    }
    {
        simd::Vec<int32_t, 4> a(0, 1, 2, 3);
        int32_t pa[] = { 0, 1, 2, 3 };
        auto b = simd::Vec<int32_t, 4>::load_unaligned(pa);
        EXPECT_TRUE(simd::all_of(a == b));
        for (int i = 0; i < 4; i ++) {
            EXPECT_EQ(pa[i], a[i]);
        }
    }
    {
        simd::Vec<int16_t, 8> a(0, 1, 2, 3, 4, 5, 6, 7);
        int16_t pa[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
        auto b = simd::Vec<int16_t, 8>::load_unaligned(pa);
        EXPECT_TRUE(simd::all_of(a == b));
        for (int i = 0; i < 8; i ++) {
            EXPECT_EQ(pa[i], a[i]);
        }
    }
    {
        simd::Vec<int8_t, 16> a(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        int8_t pa[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
        auto b = simd::Vec<int8_t, 16>::load_unaligned(pa);
        EXPECT_TRUE(simd::all_of(a == b));
        for (int i = 0; i < 16; i ++) {
            EXPECT_EQ(pa[i], a[i]);
        }
    }
    {
        simd::Vec<float, 4> a(0, 1, 2, 3);
        float pa[] = { 0, 1, 2, 3 };
        auto b = simd::Vec<float, 4>::load_unaligned(pa);
        EXPECT_TRUE(simd::all_of(a == b));
        for (int i = 0; i < 4; i ++) {
            EXPECT_FLOAT_EQ(pa[i], a[i]);
        }
    }
    {
        simd::Vec<double, 2> a(0, 1);
        double pa[] = { 0, 1 };
        auto b = simd::Vec<double, 2>::load_unaligned(pa);
        EXPECT_TRUE(simd::all_of(a == b));
        for (int i = 0; i < 2; i ++) {
            EXPECT_FLOAT_EQ(pa[i], a[i]);
        }
    }
}

TEST(vec_op_sse, test_memory_gather)
{
    {
        simd::Vec<int32_t, 4> index(0, 1, 3, 2);
        int32_t mem[] = { 4, 3, 2, 1 };
        simd::Vec<int32_t, 4> p(4, 3, 1, 2);
        auto a = simd::gather(mem, index);
        //EXPECT_TRUE(simd::all_of(p == a)) << a;
    }
    {
        simd::Vec<int32_t, 4> index(0, 1, 3, 2);
        int32_t mem[] = { 4, 3, 2, 1 };
        simd::Vec<float, 4> p(4, 3, 1, 2);
        auto a = simd::gather<float>(mem, index);
        //EXPECT_TRUE(simd::all_of(p == a));
    }
    {
        simd::Vec<float, 4> index(0, 1, 3, 2);
        float mem[] = { 4, 3, 2, 1 };
        simd::Vec<float, 4> p(4, 3, 1, 2);
        auto a = simd::gather(mem, index);
        //EXPECT_TRUE(simd::all_of(p == a));
    }
    {
        simd::Vec<double, 4> index(0, 1, 3, 2);
        double mem[] = { 4, 3, 2, 1 };
        simd::Vec<double, 4> p(4, 3, 1, 2);
        auto a = simd::gather(mem, index);
        //EXPECT_TRUE(simd::all_of(p == a));
    }
}
