#include <gtest/gtest.h>

#include "simd/simd.h"

STATIC_CHECK_ARCH_ENABLED(AVX);

TEST(vec_op_avx, test_memory_clear)
{
    {
        simd::vi32x8_t a(1);
        a.clear();
        for (auto i = 0; i < a.size(); i++) {
            EXPECT_EQ(0, a[i]);
        }
    }
    {
        simd::vf32x8_t a(1);
        a.clear();
        for (auto i = 0; i < a.size(); i++) {
            EXPECT_FLOAT_EQ(0, a[i]);
        }
    }
    {
        simd::vf64x4_t a(1);
        a.clear();
        for (auto i = 0; i < a.size(); i++) {
            EXPECT_FLOAT_EQ(0, a[i]);
        }
    }
}

TEST(vec_op_avx, test_memory_load_aligned)
{
    {
        alignas(32) int32_t pa[] = { 1, 1, 1, 1, 1, 1, 1, 1 };
        auto a = simd::vi32x4_t::load_aligned(pa);
        simd::vi32x4_t b(1);
        EXPECT_TRUE(simd::all_of(a == b));
    }
    {
        alignas(32) float pa[] = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f };
        auto a = simd::vf32x8_t::load_aligned(pa);
        simd::vf32x8_t b(1.f);
        EXPECT_TRUE(simd::all_of(b == a));
    }
    {
        alignas(32) double pa[] = { 1.f, 1.f, 1.f, 1.f };
        auto a = simd::vf64x4_t::load_aligned(pa);
        simd::vf64x4_t b(1.0);
        EXPECT_TRUE(simd::all_of(b == a));
    }
}

TEST(vec_op_avx, test_memory_load_unaligned)
{
    {
        int32_t pa[] = { 1, 1, 1, 1, 1, 1, 1, 1 };
        auto a = simd::vi32x8_t::load_unaligned(pa);
        simd::vi32x8_t b(1);
        EXPECT_TRUE(simd::all_of(a == b));
    }
    {
        float pa[] = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f };
        auto a = simd::vf32x8_t::load_unaligned(pa);
        simd::vf32x8_t b(1.f);
        EXPECT_TRUE(simd::all_of(b == a));
    }
    {
        double pa[] = { 1.0, 1.0, 1.0, 1.0 };
        auto a = simd::vf64x4_t::load_unaligned(pa);
        simd::vf64x4_t b(1.0);
        EXPECT_TRUE(simd::all_of(b == a));
    }
}

TEST(vec_op_avx, test_memory_store_aligned)
{
    {
        alignas(32) int32_t pa[] = { 1, 1, 1, 1, 1, 1, 1, 1 };
        simd::vi32x8_t a(2);
        a.store_aligned(pa);
        auto b = simd::vi32x8_t::load_aligned(pa);
        EXPECT_TRUE(simd::all_of(a == b));
    }
    {
        alignas(32) float pa[] = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f };
        simd::vf32x8_t a(2.f);
        a.store_aligned(pa);
        auto b = simd::vf32x8_t::load_aligned(pa);
        EXPECT_TRUE(simd::all_of(b == a));
    }
    {
        alignas(32) double pa[] = { 1.0, 1.0, 1.0, 1.0 };
        simd::vf64x4_t a(2.0);
        a.store_aligned(pa);
        auto b = simd::vf64x4_t::load_aligned(pa);
        EXPECT_TRUE(simd::all_of(b == a));
    }
}

TEST(vec_op_avx, test_memory_store_unaligned)
{
    {
        int32_t pa[] = { 1, 1, 1, 1, 1, 1, 1, 1 };
        simd::vi32x8_t a(2);
        a.store_unaligned(pa);
        auto b = simd::vi32x8_t::load_aligned(pa);
        EXPECT_TRUE(simd::all_of(a == b));
    }
    {
        float pa[] = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f };
        simd::vf32x8_t a(2.f);
        a.store_unaligned(pa);
        auto b = simd::vf32x8_t::load_unaligned(pa);
        EXPECT_TRUE(simd::all_of(b == a));
    }
    {
        double pa[] = { 1.0, 1.0, 1.0, 1.0 };
        simd::vf64x4_t a(2.0);
        a.store_unaligned(pa);
        auto b = simd::vf64x4_t::load_unaligned(pa);
        EXPECT_TRUE(simd::all_of(b == a));
    }
}

TEST(vec_op_avx, test_memory_set)
{
    {
        simd::vi64x4_t a([](int i) { return i; });
        int64_t pa[4];
        std::iota(std::begin(pa), std::end(pa), 0);
        auto b = simd::vi64x4_t::load_unaligned(pa);
        EXPECT_TRUE(simd::all_of(a == b));
        auto c = simd::set<int64_t, 4>(
            pa[0], pa[1], pa[2], pa[3]);
        EXPECT_TRUE(simd::all_of(a == c));
    }
    {
        simd::vi32x8_t a([](int i) { return i; });
        int32_t pa[8];
        std::iota(std::begin(pa), std::end(pa), 0);
        auto b = simd::vi32x8_t::load_unaligned(pa);
        EXPECT_TRUE(simd::all_of(a == b));
        auto c = simd::set<int32_t, 8>(
            pa[0], pa[1], pa[2], pa[3], pa[4], pa[5], pa[6], pa[7]);
        EXPECT_TRUE(simd::all_of(a == c));
    }
    {
        simd::vi16x16_t a([](int i) { return i; });
        int16_t pa[16];
        std::iota(std::begin(pa), std::end(pa), 0);
        auto b = simd::vi16x16_t::load_unaligned(pa);
        EXPECT_TRUE(simd::all_of(a == b));
        auto c = simd::set<int16_t, 16>(
            pa[0], pa[1], pa[2],  pa[3],  pa[4],  pa[5],  pa[6],  pa[7],
            pa[8], pa[9], pa[10], pa[11], pa[12], pa[13], pa[14], pa[15]);
        EXPECT_TRUE(simd::all_of(a == c));
    }
    {
        simd::vi8x32_t a([](int i) { return i; });
        int8_t pa[32];
        std::iota(std::begin(pa), std::end(pa), 0);
        auto b = simd::vi8x32_t::load_unaligned(pa);
        EXPECT_TRUE(simd::all_of(a == b));
        auto c = simd::set<int8_t, 32>(
            pa[0], pa[1], pa[2],  pa[3],  pa[4],  pa[5],  pa[6],  pa[7],
            pa[8], pa[9], pa[10], pa[11], pa[12], pa[13], pa[14], pa[15],
            pa[16], pa[17], pa[18], pa[19], pa[20], pa[21], pa[22], pa[23],
            pa[24], pa[25], pa[26], pa[27], pa[28], pa[29], pa[30], pa[31]);
        EXPECT_TRUE(simd::all_of(a == c));
    }
    {
        simd::vf32x8_t a([](int i) { return i; });
        float pa[8];
        std::iota(std::begin(pa), std::end(pa), 0);
        auto b = simd::vf32x8_t::load_unaligned(pa);
        EXPECT_TRUE(simd::all_of(a == b));
        auto c = simd::set<float, 8>(
            pa[0], pa[1], pa[2], pa[3], pa[4], pa[5], pa[6], pa[7]);
        EXPECT_TRUE(simd::all_of(a == c));
    }
    {
        simd::vf64x4_t a([](int i) { return i; });
        double pa[4];
        std::iota(std::begin(pa), std::end(pa), 0);
        auto b = simd::vf64x4_t::load_unaligned(pa);
        EXPECT_TRUE(simd::all_of(a == b));
        auto c = simd::set<double, 4>(
            pa[0], pa[1], pa[2], pa[3]);
        EXPECT_TRUE(simd::all_of(a == c));
    }
}

TEST(vec_op_avx, test_memory_gather)
{
    {
        simd::vi32x8_t index(0, 1, 3, 2, 0, 1, 3, 2 );
        int32_t mem[] = { 4, 3, 2, 1 };
        simd::vi32x8_t p(4, 3, 1, 2, 4, 3, 1, 2);
        auto a = simd::gather(mem, index);
        //EXPECT_TRUE(simd::all_of(p == a)) << a;
    }
    {
        simd::vi32x8_t index(0, 1, 3, 2, 0, 1, 3, 2);
        int32_t mem[] = { 4, 3, 2, 1 };
        simd::vf32x8_t p(4, 3, 1, 2, 4, 3, 1, 2);
        auto a = simd::gather<float>(mem, index);
        //EXPECT_TRUE(simd::all_of(p == a));
    }
    {
        simd::vf32x8_t index(0, 1, 3, 2, 0, 1, 3, 2);
        float mem[] = { 4, 3, 2, 1 };
        simd::vf32x8_t p(4, 3, 1, 2, 4, 3, 1, 2);
        auto a = simd::gather(mem, index);
        //EXPECT_TRUE(simd::all_of(p == a));
    }
    {
        simd::vf64x4_t index(0, 1, 3, 2);
        double mem[] = { 4, 3, 2, 1 };
        simd::vf64x4_t p(4, 3, 1, 2);
        auto a = simd::gather(mem, index);
        //EXPECT_TRUE(simd::all_of(p == a));
    }
}
