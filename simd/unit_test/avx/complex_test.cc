#include <gtest/gtest.h>

#include "simd/simd.h"
#include "check_arch.h"

#include <algorithm>

using namespace simd;

TEST(vec_complex_avx, test_arith_add)
{
    {
        cf32_t x(1.f, 2.f), y(2.f, 3.f);
        simd::Vec<cf32_t, 8> a(x), b(y), p(x + y);
        auto c = a + b;
        for (auto i = 0; i < c.size(); i++) {
            EXPECT_FLOAT_EQ(p.real()[i], c.real()[i]);
            EXPECT_FLOAT_EQ(p.imag()[i], c.imag()[i]);
        }
    }
    {
        cf64_t x(1.5, 2.3), y(2.1, 3.0);
        simd::Vec<cf64_t, 4> a(x), b(y), p(x + y);
        auto c = a + b;
        for (auto i = 0; i < c.size(); i++) {
            EXPECT_FLOAT_EQ(p.real()[i], c.real()[i]);
            EXPECT_FLOAT_EQ(p.imag()[i], c.imag()[i]);
        }
    }
    {
        cf64_t x(1.5, 2.3), y(2.1, 3.0);
        simd::Vec<cf64_t, 8> a(x), b(y), p(x + y);
        auto c = a + b;
        for (auto i = 0; i < c.size(); i++) {
            EXPECT_FLOAT_EQ(p.real()[i], c.real()[i]);
            EXPECT_FLOAT_EQ(p.imag()[i], c.imag()[i]);
        }
    }
}

TEST(vec_complex_avx, test_arith_sub)
{
    {
        cf32_t x(1.f, 2.f), y(2.f, 3.f);
        simd::Vec<cf32_t, 8> a(x), b(y), p(x - y);
        auto c = a - b;
        for (auto i = 0; i < c.size(); i++) {
            EXPECT_FLOAT_EQ(p.real()[i], c.real()[i]);
            EXPECT_FLOAT_EQ(p.imag()[i], c.imag()[i]);
        }
    }
    {
        cf64_t x(1.5, 2.3), y(2.1, 3.0);
        simd::Vec<cf64_t, 4> a(x), b(y), p(x - y);
        auto c = a - b;
        for (auto i = 0; i < c.size(); i++) {
            EXPECT_FLOAT_EQ(p.real()[i], c.real()[i]);
            EXPECT_FLOAT_EQ(p.imag()[i], c.imag()[i]);
        }
    }
    {
        cf64_t x(1.5, 2.3), y(2.1, 3.0);
        simd::Vec<cf64_t, 8> a(x), b(y), p(x - y);
        auto c = a - b;
        for (auto i = 0; i < c.size(); i++) {
            EXPECT_FLOAT_EQ(p.real()[i], c.real()[i]);
            EXPECT_FLOAT_EQ(p.imag()[i], c.imag()[i]);
        }
    }
}

TEST(vec_complex_avx, test_arith_mul)
{
    {
        cf32_t x(1.f, 2.f), y(2.f, 3.f);
        simd::Vec<cf32_t, 8> a(x), b(y), p(x * y);
        auto c = a * b;
        for (auto i = 0; i < c.size(); i++) {
            EXPECT_FLOAT_EQ(p.real()[i], c.real()[i]);
            EXPECT_FLOAT_EQ(p.imag()[i], c.imag()[i]);
        }
    }
    {
        cf64_t x(1.5, 2.3), y(2.1, 3.0);
        simd::Vec<cf64_t, 4> a(x), b(y), p(x * y);
        auto c = a * b;
        for (auto i = 0; i < c.size(); i++) {
            EXPECT_FLOAT_EQ(p.real()[i], c.real()[i]);
            EXPECT_FLOAT_EQ(p.imag()[i], c.imag()[i]);
        }
    }
    {
        cf64_t x(1.5, 2.3), y(2.1, 3.0);
        simd::Vec<cf64_t, 8> a(x), b(y), p(x * y);
        auto c = a * b;
        for (auto i = 0; i < c.size(); i++) {
            EXPECT_FLOAT_EQ(p.real()[i], c.real()[i]);
            EXPECT_FLOAT_EQ(p.imag()[i], c.imag()[i]);
        }
    }
}

TEST(vec_complex_avx, test_arith_div)
{
    {
        cf32_t x(1.f, 2.f), y(2.f, 3.f);
        simd::Vec<cf32_t, 4> a(x), b(y), p(x / y);
        auto c = a / b;
        for (auto i = 0; i < c.size(); i++) {
            EXPECT_FLOAT_EQ(p.real()[i], c.real()[i]);
            EXPECT_FLOAT_EQ(p.imag()[i], c.imag()[i]);
        }
    }
    {
        cf64_t x(1.5, 2.3), y(2.1, 3.0);
        simd::Vec<cf64_t, 2> a(x), b(y), p(x / y);
        auto c = a / b;
        for (auto i = 0; i < c.size(); i++) {
            EXPECT_FLOAT_EQ(p.real()[i], c.real()[i]);
            EXPECT_FLOAT_EQ(p.imag()[i], c.imag()[i]);
        }
    }
    {
        cf64_t x(1.5, 2.3), y(2.1, 3.0);
        simd::Vec<cf64_t, 4> a(x), b(y), p(x / y);
        auto c = a / b;
        for (auto i = 0; i < c.size(); i++) {
            EXPECT_FLOAT_EQ(p.real()[i], c.real()[i]);
            EXPECT_FLOAT_EQ(p.imag()[i], c.imag()[i]);
        }
    }
}

TEST(vec_complex_avx, test_memory_load_aligned)
{
    {
        alignas(32) float mem[8]{};
        std::iota(mem, mem + 8, 1.f);
        auto a = simd::Vec<cf32_t, 8>::load_aligned(mem, nullptr);
        for (auto i = 0; i < a.size(); i++) {
            EXPECT_FLOAT_EQ(mem[i], a[i].real());
            EXPECT_FLOAT_EQ(0.f, a[i].imag());
        }
    }
    {
        alignas(32) float rmem[8]{};
        std::iota(rmem, rmem + 8, 1.f);
        alignas(32) float imem[8]{};
        std::iota(imem, imem + 8, 9.f);
        auto a = simd::Vec<cf32_t, 8>::load_aligned(rmem, imem);
        for (auto i = 0; i < a.size(); i++) {
            EXPECT_FLOAT_EQ(rmem[i], a[i].real());
            EXPECT_FLOAT_EQ(imem[i], a[i].imag());
        }
    }
    {
        alignas(32) float mem[16]{};
        std::iota(mem, mem + 16, 1.f);
        auto a = simd::Vec<cf32_t, 8>::load(mem, simd::aligned_mode{});
        for (auto i = 0; i < a.size(); i++) {
            EXPECT_FLOAT_EQ(mem[2*i],   a[i].real());
            EXPECT_FLOAT_EQ(mem[2*i+1], a[i].imag());
        }
    }
    {
        alignas(32) double mem[8]{};
        std::iota(mem, mem + 8, 1.0);
        auto a = simd::Vec<cf64_t, 4>::load(mem, simd::aligned_mode{});
        for (auto i = 0; i < a.size(); i++) {
            EXPECT_FLOAT_EQ(mem[2*i],   a[i].real());
            EXPECT_FLOAT_EQ(mem[2*i+1], a[i].imag());
        }
    }
}

TEST(vec_complex_avx, test_memory_load_unaligned)
{
    {
        float mem[8]{};
        std::iota(mem, mem + 8, 1.f);
        auto a = simd::Vec<cf32_t, 8>::load_unaligned(mem, nullptr);
        for (auto i = 0; i < a.size(); i++) {
            EXPECT_FLOAT_EQ(mem[i], a[i].real());
            EXPECT_FLOAT_EQ(0.f, a[i].imag());
        }
    }
    {
        float rmem[8]{};
        std::iota(rmem, rmem + 8, 1.f);
        float imem[8]{};
        std::iota(imem, imem + 8, 9.f);
        auto a = simd::Vec<cf32_t, 8>::load_unaligned(rmem, imem);
        for (auto i = 0; i < a.size(); i++) {
            EXPECT_FLOAT_EQ(rmem[i], a[i].real());
            EXPECT_FLOAT_EQ(imem[i], a[i].imag());
        }
    }
    {
        float mem[16]{};
        std::iota(mem, mem + 16, 1.f);
        auto a = simd::Vec<cf32_t, 8>::load(mem, simd::unaligned_mode{});
        for (auto i = 0; i < a.size(); i++) {
            EXPECT_FLOAT_EQ(mem[2*i],   a[i].real());
            EXPECT_FLOAT_EQ(mem[2*i+1], a[i].imag());
        }
    }
    {
        double mem[8]{};
        std::iota(mem, mem + 8, 1.0);
        auto a = simd::Vec<cf64_t, 4>::load(mem, simd::unaligned_mode{});
        for (auto i = 0; i < a.size(); i++) {
            EXPECT_FLOAT_EQ(mem[2*i],   a[i].real());
            EXPECT_FLOAT_EQ(mem[2*i+1], a[i].imag());
        }
    }
}

TEST(vec_complex_avx, test_memory_store_aligned)
{
    {
        alignas(32) float mem[8]{};
        std::iota(mem, mem + 8, 1.f);
        auto a = simd::Vec<cf32_t, 8>::load_aligned(mem, nullptr);
        alignas(32) float mem1[8] = { 0.f };
        a.store_aligned(mem1, nullptr);
        for (auto i = 0; i < a.size(); i++) {
            EXPECT_FLOAT_EQ(mem[i], mem1[i]);
        }
    }
    {
        alignas(32) float rmem[8]{};
        std::iota(rmem, rmem + 8, 1.f);
        alignas(32) float imem[8]{};
        std::iota(imem, imem + 8, 9.f);
        auto a = simd::Vec<cf32_t, 8>::load_aligned(rmem, imem);
        alignas(32) float rmem1[8] = { 0.f };
        alignas(32) float imem1[8] = { 0.f };
        a.store_aligned(rmem1, imem1);
        for (auto i = 0; i < a.size(); i++) {
            EXPECT_FLOAT_EQ(rmem[i], rmem1[i]);
            EXPECT_FLOAT_EQ(imem[i], imem1[i]);
        }
    }
    {
        alignas(32) float mem[16]{};
        std::iota(mem, mem + 16, 1.f);
        auto a = simd::Vec<cf32_t, 8>::load(mem, simd::aligned_mode{});
        alignas(32) float mem1[16] = { 0.f };
        a.store_aligned((cf32_t*)mem1);
        for (auto i = 0; i < 2 * a.size(); i++) {
            EXPECT_FLOAT_EQ(mem[i], mem1[i]);
        }
    }
    {
        alignas(32) double mem[8]{};
        std::iota(mem, mem + 8, 1.f);
        auto a = simd::Vec<cf64_t, 4>::load(mem, simd::aligned_mode{});
        alignas(32) double mem1[8] = { 0.0 };
        a.store_aligned((cf64_t*)mem1);
        for (auto i = 0; i < 2 * a.size(); i++) {
            EXPECT_FLOAT_EQ(mem[i], mem1[i]);
        }
    }
}

TEST(vec_complex_avx, test_memory_store_unaligned)
{
    {
        float mem[8]{};
        std::iota(mem, mem + 8, 1.f);
        auto a = simd::Vec<cf32_t, 8>::load_unaligned(mem, nullptr);
        float mem1[8] = { 0.f };
        a.store_unaligned(mem1, nullptr);
        for (auto i = 0; i < a.size(); i++) {
            EXPECT_FLOAT_EQ(mem[i], mem1[i]);
        }
    }
    {
        float rmem[8]{};
        std::iota(rmem, rmem + 8, 1.f);
        float imem[8]{};
        std::iota(imem, imem + 8, 9.f);
        auto a = simd::Vec<cf32_t, 8>::load_unaligned(rmem, imem);
        float rmem1[8] = { 0.f };
        float imem1[8] = { 0.f };
        a.store_unaligned(rmem1, imem1);
        for (auto i = 0; i < a.size(); i++) {
            EXPECT_FLOAT_EQ(rmem[i], rmem1[i]);
            EXPECT_FLOAT_EQ(imem[i], imem1[i]);
        }
    }
    {
        float mem[16]{};
        std::iota(mem, mem + 16, 1.f);
        auto a = simd::Vec<cf32_t, 8>::load(mem, simd::unaligned_mode{});
        float mem1[16] = { 0.f };
        a.store_unaligned((cf32_t*)mem1);
        for (auto i = 0; i < 2 * a.size(); i++) {
            EXPECT_FLOAT_EQ(mem[i], mem1[i]);
        }
    }
    {
        double mem[8]{};
        std::iota(mem, mem + 8, 1.0);
        auto a = simd::Vec<cf64_t, 4>::load(mem, simd::unaligned_mode{});
        double mem1[8] = { 0.0 };
        a.store_unaligned((cf64_t*)mem1);
        for (auto i = 0; i < 2 * a.size(); i++) {
            EXPECT_FLOAT_EQ(mem[i], mem1[i]);
        }
    }
}
