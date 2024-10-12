#include <gtest/gtest.h>

#include "simd/simd.h"
#include "check_arch.h"

#include <algorithm>

#include <vector>

using namespace simd;

TEST(vec_complex_sse, test_ctor_multiple_complex)
{
    {
        std::vector<cf32_t> p = {
            cf32_t(1.f, 2.f),
            cf32_t(2.f, 3.f),
            cf32_t(3.f, 4.f),
            cf32_t(4.f, 5.f)
        };
        simd::Vec<cf32_t, 4> a(p[0], p[1], p[2], p[3]);
        for (auto i = 0; i < a.size(); i++) {
            EXPECT_FLOAT_EQ(p[i].real(), a[i].real());
            EXPECT_FLOAT_EQ(p[i].imag(), a[i].imag());
        }
    }
}

TEST(vec_complex_sse, test_ctor_generator)
{
    {
        std::vector<cf32_t> p = {
            cf32_t(1.f, 2.f),
            cf32_t(2.f, 3.f),
            cf32_t(3.f, 4.f),
            cf32_t(4.f, 5.f)
        };
        simd::Vec<cf32_t, 4> a([&p](int idx) { return p[idx]; });
        for (auto i = 0; i < a.size(); i++) {
            EXPECT_FLOAT_EQ(p[i].real(), a[i].real());
            EXPECT_FLOAT_EQ(p[i].imag(), a[i].imag());
        }
    }
    {  // TODO:
        std::vector<cf64_t> p = {
            cf64_t(1.0, 2.0),
            cf64_t(2.0, 3.0),
            cf64_t(3.0, 4.0),
            cf64_t(4.0, 5.0)
        };
        simd::Vec<cf64_t, 4> a([&p](int idx) { return p[idx]; });
        for (auto i = 0; i < a.size(); i++) {
            EXPECT_FLOAT_EQ(p[i].real(), a[i].real());
            EXPECT_FLOAT_EQ(p[i].imag(), a[i].imag());
        }
    }
}

TEST(vec_complex_sse, test_arith_add)
{
    {
        cf32_t x(1.f, 2.f), y(2.f, 3.f);
        simd::Vec<cf32_t, 4> a(x), b(y), p(x + y);
        auto c = a + b;
        for (auto i = 0; i < c.size(); i++) {
            EXPECT_FLOAT_EQ(p.real()[i], c.real()[i]);
            EXPECT_FLOAT_EQ(p.imag()[i], c.imag()[i]);
        }
    }
    {
        cf64_t x(1.5, 2.3), y(2.1, 3.0);
        simd::Vec<cf64_t, 2> a(x), b(y), p(x + y);
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
}

TEST(vec_complex_sse, test_arith_sub)
{
    {
        cf32_t x(1.f, 2.f), y(2.f, 3.f);
        simd::Vec<cf32_t, 4> a(x), b(y), p(x - y);
        auto c = a - b;
        for (auto i = 0; i < c.size(); i++) {
            EXPECT_FLOAT_EQ(p.real()[i], c.real()[i]);
            EXPECT_FLOAT_EQ(p.imag()[i], c.imag()[i]);
        }
    }
    {
        cf64_t x(1.5, 2.3), y(2.1, 3.0);
        simd::Vec<cf64_t, 2> a(x), b(y), p(x - y);
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
}

TEST(vec_complex_sse, test_arith_mul)
{
    {
        cf32_t x(1.f, 2.f), y(2.f, 3.f);
        simd::Vec<cf32_t, 4> a(x), b(y), p(x * y);
        auto c = a * b;
        for (auto i = 0; i < c.size(); i++) {
            EXPECT_FLOAT_EQ(p.real()[i], c.real()[i]);
            EXPECT_FLOAT_EQ(p.imag()[i], c.imag()[i]);
        }
    }
    {
        cf64_t x(1.5, 2.3), y(2.1, 3.0);
        simd::Vec<cf64_t, 2> a(x), b(y), p(x * y);
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
}

TEST(vec_complex_sse, test_arith_div)
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

TEST(vec_complex_sse, test_memory_load_aligned)
{
    {
        alignas(16) float mem[4]{};
        std::iota(mem, mem + 4, 1.f);
        auto a = simd::Vec<cf32_t, 4>::load_aligned(mem, nullptr);
        for (auto i = 0; i < a.size(); i++) {
            EXPECT_FLOAT_EQ(mem[i], a[i].real());
            EXPECT_FLOAT_EQ(0.f, a[i].imag());
        }
    }
    {
        alignas(16) float rmem[4]{};
        std::iota(rmem, rmem + 4, 1.f);
        alignas(32) float imem[4]{};
        std::iota(imem, imem + 4, 5.f);
        auto a = simd::Vec<cf32_t, 4>::load_aligned(rmem, imem);
        for (auto i = 0; i < a.size(); i++) {
            EXPECT_FLOAT_EQ(rmem[i], a[i].real());
            EXPECT_FLOAT_EQ(imem[i], a[i].imag());
        }
    }
    {
        alignas(16) float mem[8]{};
        std::iota(mem, mem + 8, 1.f);
        auto a = simd::Vec<cf32_t, 4>::load(mem, simd::aligned_mode{});
        for (auto i = 0; i < a.size(); i++) {
            EXPECT_FLOAT_EQ(mem[2*i],   a[i].real());
            EXPECT_FLOAT_EQ(mem[2*i+1], a[i].imag());
        }
    }
    {
        alignas(16) double mem[4]{};
        std::iota(mem, mem + 4, 1.0);
        auto a = simd::Vec<cf64_t, 2>::load(mem, simd::aligned_mode{});
        for (auto i = 0; i < a.size(); i++) {
            EXPECT_FLOAT_EQ(mem[2*i],   a[i].real());
            EXPECT_FLOAT_EQ(mem[2*i+1], a[i].imag());
        }
    }
}

TEST(vec_complex_sse, test_memory_load_unaligned)
{
    {
        float mem[4]{};
        std::iota(mem, mem + 4, 1.f);
        auto a = simd::Vec<cf32_t, 4>::load_unaligned(mem, nullptr);
        for (auto i = 0; i < a.size(); i++) {
            EXPECT_FLOAT_EQ(mem[i], a[i].real());
            EXPECT_FLOAT_EQ(0.f, a[i].imag());
        }
    }
    {
        float rmem[4]{};
        std::iota(rmem, rmem + 4, 1.f);
        float imem[8]{};
        std::iota(imem, imem + 4, 5.f);
        auto a = simd::Vec<cf32_t, 4>::load_unaligned(rmem, imem);
        for (auto i = 0; i < a.size(); i++) {
            EXPECT_FLOAT_EQ(rmem[i], a[i].real());
            EXPECT_FLOAT_EQ(imem[i], a[i].imag());
        }
    }
    {
        float mem[8]{};
        std::iota(mem, mem + 8, 1.f);
        auto a = simd::Vec<cf32_t, 4>::load(mem, simd::unaligned_mode{});
        for (auto i = 0; i < a.size(); i++) {
            EXPECT_FLOAT_EQ(mem[2*i],   a[i].real());
            EXPECT_FLOAT_EQ(mem[2*i+1], a[i].imag());
        }
    }
    {
        double mem[4]{};
        std::iota(mem, mem + 4, 1.0);
        auto a = simd::Vec<cf64_t, 2>::load(mem, simd::unaligned_mode{});
        for (auto i = 0; i < a.size(); i++) {
            EXPECT_FLOAT_EQ(mem[2*i],   a[i].real());
            EXPECT_FLOAT_EQ(mem[2*i+1], a[i].imag());
        }
    }
}

TEST(vec_complex_sse, test_memory_store_aligned)
{
    {
        alignas(16) float mem[4];
        std::iota(mem, mem + 4, 1.f);
        auto a = simd::Vec<cf32_t, 4>::load_aligned(mem, nullptr);
        alignas(16) float mem1[4] = { 0.f };
        a.store_aligned(mem1, nullptr);
        for (auto i = 0; i < a.size(); i++) {
            EXPECT_FLOAT_EQ(mem[i], mem1[i]);
        }
    }
    {
        alignas(16) float rmem[4]{};
        std::iota(rmem, rmem + 4, 1.f);
        alignas(16) float imem[4]{};
        std::iota(imem, imem + 4, 5.f);
        auto a = simd::Vec<cf32_t, 4>::load_aligned(rmem, imem);
        alignas(16) float rmem1[4] = { 0.f };
        alignas(16) float imem1[4] = { 0.f };
        a.store_aligned(rmem1, imem1);
        for (auto i = 0; i < a.size(); i++) {
            EXPECT_FLOAT_EQ(rmem[i], rmem1[i]);
            EXPECT_FLOAT_EQ(imem[i], imem1[i]);
        }
    }
    {
        alignas(16) float mem[8]{};
        std::iota(mem, mem + 8, 1.f);
        auto a = simd::Vec<cf32_t, 4>::load(mem, simd::aligned_mode{});
        alignas(16) float mem1[8] = { 0.f };
        a.store_aligned((cf32_t*)mem1);
        for (auto i = 0; i < 2 * a.size(); i++) {
            EXPECT_FLOAT_EQ(mem[i], mem1[i]);
        }
    }
    {
        alignas(16) double mem[4]{};
        std::iota(mem, mem + 4, 1.f);
        auto a = simd::Vec<cf64_t, 2>::load(mem, simd::aligned_mode{});
        alignas(16) double mem1[4] = { 0.0 };
        a.store_aligned((cf64_t*)mem1);
        for (auto i = 0; i < 2 * a.size(); i++) {
            EXPECT_FLOAT_EQ(mem[i], mem1[i]);
        }
    }
}

TEST(vec_complex_sse, test_memory_store_unaligned)
{
    {
        float mem[4]{};
        std::iota(mem, mem + 4, 1.f);
        auto a = simd::Vec<cf32_t, 4>::load_unaligned(mem, nullptr);
        float mem1[4] = { 0.f };
        a.store_unaligned(mem1, nullptr);
        for (auto i = 0; i < a.size(); i++) {
            EXPECT_FLOAT_EQ(mem[i], mem1[i]);
        }
    }
    {
        float rmem[4]{};
        std::iota(rmem, rmem + 4, 1.f);
        float imem[4]{};
        std::iota(imem, imem + 4, 5.f);
        auto a = simd::Vec<cf32_t, 4>::load_unaligned(rmem, imem);
        float rmem1[4] = { 0.f };
        float imem1[4] = { 0.f };
        a.store_unaligned(rmem1, imem1);
        for (auto i = 0; i < a.size(); i++) {
            EXPECT_FLOAT_EQ(rmem[i], rmem1[i]);
            EXPECT_FLOAT_EQ(imem[i], imem1[i]);
        }
    }
    {
        float mem[8]{};
        std::iota(mem, mem + 8, 1.f);
        auto a = simd::Vec<cf32_t, 4>::load(mem, simd::unaligned_mode{});
        float mem1[8] = { 0.f };
        a.store_unaligned((cf32_t*)mem1);
        for (auto i = 0; i < 2 * a.size(); i++) {
            EXPECT_FLOAT_EQ(mem[i], mem1[i]);
        }
    }
    {
        double mem[4]{};
        std::iota(mem, mem + 4, 1.0);
        auto a = simd::Vec<cf64_t, 2>::load(mem, simd::unaligned_mode{});
        double mem1[4] = { 0.0 };
        a.store_unaligned((cf64_t*)mem1);
        for (auto i = 0; i < 2 * a.size(); i++) {
            EXPECT_FLOAT_EQ(mem[i], mem1[i]);
        }
    }
}
