#include <gtest/gtest.h>

#include "simd/simd.h"
#include "check_arch.h"

using namespace simd;

TEST(vec_complex_avx, test_arith_add)
{
    {
        simd::Vec<cf32_t, 4> a(cf32_t(1.f, 2.f)), b(cf32_t(2.f, 3.f)),
                             p(cf32_t(1.f + 2.f, 2.f + 3.f));
        auto c = a + b;
        for (auto i = 0; i < 4; i++) {
            EXPECT_FLOAT_EQ(p[i].real(), c[i].real());
            EXPECT_FLOAT_EQ(p[i].imag(), c[i].imag());
        }
    }
    {
        simd::Vec<cf64_t, 2> a(cf64_t(1.5, 2.3)), b(cf64_t(2.1, 3.0)),
                             p(cf64_t(1.5 + 2.1, 2.3 + 3.0));
        auto c = a + b;
        for (auto i = 0; i < 2; i++) {
            EXPECT_FLOAT_EQ(p[i].real(), c[i].real());
            EXPECT_FLOAT_EQ(p[i].imag(), c[i].imag());
        }
    }
    {
        simd::Vec<cf64_t, 4> a(cf64_t(1.5, 2.3)), b(cf64_t(2.1, 3.0)),
                             p(cf64_t(1.5 + 2.1, 2.3 + 3.0));
        auto c = a + b;
        for (auto i = 0; i < 4; i++) {
            EXPECT_FLOAT_EQ(p[i].real(), c[i].real());
            EXPECT_FLOAT_EQ(p[i].imag(), c[i].imag());
        }
    }
}

TEST(vec_complex_avx, test_arith_sub)
{
    {
        simd::Vec<cf32_t, 4> a(cf32_t(1.f, 10.f)), b(cf32_t(2.f, 3.f)),
                             p(cf32_t(1.f - 2.f, 10.f - 3.f));
        auto c = a - b;
        for (auto i = 0; i < 4; i++) {
            EXPECT_FLOAT_EQ(p[i].real(), c[i].real());
            EXPECT_FLOAT_EQ(p[i].imag(), c[i].imag());
        }
    }
    {
        simd::Vec<cf64_t, 2> a(cf64_t(1.5, 2.3)), b(cf64_t(2.1, 3.0)),
                             p(cf64_t(1.5 - 2.1, 2.3 - 3.0));
        auto c = a - b;
        for (auto i = 0; i < 2; i++) {
            EXPECT_FLOAT_EQ(p[i].real(), c[i].real());
            EXPECT_FLOAT_EQ(p[i].imag(), c[i].imag());
        }
    }
    {
        simd::Vec<cf64_t, 4> a(cf64_t(1.5, 2.3)), b(cf64_t(2.1, 3.0)),
                             p(cf64_t(1.5 - 2.1, 2.3 - 3.0));
        auto c = a - b;
        for (auto i = 0; i < 4; i++) {
            EXPECT_FLOAT_EQ(p[i].real(), c[i].real());
            EXPECT_FLOAT_EQ(p[i].imag(), c[i].imag());
        }
    }
}
