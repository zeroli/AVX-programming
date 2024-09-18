#include <gtest/gtest.h>

#include "simd/simd.h"

TEST(vec_op_sse, test_arith_add)
{
    {
        simd::Vec<int32_t, 4> a(1), b(2), p(3);
        auto c = a + b;
        std::cout << a << "(a)" << " + " << b << "(b) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
        auto d = simd::add(a, b);
        EXPECT_TRUE(simd::all(p == d));
    }
    {
        simd::Vec<int32_t, 4> a(1), p(3);
        int b = 2;
        auto c = a + b;
        std::cout << a << "(a)" << " + " << b << "(b) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
        auto d = simd::add(a, b);
        EXPECT_TRUE(simd::all(p == d));
    }
    {
        simd::Vec<int32_t, 4> a(1), p(3);
        int b = 2;
        auto c = b + a;
        std::cout << b << "(b)" << " + " << a << "() = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
        auto d = simd::add(b, a);
        EXPECT_TRUE(simd::all(p == d));
    }
    {
        simd::Vec<float, 4> a(1.f), b(2.f), p(3.f);
        auto c = a + b;
        EXPECT_TRUE(simd::all(p == c));
        auto d = a + 2.f;
        EXPECT_TRUE(simd::all(p == d));
        std::cout << a << "(a)" << " + " << b << "(b) = " << c << "(c)\n";
        auto e = simd::add(a, b);
        EXPECT_TRUE(simd::all(p == e));
        auto f = simd::add(a, 2.f);
        EXPECT_TRUE(simd::all(p == f));
        auto g = simd::add(1.f, b);
        EXPECT_TRUE(simd::all(p == g));
    }
    {
        simd::Vec<double, 2> a(1.0), b(2.0), p(3.);
        auto c = a + b;
        EXPECT_TRUE(simd::all(p == c));
        auto d = a + 2.0;
        EXPECT_TRUE(simd::all(p == d));
        std::cout << a << "(a)" << " + " << b << "(b) = " << c << "(c)\n";
        auto e = simd::add(a, b);
        EXPECT_TRUE(simd::all(p == e));
        auto f = simd::add(a, 2.0);
        EXPECT_TRUE(simd::all(p == f));
        auto g = simd::add(1.0, b);
        EXPECT_TRUE(simd::all(p == g));
    }
    {
        simd::Vec<std::complex<float>, 2> a(std::complex<float>{1.f, 2.f}), b(std::complex<float>{1.f, 2.f});
        auto c = a + b;
        auto d = simd::add(a, b);
    }
    {
        simd::Vec<std::complex<double>, 1> a(std::complex<double>{1.0, 2.0}), b(std::complex<double>{1.0, 2.0});
        auto c = a + b;
        auto d = simd::add(a, b);
    }
}

TEST(vec_op_sse, test_arith_add_inplace)
{
    {
        simd::Vec<int32_t, 4> a(1), b(2), p(3);
        a += b;
        std::cout << a << "(a)\n";
        EXPECT_TRUE(simd::all(p == a));
    }
    {
        simd::Vec<float, 4> a(1.f), b(2.f), p(3.f);
        a += b;
        std::cout << a << "(a)\n";
        EXPECT_TRUE(simd::all(p == a));
    }
    {
        simd::Vec<double, 2> a(1.0), b(2.0), p(3.0);
        a += b;
        std::cout << a << "(a)\n";
        EXPECT_TRUE(simd::all(p == a)) << a << ", " << p;
    }
}

TEST(vec_op_sse, test_arith_sub)
{
    {
        simd::Vec<int32_t, 4> a(1), b(2), p(-1);
        auto c = a - b;
        std::cout << a << "(a)" << " - " << b << "(b) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
        auto d = simd::sub(a, b);
        EXPECT_TRUE(simd::all(p == d));
    }
    {
        simd::Vec<int32_t, 4> a(1), p(-1);
        int b = 2;
        auto c = a - b;
        std::cout << a << "(a)" << " - " << b << "(b) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
        auto d = simd::sub(a, b);
        EXPECT_TRUE(simd::all(p == d));
    }
    {
        simd::Vec<int32_t, 4> a(1), p(1);
        int b = 2;
        auto c = b - a;
        std::cout << b << "(b)" << " - " << a << "(a) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
        auto d = simd::sub(b, a);
        EXPECT_TRUE(simd::all(p == d));
    }
    {
        simd::Vec<float, 4> a(2.f), b(1.f), p(1.f);
        auto c = a - b;
        std::cout << a << "(a)" << " - " << b << "(b) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
        auto d = simd::sub(a, b);
        EXPECT_TRUE(simd::all(p == d));
        auto e = simd::sub(a, 1.f);
        EXPECT_TRUE(simd::all(p == e));
        auto f = simd::sub(2.f, b);
        EXPECT_TRUE(simd::all(p == f));
    }
    {
        simd::Vec<double, 2> a(2.0), b(1.0), p(1.0);
        auto c = a - b;
        std::cout << a << "(a)" << " - " << b << "(b) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
        auto d = simd::sub(a, b);
        EXPECT_TRUE(simd::all(p == d));
        auto e = simd::sub(a, 1.0);
        EXPECT_TRUE(simd::all(p == e));
        auto f = simd::sub(2.0, b);
        EXPECT_TRUE(simd::all(p == f));
    }
    {
        simd::Vec<std::complex<float>, 2> a(std::complex<float>{1.f, 2.f}), b(std::complex<float>{1.f, 2.f});
        auto c = a - b;
        auto d = simd::sub(a, b);
    }
    {
        simd::Vec<std::complex<double>, 1> a(std::complex<double>{1.0, 2.0}), b(std::complex<double>{1.0, 2.0});
        auto c = a - b;
        auto d = simd::sub(a, b);
    }
}

TEST(vec_op_sse, test_arith_sub_inplace)
{
    {
        simd::Vec<int32_t, 4> a(1), b(2), p(-1);
        a -= b;
        std::cout << a << "(a)\n";
        EXPECT_TRUE(simd::all(p == a));
    }
    {
        simd::Vec<float, 4> a(1.f), b(2.f), p(-1);
        a -= b;
        std::cout << a << "(a)\n";
        EXPECT_TRUE(simd::all(p == a));
    }
    {
        simd::Vec<double, 2> a(1.0), b(2.0), p(-1);
        a -= b;
        std::cout << a << "(a)\n";
        EXPECT_TRUE(simd::all(p == a));
    }
}

TEST(vec_op_sse, test_arith_mul)
{
    {
        simd::Vec<float, 4> a(2.f), b(1.f), p(2.f);
        auto c = a * b;
        std::cout << a << "(a)" << " * " << b << "(b) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
        auto d = simd::mul(a, b);
        EXPECT_TRUE(simd::all(p == d));
        auto e = simd::mul(a, 1.f);
        EXPECT_TRUE(simd::all(p == e));
        auto f = simd::mul(2.f, b);
        EXPECT_TRUE(simd::all(p == f));
    }
    {
        simd::Vec<double, 2> a(2.0), b(1.0), p(2.0);
        auto c = a * b;
        std::cout << a << "(a)" << " * " << b << "(b) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
        auto d = simd::mul(a, b);
        EXPECT_TRUE(simd::all(p == d));
        auto e = simd::mul(a, 1.0);
        EXPECT_TRUE(simd::all(p == e));
        auto f = simd::mul(2.0, b);
        EXPECT_TRUE(simd::all(p == f));
    }
}

TEST(vec_op_sse, test_arithmul_inplace)
{
    {
        simd::Vec<float, 4> a(1.f), b(2.f), p(2.f);
        a *= b;
        std::cout << a << "(a)\n";
        EXPECT_TRUE(simd::all(p == a));
    }
    {
        simd::Vec<double, 2> a(1.0), b(2.0), p(2.0);
        a *= b;
        std::cout << a << "(a)\n";
        EXPECT_TRUE(simd::all(p == a));
    }
}

TEST(vec_op_sse, test_arith_div)
{
    {
        simd::Vec<float, 4> a(4.f), b(2.f), p(2.f);
        auto c = a / b;
        std::cout << a << "(a)" << " / " << b << "(b) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
        auto d = simd::div(a, b);
        EXPECT_TRUE(simd::all(p == d));
        auto e = simd::div(a, 2.f);
        EXPECT_TRUE(simd::all(p == e));
        auto f = simd::div(4.f, b);
        EXPECT_TRUE(simd::all(p == f));
    }
    {
        simd::Vec<double, 2> a(8.0), b(2.0), p(4.0);
        auto c = a / b;
        std::cout << a << "(a)" << " / " << b << "(b) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
        auto d = simd::div(a, b);
        EXPECT_TRUE(simd::all(p == d));
        auto e = simd::div(a, 2.0);
        EXPECT_TRUE(simd::all(p == e));
        auto f = simd::div(8.0, b);
        EXPECT_TRUE(simd::all(p == f));
    }
}

TEST(vec_op_sse, test_arithdiv_inplace)
{
    {
        simd::Vec<float, 4> a(1.f), b(2.f), p(0.5f);
        a /= b;
        std::cout << a << "(a)\n";
        EXPECT_TRUE(simd::all(p == a));
    }
    {
        simd::Vec<double, 2> a(1.0), b(2.0), p(0.5);
        a /= b;
        std::cout << a << "(a)\n";
        EXPECT_TRUE(simd::all(p == a));
    }
}

TEST(vec_op_sse, test_arithinc_by1)
{
    {
        simd::Vec<int32_t, 4> a(1), p(2);
        auto c = ++a;
        EXPECT_TRUE(simd::all(p == c));
        EXPECT_TRUE(simd::all(p == a));
        std::cout << "++a: " << a << "(a) = " << c << "(c)\n";
        auto d = a++;
        EXPECT_TRUE(simd::all(p == d));
        p++;
        EXPECT_TRUE(simd::all(p == a));
    }
    {
        simd::Vec<float, 4> a(1.f), p(2.f);
        auto c = ++a;
        std::cout << "++a: " << a << "(a) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
        EXPECT_TRUE(simd::all(p == a));
        auto d = a++;
        EXPECT_TRUE(simd::all(p == d));
        p++;
        EXPECT_TRUE(simd::all(p == a));
    }
    {
        simd::Vec<double, 2> a(1.0), p(2.0);
        auto c = ++a;
        std::cout << "++a: " << a << "(a) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
        EXPECT_TRUE(simd::all(p == a));
        auto d = a++;
        EXPECT_TRUE(simd::all(p == d));
        p++;
        EXPECT_TRUE(simd::all(p == a));
    }
}

TEST(vec_op_sse, test_arithdec_by1)
{
    {
        simd::Vec<int32_t, 4> a(1), p(0);
        auto c = --a;
        std::cout << "--a: " << a << "(a) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
        EXPECT_TRUE(simd::all(p == a));
        auto d = a--;
        EXPECT_TRUE(simd::all(p == d));
        p--;
        EXPECT_TRUE(simd::all(p == a));
    }
    {
        simd::Vec<float, 4> a(1.f), p(0.f);
        auto c = --a;
        std::cout << "--a: " << a << "(a) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
        EXPECT_TRUE(simd::all(p == a));
        auto d = a--;
        EXPECT_TRUE(simd::all(p == d));
        p--;
        EXPECT_TRUE(simd::all(p == a));
    }
    {
        simd::Vec<double, 2> a(1.0), p(0.0);
        auto c = --a;
        std::cout << "--a: " << a << "(a) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
        EXPECT_TRUE(simd::all(p == a));
        auto d = a--;
        EXPECT_TRUE(simd::all(p == d));
        p--;
        EXPECT_TRUE(simd::all(p == a));
    }
}

TEST(vec_op_sse, test_arithneg)
{
    {
        simd::Vec<int32_t, 4> a(1), p(-1);
        auto c = -a;
        std::cout << "-a: " << a << "(a) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
    }
    {
        simd::Vec<float, 4> a(1.f), p(-1.f);
        auto c = -a;
        std::cout << "-a: " << a << "(a) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
    }
    {
        simd::Vec<double, 2> a(1.0), p(-1.0);
        auto c = -a;
        std::cout << "-a: " << a << "(a) = " << c << "(c)\n";
        EXPECT_TRUE(simd::all(p == c));
    }
}
