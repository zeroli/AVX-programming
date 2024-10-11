#include <gtest/gtest.h>

#include "simd/simd.h"

#include <sstream>

TEST(vec_complex, test_complex_type)
{
    {
        simd::Vec<std::complex<float>, 4> a(std::complex<float>(1.f, 2.f));
        std::ostringstream os;
        os << a;
        //EXPECT_EQ("vcf32x2[(1,2), (1,2)]", os.str());
    }
    {
        simd::Vec<std::complex<double>, 2> a(std::complex<double>{1.0, 2.0});
        std::ostringstream os;
        os << a;
        //EXPECT_EQ("vcf64x1[(1,2)]", os.str());
    }
    {
        simd::Vec<std::complex<float>, 4> a(std::complex<float>{1.f, 2.f});
        //EXPECT_FLOAT_EQ(1.f, a[0].real());
        //EXPECT_FLOAT_EQ(2.f, a[0].imag());
    }
    {
        simd::Vec<std::complex<double>, 2> a(std::complex<double>{1.0, 2.0});
        //EXPECT_FLOAT_EQ(1.0, a[0].real());
        //EXPECT_FLOAT_EQ(2.0, a[0].imag());
    }
}
