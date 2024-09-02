#pragma once

#include "simd/arch/kernel_impl.h"
#include "simd/types/sse_register.h"
#include "simd/types/traits.h"

#include <complex>

namespace simd {
namespace kernel {
namespace impl {
template <size_t W>
struct add<std::complex<float>, W>
{
    static Vec<std::complex<float>, W> apply(const Vec<std::complex<float>, W>& lhs, const Vec<std::complex<float>, W>& rhs) noexcept
    {
        return {};
    }
};

template <size_t W>
struct add<std::complex<double>, W>
{
    static Vec<std::complex<double>, W> apply(const Vec<std::complex<double>, W>& lhs, const Vec<std::complex<double>, W>& rhs) noexcept
    {
        return {};
    }
};

template <size_t W>
struct sub<std::complex<float>, W>
{
    static Vec<std::complex<float>, W> apply(const Vec<std::complex<float>, W>& lhs, const Vec<std::complex<float>, W>& rhs) noexcept
    {
        return {};
    }
};

template <size_t W>
struct sub<std::complex<double>, W>
{
    static Vec<std::complex<double>, W> apply(const Vec<std::complex<double>, W>& lhs, const Vec<std::complex<double>, W>& rhs) noexcept
    {
        return {};
    }
};

template <size_t W>
struct broadcast<std::complex<float>, W>
{
    static Vec<std::complex<float>, W> apply(const std::complex<float>& val) noexcept
    {
        Vec<std::complex<float>, W> ret;
        constexpr int nregs = Vec<std::complex<float>, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_set1_ps(val.real());
        }
        return ret;
    }
};

template <size_t W>
struct broadcast<std::complex<double>, W>
{
    static Vec<std::complex<double>, W> apply(const std::complex<double>& val) noexcept
    {
        Vec<std::complex<double>, W> ret;
        constexpr int nregs = Vec<std::complex<double>, W>::n_regs();
        #pragma unroll
        for (auto idx = 0; idx < nregs; idx++) {
            ret.reg(idx) = _mm_set1_pd(val.real());
        }
        return ret;
    }
};
}  // namespace impl
}  // namespace kernel
}  // namespace simd
