#pragma once

#include "simd/config/config.h"
#include "simd/types/register.h"

#include <cstddef>

namespace simd {
/// generic architecture
struct Generic {
    static constexpr bool supported() noexcept { return true; }
    static constexpr bool available() noexcept { return true; }
    static constexpr size_t alignment() noexcept { return 1; }
    static constexpr bool require_alignment() noexcept { return false; }
    static constexpr const char* name() noexcept { return "Generic"; }
};

namespace types {
struct float2 { float x, y; };

using generic_reg_i = uint64_t;
using generic_reg_f = float2;
using generic_reg_d = double;

template <typename T, typename Enable = void>
struct generic_reg_traits;

template <typename T>
using generic_reg_traits_t = typename generic_reg_traits<T>::type;

#define DECLARE_SIMD_GENERIC_REGISTER(SCALAR_TYPE, ISA, VECTOR_TYPE) \
template <> \
struct generic_reg_traits<SCALAR_TYPE> { \
    using type = VECTOR_TYPE; \
}; \
DECLARE_SIMD_REGISTER(SCALAR_TYPE, ISA, VECTOR_TYPE) \
///###

DECLARE_SIMD_GENERIC_REGISTER(int8_t,               Generic, generic_reg_i);
DECLARE_SIMD_GENERIC_REGISTER(uint8_t,              Generic, generic_reg_i);
DECLARE_SIMD_GENERIC_REGISTER(int16_t,              Generic, generic_reg_i);
DECLARE_SIMD_GENERIC_REGISTER(uint16_t,             Generic, generic_reg_i);
DECLARE_SIMD_GENERIC_REGISTER(int32_t,              Generic, generic_reg_i);
DECLARE_SIMD_GENERIC_REGISTER(uint32_t,             Generic, generic_reg_i);
DECLARE_SIMD_GENERIC_REGISTER(int64_t,              Generic, generic_reg_i);
DECLARE_SIMD_GENERIC_REGISTER(uint64_t,             Generic, generic_reg_i);
DECLARE_SIMD_GENERIC_REGISTER(float,                Generic, generic_reg_f);
DECLARE_SIMD_GENERIC_REGISTER(double,               Generic, generic_reg_d);
DECLARE_SIMD_GENERIC_REGISTER(std::complex<float>,  Generic, generic_reg_f);
DECLARE_SIMD_GENERIC_REGISTER(std::complex<double>, Generic, generic_reg_d);

}  // namespace types
}  // namespace simd
