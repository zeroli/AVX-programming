#pragma once

#include <limits>
#include <type_traits>

#include "simd/config/inline.h"
#include "simd/memory/bits.h"

namespace simd {
namespace constants {
#define SIMD_DEFINE_CONSTANT(NAME, SINGLE, DOUBLE) \
    template <class T>                              \
    SIMD_INLINE \
    T NAME() noexcept                  \
    {                                               \
        return T(NAME<typename T::scalar_t>());   \
    }                                               \
    template <>                                     \
    SIMD_INLINE \
    float NAME<float>() noexcept       \
    {                                               \
        return SINGLE;                              \
    }                                               \
    template <>                                     \
    SIMD_INLINE \
    double NAME<double>() noexcept     \
    {                                               \
        return DOUBLE;                              \
    }
///

#define SIMD_DEFINE_CONSTANT_HEX(NAME, SINGLE, DOUBLE) \
    template <class T>                                  \
    SIMD_INLINE \
    T NAME() noexcept                      \
    {                                                   \
        return T(NAME<typename T::scalar_t>());       \
    }                                                   \
    template <>                                         \
    SIMD_INLINE \
    float NAME<float>() noexcept           \
    {                                                   \
        return bits::cast<float>((uint32_t)SINGLE);       \
    }                                                   \
    template <>                                         \
    SIMD_INLINE \
    double NAME<double>() noexcept         \
    {                                                   \
        return bits::cast<double>((uint64_t)DOUBLE);     \
    }
///

SIMD_DEFINE_CONSTANT(infinity, (std::numeric_limits<float>::infinity()), (std::numeric_limits<double>::infinity()));
SIMD_DEFINE_CONSTANT_HEX(nan, 0xffffffff, 0xffffffffffffffff);
SIMD_DEFINE_CONSTANT(log_2, 0.6931471805599453094172321214581765680755001343602553f, 0.6931471805599453094172321214581765680755001343602553);
SIMD_DEFINE_CONSTANT_HEX(signmask, 0x80000000, 0x8000000000000000);

#undef SIMD_DEFINE_CONSTANT
#undef SIMD_DEFINE_CONSTANT_HEX

namespace detail {
template <typename T, bool = std::is_integral<T>::value>
struct allbits_impl
{
    static constexpr T get() {
        return T(~0);
    }
};
template <typename T>
struct allbits_impl<T, false>
{
    static constexpr T get() {
        return nan<T>();
    }
};
}  // namespace detail

template <typename T>
constexpr T allbits() noexcept
{
    return T(detail::allbits_impl<typename T::scalar_t>::get());
}
template <class T>
constexpr T ones() noexcept
{
    return allbits<T>();
}

template <class T>
constexpr T zeros() noexcept
{
    return T(typename T::scalar_t(0));
}

}  // namespace constants
}  // namespace simd
