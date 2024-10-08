#pragma once

#include <type_traits>
#include <cstddef>
#include <complex>

namespace simd {
namespace detail {
template <typename T, typename A>
void static_check_supported_config()
{
    // TODO
}

template <size_t N>
constexpr bool is_pow_of_2() {
    return N != 0 && (N & (N - 1)) == 0;
}

template <size_t... Idx>
struct index_sequence { };

template <size_t N, size_t... Vs>
struct index_sequence_helper : index_sequence_helper<N-1, N-1, Vs...> {};

template <size_t... Vs>
struct index_sequence_helper<0, Vs...> {
    using type = index_sequence<Vs...>;
};

template <size_t Num>
using make_index_sequence = typename index_sequence_helper<Num>::type;

}  // namespace detail

template <typename T, size_t maxsize = 8>
void static_check_supported_type()
{
    static_assert(sizeof(T) <= maxsize && detail::is_pow_of_2<sizeof(T)>(),
        "sizeof(T) must be satisfied");
}

#define STATIC_CHECK_ARCH_ENABLED(ARCH) \
    static_assert(simd::ARCH::supported(), "simd with arch '" #ARCH "' not supported")

namespace traits {
template <bool cond, typename V = void>
using enable_if_t = typename std::enable_if<cond, V>::type;

#define REQUIRES(cond) \
    traits::enable_if_t<cond>* = nullptr

#define ENABLE_IF(cond) \
    traits::enable_if_t<cond>

#define REQUIRE_INTEGRAL(T) \
    traits::enable_if_t<std::is_integral<T>::value>

#define REQUIRE_SIGNED_INTEGRAL(T) \
    traits::enable_if_t<std::is_signed<T>::value && std::is_integral<T>::value>

#define REQUIRE_UNSIGNED_INTEGRAL(T) \
    traits::enable_if_t<std::is_unsigned<T>::value && std::is_integral<T>::value>

#define REQUIRE_FLOATING(T) \
    traits::enable_if_t<std::is_floating_point<T>::value>

#define REQUIRE_FLOAT32(T) \
    traits::enable_if_t<std::is_same<T, float>::value>

#define REQUIRE_FLOAT64(T) \
    traits::enable_if_t<std::is_same<T, double>::value>

#define EVAL_INT_SIZE(T, SIZE) (std::is_integral<T>::value && sizeof(T) == SIZE)

#define REQUIRE_INTEGRAL_SIZE_MATCH(T, SIZE) \
    traits::enable_if_t<EVAL_INT_SIZE(T, SIZE)>

#define IS_INT_SIZE_1(T) (EVAL_INT_SIZE(T, 1))
#define IS_INT_SIZE_2(T) (EVAL_INT_SIZE(T, 2))
#define IS_INT_SIZE_4(T) (EVAL_INT_SIZE(T, 4))
#define IS_INT_SIZE_8(T) (EVAL_INT_SIZE(T, 8))

#define REQUIRE_INTEGRAL_SIZE_1(T) REQUIRE_INTEGRAL_SIZE_MATCH(T, 1)
#define REQUIRE_INTEGRAL_SIZE_2(T) REQUIRE_INTEGRAL_SIZE_MATCH(T, 2)
#define REQUIRE_INTEGRAL_SIZE_4(T) REQUIRE_INTEGRAL_SIZE_MATCH(T, 4)
#define REQUIRE_INTEGRAL_SIZE_8(T) REQUIRE_INTEGRAL_SIZE_MATCH(T, 8)

namespace detail {
template <typename T, typename Enable = void>
struct to_integral { };
template <typename T>
struct to_integral<T, REQUIRE_INTEGRAL(T)> {
    using type = T;
};
template <>
struct to_integral<float> {
    using type = int32_t;  // keep sign bit
};
template <>
struct to_integral<double> {
    using type = int64_t;  // keep sign bit
};
}  // namespace detail

template <typename T>
using to_integral_t = typename detail::to_integral<T>::type;

template <typename T, size_t W>
struct vec_type_traits;

#define stringify(X) #X

#define DEFINE_VEC_TYPE_TRAITS(T, W, STR_T) \
template <> \
struct vec_type_traits<T, W> { \
    static constexpr const char* type() { \
        /* vi32x4 */ \
        return "v" STR_T "x" stringify(W); \
    } \
    static constexpr const char* bool_type() { \
        /* vi32bx4 */ \
        return "v" STR_T "bx" stringify(W); \
    } \
} \
///

/// 128bits
DEFINE_VEC_TYPE_TRAITS(int8_t,   16, "i8" );
DEFINE_VEC_TYPE_TRAITS(uint8_t,  16, "u8" );
DEFINE_VEC_TYPE_TRAITS(int16_t,   8, "i16");
DEFINE_VEC_TYPE_TRAITS(uint16_t,  8, "u16");
DEFINE_VEC_TYPE_TRAITS(int32_t,   4, "i32");
DEFINE_VEC_TYPE_TRAITS(uint32_t,  4, "u32");
DEFINE_VEC_TYPE_TRAITS(int64_t,   2, "i64");
DEFINE_VEC_TYPE_TRAITS(uint64_t,  2, "u64");
DEFINE_VEC_TYPE_TRAITS(float,     4, "f32");
DEFINE_VEC_TYPE_TRAITS(double,    2, "f64");
DEFINE_VEC_TYPE_TRAITS(std::complex<float>,  2, "cf32");
DEFINE_VEC_TYPE_TRAITS(std::complex<double>, 1, "cf64");

/// 256bits
DEFINE_VEC_TYPE_TRAITS(int8_t,   32, "i8" );
DEFINE_VEC_TYPE_TRAITS(uint8_t,  32, "u8" );
DEFINE_VEC_TYPE_TRAITS(int16_t,  16, "i16");
DEFINE_VEC_TYPE_TRAITS(uint16_t, 16, "u16");
DEFINE_VEC_TYPE_TRAITS(int32_t,   8, "i32");
DEFINE_VEC_TYPE_TRAITS(uint32_t,  8, "u32");
DEFINE_VEC_TYPE_TRAITS(int64_t,   4, "i64");
DEFINE_VEC_TYPE_TRAITS(uint64_t,  4, "u64");
DEFINE_VEC_TYPE_TRAITS(float,     8, "f32");
DEFINE_VEC_TYPE_TRAITS(double,    4, "f64");
DEFINE_VEC_TYPE_TRAITS(std::complex<float>,  4, "cf32");
DEFINE_VEC_TYPE_TRAITS(std::complex<double>, 2, "cf64");

/// 512bits
DEFINE_VEC_TYPE_TRAITS(int8_t,   64, "i8" );
DEFINE_VEC_TYPE_TRAITS(uint8_t,  64, "u8" );
DEFINE_VEC_TYPE_TRAITS(int16_t,  32, "i16");
DEFINE_VEC_TYPE_TRAITS(uint16_t, 32, "u16");
DEFINE_VEC_TYPE_TRAITS(int32_t,  16, "i32");
DEFINE_VEC_TYPE_TRAITS(uint32_t, 16, "u32");
DEFINE_VEC_TYPE_TRAITS(int64_t,   8, "i64");
DEFINE_VEC_TYPE_TRAITS(uint64_t,  8, "u64");
DEFINE_VEC_TYPE_TRAITS(float,    16, "f32");
DEFINE_VEC_TYPE_TRAITS(double,    8, "f64");
DEFINE_VEC_TYPE_TRAITS(std::complex<float>,  8, "cf32");
DEFINE_VEC_TYPE_TRAITS(std::complex<double>, 4, "cf64");

}  // namespace traits
}  // namespace simd
