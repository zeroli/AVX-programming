#pragma once

#include <type_traits>

namespace simd {
namespace detail {
template <typename T, typename A>
void static_check_supported_config()
{
    // TODO
}
}  // namespace detail

namespace traits {
template <bool cond, typename V = void>
using enable_if_t = typename std::enable_if<cond, V>::type;

#define REQUIRE_INTEGRAL(T) \
    traits::enable_if_t<std::is_integral<T>::value>* = nullptr

#define REQUIRE_INTEGRAL_SIZE_MATCH(T, SIZE) \
    traits::enable_if_t<std::is_integral<T>::value && sizeof(T) == SIZE>* = nullptr

#define REQUIRE_INTEGRAL_SIZE_1(T) REQUIRE_INTEGRAL_SIZE_MATCH(T, 1)
#define REQUIRE_INTEGRAL_SIZE_2(T) REQUIRE_INTEGRAL_SIZE_MATCH(T, 2)
#define REQUIRE_INTEGRAL_SIZE_4(T) REQUIRE_INTEGRAL_SIZE_MATCH(T, 4)
#define REQUIRE_INTEGRAL_SIZE_8(T) REQUIRE_INTEGRAL_SIZE_MATCH(T, 8)

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
} \
///

DEFINE_VEC_TYPE_TRAITS(float, 4, "f32");
DEFINE_VEC_TYPE_TRAITS(float, 8, "f32");
DEFINE_VEC_TYPE_TRAITS(float, 16, "f32");

DEFINE_VEC_TYPE_TRAITS(double, 8, "f64");
DEFINE_VEC_TYPE_TRAITS(double, 16, "f64");

DEFINE_VEC_TYPE_TRAITS(int8_t, 16, "i8");
DEFINE_VEC_TYPE_TRAITS(int8_t, 32, "i8");
DEFINE_VEC_TYPE_TRAITS(int8_t, 64, "i8");

DEFINE_VEC_TYPE_TRAITS(int16_t, 8, "i16");
DEFINE_VEC_TYPE_TRAITS(int16_t, 16, "i16");
DEFINE_VEC_TYPE_TRAITS(int16_t, 32, "i16");

DEFINE_VEC_TYPE_TRAITS(int32_t, 4, "i32");
DEFINE_VEC_TYPE_TRAITS(int32_t, 8, "i32");
DEFINE_VEC_TYPE_TRAITS(int32_t, 16, "i32");

DEFINE_VEC_TYPE_TRAITS(int64_t, 2, "i64");
DEFINE_VEC_TYPE_TRAITS(int64_t, 4, "i64");
DEFINE_VEC_TYPE_TRAITS(int64_t, 8, "i64");

}  // namespace traits
}  // namespace simd
