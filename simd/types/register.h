#pragma once

#include <type_traits>

namespace simd {
namespace types {
template <typename T, typename Arch>
struct has_simd_register : std::false_type { };

template <typename T, typename Arch>
struct simd_register {
    struct register_t { };
};

#define DECLARE_SIMD_REGISTER(SCALAR_TYPE, ISA, VECTOR_TYPE) \
template <> \
struct simd_register<SCALAR_TYPE, ISA> \
{ \
    using register_t = VECTOR_TYPE; \
    register_t data; \
    operator register_t() const noexcept { \
        return data; \
    } \
}; \
template <> \
struct has_simd_register<SCALAR_TYPE, ISA> : std::true_type \
{ \
}; \
/// #####

#define DECLARE_SIMD_REGISTER_ALIAS(ISA, ISA_BASE)  \
template <class T>  \
struct simd_register<T, ISA> : simd_register<T, ISA_BASE>  \
{  \
    using register_t = typename simd_register<T, ISA_BASE>::register_t; \
    simd_register(register_t reg) noexcept  \
        : simd_register<T, ISA_BASE> { reg }  \
    {  \
    }  \
    simd_register() = default;  \
};  \
template <class T> \
struct has_simd_register<T, ISA> : has_simd_register<T, ISA_BASE> \
{ \
} \
/// #######

template <typename T, typename Arch>
struct get_bool_simd_register {
    using type = simd_register<T, Arch>;
};

template <typename T, typename Arch>
using get_bool_simd_register_t = typename get_bool_simd_register<T, Arch>::type;

}  // namespace types
}  // namespace simd
