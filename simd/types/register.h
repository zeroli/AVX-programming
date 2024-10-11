#pragma once

#include "simd/config/config.h"
#include "simd/config/inline.h"
#include "simd/types/register_base.h"

namespace simd {
namespace types {
template <typename T, size_t W, typename A>
struct has_simd_register : std::false_type { };

template <typename T, size_t W, typename A, typename Enable = void>
struct simd_register;


#define DECLARE_SIMD_REGISTER(SCALAR_TYPE, ISA, VECTOR_TYPE) \
template <size_t W> \
struct simd_register<SCALAR_TYPE, W, ISA> \
    : public simd_register_base<SCALAR_TYPE, W, ISA, VECTOR_TYPE> \
{ \
    using base_t = simd_register_base<SCALAR_TYPE, W, ISA, VECTOR_TYPE>; \
    using arch_t = ISA; \
    using base_t::base_t; \
}; \
template <size_t W> \
struct has_simd_register<SCALAR_TYPE, W, ISA> \
    : std::integral_constant<bool, (simd_register<SCALAR_TYPE, W, ISA>::n_regs() > 0)> \
{ \
}; \
/// #######

#define DECLARE_SIMD_REGISTER_ALIAS(ISA, ISA_BASE)  \
template <typename T, size_t W>  \
struct simd_register<T, W, ISA> : simd_register<T, W, ISA_BASE>  \
{  \
    using base_t = simd_register<T, W, ISA_BASE>; \
    using arch_t = ISA; /* this is derived arch */ \
    using base_t::base_t; \
};  \
template <typename T, size_t W> \
struct has_simd_register<T, W, ISA> : has_simd_register<T, W, ISA_BASE> \
{ \
}; \
/// #######

template <typename T, size_t W, typename A>
struct get_bool_simd_register {
    using type = simd_register<T, W, A>;
};

template <typename T, size_t W, typename A>
using get_bool_simd_register_t = typename get_bool_simd_register<T, W, A>::type;

}  // namespace types

namespace kernel {
template <typename A>
using requires_arch = typename std::add_lvalue_reference<
                                    typename std::add_const<A>::type>::type;

}  // namespace kernel
}  // namespace simd
