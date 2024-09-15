#pragma once

#include <type_traits>
#include <cstddef>
#include <cstdint>

namespace simd {
namespace types {
template <typename T, size_t W, typename A>
struct has_simd_register : std::false_type { };

template <typename T, size_t W, typename A>
struct simd_register {
    struct register_t { };
};

#define DECLARE_SIMD_REGISTER(SCALAR_TYPE, ISA, VECTOR_TYPE) \
template <size_t W> \
struct simd_register<SCALAR_TYPE, W, ISA> \
{ \
    using scalar_t = SCALAR_TYPE; \
    using arch_t = ISA; \
    using register_t = VECTOR_TYPE; \
    /* how many registers for this vector */ \
    static constexpr size_t n_regs() { \
        return sizeof(scalar_t) * W / sizeof(register_t); \
    } \
    /* how many lanes per register */ \
    static constexpr size_t reg_lanes() { \
        return W / n_regs(); \
    } \
\
    /* aligned to the whole vector */ \
    union alignas(n_regs() * ISA::alignment()) { \
        register_t regs_[n_regs()]; \
        scalar_t array_[W]; \
    }; \
\
    simd_register() noexcept {} \
    template <typename... Regs> \
    simd_register(register_t val, Regs... others) \
        : regs_{val, others...} \
    { \
    } \
\
    operator register_t() const noexcept { \
        return regs_[0]; \
    } \
    register_t reg(size_t idx = 0) const noexcept { \
        return regs_[idx]; \
    } \
    register_t& reg(size_t idx = 0) noexcept { \
        return regs_[idx]; \
    } \
    scalar_t operator[](size_t idx) const noexcept { \
        return array_[idx]; \
    } \
    scalar_t& operator[](size_t idx) noexcept { \
        return array_[idx]; \
    } \
    /* exception out-of-range may be thrown, same as std::array::at */ \
    scalar_t at(size_t idx) const { \
        return array_[idx]; \
    } \
    scalar_t get(size_t idx) const noexcept { \
        return array_[idx]; \
    } \
}; \
template <size_t W> \
struct has_simd_register<SCALAR_TYPE, W, ISA> \
    : std::integral_constant<bool, (simd_register<SCALAR_TYPE, W, ISA>::n_regs() > 0)> \
{ \
}; \
/// #####

#define DECLARE_SIMD_REGISTER_ALIAS(ISA, ISA_BASE)  \
template <typename T, size_t W>  \
struct simd_register<T, W, ISA> : simd_register<T, W, ISA_BASE>  \
{  \
    using base_t = simd_register<T, W, ISA_BASE>; \
    using register_t = typename simd_register<T, W, ISA_BASE>::register_t; \
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
