#pragma once

#include "simd/config/config.h"
#include "simd/config/inline.h"

#include <type_traits>
#include <iterator>
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

template <typename ST, size_t W, typename A, typename VT>
struct simd_register_base {
    using scalar_t = ST;
    using arch_t = A;
    using register_t = VT;
    /* how many registers for this vector */
    static constexpr size_t n_regs() {
        return sizeof(scalar_t) * W / sizeof(register_t);
    }
    /* how many lanes per register */
    static constexpr size_t reg_lanes() {
        return W / n_regs();
    }

    /* aligned to the whole vector */
    union alignas(n_regs() * arch_t::alignment()) {
        register_t regs_[n_regs()];
        scalar_t array_[W];
    };

    simd_register_base() noexcept {}

    template <typename... Regs>
    simd_register_base(register_t val, Regs... others) noexcept
        : regs_{val, others...}
    {
    }

    register_t reg(size_t idx = 0) const noexcept {
        return regs_[idx];
    }
    register_t& reg(size_t idx = 0) noexcept {
        return regs_[idx];
    }
    scalar_t operator[](size_t idx) const noexcept {
        return array_[idx];
    }
    scalar_t& operator[](size_t idx) noexcept {
        return array_[idx];
    }
    scalar_t at(size_t idx) const {
        return array_[idx];
    }
    scalar_t get(size_t idx) const noexcept {
        return array_[idx];
    }

    using value_type = ST;
    using pointer = ST*;
    using const_pointer = const ST*;
    using reference = ST&;
    using const_reference = const ST&;
    using iterator = pointer;
    using const_iterator = const_pointer;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    using reverse_iterator = std::reverse_iterator<iterator>;

    /// make vector iteratorable
    const_iterator begin() const noexcept {
        return &array_[0];
    }
    const_iterator end() const noexcept {
        return &array_[W];
    }
    iterator begin() noexcept {
        return &array_[0];
    }
    iterator end() noexcept {
        return &array_[W];
    }

    const_iterator cbegin() const noexcept {
        return &array_[0];
    }
    const_iterator cend() const noexcept {
        return &array_[W];
    }

    const_reverse_iterator rbegin() const noexcept {
        return const_reverse_iterator(end());
    }
    const_reverse_iterator rend() const noexcept {
        return const_reverse_iterator(begin());
    }
    reverse_iterator rbegin() noexcept {
        return reverse_iterator(end());
    }
    reverse_iterator rend() noexcept {
        return reverse_iterator(begin());
    }

    const_reverse_iterator crbegin() const noexcept {
        return const_reverse_iterator(end());
    }
    const_reverse_iterator crend() const noexcept {
        return const_reverse_iterator(begin());
    }
};

#define DECLARE_SIMD_REGISTER(SCALAR_TYPE, ISA, VECTOR_TYPE) \
template <size_t W> \
struct simd_register<SCALAR_TYPE, W, ISA> \
    : public simd_register_base<SCALAR_TYPE, W, ISA, VECTOR_TYPE> \
{ \
    using base_t = simd_register_base<SCALAR_TYPE, W, ISA, VECTOR_TYPE>; \
    using base_t::base_t; \
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
