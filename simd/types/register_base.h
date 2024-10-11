#pragma once

#include <iterator>
#include <type_traits>
#include <cstddef>
#include <cstdint>

namespace simd {
namespace types {
template <typename ST, size_t W, typename A, typename VT, typename Enable = void>
struct simd_register_base {
    using scalar_t = ST;
    using arch_t = A;
    using register_t = VT;
    /// how many registers for this vector
    static constexpr size_t n_regs() {
        return sizeof(scalar_t) * W / sizeof(register_t);
    }
    /// how many lanes per register
    static constexpr size_t reg_lanes() {
        return W / n_regs();
    }

    /// aligned to the whole vector
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

}  // namespace types
}  // namespace simd
