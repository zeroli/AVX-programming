#pragma once

namespace simd {
template <typename T, size_t W>
class VecBool
    : public types::get_bool_simd_register_t<T, W, types::arch_traits_t<T, W>>
{
public:
    static constexpr size_t size() { return W; }
    static constexpr const char* type() {
        return traits::vec_type_traits<T, W>::bool_type();
    }

    using A = types::arch_traits_t<T, W>;
    using arch_t = A;
    using base_t = types::get_bool_simd_register_t<T, W, A>;
    using self_t = VecBool;
    using scalar_t = bool;
    using register_t = typename base_t::register_t;
    using vec_t = Vec<T, W>;

    SIMD_INLINE
    VecBool() = default;

    SIMD_INLINE
    VecBool(bool val) noexcept;

    template <typename... Regs>
    SIMD_INLINE
    VecBool(register_t arg, Regs... others) noexcept;

    template <typename... Ts>
    SIMD_INLINE
    VecBool(bool val0, bool val1, Ts... vals) noexcept;

    template <typename Tp>
    SIMD_INLINE
    VecBool(const Tp* ptr) = delete;

    /// load/store
    SIMD_INLINE
    void store_aligned(bool* mem) const noexcept;
    SIMD_INLINE
    void store_unaligned(bool* mem) const noexcept;
    SIMD_INLINE
    static VecBool load_aligned(const bool* mem) noexcept;
    SIMD_INLINE
    static VecBool load_unaligned(const bool* mem) noexcept;

    /// short/convenience functions for load/store
    SIMD_INLINE
    void store(bool* mem) const noexcept {
        store_aligned(mem);
    }
    SIMD_INLINE
    void storeu(bool* mem) const noexcept {
        store_unaligned(mem);
    }
    SIMD_INLINE
    void load(const bool* mem) noexcept {
        *this = load_aligned(mem);
    }
    SIMD_INLINE
    void loadu(const bool* mem) noexcept {
        *this = load_unaligned(mem);
    }

    /// mask operators
    /// Extract a scalar mask representation from this vec bool
    SIMD_INLINE
    uint64_t to_mask() const noexcept;
    SIMD_INLINE
    static VecBool from_mask(uint64_t mask) noexcept;

    /// comparison operators
    SIMD_INLINE
    VecBool operator ==(const VecBool& other) const noexcept;
    SIMD_INLINE
    VecBool operator !=(const VecBool& other) const noexcept;

    /// logical operators
    SIMD_INLINE
    VecBool operator ~() const noexcept;
    SIMD_INLINE
    VecBool operator !() const noexcept {
        return operator ==(self_t(false));
    }
    SIMD_INLINE
    VecBool operator &(const VecBool& other) const noexcept;
    SIMD_INLINE
    VecBool operator |(const VecBool& other) const noexcept;
    SIMD_INLINE
    VecBool operator ^(const VecBool& other) const noexcept;

    SIMD_INLINE
    VecBool operator &&(const VecBool& other) const noexcept {
        return operator &(other);
    }
    SIMD_INLINE
    VecBool operator ||(const VecBool& other) const noexcept {
        return operator |(other);
    }

    /// in-place update operators
    SIMD_INLINE
    VecBool& operator &=(const VecBool& other) noexcept {
        return (*this) = (*this) & other;
    }
    SIMD_INLINE
    VecBool& operator |=(const VecBool& other) noexcept {
        return (*this) = (*this) | other;
    }
    SIMD_INLINE
    VecBool& operator ^=(const VecBool& other) noexcept {
        return (*this) = (*this) ^ other;
    }

private:
    template <typename U, typename... V, size_t I, size_t... Is>
    SIMD_INLINE
    static register_t make_register(detail::index_sequence<I, Is...>, U u, V... v) noexcept
    {
        return make_register(detail::index_sequence<Is...>(), u, u, v...);
    }

    template <typename... V>
    SIMD_INLINE
    static register_t make_register(detail::index_sequence<>, V... v) noexcept;
};

}  // namespace simd
