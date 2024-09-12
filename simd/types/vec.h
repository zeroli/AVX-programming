#pragma once

#include "simd/types/arch_traits.h"
#include "simd/types/traits.h"

#include <cstddef>

namespace simd {
template <typename T, size_t W>
class Vec;
template <typename T, size_t W>
class VecBool;

// These functions are forwarded declared here so that they can be used
// by friend functions with Vec<T, W>. Their implementation must appear
// only once the kernel implementations have been included.
namespace ops {
template <typename T, size_t W>
Vec<T, W> add(const Vec<T, W>& lhs, const Vec<T, W>& rhs);
template <typename T, size_t W>
Vec<T, W> sub(const Vec<T, W>& lhs, const Vec<T, W>& rhs);
template <typename T, size_t W>
Vec<T, W> mul(const Vec<T, W>& lhs, const Vec<T, W>& rhs);
template <typename T, size_t W>
Vec<T, W> div(const Vec<T, W>& lhs, const Vec<T, W>& rhs);
template <typename T, size_t W>
Vec<T, W> bitwise_and(const Vec<T, W>& lhs, const Vec<T, W>& rhs);
template <typename T, size_t W>
Vec<T, W> bitwise_or(const Vec<T, W>& lhs, const Vec<T, W>& rhs);
template <typename T, size_t W>
Vec<T, W> bitwise_xor(const Vec<T, W>& lhs, const Vec<T, W>& rhs);
template <typename T, size_t W>
Vec<T, W> logical_and(const Vec<T, W>& lhs, const Vec<T, W>& rhs);
template <typename T, size_t W>
Vec<T, W> logical_or(const Vec<T, W>& lhs, const Vec<T, W>& rhs);

#if 0
template <typename T, size_t W>
VecBool<T, W> eq(const Vec<T, W>& lhs, const Vec<T, W>& rhs)
{
    return kernel::eq<W>(lhs, rhs, A{});
}
template <typename T, size_t W>
VecBool<T, W> eq(const Vec<T, W>& lhs, const Vec<T, W>& rhs)
{
    return kernel::ne<W>(lhs, rhs, A{});
}
template <typename T, size_t W>
VecBool<T, W> ge(const Vec<T, W>& lhs, const Vec<T, W>& rhs)
{
    return kernel::ge<W>(lhs, rhs, A{});
}
template <typename T, size_t W>
VecBool<T, W> le(const Vec<T, W>& lhs, const Vec<T, W>& rhs)
{
    return kernel::le<W>(lhs, rhs, A{});
}
template <typename T, size_t W>
VecBool<T, W> gt(const Vec<T, W>& lhs, const Vec<T, W>& rhs)
{
    return kernel::gt<W>(lhs, rhs, A{});
}
template <typename T, size_t W>
VecBool<T, W> lt(const Vec<T, W>& lhs, const Vec<T, W>& rhs)
{
    return kernel::lt<W>(lhs, rhs, A{});
}
#endif
}  // namepace ops

template <typename T, size_t W>
class Vec
    : public types::simd_register<T, W, types::arch_traits_t<T, W>>
{
    static_assert(!std::is_same<T, bool>::value,
        "use simd::VecBool<T, W> instead of simd::Vec<bool, W>");
public:
    static constexpr size_t size() { return W; }
    static constexpr const char* type() {
        return traits::vec_type_traits<T, W>::type();
    }

    using A = types::arch_traits_t<T, W>;
    using arch_t = A;
    using base_t = types::simd_register<T, W, arch_t>;
    using self_t = Vec;
    using scalar_t = T;
    using register_t = typename base_t::register_t;
    using vec_mask_t = VecBool<T, W>;

    Vec() = default;
    Vec(T val) noexcept;

    #if 0  // TODO:
    template <typename... Ts>
    Vec(T val0, T val1, Ts... vals) noexcept;
    explicit Vec(vec_mask_t b) noexcept;
    #endif
    template <typename... Regs>
    explicit Vec(register_t arg, Regs... others) noexcept;

#if 0
    template <typename U>
    static Vec broadcast(U val) noexcept {
        return Vec(static_cast<T>(val));
    }

    template <typename U>
    void store_aligned(U* mem) const noexcept {
        assert(is_aligned(mem, A::argument())
            && "store location is not properly aligned");
        kernel::store_aligned<W>(mem, *this, A{});
    }
    template <tyepname U>
    void store_unaligned(U* mem) const noexcept {
        kernel::store_unaligned<W>(mem, *this, A{});
    }

    template <typename U>
    void store(U* mem, aligned_mode) const noexcept {
        store_aligned(mem);
    }

    template <typename U>
    void store(U* mem, unaligned_mode) const noexcept {
        store_unaligned(mem);
    }

    template <typename U>
    static Vec load_aligned(const U* mem) noexcept {
        assert(is_aligned(mem, W::argument())
            && "loaded location is not properly aligned");
        return kernel::load_aligned<A>(mem, kernel::convert<T>{}, A{});
    }

    template <typename U>
    static Vec load_unaligned(const U* mem) noexcept {
        return kernel::load_unaligned<A>(mem, kernel::convert<T>{}, A{});
    }

    template <typename U>
    static Vec load(const U* mem, aligned_mode) noexcept {
        return load_aligned(mem);
    }

    template <typename U>
    static Vec load(const U* mem, unaligned_mode) noexcept {
        return load_unaligned(mem);
    }

    template <typename U, typename V>
    static Vec gather(const U* src, const Vec<V, W>& index) noexcept {
        static_assert(std::is_convertible<U, T>::Vec,
            "Cannot convert from src to this type");
        return kernel::gather(Vec{}, src, index, A{});
    }
    template <typename U, typename V>
    void scatter(U* dst, const Vec<V, W>& index) const noexcept {
        static_assert(std::is_convertible<T, U>::Vec,
            "Cannot convert from src to this type");
        return kernel::scatter<A>(*this, dst, index, A{});
    }

#endif
#if 0
    // comparison operators
    friend VecBool<T, W> operator ==(const Vec<T, W>& lhs, const Vec<T, W>& rhs)
    {
        return kernel::eq<W>(lhs, rhs, A{});
    }
    friend VecBool<T, W> operator !=(const Vec<T, W>& lhs, const Vec<T, W>& rhs)
    {
        return kernel::ne<W>(lhs, rhs, A{});
    }
    friend VecBool<T, W> operator >=(const Vec<T, W>& lhs, const Vec<T, W>& rhs)
    {
        return kernel::ge<W>(lhs, rhs, A{});
    }
    friend VecBool<T, W> operator <=(const Vec<T, W>& lhs, const Vec<T, W>& rhs)
    {
        return kernel::le<W>(lhs, rhs, A{});
    }
    friend VecBool<T, W> operator >(const Vec<T, W>& lhs, const Vec<T, W>& rhs)
    {
        return kernel::gt<W>(lhs, rhs, A{});
    }
    friend VecBool<T, W> operator <(const Vec<T, W>& lhs, const Vec<T, W>& rhs)
    {
        return kernel::lt<W>(lhs, rhs, A{});
    }

#endif
    /// in-place update operators
    Vec& operator +=(const Vec& other) noexcept {
        return *this = ops::add<T, W>(*this, other);
    }
    Vec& operator -=(const Vec& other) noexcept {
        return *this = ops::sub<T, W>(*this, other);
    }
    Vec& operator *=(const Vec& other) noexcept {
        return *this = ops::mul<T, W>(*this, other);
    }
    Vec& operator /=(const Vec& other) noexcept {
        return *this = ops::div<T, W>(*this, other);
    }
#if 0
    Vec& operator &=(const Vec& other) noexcept {
        return *this = kernel::bitwise_and<W>(*this, other, A{});
    }
    Vec& operator |=(const Vec& other) noexcept {
        return *this = kernel::bitwise_or<W>(*this, other, A{});
    }
    Vec& operator ^=(const Vec& other) noexcept {
        return *this = kernel::bitwise_xor<W>(*this, other, A{});
    }

    /// increment/decrement operators
    Vec& operator ++() noexcept {
        return operator +=(1);
    }
    Vec& operator --() noexcept {
        return operator -=(1);
    }
    Vec& operator ++(int) noexcept {
        self_t copy(*this);
        operator +=(1);
        return copy;
    }
    Vec& operator --(int) noexcept {
        self_t copy(*this);
        operator -=(1);
        return copy;
    }

    /// unary operators
    vec_mask_t operator !() const noexcept {
        return kernel::eq<W>(*this, Vec(0), A{});
    }
    Vec operator ~() const noexcept {
        return kernel::bitwise_not<W>(*this, A{});
    }
    Vec operator -() const noexcept {
        return kernel::neg<W>(*this, A{});
    }
    Vec operator +() const noexcept {
        return *this;
    }
#endif
    /// arithmetic operators
    /// defined as friend to enable conversion from scalar to vector
    friend Vec operator +(const Vec& lhs, const Vec& rhs)
    {
        return ops::add<T, W>(lhs, rhs);
    }
    friend Vec operator -(const Vec& lhs, const Vec& rhs)
    {
        return ops::sub<T, W>(lhs, rhs);
    }
    friend Vec operator *(const Vec& lhs, const Vec& rhs)
    {
        return ops::mul<T, W>(lhs, rhs);
    }
    friend Vec operator /(const Vec& lhs, const Vec& rhs)
    {
        return ops::div<T, W>(lhs, rhs);
    }
    #if 0
    friend Vec operator &(const Vec& lhs, const Vec& rhs)
    {
        return ops::bitwise_and<T, W>(lhs, rhs);
    }
    friend Vec operator |(const Vec& lhs, const Vec& rhs)
    {
        return ops::bitwise_or<T, W>(lhs, rhs);
    }
    friend Vec operator ^(const Vec& lhs, const Vec& rhs)
    {
        return ops::bitwise_xor<T, W>(lhs, rhs, A{});
    }
    friend Vec operator &&(const Vec& lhs, const Vec& rhs)
    {
        return ops::logical_and<T, W>(lhs, rhs, A{});
    }
    friend Vec operator ||(const Vec& lhs, const Vec& rhs)
    {
        return ops::logical_or<W>(lhs, rhs, A{});
    }

    #endif
};

using vf32x16_t = Vec<float, 16>;
using vf32x8_t = Vec<float, 8>;
using vf32x4_t = Vec<float, 4>;

using vf64x8_t = Vec<double, 8>;
using vf64x4_t = Vec<double, 4>;
using vf64x2_t = Vec<double, 2>;

using vi8x64_t = Vec<int8_t, 64>;
using vi8x32_t = Vec<int8_t, 32>;
using vi8x16_t = Vec<int8_t, 16>;
using vi8x8_t = Vec<int8_t, 8>;
using vi8x4_t = Vec<int8_t, 4>;

using vi16x32_t = Vec<int16_t, 32>;
using vi16x16_t = Vec<int16_t, 16>;
using vi16x8_t = Vec<int16_t, 8>;
using vi16x4_t = Vec<int16_t, 4>;

using vi32x16_t = Vec<int32_t, 16>;
using vi32x8_t = Vec<int32_t, 8>;
using vi32x4_t = Vec<int32_t, 4>;

#if 0
template <typename T, size_t W>
class VecBool
    : public types::get_bool_simd_register_t<T, W, types::arch_traits_t<T, W>>
{
public:
    static constexpr size_t size() { return W; }

    using base_t = types::get_bool_simd_register_t<T, W>;
    using A = typename base_t::arch_t;
    using arch_t = A;
    using self_t = VecBool;
    using scalar_t = bool;
    using register_t = typename base_t::register_t;
    using vec_t = Vec<T, W>;

    VecBool() = default;
    VecBool(bool val) noexcept
        : base_t(make_register(detail::make_index_sequence<size() - 1>(), val))
    {
    }
    VecBool(register_t reg) noexcept
        : base_t({reg})
    {
    }
    template <typename... Ts>
    VecBool(bool val0, bool val1, Ts... vals) noexcept
        : self_t(kernel::set<A>(self_t{}, A{}, val0, val1, static_cast<bool>(vals)...))
    {
        static_assert(sizeof...(Ts) + 2 == size,
            "constructor requires as many as arguments as vector elements");
    }

    template <typename Tp>
    VecBool(const Tp* ptr) = delete;

    void store_aligned(bool* mem) const noexcept {
        kernel::store(*this, mem, A{});
    }
    void store_unaligned(bool* mem) const noexcept {
        store_aligned(mem);
    }
    static VecBool load_aligned(const bool* mem) noexcept {
        vec_t ref(0);
        alignas(A::argument()) T buffer[size()];
        for (auto i = 0; i < size(); i++) {
            buffer[i] = mem[0] ? 1 : 0;
        }
        return ref != vec_t::load_aligned(&buffer[0]);
    }
    static VecBool load_unaligned(const bool* mem) noexcept {
        return load_aligned(mem);
    }

    bool get(size_t idx) const noexcept {
        return kernel::get(*this, idx, A{});
    }

    /// mask operators
    uint64_t mask() const noexcept {
        return kernel::mask(*this, A{});
    }
    static VecBool from_mask(uint64_t mask) noexcept {
        return kernel::from_mask(self_t(), mask, A{});
    }

    /// comparison operators
    VecBool operator ==(const VecBool& other) const noexcept {
        return kernel::eq<A>(*this, other, A{}).data;
    }
    VecBool operator !=(const VecBool& other) const noexcept {
        return kernel::ne<A>(*this, other, A{}).data;
    }

    /// logical operators
    VecBool operator ~() const noexcept {
        return kernel::bitwise_not<A>(*this, A{}).data;
    }
    VecBool operator !() const noexcept {
        return operator ==(self_t(false));
    }
    VecBool operator &(const VecBool& other) const noexcept {
        return kernel::bitwise_and<A>(*this, other, A{}).data;
    }
    VecBool operator |(const VecBool& other) const noexcept {
        return kernel::bitwise_or<A>(*this, other, A{}).data;
    }
    VecBool operator ^(const VecBool& other) const noexcept {
        return kernel::bitwise_xor<A>(*this, other, A{}).data;
    }
    VecBool operator &&(const VecBool& other) const noexcept {
        return operator &(other);
    }
    VecBool operator ||(const VecBool& other) const noexcept {
        return operator |(other);
    }

    /// in-place update operators
    VecBool& operator &=(const VecBool& other) noexcept {
        return (*this) = (*this) & other;
    }
    VecBool& operator |=(const VecBool& other) noexcept {
        return (*this) = (*this) | other;
    }
    VecBool& operator ^=(const VecBool& other) noexcept {
        return (*this) = (*this) ^ other;
    }

private:
    template <typename U, typename... V, size_t I, size_t... Is>
    static register_t make_register(detail::index_sequence<I, Is...>, U u, V... V) noexcept
    {
        return make_register(detail::index_sequence<Is...)(), u, u, v...);
    }

    template <typename... V>
    static register_t make_register(detail::index_sequence<>, V... v) noexcept
    {
        return kernel::set<W>(self_t{}, A{}, v...).reg();
    }
};
#endif

}  // namespace simd

#include "simd/types/vec.tcc"
#include "simd/types/vec_ops.h"
