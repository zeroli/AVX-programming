#pragma once

#include <cstddef>

#include "simd/types/arch_traits.h"
#include "simd/types/traits.h"
#include "simd/types/vec_ops_fwd.h"
#include "simd/types/integral_only_ops.h"

namespace simd {
template <typename T, size_t W>
class Vec
    : public types::simd_register<T, W, types::arch_traits_t<T, W>>
    , public types::integral_only_ops<T, W>
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
    using vec_bool_t = VecBool<T, W>;

    Vec() noexcept;
    Vec(T val) noexcept;

    template <typename... Ts>
    Vec(T val0, T val1, Ts... vals) noexcept;

    explicit Vec(const vec_bool_t& b) noexcept;

    template <typename... Regs>
    Vec(register_t arg, Regs... others) noexcept;

    template <size_t... Ws>
    Vec(const Vec<T, Ws>&... vecs) noexcept;

    template <typename U>
    static Vec broadcast(U val) noexcept {
        return Vec(static_cast<T>(val));
    }

    template <typename U>
    void store_aligned(U* mem) const noexcept;
    template <typename U>
    void store_unaligned(U* mem) const noexcept;

    template <typename U>
    void store(U* mem, aligned_mode) const noexcept {
        store_aligned(mem);
    }

    template <typename U>
    void store(U* mem, unaligned_mode) const noexcept {
        store_unaligned(mem);
    }

    template <typename U>
    static Vec load_aligned(const U* mem) noexcept;

    template <typename U>
    static Vec load_unaligned(const U* mem) noexcept;

    template <typename U>
    static Vec load(const U* mem, aligned_mode) noexcept {
        return load_aligned(mem);
    }

    template <typename U>
    static Vec load(const U* mem, unaligned_mode) noexcept {
        return load_unaligned(mem);
    }

#if 0
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
    // comparison operators
    friend VecBool<T, W> operator ==(const Vec<T, W>& lhs, const Vec<T, W>& rhs)
    {
        return ops::eq<T, W>(lhs, rhs);
    }
    friend VecBool<T, W> operator !=(const Vec<T, W>& lhs, const Vec<T, W>& rhs)
    {
        return ops::ne<T, W>(lhs, rhs);
    }
    friend VecBool<T, W> operator >=(const Vec<T, W>& lhs, const Vec<T, W>& rhs)
    {
        return ops::ge<T, W>(lhs, rhs);
    }
    friend VecBool<T, W> operator <=(const Vec<T, W>& lhs, const Vec<T, W>& rhs)
    {
        return ops::le<T, W>(lhs, rhs);
    }
    friend VecBool<T, W> operator >(const Vec<T, W>& lhs, const Vec<T, W>& rhs)
    {
        return ops::gt<T, W>(lhs, rhs);
    }
    friend VecBool<T, W> operator <(const Vec<T, W>& lhs, const Vec<T, W>& rhs)
    {
        return ops::lt<T, W>(lhs, rhs);
    }

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
    Vec& operator &=(const Vec& other) noexcept {
        return *this = ops::bitwise_and<T, W>(*this, other);
    }
    Vec& operator |=(const Vec& other) noexcept {
        return *this = ops::bitwise_or<T, W>(*this, other);
    }
    Vec& operator ^=(const Vec& other) noexcept {
        return *this = ops::bitwise_xor<T, W>(*this, other);
    }

    /// increment/decrement operators
    Vec& operator ++() noexcept {
        return operator +=(1);
    }
    Vec& operator --() noexcept {
        return operator -=(1);
    }
    Vec operator ++(int) noexcept {
        self_t copy(*this);
        operator +=(1);
        return copy;
    }
    Vec operator --(int) noexcept {
        self_t copy(*this);
        operator -=(1);
        return copy;
    }

    /// unary operators
    vec_bool_t operator !() const noexcept {
        return ops::eq<T, W>(*this, Vec(0));
    }
    /// bitwise not
    Vec operator ~() const noexcept;
    /// negation
    Vec operator -() const noexcept;
    Vec operator +() const noexcept {
        return *this;
    }

    /// arithmetic operators
    /// defined as friend to enable conversion from scalar to vector
    friend Vec operator +(const Vec& lhs, const Vec& rhs) noexcept
    {
        return ops::add<T, W>(lhs, rhs);
    }
    friend Vec operator -(const Vec& lhs, const Vec& rhs) noexcept
    {
        return ops::sub<T, W>(lhs, rhs);
    }
    friend Vec operator *(const Vec& lhs, const Vec& rhs) noexcept
    {
        return ops::mul<T, W>(lhs, rhs);
    }
    friend Vec operator /(const Vec& lhs, const Vec& rhs) noexcept
    {
        return ops::div<T, W>(lhs, rhs);
    }
    friend Vec operator &(const Vec& lhs, const Vec& rhs) noexcept
    {
        return ops::bitwise_and<T, W>(lhs, rhs);
    }
    friend Vec operator |(const Vec& lhs, const Vec& rhs) noexcept
    {
        return ops::bitwise_or<T, W>(lhs, rhs);
    }
    friend Vec operator ^(const Vec& lhs, const Vec& rhs) noexcept
    {
        return ops::bitwise_xor<T, W>(lhs, rhs);
    }
    friend Vec operator &&(const Vec& lhs, const Vec& rhs) noexcept
    {
        return ops::logical_and<T, W>(lhs, rhs);
    }
    friend Vec operator ||(const Vec& lhs, const Vec& rhs) noexcept
    {
        return ops::logical_or<T, W>(lhs, rhs);
    }
};

using vi8x64_t  = Vec<int8_t,  64>;    // 512 bits
using vi8x32_t  = Vec<int8_t,  32>;    // 256 bits
using vi8x16_t  = Vec<int8_t,  16>;    // 128 bits
using vu8x64_t  = Vec<uint8_t, 64>;    // 512 bits
using vu8x32_t  = Vec<uint8_t, 32>;    // 256 bits
using vu8x16_t  = Vec<uint8_t, 16>;    // 128 bits

using vi16x32_t = Vec<int16_t,  32>;   // 512 bits
using vi16x16_t = Vec<int16_t,  16>;   // 256 bits
using vi16x8_t  = Vec<int16_t,   8>;   // 128 bits
using vu16x32_t = Vec<uint16_t, 32>;   // 512 bits
using vu16x16_t = Vec<uint16_t, 16>;   // 256 bits
using vu16x8_t  = Vec<uint16_t,  8>;   // 128 bits

using vi32x16_t = Vec<int32_t,  16>;   // 512 bits
using vi32x8_t  = Vec<int32_t,   8>;   // 256 bits
using vi32x4_t  = Vec<int32_t,   4>;   // 128 bits
using vu32x16_t = Vec<uint32_t, 16>;   // 512 bits
using vu32x8_t  = Vec<uint32_t,  8>;   // 256 bits
using vu32x4_t  = Vec<uint32_t,  4>;   // 128 bits

using vi64x8_t  = Vec<int64_t,  8>;    // 512 bits
using vi64x4_t  = Vec<int64_t,  4>;    // 256 bits
using vi64x2_t  = Vec<int64_t,  2>;    // 128 bits
using vu64x8_t  = Vec<uint64_t, 8>;    // 512 bits
using vu64x4_t  = Vec<uint64_t, 4>;    // 256 bits
using vu64x2_t  = Vec<uint64_t, 2>;    // 128 bits

using vf32x16_t = Vec<float, 16>;      // 512 bits
using vf32x8_t  = Vec<float,  8>;      // 256 bits
using vf32x4_t  = Vec<float,  4>;      // 128 bits

using vf64x8_t  = Vec<double, 8>;      // 512 bits
using vf64x4_t  = Vec<double, 4>;      // 256 bits
using vf64x2_t  = Vec<double, 2>;      // 128 bits

template <typename T, size_t W>
class VecBool
    : public types::get_bool_simd_register_t<T, W, types::arch_traits_t<T, W>>
{
public:
    static constexpr size_t size() { return W; }

    using A = types::arch_traits_t<T, W>;
    using arch_t = A;
    using base_t = types::get_bool_simd_register_t<T, W, A>;
    using self_t = VecBool;
    using scalar_t = bool;
    using register_t = typename base_t::register_t;
    using vec_t = Vec<T, W>;

    VecBool() = default;

    VecBool(bool val) noexcept;

    template <typename... Regs>
    VecBool(register_t arg, Regs... others) noexcept;

    template <typename... Ts>
    VecBool(bool val0, bool val1, Ts... vals) noexcept;

    template <typename Tp>
    VecBool(const Tp* ptr) = delete;

#if 0
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

    /// mask operators
    uint64_t mask() const noexcept {
        return kernel::mask(*this, A{});
    }
    static VecBool from_mask(uint64_t mask) noexcept {
        return kernel::from_mask(self_t(), mask, A{});
    }
#endif

    /// comparison operators
    VecBool operator ==(const VecBool& other) const noexcept;
    VecBool operator !=(const VecBool& other) const noexcept;

    /// logical operators
    VecBool operator ~() const noexcept;
    VecBool operator !() const noexcept {
        return operator ==(self_t(false));
    }
    VecBool operator &(const VecBool& other) const noexcept;
    VecBool operator |(const VecBool& other) const noexcept;
    VecBool operator ^(const VecBool& other) const noexcept;

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
    static register_t make_register(detail::index_sequence<I, Is...>, U u, V... v) noexcept
    {
        return make_register(detail::index_sequence<Is...>(), u, u, v...);
    }

    template <typename... V>
    static register_t make_register(detail::index_sequence<>, V... v) noexcept;
};

}  // namespace simd

#include "simd/types/vec.tcc"
#include "simd/types/vec_ops.h"
