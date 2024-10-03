#pragma once

#include <cstddef>

#include "simd/config/inline.h"
#include "simd/memory/alignment.h"
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
    using A = types::arch_traits_t<T, W>;
    using arch_t = A;
    using base_t = types::simd_register<T, W, arch_t>;
    using self_t = Vec;
    using scalar_t = T;
    using register_t = typename base_t::register_t;
    using vec_bool_t = VecBool<T, W>;

    /// query the number of elements of this vector
    /// equavalent as `W`
    /// compile-time const expression
    static constexpr size_t size() { return W; }

    /// query the name string represented for this vector
    /// for example: vi32x4 (element type: int32_t, 4 elements)
    /// compile-time const expression
    static constexpr const char* type() {
        return traits::vec_type_traits<T, W>::type();
    }
    /// query the name string represented for underlying arch/reg backed for this vector
    /// for example:
    /// if AVX enabled, Vec<float, 8>, 8xfloats fit in AVX YMM register(__m256, 256bits)
    /// if SSE enabled, Vec<float, 8>, 8xfloats fit by 2 SSE XMM reigsters(__m128, 128bits)
    /// if AVX512F enabled, Vec<float, 8>, 8xfloats still backed by AVX YMM register(__m256, 256bits)
    /// Above, "AVX", "SSE", or "AVX512F" returned
    /// compile-time const expression
    static constexpr const char* arch_name() {
        return A::name();
    }

    SIMD_INLINE
    Vec() noexcept = default;
    SIMD_INLINE
    Vec(T val) noexcept;

    Vec(const Vec&) noexcept = default;
    Vec& operator =(const Vec&) noexcept = default;
    Vec(Vec&&) noexcept = default;
    Vec& operator =(Vec&&) noexcept = default;

    template <typename... Ts>
    SIMD_INLINE
    Vec(T val0, T val1, Ts... vals) noexcept;

    SIMD_INLINE
    explicit Vec(const vec_bool_t& b) noexcept;

    template <typename... Regs>
    SIMD_INLINE
    Vec(const register_t& arg, Regs&&... others) noexcept;

    template <size_t... Ws>
    SIMD_INLINE
    Vec(const Vec<T, Ws>&... vecs) noexcept;

    /// generate values for each slot through generator,
    /// which must satisfy below operation:
    /// `T operator ()(int idx);`
    template <typename G,
     REQUIRES((std::is_convertible<
                decltype(std::declval<G>()(0)),
                T
            >::value))
    >
    SIMD_INLINE
    Vec(G&& generator) noexcept {
        gen_values(std::forward<G>(generator));
    }

    /// set all elements to 0
    SIMD_INLINE
    void clear() noexcept;

    template <typename U>
    SIMD_INLINE
    static Vec broadcast(U val) noexcept {
        return Vec(static_cast<T>(val));
    }

    /// store/load
    template <typename U>
    SIMD_INLINE
    void store_aligned(U* mem) const noexcept;
    template <typename U>
    SIMD_INLINE
    void store_unaligned(U* mem) const noexcept;

    template <typename U>
    SIMD_INLINE
    void store(U* mem, aligned_mode) const noexcept {
        store_aligned(mem);
    }
    template <typename U>
    SIMD_INLINE
    void store(U* mem, unaligned_mode) const noexcept {
        store_unaligned(mem);
    }

    template <typename U>
    SIMD_INLINE
    static Vec load_aligned(const U* mem) noexcept;
    template <typename U>
    SIMD_INLINE
    static Vec load_unaligned(const U* mem) noexcept;

    template <typename U>
    SIMD_INLINE
    static Vec load(const U* mem, aligned_mode) noexcept {
        return load_aligned(mem);
    }
    template <typename U>
    SIMD_INLINE
    static Vec load(const U* mem, unaligned_mode) noexcept {
        return load_unaligned(mem);
    }

    /// short/convenience functions for load/store
    template <typename U>
    SIMD_INLINE
    void load(const U* mem) noexcept {
        *this = load_aligned(mem);
    }
    template <typename U>
    SIMD_INLINE
    void loadu(const U* mem) noexcept {
        *this = load_unaligned(mem);
    }
    template <typename U>
    SIMD_INLINE
    void store(U* mem) const noexcept {
        store_aligned(mem);
    }
    template <typename U>
    SIMD_INLINE
    void storeu(U* mem) const noexcept {
        store_unaligned(mem);
    }

    template <typename U, typename V>
    SIMD_INLINE
    static Vec gather(const U* src, const Vec<V, W>& index) noexcept;

    template <typename U, typename V>
    SIMD_INLINE
    void scatter(U* dst, const Vec<V, W>& index) const noexcept;

    /// comparison operators
    SIMD_INLINE
    friend VecBool<T, W> operator ==(const Vec<T, W>& lhs, const Vec<T, W>& rhs)
    {
        return ops::eq<T, W>(lhs, rhs);
    }
    SIMD_INLINE
    friend VecBool<T, W> operator !=(const Vec<T, W>& lhs, const Vec<T, W>& rhs)
    {
        return ops::ne<T, W>(lhs, rhs);
    }
    SIMD_INLINE
    friend VecBool<T, W> operator >=(const Vec<T, W>& lhs, const Vec<T, W>& rhs)
    {
        return ops::ge<T, W>(lhs, rhs);
    }
    SIMD_INLINE
    friend VecBool<T, W> operator <=(const Vec<T, W>& lhs, const Vec<T, W>& rhs)
    {
        return ops::le<T, W>(lhs, rhs);
    }
    SIMD_INLINE
    friend VecBool<T, W> operator >(const Vec<T, W>& lhs, const Vec<T, W>& rhs)
    {
        return ops::gt<T, W>(lhs, rhs);
    }
    SIMD_INLINE
    friend VecBool<T, W> operator <(const Vec<T, W>& lhs, const Vec<T, W>& rhs)
    {
        return ops::lt<T, W>(lhs, rhs);
    }

    /// in-place update operators
    SIMD_INLINE
    Vec& operator +=(const Vec& other) noexcept {
        return *this = ops::add<T, W>(*this, other);
    }
    SIMD_INLINE
    Vec& operator -=(const Vec& other) noexcept {
        return *this = ops::sub<T, W>(*this, other);
    }
    SIMD_INLINE
    Vec& operator *=(const Vec& other) noexcept {
        return *this = ops::mul<T, W>(*this, other);
    }
    SIMD_INLINE
    Vec& operator /=(const Vec& other) noexcept {
        return *this = ops::div<T, W>(*this, other);
    }
    SIMD_INLINE
    Vec& operator &=(const Vec& other) noexcept {
        return *this = ops::bitwise_and<T, W>(*this, other);
    }
    SIMD_INLINE
    Vec& operator |=(const Vec& other) noexcept {
        return *this = ops::bitwise_or<T, W>(*this, other);
    }
    SIMD_INLINE
    Vec& operator ^=(const Vec& other) noexcept {
        return *this = ops::bitwise_xor<T, W>(*this, other);
    }

    /// increment/decrement operators
    SIMD_INLINE
    Vec& operator ++() noexcept {
        return operator +=(1);
    }
    SIMD_INLINE
    Vec& operator --() noexcept {
        return operator -=(1);
    }
    SIMD_INLINE
    Vec operator ++(int) noexcept {
        self_t copy(*this);
        operator +=(1);
        return copy;
    }
    SIMD_INLINE
    Vec operator --(int) noexcept {
        self_t copy(*this);
        operator -=(1);
        return copy;
    }

    /// unary operators
    SIMD_INLINE
    vec_bool_t operator !() const noexcept {
        return ops::eq<T, W>(*this, Vec(0));
    }
    /// bitwise not
    SIMD_INLINE
    Vec operator ~() const noexcept;
    /// negation
    SIMD_INLINE
    Vec operator -() const noexcept;
    SIMD_INLINE
    Vec operator +() const noexcept {
        return *this;
    }

    /// arithmetic operators
    /// defined as friend to enable conversion from scalar to vector
    SIMD_INLINE
    friend Vec operator +(const Vec& lhs, const Vec& rhs) noexcept
    {
        return ops::add<T, W>(lhs, rhs);
    }
    SIMD_INLINE
    friend Vec operator -(const Vec& lhs, const Vec& rhs) noexcept
    {
        return ops::sub<T, W>(lhs, rhs);
    }
    SIMD_INLINE
    friend Vec operator *(const Vec& lhs, const Vec& rhs) noexcept
    {
        return ops::mul<T, W>(lhs, rhs);
    }
    SIMD_INLINE
    friend Vec operator /(const Vec& lhs, const Vec& rhs) noexcept
    {
        return ops::div<T, W>(lhs, rhs);
    }
    SIMD_INLINE
    friend Vec operator &(const Vec& lhs, const Vec& rhs) noexcept
    {
        return ops::bitwise_and<T, W>(lhs, rhs);
    }
    SIMD_INLINE
    friend Vec operator |(const Vec& lhs, const Vec& rhs) noexcept
    {
        return ops::bitwise_or<T, W>(lhs, rhs);
    }
    SIMD_INLINE
    friend Vec operator ^(const Vec& lhs, const Vec& rhs) noexcept
    {
        return ops::bitwise_xor<T, W>(lhs, rhs);
    }

private:
    template <typename G>
    SIMD_INLINE
    void gen_values(G&& generator) noexcept;
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

#include "simd/types/vec.tcc"
#include "simd/types/vec_ops.h"
