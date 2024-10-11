#pragma once

#include "simd/types/vec.h"

#include <complex>

namespace simd {

/// An abstraction vector for complex<T>
template <typename T, size_t W>
class Vec<std::complex<T>, W>
{
public:
    using real_vec_t = Vec<T, W>;
    using imag_vec_t = Vec<T, W>;

    using self_t = Vec<std::complex<T>, W>;
    using value_type = std::complex<T>;

    /// always Generic, no native arch to support this complex<T> type
    using A = Generic;
    using arch_t = A;
    using scalar_t = T;
    using register_t = typename Vec<T, W>::register_t;
    using vec_bool_t = VecBool<T, W>;

    /// query the number of elements of this vector
    /// equavalent as `W`
    /// compile-time const expression
    static constexpr size_t size() { return W; }

    /// query the name string represented for this vector
    /// for example: vcf32x4 (element type: float, 4 elements)
    /// compile-time const expression
    static constexpr const char* type() {
        return traits::vec_type_traits<value_type, W>::type();
    }
    /// query the name string represented for underlying arch/reg backed for this vector
    /// compile-time const expression
    static constexpr const char* arch_name() {
        return A::name();
    }

    /// how many registers for this vector (real or imag)
    static constexpr size_t n_regs() {
        return sizeof(scalar_t) * W / sizeof(register_t);
    }
    /// how many lanes per register
    static constexpr size_t reg_lanes() {
        return W / n_regs();
    }
private:
    /// store real and imag in separate vectors
    real_vec_t real_;
    imag_vec_t imag_;

public:
    // create a vector initialized with undefined values
    SIMD_INLINE
    Vec() noexcept = default;

    /// initialize all elements with same value `val`
    SIMD_INLINE
    Vec(const value_type& val) noexcept;

    SIMD_INLINE
    Vec(const real_vec_t& real, const imag_vec_t& imag) noexcept;

    SIMD_INLINE
    Vec(const real_vec_t& real) noexcept;

    /// initialize all elements with same value (val, 0)
    SIMD_INLINE
    Vec(const T& val) noexcept;

    template <typename... Ts>
    SIMD_INLINE
    Vec(const value_type& val0, const value_type& val1, Ts... vals) noexcept;

    template <typename... Regs>
    SIMD_INLINE
    Vec(const register_t& arg, Regs&&... others) noexcept;

    /// generate values for each slot through generator,
    /// which must satisfy below operation:
    /// `T operator ()(int idx);`
    template <typename G,
     REQUIRES((std::is_convertible<
                decltype(std::declval<G>()(0)),
                value_type
            >::value))
    >
    SIMD_INLINE
    Vec(G&& generator) noexcept {
        gen_values(std::forward<G>(generator));
    }

    SIMD_INLINE
    explicit Vec(const vec_bool_t& b) noexcept;

    /// set all elements to (0,0)
    SIMD_INLINE
    void clear() noexcept;

    template <typename U>
    SIMD_INLINE
    static Vec broadcast(U val) noexcept {
        return Vec(static_cast<value_type>(val));
    }

    /// load
    SIMD_INLINE
    static Vec load_aligned(const T* real, const T* imag = nullptr) noexcept;

    SIMD_INLINE
    static Vec load_unaligned(const T* real, const T* imag = nullptr) noexcept;

    template <typename U>
    SIMD_INLINE
    static Vec load(const U* mem, aligned_mode) noexcept;

    template <typename U>
    SIMD_INLINE
    static Vec load(const U* mem, unaligned_mode) noexcept;

    SIMD_INLINE
    static Vec load_aligned(const value_type* src) noexcept {
        return load(src, aligned_mode{});
    }

    SIMD_INLINE
    static Vec load_unaligned(const value_type* src) noexcept {
        return load(src, unaligned_mode{});
    }

    /// store
    SIMD_INLINE
    void store_aligned(T* real, T* imag) const noexcept;

    SIMD_INLINE
    void store_unaligned(T* real, T* imag) const noexcept;

    template <typename U>
    SIMD_INLINE
    void store(U* mem, aligned_mode) const noexcept;

    template <typename U>
    SIMD_INLINE
    void store(U* mem, unaligned_mode) const noexcept;

    SIMD_INLINE
    void store_aligned(value_type* dst) const noexcept {
        store(dst, aligned_mode{});
    }

    SIMD_INLINE
    void store_unaligned(value_type* dst) const noexcept {
        store(dst, unaligned_mode{});
    }

    /// short/convenience functions for load/store
    template <typename U>
    SIMD_INLINE
    void load(const U* mem) noexcept {
        *this = load(mem, aligned_mode{});
    }
    template <typename U>
    SIMD_INLINE
    void loadu(const U* mem) noexcept {
        *this = load(mem, unaligned_mode{});
    }

    SIMD_INLINE
    void load(const T* real_mem, const T* imag_mem = nullptr) noexcept {
        *this = load_aligned(real_mem, imag_mem);
    }
    SIMD_INLINE
    void loadu(const T* real_mem, const T* imag_mem = nullptr) noexcept {
        *this = load_unaligned(real_mem, imag_mem);
    }

    template <typename U>
    SIMD_INLINE
    void store(U* mem) const noexcept {
        store(mem, aligned_mode{});
    }
    template <typename U>
    SIMD_INLINE
    void storeu(U* mem) const noexcept {
        store(mem, unaligned_mode{});
    }
    SIMD_INLINE
    void store(T* real_mem, T* imag_mem) const noexcept {
        store_aligned(real_mem, imag_mem);
    }
    SIMD_INLINE
    void storeu(T* real_mem, T* imag_mem) const noexcept {
        store_unaligned(real_mem, imag_mem);
    }

    SIMD_INLINE
    const real_vec_t& real() const noexcept {
        return real_;
    }

    SIMD_INLINE
    const imag_vec_t& imag() const noexcept {
        return imag_;
    }

    SIMD_INLINE
    real_vec_t& real() noexcept {
        return real_;
    }

    SIMD_INLINE
    imag_vec_t& imag() noexcept {
        return imag_;
    }

    /// comparison operators
    SIMD_INLINE
    friend VecBool<T, W> operator ==(const Vec& lhs, const Vec& rhs)
    {
        return ops::eq<value_type, W>(lhs, rhs);
    }
    SIMD_INLINE
    friend VecBool<T, W> operator !=(const Vec& lhs, const Vec& rhs)
    {
        return ops::ne<value_type, W>(lhs, rhs);
    }
    SIMD_INLINE
    friend VecBool<T, W> operator >=(const Vec& lhs, const Vec& rhs)
    {
        return ops::ge<value_type, W>(lhs, rhs);
    }
    SIMD_INLINE
    friend VecBool<T, W> operator <=(const Vec& lhs, const Vec& rhs)
    {
        return ops::le<value_type, W>(lhs, rhs);
    }
    SIMD_INLINE
    friend VecBool<T, W> operator >(const Vec& lhs, const Vec& rhs)
    {
        return ops::gt<value_type, W>(lhs, rhs);
    }
    SIMD_INLINE
    friend VecBool<T, W> operator <(const Vec& lhs, const Vec& rhs)
    {
        return ops::lt<value_type, W>(lhs, rhs);
    }

    /// in-place update operators
    SIMD_INLINE
    Vec& operator +=(const Vec& other) noexcept {
        return *this = ops::add<value_type, W>(*this, other);
    }
    SIMD_INLINE
    Vec& operator -=(const Vec& other) noexcept {
        return *this = ops::sub<value_type, W>(*this, other);
    }
    SIMD_INLINE
    Vec& operator *=(const Vec& other) noexcept {
        return *this = ops::mul<value_type, W>(*this, other);
    }
    SIMD_INLINE
    Vec& operator /=(const Vec& other) noexcept {
        return *this = ops::div<value_type, W>(*this, other);
    }

    /// no logical bitwise operations
    #if 0
    SIMD_INLINE
    Vec& operator &=(const Vec& other) noexcept {
        return *this = ops::bitwise_and<value_type, W>(*this, other);
    }
    SIMD_INLINE
    Vec& operator |=(const Vec& other) noexcept {
        return *this = ops::bitwise_or<value_type, W>(*this, other);
    }
    SIMD_INLINE
    Vec& operator ^=(const Vec& other) noexcept {
        return *this = ops::bitwise_xor<value_type, W>(*this, other);
    }
    #endif

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

    /// no below 2 unary operations
    #if 0
    /// unary operators
    SIMD_INLINE
    vec_bool_t operator !() const noexcept {
        return ops::eq<value_type, W>(*this, Vec(0));
    }

    /// bitwise not
    SIMD_INLINE
    Vec operator ~() const noexcept;
    #endif

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
        return ops::add<value_type, W>(lhs, rhs);
    }
    SIMD_INLINE
    friend Vec operator -(const Vec& lhs, const Vec& rhs) noexcept
    {
        return ops::sub<value_type, W>(lhs, rhs);
    }
    SIMD_INLINE
    friend Vec operator *(const Vec& lhs, const Vec& rhs) noexcept
    {
        return ops::mul<value_type, W>(lhs, rhs);
    }
    SIMD_INLINE
    friend Vec operator /(const Vec& lhs, const Vec& rhs) noexcept
    {
        return ops::div<value_type, W>(lhs, rhs);
    }

    /// no bitwise logical operations
    #if 0
    SIMD_INLINE
    friend Vec operator &(const Vec& lhs, const Vec& rhs) noexcept
    {
        return ops::bitwise_and<value_type, W>(lhs, rhs);
    }
    SIMD_INLINE
    friend Vec operator |(const Vec& lhs, const Vec& rhs) noexcept
    {
        return ops::bitwise_or<value_type, W>(lhs, rhs);
    }
    SIMD_INLINE
    friend Vec operator ^(const Vec& lhs, const Vec& rhs) noexcept
    {
        return ops::bitwise_xor<value_type, W>(lhs, rhs);
    }
    #endif
private:
    template <typename G>
    SIMD_INLINE
    void gen_values(G&& generator) noexcept;
};

using cf32_t = std::complex<float>;
using cf64_t = std::complex<double>;

using vcf32x16_t = Vec<std::complex<float>, 16>;    // 2 * 512 bits
using vcf32x8_t  = Vec<std::complex<float>, 8>;     // 2 * 256 bits
using vcf32x4_t  = Vec<std::complex<float>, 4>;     // 2 * 128 bits

using vcf64x8_t  = Vec<std::complex<double>, 8>;    // 2 * 512 bits
using vcf64x4_t  = Vec<std::complex<double>, 4>;    // 2 * 256 bits
using vcf64x2_t  = Vec<std::complex<double>, 2>;    // 2 * 128 bits

}  // namespace simd
