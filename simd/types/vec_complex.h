#pragma once

#include "simd/types/vec.h"

#include <complex>

namespace simd {

template <typename T, size_t W>
class Vec<std::complex<T>, W>
    : public types::simd_register<std::complex<T>, W, types::arch_traits_t<std::complex<T>, W>>
{
public:
    using self_t = Vec<std::complex<T>, W>;
    using value_type = std::complex<T>;
    using A = types::arch_traits_t<value_type, W>;
    using arch_t = A;
    using base_t = types::simd_register<value_type, W, arch_t>;
    using scalar_t = value_type;
    using register_t = typename base_t::register_t;
    /// still use complex<T> for mask bool, so that it has same type as vec
    using vec_bool_t = VecBool<value_type, W>;
    using real_vec_t = Vec<T, W>;
    using imag_vec_t = Vec<T, W>;

    /// query the number of elements of this vector
    /// equavalent as `W`
    /// compile-time const expression
    static constexpr size_t size() { return W; }

    /// query the name string represented for this vector
    /// for example: vi32x4 (element type: int32_t, 4 elements)
    /// compile-time const expression
    static constexpr const char* type() {
        return traits::vec_type_traits<value_type, W>::type();
    }
    /// query the name string represented for underlying arch/reg backed for this vector
    /// for example:
    /// if AVX enabled, Vec<std::complex<float>, 8>, 8xfloats fit in AVX YMM register(__m256, 256bits)
    /// if SSE enabled, Vec<std::complex<float>, 8>, 8xfloats fit by 2 SSE XMM reigsters(__m128, 128bits)
    /// if AVX512 enabled:
    ///   Vec<std::complex<float>, 8>, 8xfloats still backed by AVX YMM register(__m256, 256bits)
    ///   Vec<std::complex<float>, 16>, 16xfloats fit in AVX512 ZMM register(__m512, 512bits)
    /// Above, "AVX", "SSE", or "AVX512" returned
    /// compile-time const expression
    static constexpr const char* arch_name() {
        return A::name();
    }

    // create a vector initialized with undefined values
    SIMD_INLINE
    Vec() noexcept = default;
    SIMD_INLINE
    Vec(const value_type& val) noexcept;

    SIMD_INLINE
    Vec(const real_vec_t& real, const imag_vec_t& imag) noexcept;

    SIMD_INLINE
    Vec(const real_vec_t& real) noexcept;

    SIMD_INLINE
    Vec(const T& val) noexcept;

    template <typename... Ts>
    SIMD_INLINE
    Vec(const value_type& val0, const value_type& val1, Ts... vals) noexcept;

    template <typename... Regs>
    SIMD_INLINE
    Vec(const register_t& arg, Regs&&... others) noexcept;

    template <size_t... Ws>
    SIMD_INLINE
    Vec(const Vec<value_type, Ws>&... vecs) noexcept;

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
    real_vec_t real() const noexcept;

    SIMD_INLINE
    imag_vec_t imag() const noexcept;

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

using vcf32x8_t = Vec<std::complex<float>, 8>;     // 512 bits
using vcf32x4_t = Vec<std::complex<float>, 4>;     // 256 bits
using vcf32x2_t = Vec<std::complex<float>, 2>;     // 128 bits

using vcf64x4_t = Vec<std::complex<double>, 4>;    // 512 bits
using vcf64x2_t = Vec<std::complex<double>, 2>;    // 256 bits
using vcf64x1_t = Vec<std::complex<double>, 1>;    // 128 bits

}  // namespace simd
