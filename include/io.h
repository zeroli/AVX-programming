#pragma once

#include "immintrin.h"
#include <cstdint>
#include <ostream>
#include <type_traits>

namespace simd {
struct AVX { enum { alignment = 32 }; };
struct AVX2 { enum { alignment = 32 }; };
struct AVX512 { enum { alignment = 64 }; };

template <typename Scalar, typename Arch>
struct Value;

template <>
struct alignas(AVX::alignment) Value<int32_t, AVX> {
    using scalar_t = int32_t;
    using arch_t = AVX;
    using vec_t = __m256i;
    vec_t val_;

    static constexpr const char* name() {
        return "veci32x8";
    }
    static constexpr const int size() {
        return 8;
    }
    using self_t = Value<int32_t, AVX>;
    Value(vec_t v)
        : val_(v)
    { }

    operator vec_t() const {
        return val_;
    }
};

template <>
struct alignas(AVX::alignment) Value<float, AVX> {
    using scalar_t = float;
    using arch_t = AVX;
    using vec_t = __m256;
    vec_t val_;

    static constexpr const char* name() {
        return "vecf32x8";
    }
    static constexpr const int size() {
        return 8;
    }

    using self_t = Value<float, AVX>;
    Value(vec_t v)
        : val_(v)
    { }

    operator vec_t() const {
        return val_;
    }
};

template <>
struct alignas(AVX::alignment) Value<double, AVX> {
    using scalar_t = double;
    using arch_t = AVX;
    using vec_t = __m256d;
    vec_t val_;

    static constexpr const char* name() {
        return "vecf64x4";
    }
    static constexpr const int size() {
        return 4;
    }
    using self_t = Value<double, AVX>;
    Value(vec_t v)
        : val_(v)
    { }

    operator vec_t() const {
        return val_;
    }
};

template <typename SIMD>
struct SIMDTraits {
    using scalar_t = typename SIMD::scalar_t;
    using vec_t = typename SIMD::vec_t;
    using arch_t = typename SIMD::vec_t;
};

Value<int32_t, AVX> Make(__m256i vec)
{
    return Value<int32_t, AVX>(vec);
}

Value<float, AVX> Make(__m256 vec)
{
    return Value<float, AVX>(vec);
}

Value<double, AVX> Make(__m256d vec)
{
    return Value<double, AVX>(vec);
}

template <typename T, typename Arch>
inline std::ostream& operator <<(std::ostream& os, const Value<T, Arch>& x)
{
    using simd_t = Value<T, Arch>;
    using scalar_t = typename simd_t::scalar_t;
    using vec_t = typename simd_t::vec_t;
    const scalar_t* p = (const scalar_t*)&(const vec_t&)x;
    os << simd_t::name() << "{[0]=" << p[0];
    for (int i = 1; i < simd_t::size(); i++) {
        os << ", [" << i << "]=" << p[i];
    }
    os << "}";
    return os;
}
}  // namespace simd

template <typename VT>
struct is_vec_t : std::false_type { };
template <>
struct is_vec_t<__m256i> : std::true_type { };
template <>
struct is_vec_t<__m256> : std::true_type { };
template <>
struct is_vec_t<__m256d> : std::true_type { };

template <typename VecT,
    std::enable_if_t<is_vec_t<VecT>::value>* = nullptr>
inline std::ostream& operator <<(std::ostream& os, const VecT& x)
{
    return os << simd::Make(x);
}
