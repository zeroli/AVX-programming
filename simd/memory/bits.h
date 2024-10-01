#pragma once

#include "simd/config/inline.h"
#include "simd/types/traits.h"

#include <cstring>
#include <cstdint>

namespace simd {
namespace bits {
namespace detail {
template <typename TO, typename FROM>
struct cast {
    SIMD_INLINE
    static TO apply(FROM from) noexcept
    {
        static_assert(sizeof(FROM) == sizeof(TO),
            "bit cast from `FROM` to `TO` must be same size");
        TO to{};
        std::memcpy(&to, &from, sizeof(to));
        return to;
    }
};
template <typename T>
struct cast<T, T> {
    SIMD_INLINE
    static T apply(T from) noexcept { return from; }
};
}  // namespace detail

template <typename TO, typename FROM>
SIMD_INLINE
TO cast(FROM from) noexcept {
    return detail::cast<TO, FROM>::apply(from);
}

/// make one T value with bits pattern:
/// 111....111
template <typename T>
SIMD_INLINE
T ones()
{
    union {
        char b[sizeof(T)];
        T t;
    };
    #pragma unroll
    for (int i = 0; i < sizeof(T); i++) {
        b[i] = 0xFF;
    }
    return t;
}

/// make one T value with bits pattern:
/// 0000....000
template <typename T>
SIMD_INLINE
T zeros()
{
    return T{};
}

/// make one T value with bits pattern:
/// 1000....000
template <typename T>
SIMD_INLINE
T one_zeros()
{
    union {
        char b[sizeof(T)];
        T t;
    };
    t = T{};
    b[sizeof(T) - 1] |= 0x80;
    return t;
}

template <typename T>
SIMD_INLINE
T signmask()
{
    return one_zeros<T>();
}

/// extend bit to full bits for T
/// 1 => 1111...1
/// 0 => 0000...0
template <typename T>
SIMD_INLINE
T extend(bool lsb)
{
    return lsb ? ones<T>() : zeros<T>();
}

/// return bit state at msb: true for 1, false for 0
template <typename T>
SIMD_INLINE
bool at_msb(T d)
{
    union {
        char b[sizeof(T)];
        T t;
    };
    t = d;
    return (b[sizeof(T) - 1] & 0x80) != 0;
}

SIMD_INLINE
int count1(uint64_t x)
{
    int cnt = 0;
    while (x) {
        cnt++;
        x &= (x - 1);
    }
    return cnt;
}

/// scalar bitwise operations
template <typename T>
SIMD_INLINE
T bitwise_and(T x, T y)
{
    return cast<T>(
        cast<traits::to_integral_t<T>>(x) &
        cast<traits::to_integral_t<T>>(y)
    );
}

template <typename T>
SIMD_INLINE
T bitwise_or(T x, T y)
{
    return cast<T>(
        cast<traits::to_integral_t<T>>(x) |
        cast<traits::to_integral_t<T>>(y)
    );
}

template <typename T>
SIMD_INLINE
T bitwise_xor(T x, T y)
{
    return cast<T>(
        cast<traits::to_integral_t<T>>(x) ^
        cast<traits::to_integral_t<T>>(y)
    );
}

template <typename T>
SIMD_INLINE
T bitwise_not(T x)
{
    return cast<T>(
        ~cast<traits::to_integral_t<T>>(x)
    );
}

template <typename T>
SIMD_INLINE
T bitwise_andnot(T x, T y)
{
    return cast<T>(
        (~cast<traits::to_integral_t<T>>(x)) &
          cast<traits::to_integral_t<T>>(y)
    );
}

template <typename T>
SIMD_INLINE
T bitwise_lshift(T x, int32_t y)
{
    /// cast to `to_integral<T>` to prevent promotion of operator `<<`
    return cast<T>(
        (traits::to_integral_t<T>)(cast<traits::to_integral_t<T>>(x) << y)
    );
}

template <typename T>
SIMD_INLINE
T bitwise_rshift(T x, int32_t y)
{
    /// cast to `to_integral<T>` to prevent promotion of operator `<<`
    return cast<T>(
        (traits::to_integral_t<T>)(cast<traits::to_integral_t<T>>(x) >> y)
    );
}

}  // namespace bits

template <typename TO, typename FROM>
SIMD_INLINE
TO bits_cast(FROM from)
{
    return bits::cast<TO>(from);
}

}  // namespace simd
